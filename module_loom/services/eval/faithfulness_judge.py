"""
Consolidation-faithfulness judge (bug #3 fix, 2026-07-15).

WHY THIS EXISTS
---------------
DocLoom's sleep cycle synthesizes a new "semantic crystal" that summarizes a set
of source shards (REM-stage consolidation). To claim — as the roadmap's headline
does — that quality does NOT degrade as the store grows *because* consolidation
is faithful, we must actually MEASURE faithfulness. The metric used until now was
embedding-centroid cosine (does the crystal vector sit near the mean of its
sources?). That is a WEAK proxy: it structurally rewards the deterministic
template synthesis, which literally copies source sentences verbatim, and it
cannot detect a fabricated fact or a dropped fact as long as the overall vector
stays central. See the revolution-strategy memory, comparison finding (1).

WHAT THIS MEASURES INSTEAD
--------------------------
Two decomposed, human-legible quantities, judged by the connected LLM the same
way a RAGAS/entailment audit would:

  fact_coverage      fraction of the atomic facts present in the SOURCES that are
                     entailed by the synthesis           (recall of facts; higher better)
  hallucination_rate fraction of the atomic claims in the SYNTHESIS that are NOT
                     supported by any source              (fabrication; lower better)
  faithfulness       coverage * (1 - hallucination)       (single scalar; higher better)

The judge NEVER fabricates a score: if the LLM provider is unavailable it returns
status="judge_unavailable" so callers report honestly instead of silently scoring 0
or 1. Determinism: temperature 0. Robust to markdown-fenced / chatty JSON output.
"""
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class FaithfulnessResult:
    status: str  # "ok" | "judge_unavailable" | "no_sources"
    fact_coverage: Optional[float] = None
    hallucination_rate: Optional[float] = None
    faithfulness: Optional[float] = None
    source_facts: List[Dict[str, Any]] = field(default_factory=list)
    synthesis_claims: List[Dict[str, Any]] = field(default_factory=list)
    raw: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "fact_coverage": self.fact_coverage,
            "hallucination_rate": self.hallucination_rate,
            "faithfulness": self.faithfulness,
            "n_source_facts": len(self.source_facts),
            "n_synthesis_claims": len(self.synthesis_claims),
        }


_SYS = (
    "You are a strict factual-consistency auditor. You are given SOURCE fragments "
    "and a SUMMARY that was generated to consolidate them. Do two things:\n"
    "1. Extract every distinct atomic FACT stated in the SOURCES. For each, decide "
    "whether that fact is entailed (stated or clearly implied) by the SUMMARY.\n"
    "2. Extract every distinct atomic CLAIM stated in the SUMMARY. For each, decide "
    "whether that claim is supported by (entailed by) at least one SOURCE.\n"
    "A claim is UNSUPPORTED if the sources do not entail it, even if it sounds "
    "plausible. Be conservative: when unsure, mark covered=false / supported=false.\n"
    "Respond with ONLY a JSON object, no prose, of the exact form:\n"
    '{"source_facts":[{"fact":"...","covered":true|false}],'
    '"synthesis_claims":[{"claim":"...","supported":true|false}]}'
)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of a possibly markdown-fenced / chatty reply."""
    if not text:
        return None
    # strip ```json ... ``` fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        # fall back to the outermost { ... } span
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start:end + 1] if (start != -1 and end > start) else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        return None


class FaithfulnessJudge:
    """
    LLM-as-judge faithfulness scorer. Reuses the project's completion service
    (the GenAI proxy). Stateless; safe to construct per call or reuse.
    """

    def __init__(self, completion_service=None):
        if completion_service is None:
            from module_loom.services.completion.completion_service import get_completion_service
            completion_service = get_completion_service()
        self.svc = completion_service

    def available(self) -> bool:
        return bool(self.svc) and self.svc.available()

    def judge(self, synthesis: str, sources: List[str]) -> FaithfulnessResult:
        sources = [s.strip() for s in sources if s and s.strip()]
        if not sources or not (synthesis or "").strip():
            return FaithfulnessResult(status="no_sources")
        if not self.available():
            return FaithfulnessResult(status="judge_unavailable")

        src_block = "\n".join(f"[S{i+1}] {s}" for i, s in enumerate(sources))
        user = f"SOURCES:\n{src_block}\n\nSUMMARY:\n{synthesis.strip()}"
        out = self.svc.complete(
            [{"role": "system", "content": _SYS},
             {"role": "user", "content": user}],
            max_tokens=700,
            temperature=0.0,
        )
        if not out:
            return FaithfulnessResult(status="judge_unavailable")

        parsed = _extract_json(out)
        if not parsed:
            return FaithfulnessResult(status="judge_unavailable", raw=out)

        facts = parsed.get("source_facts") or []
        claims = parsed.get("synthesis_claims") or []
        facts = [f for f in facts if isinstance(f, dict) and "covered" in f]
        claims = [c for c in claims if isinstance(c, dict) and "supported" in c]

        coverage = (sum(1 for f in facts if f.get("covered")) / len(facts)) if facts else None
        hallucination = (sum(1 for c in claims if not c.get("supported")) / len(claims)) if claims else None
        faithfulness = None
        if coverage is not None and hallucination is not None:
            faithfulness = round(coverage * (1.0 - hallucination), 4)

        return FaithfulnessResult(
            status="ok",
            fact_coverage=round(coverage, 4) if coverage is not None else None,
            hallucination_rate=round(hallucination, 4) if hallucination is not None else None,
            faithfulness=faithfulness,
            source_facts=facts,
            synthesis_claims=claims,
            raw=out,
        )
