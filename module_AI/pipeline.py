import time
from typing import Any, Dict, Optional

from module_AI.config import AI_RECALL_TOP_K, AI_RECALL_MIN_SIMILARITY, AI_CONTEXT_SNIPPET_CHARS
from module_AI.llm_client import LMStudioClient
from module_AI.memory_bridge import LoomMemory
from module_AI.training_logger import log_interaction

SYSTEM_PROMPT = (
    "You are DocLoom's assistant. You are given context retrieved from a growing "
    "memory base (the Loom). If the context actually answers the question, answer "
    "using it and say the answer came from memory. If the context is empty or "
    "irrelevant, say you found nothing relevant in memory and answer from your own "
    "knowledge instead. Be concise."
)


class CognitivePipeline:
    """User prompt -> Loom recall -> AI (LM Studio) -> answer -> Loom insert."""

    def __init__(self, memory: Optional[LoomMemory] = None, llm: Optional[LMStudioClient] = None):
        self.memory = memory or LoomMemory()
        self.llm = llm or LMStudioClient()

    def ask(self, query: str, top_k: int = AI_RECALL_TOP_K, remember_interaction: bool = True) -> Dict[str, Any]:
        t0 = time.perf_counter()
        # learn=True (not False): every real chat query now reinforces the
        # Loom shards it actually used, via Loom's own activation/decay
        # physics -- personalization for free, no extra ML code.
        matches = self.memory.recall(query, top_k=top_k, learn=True)
        t_recall = time.perf_counter() - t0

        used_memory = bool(matches) and matches[0].get("score", 0.0) >= AI_RECALL_MIN_SIMILARITY

        def _snippet(text: str) -> str:
            text = text or ""
            return text[:AI_CONTEXT_SNIPPET_CHARS] + ("..." if len(text) > AI_CONTEXT_SNIPPET_CHARS else "")

        context_block = "\n".join(
            f"- ({m.get('score', 0):.2f}) {_snippet(m.get('text', ''))}" for m in matches
        ) if matches else "(no relevant memory found)"

        user_prompt = (
            f"Retrieved memory context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            "Answer using the memory context above if it is relevant; otherwise answer "
            "from your own knowledge."
        )

        t1 = time.perf_counter()
        answer = self.llm.chat(SYSTEM_PROMPT, user_prompt)
        t_llm = time.perf_counter() - t1

        shard_id = None
        t2 = time.perf_counter()
        if remember_interaction:
            shard_id = self.memory.remember(
                f"Q: {query}\nA: {answer}",
                metadata={"kind": "interaction", "source": "ai_pipeline"},
            )
        t_insert = time.perf_counter() - t2

        excited_shards = [
            {"shard_id": m.get("shard_id"), "text": m.get("text"), "score": m.get("score"),
             # recall() doesn't return the content vector — re-embed the (short,
             # already-recalled) text rather than touching module_loom's recall
             # signature. Cheap relative to the LLM call, and what Stage 1
             # (module_AI/model/memory_projection.py) needs to train on.
             "vector": self.memory.embed(m.get("text", "")).tolist(),
             **self.memory.get_shard_physics(m.get("shard_id"))}
            for m in matches
        ]
        log_interaction(query, excited_shards, answer, used_memory)

        total = time.perf_counter() - t0
        return {
            "query": query,
            "answer": answer,
            "used_memory": used_memory,
            "sources": [
                {"shard_id": m.get("shard_id"), "text": m.get("text"), "score": m.get("score")}
                for m in matches
            ],
            "new_shard_id": shard_id,
            "timings_ms": {
                "recall": round(t_recall * 1000, 2),
                "llm_inference": round(t_llm * 1000, 2),
                "memory_insert": round(t_insert * 1000, 2),
                "total": round(total * 1000, 2),
            },
        }

    def remember(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self.memory.remember(text, metadata=metadata)
