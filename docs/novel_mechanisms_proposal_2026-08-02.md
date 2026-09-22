# DocLoom — Novel Mechanisms Proposal: Toward Advanced RAG

2026-08-02. This is the ideation/spec document: what to build and why, not
finished algorithms. Each mechanism below states the gap, the goal, and what
"solved" looks like — the actual formula/algorithm is intentionally left for
you to invent, per the instruction not to just re-ship literature equations.

## The framing: what "advanced RAG" means here

Standard RAG = embed once, index once, retrieve by static cosine similarity,
forever. Every reorganization decision a RAG-like system could make — when
to re-embed, when to consolidate, when to forget, when to trust its own
retrieved set — is either absent, or hardcoded to a schedule/count/config
value picked once at design time.

**The single unifying idea behind everything below: replace every
fixed/arbitrary threshold in the substrate with a signal the substrate
already computes about its own state.** Not "smarter embeddings" or
"bigger index" — self-organization driven by internal state instead of
external configuration. That's the actual "advanced" in advanced RAG here:
the memory watches itself and reorganizes/forgets/self-corrects because of
what it observes, not because a human set `MAX_SIZE=500000` once.

---

## 1. Entropy-triggered division (replaces count-based crystal splitting)

**Gap**: `MAX_CRYSTAL_SIZE` is a flat item-count threshold. A crystal splits
when it gets big, never because it got incoherent. This is *why* the
2026-08-02 growth-sweep showed nothing at 20k items — the store never
divided (`crystals=1` for the entire run), so the "self-organizing brain"
mechanism was never actually exercised by that test.

**Goal**: a crystal should split when it stops being a coherent unit of
knowledge, regardless of how many items it holds. A crystal with 5 tightly
related facts should NEVER split; a crystal with 400 loosely related facts
maybe should.

**What "solved" looks like**: a measurable internal signal (derived from
existing per-member vectors/coherence data already computed for the
assembly-birth coherence gate) that rises as a crystal's members diverge,
with a split triggered when that signal crosses a threshold — tested by
re-running the growth-sweep and confirming crystals actually divide before
20k items, and that division improves (or at minimum doesn't hurt) R@1/R@5
versus the current fixed-threshold baseline.

**Where it plugs in**: `module_loom/services/weaver/weaver_coordinator.py`,
wherever `MAX_CRYSTAL_SIZE` currently gates `_route_for_ingest`/division.

---

## 2. Surprise-gated consolidation (replaces clock/count-based sleep)

**Gap**: `maybe_sleep_cycle` fires on a schedule/count (`CONSOLIDATE_EVERY`),
not because a region of memory actually accumulated something worth
re-thinking. A region that's had nothing surprising happen to it still gets
"slept on" at the same rate as a region under active contradiction.

**Goal**: consolidate a region when it has accumulated enough *prediction
error* — new ingests that don't match what that region "expected" — not on
a fixed cadence. This is the real analog of biological sleep-pressure: you
consolidate more when more unexpected things happened while awake, not on
a timer.

**What "solved" looks like**: a per-region running expectation (some
function of recent members' vectors — the exact form is yours to invent)
against which new ingests are scored for surprise; consolidation triggers
when accumulated surprise crosses a threshold. Testable by comparing
consolidation timing/frequency under a stream of consistent vs.
contradicting facts — the contradicting stream should trigger
consolidation measurably sooner.

**Where it plugs in**: `maybe_sleep_cycle` in `weaver_coordinator.py`,
alongside the existing `PRESSURE_THRESHOLD`/`PRESSURE_DECAY` tuning keys
(there's already a "pressure" concept — this replaces its trigger source).

---

## 3. Faithfulness-gated consolidation loop (closes an existing measurement into a decision)

**Gap**: `eval/faithfulness_judge.py` can already detect that a consolidated
summary hallucinated a claim (measured 2026-07-15: ~3.3% hallucination
rate) — but that number only ever reaches a benchmark report. The actual
`maybe_sleep_cycle` synthesis path never calls it. A bad synthesis ships
identically to a good one.

**Goal**: make the faithfulness judge part of the accept/reject decision
for a synthesis, not just a post-hoc metric.

**What "solved" looks like**: after `_synthesize_semantic_text` produces a
candidate summary, the faithfulness judge scores it inline; below a
threshold, the system either (a) falls back to the deterministic template
synthesis for that crystal, or (b) retries synthesis once with the specific
unsupported claim identified and flagged out. Testable: re-run
`run_faithfulness.py` and confirm the shipped hallucination rate drops
versus the current ungated 3.3%, without giving up the template's known
compression/readability loss.

**Where it plugs in**: `maybe_sleep_cycle`'s synthesis step, wiring in
`FaithfulnessJudge` from `module_loom/services/eval/faithfulness_judge.py`.

---

## 4. Forgetting ledger with provenance (replaces silent deletion)

**Gap**: `enforce_spatiotemporal_decay` deletes a `ram_ledger` entry outright
once its activation decays below `SLEEP_EVICTION_LIMIT`. Nothing records
what was known or why it faded. "Graceful forgetting with provenance" is a
metric the July 2026 research sweep explicitly found NO published
memory-store system currently reports — this is genuinely open ground, not
a gap other systems already closed.

**Goal**: forgetting should be auditable. When a shard is evicted, keep a
cheap trace of what it was and why it went, without keeping the full
content (that would defeat the point of forgetting).

**What "solved" looks like**: an append-only, compact tombstone log
(gist/summary, decay reason, last-recalled time, and — if it was absorbed
into a sleep-cycle consolidation — a pointer to what replaced it) written
at eviction time instead of a bare `del`. Testable via a new metric: given
a query about something that's been forgotten, can the system explain
*that* it once knew this and roughly why it no longer does, instead of
either hallucinating an answer or returning nothing with no explanation.

**Where it plugs in**: `enforce_spatiotemporal_decay` in
`weaver_coordinator.py`, at the `del self.ram_ledger[shard_id]` site.

---

## 5. Coherence-seeded causal edges (connects two mechanisms that don't talk today)

**Gap**: the coherence gate (`ASSEMBLY_COHERENCE_MIN`/
`SLEEP_CRYSTAL_COHERENCE_MIN`, built 2026-07-15) governs crystal/assembly
*membership*. Causal-graph edges (which drive multi-hop recall) are formed
from raw ingest-order adjacency — which is exactly why `CAUSAL_SPREAD_WEIGHT`
ships disabled (`0.0`) by default: sequential-adjacency edges are noise on
shuffled bulk data. These two mechanisms were built independently and never
connected.

**Goal**: causal edges should form (or be weighted more strongly) between
members that the coherence gate has ALREADY validated as topically related
— not just "ingested next to each other in time."

**What "solved" looks like**: edge formation/strengthening keyed off
coherence-gate membership (same crystal/assembly pass) rather than pure
ingest-order adjacency, so that turning `CAUSAL_SPREAD_WEIGHT` on doesn't
reintroduce the noise problem that got it disabled in the first place.
Testable: re-run `run_bridge_multihop.py` with the new edge rule and
`CAUSAL_SPREAD_WEIGHT>0` on shuffled (non-narrative) bulk data, and confirm
the multi-hop win holds without the adjacency-noise penalty that would
show up on a shuffled corpus today.

**Where it plugs in**: wherever causal edges are currently recorded
(`CausalGraphs.record_transitions_batch` per the 2026-07-13 notes) —
add a coherence check as a gate/weight multiplier on edge formation.

---

## 6. (additional) Retrieval-adequacy self-check, native to the storage layer

**Gap**: "agentic RAG" today (retrieve, judge if it's enough, retrieve
again) is implemented as a prompting loop OUTSIDE the memory system — an
LLM decides whether to issue a second retrieval call. That costs a full LLM
round-trip just to decide "was that enough."

**Goal**: push the adequacy check down into the substrate itself, using
signals it already has cheaply (candidate-set coverage/coherence against
the query) instead of requiring an LLM call to decide whether to hop again.

**What "solved" looks like**: after assembling a candidate set for a query,
a cheap internal estimate of "is this set likely sufficient" (using
existing coverage/coherence machinery, not a new LLM call) triggers an
automatic causal-hop expansion when the estimate is low — an agentic loop
that lives in the storage layer, measurably cheaper than an LLM-mediated
retrieve-judge-retrieve loop. Testable: compare answer coverage and latency
against an LLM-judged retrieve-again baseline on the multi-hop benchmark.

---

## 7. (additional) Usage-pattern reorganization ("working set" memory)

**Gap**: every reorganization mechanism above (1-5) reorganizes by CONTENT
(coherence, surprise). None reorganize by ACCESS PATTERN — which facts get
recalled TOGETHER across sessions, regardless of topic similarity. Human
associative memory binds things by co-use (you remember your keys and your
coat together because you use them together, not because they're
semantically similar) — DocLoom's causal graph gets partway there via
co-occurrence, but nothing currently promotes a genuinely cross-topic
"working set" the way an OS promotes a hot page set.

**Goal**: let recall patterns across time — not just content — reshape
what's grouped/kept resident, independent of semantic similarity.

**What "solved" looks like**: a cross-crystal linking mechanism keyed
purely on temporal co-recall frequency (this pair of facts keeps getting
retrieved in the same query/session, even though they're topically
unrelated), tested by a scenario where a user's real query pattern
repeatedly needs two semantically-distant facts together, and the
substrate should get measurably faster/more accurate at serving that pair
over repeated use, the way any cache warms.

---

## Suggested problem statement structure (fill in as you invent solutions)

Use this shape for whichever mechanism(s) you actually build — reviewers
and professors both look for these six pieces, in this order:

1. **Problem**: what does static/standard RAG structurally fail at, stated
   as a concrete failure mode, not a vague "it's not smart enough."
2. **Gap**: why the obvious existing fixes (name the closest real prior
   art per mechanism — e.g. HippoRAG for causal edges, Zep/Graphiti for
   supersession, ACT-R/Generative Agents for activation decay) don't
   already close this. Say precisely what's different about your approach.
3. **Hypothesis**: one falsifiable sentence — "replacing [fixed threshold]
   with [internal-state signal] improves [specific measured quantity]
   under [specific growth/stress condition]."
4. **Approach**: the mechanism (one of 1-7 above, or your own variant).
5. **Evaluation**: the specific re-runnable benchmark and metric that
   would prove or kill the hypothesis (each mechanism above names one).
6. **Honest limits**: state what you tested and what you didn't, and what
   would have to be true for this to fail — a pre-registered kill
   condition is what makes a negative result still count as research
   instead of reading as a failed demo.

## What we're actually solving, in one sentence

**DocLoom's self-organization (division, consolidation, forgetting, and
edge formation) is currently driven by arbitrary, fixed, externally-set
thresholds; replacing each of them with a signal the substrate computes
about its own state — coherence, surprise, faithfulness, and usage — is
the mechanism by which a memory-augmented RAG system could plausibly grow
without degrading, which is the one claim the field has not yet
demonstrated under adversarial/interference-dominated growth.**
