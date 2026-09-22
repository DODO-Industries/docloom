# DocLoom: Self-Organizing Memory Substrates for LLM Agents
### A Pre-Doctoral Research Proposal

Prepared 2026-08-02. Candidate status: pursuing NET/GATE, expected 2027.
This document is written for that timeline honestly — it is a proposal for
collaboration/mentorship now, building toward a formal PhD/JRF application
once qualification is complete, not a claim of present eligibility.

---

## 1. Who I am and what I'm asking for

I am an independent researcher building DocLoom, an open-source
self-organizing memory substrate for LLM agents (repository: [link]). I am
preparing for NET/GATE (expected 2027) with the goal of pursuing a PhD.

I am not asking to be taken on as JRF today — I understand that requires
formal qualification I don't yet have. What I am asking for: feedback on
the research direction below, and the chance to build a working
relationship with your group over the next year, so that by the time I
qualify, this is a continuation of a known collaboration rather than a
cold application.

---

## 2. Motivation

Memory-augmented LLM agents are one of the most active subfields in AI
right now (see §4). Nearly every published system in 2026 solves ONE piece
of the problem — eviction, or consolidation, or provenance, or
decomposition — in isolation, with a clean positive result. Almost none
combine multiple mechanisms into one working system and ablate it to show
which piece is actually responsible for any improvement, and very few
report negative results honestly. That gap — rigor and integration, not a
new primitive — is where this proposal sits.

---

## 3. Problem statement

**Problem.** Static retrieval (vector search / RAG) cannot update itself
when facts change, and cannot find answers that are only reachable through
an intermediate fact rather than direct similarity. Both are structural
limitations, not tuning issues.

**Gap.** Existing memory-augmented systems address pieces of this
(see §4) but almost never integrate multiple self-organization mechanisms
into one ablated system, and rarely publish honest negative results.

**Hypothesis.** Replacing every fixed, externally-set threshold in a
memory substrate's self-organization (when to reorganize, when to
consolidate, when to trust a consolidated summary, when to forget) with a
signal derived from the substrate's own internal state (coherence,
surprise, faithfulness, usage) improves retrieval quality under
adversarial/interference-dominated growth (contradiction and supersession
streams at scale) — measurably, and attributably via ablation to specific
mechanisms rather than the system as a whole.

**Approach.** Five concrete mechanisms (detailed technical spec available
on request / in repo docs):
1. Entropy-triggered store division (coherence-gated, not count-gated)
2. Surprise-gated consolidation (prediction-error-triggered, not clock-gated)
3. Faithfulness-gated consolidation (inline hallucination check-and-reject)
4. Forgetting ledger with provenance (auditable eviction, not silent delete)
5. Coherence-seeded causal edges (validated-topical, not raw temporal adjacency)

**Evaluation.** LongMemEval (the field's current standard benchmark) plus
a from-scratch ablation: each mechanism on/off, individually and combined,
against named 2026 baselines (not just a generic vector-search baseline).

**Honest limits, stated up front.** A preliminary growth-sweep test
(2026-08-02, 20-Newsgroups corpus, 2k/8k/20k scale) came back NEGATIVE
under benign growth — the store's self-organization did not engage at
that scale, and recall trailed a static FAISS index rather than beating
it. This is reported here because it's the actual reason mechanism #1 is
proposed: the current fixed-count division threshold is diagnosably the
cause, and this proposal exists to fix and re-test that, not to hide that
the first test failed.

---

## 4. Related work (what I've read, and what's actually different)

- **"Language Models Need Sleep"** (2606.03979) — the closest work to
  DocLoom's core metaphor; consolidation via self-modification. Difference
  proposed here: the consolidation *trigger* is surprise-gated rather than
  scheduled, and gated inline by a faithfulness check before shipping.
- **Mem0 / AtomMem** — dynamic consolidation optimizing storage structure
  continuously. Difference: this proposal's division/consolidation
  triggers are derived from coherence/surprise signals specifically, with
  a stated, testable, falsifiable hypothesis and ablation, not just
  "continuously optimized."
- **MAGMA** — orthogonal decomposition into semantic/temporal/causal/entity
  subspaces. Difference: DocLoom's causal edges are proposed to be
  coherence-*seeded* (validated by the same gate as store membership)
  rather than a separately-decomposed subspace.
- **Memory-R1** — RL-learned policy for memory operations. Difference:
  this proposal is deliberately RL-free and LLM-call-free for the
  self-organization signals themselves (interpretable, cheap, no training
  loop) — closer in spirit to MemDecay's "training-free interpretable
  eviction" result, applied to a full memory substrate rather than only a
  KV-cache.
- **Hakuya / "Auditing Forgetting in Limited Memory Language Models"** —
  already do provenance-on-forgetting and per-fact audit. Difference:
  this proposal's forgetting ledger is tied specifically to the
  consolidation/absorption event (what a forgotten fact was subsumed
  into, if anything), not just deletion tracking in isolation.

I am citing these because I read them before writing this, not around
them — if any of the above already fully subsumes one of these five
mechanisms, I would want to know that before spending time re-deriving it.

---

## 5. What already exists (not a proposal — working code, today)

- Full open-source implementation: append-only storage engine, ACT-R-style
  activation decay, causal-graph spreading activation, LLM-driven
  consolidation, a faithfulness judge, and a re-runnable benchmark suite.
- Measured, reproducible results against FAISS (exact + HNSW) on identical
  corpora/queries: knowledge-updating current-fact@1 1.00 vs FAISS's 0.15;
  multi-hop answer@5 on cosine-unreachable bridge queries 0.56 vs FAISS's
  0.00.
- A negative result from today's own growth-sweep test, reported honestly
  (§3), which directly motivates mechanism #1 above.

---

## 6. Proposed timeline

- **Now – early 2027**: implement and ablate the five mechanisms above,
  one at a time, each with a re-run of the relevant benchmark; move
  evaluation from the homemade corpus onto LongMemEval; keep this document
  and the repo updated as a running log, shared periodically for feedback.
- **2027**: clear NET/GATE; by this point the working relationship and
  the body of work should already exist rather than starting from zero.
- **Post-qualification**: formal PhD/JRF application, with this proposal
  (updated with real ablation numbers by then) as the research statement.

---

## 7. What I'm asking for, concretely

- Feedback on whether this problem framing holds up, and what I'm missing
  in the related work.
- Whether this is a fit for your group's ongoing work, and whether you'd
  be open to occasional check-ins over the next year as the ablation
  results come in.
- Any papers/preprints you think I should read before going further.

Full repository, benchmark harness, and mechanism specs available on
request or at [repo link].
