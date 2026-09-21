# DocLoom — Full Plan: Memory Substrate → Small-Model Companion

Consolidated plan from this session's work and discussion. Ordered by dependency,
not by importance — earlier phases unblock later ones.

## Phase 0 — Done this session

- `module_AI/` built: LM Studio client, Loom read/write bridge, full
  `prompt → recall → LLM → answer → insert` pipeline, HTTP routes
- Root-cause perf fixes: vectorized O(m²) physics loop + debounced config
  reload → **2.86x** isolated, **~2x** real-world ingest throughput
- Stress-tested at 5,056 shards (clean run), confirmed flat scaling, no leaks
- Confirmed: recall/memory overhead is <0.5% of total latency — LLM generation
  is 100% of the current bottleneck, not Loom

## Phase 1 — Immediate (do today/this week, near-zero cost)

1. ⬜ **Turn off Qwen3 "thinking" mode in LM Studio.** Single biggest latency win
   available right now — measured 284 of 300 tokens were invisible reasoning
   for a trivial question. (In progress — your action, on LM Studio's side.)
2. ⬜ **Trim the prompt context** sent to the model: top-1-2 matches instead of
   top-5, short snippets not full paragraphs. Cuts prompt-processing time.
3. ✅ **DONE — training data logger is live.** Every `/ai/ask` call logs a full
   record to `module_AI/data/training_log/interactions.jsonl`: the question,
   every excited shard (content vector + physics state: activation, energy,
   phase, momentum, stability, resonance, attention), the answer, and whether
   memory was used. Code: `module_AI/training_logger.py`, wired into
   `module_AI/pipeline.py`. Costs one JSON line append per call. Verified with
   real interactions, not just a smoke test.

## Phase 2 — Near-term (1-2 weeks): formalize the substrate as a model input

1. Decide the exact **fixed-shape input** the future small model will consume:
   e.g. `working_latent` vector + top-k shard vectors, concatenated to one
   fixed-size tensor. This is the "translate everything to one point of
   understanding" contract — pin it down before training anything.
2. If multimodal input (photo/audio) matters soon, add the modality-specific
   encoder (e.g. a small image/audio embedding model) that projects into the
   *same* vector space Loom already uses for text. Same pattern CLIP uses.
3. Keep collecting the (loom_state, question, answer) dataset from Phase 1 —
   aim for a few thousand real examples before starting Phase 3.

## Phase 3 — Small-model training (the real ML project, weeks)

1. ✅ **DONE — architecture built and verified, not just designed.** Real,
   runnable PyTorch code in `module_AI/model/`, entirely separate from
   `module_loom/`:
   - `config.py` — `DocLoomModelConfig` (all hyperparameters in one place)
   - `memory_projection.py` — Stage 1 (physics-aware memory tokens)
   - `lora.py` — Stage 4 (growable adapters, `freeze_backbone()` for the
     Phase 3→4 transition)
   - `backbone.py` — Stage 3+5 (transformer blocks + output head)
   - `docloom_model.py` — the assembled model (`forward`, `generate`,
     `add_adapter`, `use_adapter`)
   - `tokenizer.py` — GPT-2 BPE via `tiktoken`
   - `train.py` — the distillation training loop, reads
     `module_AI/data/training_log/interactions.jsonl` directly

   **Verified, real numbers:** 17,771,008 total params (on target).
   Forward pass produces correct shapes. Adapter add/switch confirmed (48
   trainable tensors per adapter set — matches the math: 24 LoRA injection
   points × A/B). **Caught and fixed two real bugs before they could bite:**
   LoRA was freezing base weights unconditionally (would have silently broken
   Phase 3 pretraining — fixed with an explicit `freeze_backbone()` step), and
   the training logger wasn't capturing content vectors Stage 1 needs (fixed
   by re-embedding recalled text in `pipeline.py`, module_AI-only). Ran a real
   training pass on 3 logged interactions: loss dropped 155.7 → 33.1 over 5
   epochs — confirms every part of the chain (recall → memory tokens → model →
   masked loss → backprop → checkpoint) is correctly wired. Checkpoint saved
   to `module_AI/data/checkpoints/docloom_prototype.pt`.
2. ⬜ **Let Phase 1.3 accumulate real data**, then re-run `python -m
   module_AI.model.train` on hundreds/thousands of examples — the 3-example
   run above is a wiring check, not a real training result yet.
3. ⬜ Evaluate against Qwen3 on real queries once meaningfully trained: does
   the small model match answer quality for *grounded* (memory-based)
   questions, at a fraction of the latency? This is the actual "beat the big
   model" test — on speed/cost for grounded answers, not general trivia.
4. ⬜ Iterate on input format/model size based on real eval results.

## Model Architecture — detailed layer spec (the main design, Phase 3's core)

Decoder-only Transformer, trained from scratch — genuinely our own architecture,
not a fine-tune. Built as **stages**, each stage containing multiple real
layers underneath (a "6-stage plan" is not "6 layers" — spelled out here so
the numbers are honest, not a rough sketch).

### Stage 0 — Input encoder(s), pluggable
Text encoder only today (the existing embedding pipeline, already built and
reused as-is). Fixed interface (`modality → vector`) so photo/audio encoders
slot in later without touching anything downstream.

### Stage 1 — Loom memory projection (2-3 real layers)
Turns Loom's compressed state into "memory tokens" the backbone can read.
**Input is not just shard content vectors — it includes each shard's physics
state:** activation level, energy, Kuramoto phase, momentum, stability, hit
count, concatenated onto that shard's content vector before projection. This
is a genuine, concrete advantage over plain-text RAG systems, which only pass
retrieved text and throw away everything about *how alive* that memory
currently is. The model sees "relevant and currently highly active" versus
"relevant but fading" — information standard retrieval-augmented models never
get. Caught late in design — make sure it's in the first prototype, not an
afterthought bolted on later.
- Linear layer: Loom's vector dimension (content + physics fields) → model's
  embedding dimension
- Activation (GELU)
- Second linear layer: refine into final memory-token shape
- *Future extension*: if a variable number of excited shards needs folding
  into a fixed number of memory tokens, add a small attention-pooling layer
  here. Start without it — only add if the fixed-count version proves rigid.

### Stage 2 — Router/switch (0-1 layers)
Decides which adapter set (Stage 4) should activate.
- **Start free:** reuse Loom's existing atlas/crystal routing directly — zero
  trainable parameters, the decision already exists.
- **Upgrade path:** only if that proves imprecise once real data exists,
  replace with a learned router (1 linear + softmax gate, Mixture-of-Experts
  style). Prove the free version insufficient before building the paid one.

### Stage 3 — Backbone (the real depth: ~16 sub-layers to start)
Standard decoder-only transformer blocks — this is where "8 layers" means
**8 blocks**, each with 2 sub-layers (self-attention + feedforward, each with
residual + norm) = ~16 actual sub-layers. Starting hyperparameters (tune once
training data exists):
- 6-8 transformer blocks
- Embedding dim ~256-384
- 4-6 attention heads
- Feedforward dim ~4x embedding dim
- Reuse an existing small tokenizer (e.g. GPT-2's BPE) — don't build one

### Stage 4 — Adapters, growable (LoRA-style, ~16 injection points per set)
Not a separate sequential stage — small trainable matrices injected **inside**
Stage 3's blocks (attention and/or feedforward weights). One adapter *set* =
one domain/skill; applying to all 8 blocks means up to 16 injection points
per set — this is why "adding an adapter" adds ~16 small matrices, not 1.
- **Growth trigger:** Phase 6's self-assessment flags a wrong/corrected
  answer → becomes training data for that domain's adapter set → retrained
  offline, periodically (never live on-device)
- **Growth control:** consolidation/pruning (Phase 4 below) is mandatory,
  or "grows as needed" quietly breaks the small-model goal

### Stage 5 — Output head (1 layer)
Linear projection to vocabulary size + softmax.

### Stage 6 — Self-assessment hook (0 layers, logging only)
Logs (input, output, confidence, later corrected-or-not) back into Loom as a
shard — feeds Stage 4's adapter growth trigger. See Phase 6 below.

### Honest total, first prototype
Stage 1 (2-3) + Stage 2 (0-1) + Stage 3 (~16) + Stage 5 (1) ≈ **~20 real
layers**, before any adapters — each adapter set then adds ~16 more small
injection points on top.

## Scaling study — 30M through 3B, honestly staged by what's actually feasible

Testing across sizes is real, standard ML practice (it's how you prove a
mechanism's benefit is real, not a fluke of one size) — but "train 3B on a
single RTX 3050" is not physically feasible, so the plan is split by what
each size actually requires:

| Size | Method | Hardware | What it proves |
|---|---|---|---|
| 30M | Full training from scratch | RTX 3050 | Does the memory-token mechanism work at all |
| 100M | Full training from scratch | RTX 3050 | Does the benefit hold as size grows |
| 350M | Full training from scratch | RTX 3050 | Upper end of what's practical to fully own-train here |
| 1B-3B | **LoRA fine-tune onto an existing open pretrained model** (e.g. Qwen2.5-3B, Llama-3.2-3B) — not full pretraining | RTX 3050 (LoRA is far cheaper than full training) | Does the Loom-reading mechanism still help when attached to a much larger, already-capable base |

The 30M/100M/350M points are fully ours end-to-end. The 1B-3B point tests the
*mechanism* (memory tokens + router + adapters) bolted onto someone else's
pretrained weights via LoRA — a different, still legitimate way to validate
at that scale without needing a data center.

## Is this genuinely "our own architecture" — like DeepSeek, ChatGPT?

Yes, with one honest caveat worth stating plainly (so the claim holds up under
scrutiny, not just internally):

- **Not novel:** the base transformer block math (attention, feedforward,
  residuals). Nobody — not DeepSeek, not OpenAI, not us — invented this; it's
  shared, published, foundational math every LLM uses.
- **What actually makes an architecture "someone's own":** the specific,
  novel combination layered on top. DeepSeek's claim rests on Multi-head
  Latent Attention + their MoE design + training recipe. GPT's rests on scale
  + RLHF + engineering. **Ours rests on:** Loom's physics-based memory
  substrate feeding the model as activation-aware memory tokens (not just
  retrieved text — actual energy/phase/momentum state), crystal-routed
  adapters, and growable continual learning from self-assessed mistakes.
- This specific combination does not exist elsewhere. That is the real,
  defensible "own architecture" claim — state it this precisely, not as "we
  invented transformers," which would not hold up and isn't necessary anyway.

## Phase 4 — Adapters for continual learning (after Phase 3 has a working model)

1. Add LoRA-style adapters for **skills/behavior** that retrieval can't teach
   (Loom handles facts; adapters handle "how to reason/respond").
2. **Reuse Loom's atlas routing to pick the adapter** — one adapter per
   crystal/domain. When Loom routes a query to a crystal, that's also the
   adapter-selection decision. Don't build separate adapter-routing logic —
   the routing already exists.
3. Adapter updates happen **offline/periodically**, not live on the chip
   (training needs a backward pass — too expensive for real-time edge
   inference). Push updated adapters to the device like a model update.
4. Add **adapter consolidation** — merge/prune old adapters periodically, the
   same way Loom's own "sleep" cycle decays/compresses old memory. Without
   this, the model stops being small over time. Non-negotiable if the goal is
   staying edge-deployable.

## Phase 5 — Deployment target (pick based on actual use case)

- **Edge companion** (tiny chip, low power, personal): revisit the existing
  Phase-3-hardware-milestone roadmap item (`libdocloom`, native C++/Rust port)
  once the model/substrate design is proven in Python — porting a moving
  target is wasted work, prove it first.
- **Data-center scale** (millions of shards, many users/agents): the win here
  is cost/throughput, not tiny hardware. Needs the parallel-writer
  architecture discussed earlier (partition into N independent brain shards,
  fan-out recall across them) since the current single-writer design won't
  scale to that concurrency on its own.

## Phase 6 — Self-assessment layer (last, after the rest works)

Builds on `module_loom/services/cortex/latent_field_cognition/metacognitive_reflection.py`,
which already tracks the system's own internal state (energy, coherence,
pressure) — this phase extends that from "reflecting on internal physics" to
"reflecting on its own past decisions." Deliberately last: it needs Phases 3-4
(a working model + adapters) already in place to have real decisions worth
assessing.

1. **Log the system's own actions as first-class shards**, not just
   perceived facts — "I answered X with Y, using adapter Z, confidence W" goes
   into Loom the same way a fact does. This is the concrete difference from
   Phase 1's plain Q&A logging: it's the system observing *itself*, not the
   world.
2. **Track outcomes against those logged decisions** — did the answer get
   corrected, repeated, ignored? Feed that signal back into
   `metacognitive_reflection.py`'s existing pressure/energy mechanism instead
   of building a parallel system.
3. **Use it to gate, not just report** — e.g. low self-assessed confidence on
   a topic routes to Loom recall + the big model instead of the small model's
   own answer; a repeated pattern of self-corrected mistakes on a
   crystal/domain flags that adapter (Phase 4) for retraining.
4. Keep calling this "self-assessment" / "decision journaling," not
   "self-aware" — same reasoning as before: it needs to be testable
   (did it log the decision, did it use the outcome) to be buildable.

## Explicit non-goals (don't chase these — already ruled out)

- Feeding raw vectors into Qwen3 (or any frozen pretrained model) without
  training — doesn't work, model was never trained to read them.
- Expecting the small model to beat big models on general world knowledge —
  won't happen, and isn't the actual goal (speed/cost on grounded answers is).
- Unbounded adapter growth or live on-device training — breaks the
  small/edge-friendly goal if left unchecked.
- Introducing real Cap'n Proto into storage — current format already gets the
  same zero-copy benefit; revisit only inside the native port (Phase 5).

## Future idea, parked (not yet — revisit once data volume is solved)

**Separate encoder streams instead of one fused memory projection.** Idea:
one sub-layer that only decodes Loom's physics state (activation/energy/
phase/momentum/stability), a second for the temporal/graph part (how a
shard's state evolved — phase coupling, causal-graph connections over time),
a third for plain text, then merge/fuse the three into the final answer —
instead of today's single Linear(135 -> n_embd) that flattens content +
physics into one vector (module_AI/model/memory_projection.py).

Real idea, standard multi-modal pattern (separate encoders, late fusion).
Two blockers before building it:
1. We haven't yet proven the *current*, simple fused physics signal actually
   improves answers over plain-text RAG (zeroed-vs-real physics A/B test,
   still not run). Building specialized capacity to process a signal we
   haven't confirmed matters is premature.
2. The "temporal" part described doesn't exist as stored data yet — Loom
   currently gives a snapshot of physics state at recall time, not a
   history/trajectory per shard. Needs a storage change before it's a model
   change.
3. More sub-layers = more parameters = needs more data, and data volume is
   already this project's binding constraint (see Stage B collapse notes
   above). Don't add capacity before the data problem is solved.

## What to do right now, concretely

1. Turn off Qwen3 thinking mode (you're already doing this).
2. I can wire up the training-data logger (Phase 1.3) next — small, immediate,
   unblocks everything downstream. Say the word and I'll build it.
