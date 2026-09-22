# module_AI — DocLoom's custom Loom-native small language model

**Status as of 2026-09-23: infrastructure complete and correct; the model does not yet produce coherent factual answers. This document is a full, honest handoff — what's built, what's proven, what's broken, and exactly what to do next.**

This is not a wrapper around GPT-2/GPT-Neo/any pretrained architecture. Every layer (memory projection, backbone, LoRA adapters, tokenizer usage, chat format) is custom-built for DocLoom, on purpose — see "Why custom architecture" below. The only borrowed component anywhere in this module is `tiktoken`'s GPT-2 **byte-pair tokenizer** (just the text↔integer lookup table, no weights, no architecture).

---

## 1. The big picture

DocLoom (`module_loom/`) is a physics-based memory engine: text becomes a "shard," shards live in binary "crystal" files, and recall uses real physics — gravity, Kuramoto phase coupling, momentum, decay, a causal graph — instead of plain vector-similarity RAG. `module_AI/` is the AI layer that sits on top: a small (17.7M parameter) transformer that reads Loom's memory **including its physics state** (activation, energy, phase, momentum, stability, resonance), not just retrieved text, and answers questions.

The goal, stated by the project owner: build DocLoom's own model architecture from scratch, trained specifically to read Loom's memory dynamics — not a fine-tune of an existing architecture, not weights copied from anywhere.

## 2. Why custom architecture (do not "fix" this by importing a pretrained model)

This was explicitly decided and re-confirmed multiple times during development:
- TinyStories-8M (GPT-Neo architecture: 16 heads, alternating local/global attention, 8 layers) was evaluated as a possible weight-transfer source and **rejected** — copying its weights would mean adopting GPT-Neo's structure, which contradicts the goal.
- TinyStories-8M is used ONLY as a **local text generator**: it writes short stories, and those stories get ingested into Loom through the real `ingest_shard()` physics pipeline (see `module_AI/scripts/generate_tinystories_data.py`). Zero weights or architecture from it end up in `DocLoomModel`.
- SQuAD (public, free QA dataset) is used only as **text data** (context/question/answer triples), never as an architecture or model.

If a future agent is tempted to "just use a pretrained small LM," don't — that's the one constraint this project cannot compromise on, per direct instruction from the project owner.

## 3. Architecture — the 5 stages (see `ROADMAP.md` for the original design doc)

```
Loom recall (module_AI/memory_bridge.py)
    → excited_shards: [{text, vector(128d), activation, energy, phase_angle,
                         momentum, stability, resonance, attention}, ...]
    ↓
Stage 1: MemoryProjection (module_AI/model/memory_projection.py)
    pack_shard_features(): concatenate content_vector(128d) + physics(7d)
    = 135d per shard ("loom_shard_dim") → Linear(135→256) → GELU → Linear(256→256)
    → n_memory_tokens (4) "memory tokens" the backbone reads like any other token
    ↓
Stage 2: Router (module_AI/model/lora.py: set_active_adapter)
    Picks which LoRA adapter set is active (one per domain/crystal). External —
    not a separate network, just which adapter is selected before forward().
    ↓
Stage 3: Backbone (module_AI/model/backbone.py)
    6-layer causal transformer, 4 heads, n_embd=256, block_size=256.
    Standard causal self-attention + MLP per block, all-global attention
    (deliberately NOT GPT-Neo's alternating local/global pattern).
    Attention qkv/proj and MLP fc/proj are LoRALinear-wrapped (see Stage 4).
    ↓
Stage 4: LoRA adapters (module_AI/model/lora.py)
    Growable, one adapter set per domain/crystal (rank=8, alpha=16).
    Base weights trainable by default (Stage A needs this); freeze_base()
    switches to adapter-only training for Stage B (see Section 6 — this
    was a real, critical bug fix).
    ↓
Stage 5: Output head (module_AI/model/backbone.py: Backbone.head)
    Linear(256→50257), weight-tied to the token embedding (standard GPT-2 trick).
```

Total params: **17,771,008** (Stage A) / **17,967,616** (Stage B, +196,608 for one LoRA adapter set). Config: `module_AI/model/config.py` (`DocLoomModelConfig`).

### What makes this genuinely different from RAG (and what's NOT yet proven)

Plain RAG feeds an LLM retrieved **text only**. Loom's Stage 1 feeds the model retrieved text's content vector **plus 7 real physics numbers** (activation/energy/phase/momentum/stability/resonance/attention) as actual input features. This is a real architectural difference, not just RAG with extra steps.

**Honest caveat, unresolved:** we have never run the controlled experiment (physics features zeroed vs. real) to prove this signal actually improves answers over plain-text RAG. It's a genuine, novel design choice; whether it earns its complexity is still unverified. This is a good, cheap experiment for whoever picks this up next.

## 4. The two-stage training curriculum

**Stage A — self-supervised pretraining** (`module_AI/model/pretrain.py`, `pretrain_dataset.py`)
Zero LLM calls. Reads Loom's own stored shards; for each shard, given the preceding N shards as memory-token context (their content vector + physics state), predict the next shard's text verbatim. Teaches basic language modeling AND how to read memory tokens, using data that costs nothing to generate (it's already in Loom).

**Stage B — distillation / QA fine-tuning** (`module_AI/model/train.py`)
Two sequential passes on top of the Stage A checkpoint (NOT one flat mixed dataset — kept separate so the tiny precious real-Loom data doesn't get drowned by SQuAD's much larger volume):
1. **SQuAD pass**: 87,599 (context, question, answer) triples, general "read context, answer" skill. `NEUTRAL_PHYSICS` placeholder values (no real ingestion history for these).
2. **Real Loom/Qwen pass**: real questions asked through the live pipeline, real Loom recall, real physics state, answered by a local Qwen3 model (LM Studio) and logged automatically (`module_AI/training_logger.py`).

Both passes now train **only a small LoRA adapter** (369,408 params) on top of a **frozen backbone + frozen tied embedding** — see Section 6, this was the single most important fix made this session.

## 5. Current state of the checkpoints (as of 2026-09-23)

| File | What it is | Final validation loss |
|---|---|---|
| `module_AI/data/checkpoints/docloom_pretrained_stageA.safetensors` | Stage A, trained on ~54,000 story shards + 1,000 in-memory chat-format examples | **4.94** |
| `module_AI/data/checkpoints/docloom_stageB.safetensors` | Stage B on top of the above: SQuAD pass + 547-example real-Loom pass, frozen backbone | SQuAD: **7.55**, real-Loom: **5.65** |

Both are safetensors (not pickle — see Section 6). Config sits alongside each as `<name>.safetensors.config.json`.

**Training data on disk right now:**
- `module_AI/data/training_log/interactions.jsonl` — 607 real Loom/Qwen examples (post-cleanup; originals backed up as `.bak_before_cleanup` / `.bak_v2` in the same folder — nothing was destroyed, just filtered)
- `module_AI/data/training_log/squad_examples.jsonl` — 87,599 SQuAD examples

## 6. Bugs found and fixed this session (read this before touching training code again)

Each of these was found by direct testing/profiling, not guessed — and each one measurably changed behavior when fixed.

1. **`recall()` key mismatch** — code assumed `"similarity"`, actual key is `"score"`. Fixed in `pipeline.py`.
2. **O(n²) repulsion physics loop** — vectorized with numpy broadcasting in `weaver_coordinator.py`, verified bit-identical scores before/after. ~2.9x speedup.
3. **Config hot-reload overhead** — was doing an `os.stat()` check on every recall call (38k stat calls for 200 shards); debounced to check wall-clock time once/second (`tuning_config.py`).
4. **`batch_size=1` everywhere** — caused noisy gradients AND CPU underuse. Fixed with real batching, length-bucketed batches (sort by length before batching — fixes a real quadratic-attention-cost slowdown: 60→250 token seq_len at batch=16 went 2.36s→23.7s per step), LR warmup+cosine decay, gradient clipping. See `module_AI/model/batching.py`.
5. **`PretrainDataset` only read `crystal_1.loom`** — after the corpus grew and split into multiple crystal files (Loom's own "cellular division"), this silently dropped most of the corpus. Fixed by auto-discovering all `.loom` files in the brain dir.
6. **Stage B always trained from random init**, never loaded the Stage A checkpoint. Fixed by adding `STAGE_A_CKPT` loading in `train.py`. This was the single most impactful fix in an earlier session: cut Stage B's starting loss from 155.3→13.88.
7. **safetensors save crash on weight-tying** — `backbone.tok_emb.weight` and `backbone.head.weight` are the same tensor (weight tying); plain `save_file` on a raw state_dict refuses to save shared tensors. Fixed using `safetensors.torch.save_model`/`load_model` (`module_AI/model/checkpoint.py`), which handle this correctly. **Never switch this back to `save_file`/`load_file` on a raw state_dict.**
8. **SQuAD/real-data imbalance** — 87,599 vs ~100 real examples would drown the precious real signal in one flat mix. Fixed with the two-sequential-passes design in `train.py` (Section 4).
9. **LoRA base weights frozen too early** — original `LoRALinear.__init__` unconditionally froze base weights, which would have silently broken Stage A pretraining (which needs those weights trainable). Fixed: base is trainable by default, `freeze_base()` is an explicit method called only at the Stage A→B transition.
10. **CATASTROPHIC FORGETTING (the big one)** — Stage B was doing full fine-tuning (every parameter, including the frozen-in-theory backbone) at LR 3e-4 over 78,840 SQuAD examples. This **destroyed** the fluent language ability Stage A had already learned — verified by raw unconditional generation before/after: Stage A alone wrote coherent short stories, Stage B's full-fine-tune output degenerated into repeating one word ("memory memory memory..."). **Fix: `train.py` now calls `model.freeze_backbone()` + `model.add_adapter("qa")` + `model.use_adapter("qa")` right after loading the Stage A checkpoint**, so only the small LoRA adapter (369,408 params) + output head + `memory_proj` train. Verify this call is still there before changing `train.py`.
11. **Tied embedding also needed freezing** — freezing only attention/MLP wasn't enough; the tied token embedding (`backbone.tok_emb.weight`, 12.87M params, shared with the output head) was still trainable and drifting, which decalibrated the frozen attention/MLP layers even though their own weights never changed (they were pretrained against a specific embedding space that had since moved). Fixed: `DocLoomModel.freeze_backbone()` now also calls `self.backbone.tok_emb.weight.requires_grad_(False)`.
12. **`load_checkpoint` couldn't load adapter-containing checkpoints** — a freshly-constructed model has no adapter submodules, so `load_model` would fail on missing keys for any checkpoint saved after `add_adapter()` was called. Fixed: `load_checkpoint` (`module_AI/model/checkpoint.py`) now peeks at the saved tensor keys, detects any `adapters.<name>.` prefixes, and calls `add_adapter(name)` + `use_adapter(name)` before loading weights. Backward compatible with adapter-free (Stage A) checkpoints.
13. **`generate_tinystories_data.py` never used the GPU** — hardcoded to CPU even when CUDA was available, would have wasted a GPU entirely on the biggest data-generation lever. Fixed: `device = "cuda" if torch.cuda.is_available() else "cpu"`, model and inputs both moved to `device`.
14. **Real training questions were generated from the WRONG Loom instance** — `grow_grounded_qa_data.py`'s first version sampled source passages from `module_AI/data/stress_brain` (69,000+ shards, used for Stage A training) but asked the question through the live `/ai/ask` server, which recalls from `assets/.brain_data` (only 2,090 shards — the actual "production" brain the live pipeline uses, via `module_loom/routes/visualizer_routes.py:get_coordinator()`). These are two **completely separate Loom instances**. Result: ~50-67% of "grounded" questions couldn't actually be found by recall, teaching the model "ignore memory" on that fraction. **Fixed by pointing the generator at `assets/.brain_data` directly** (`--brain-dir assets/.brain_data`). Hit rate went from ~33-50% (lucky overlap) to ~67% (correct match). **If you build any new data-generation script, always sample source passages from the same brain the live server actually recalls from — check `module_loom/routes/visualizer_routes.py:_brain_dir()` for the real path, don't assume.**

### Data cleanup performed
After finding bug #14, `interactions.jsonl` was filtered to drop any example whose answer contains phrases like "not present in memory", "does not mention", "i found nothing relevant" (i.e., examples where recall found nothing useful — mostly the mismatched-brain artifacts). Original files preserved as `interactions.jsonl.bak_before_cleanup` and `interactions.jsonl.bak_v2` in the same directory. Kept 603→416, then 838→603 after the second corrected generation round. **If you add more real data, consider re-running this same filter** (see the inline filter script logic — search this session's history or reconstruct: drop answers containing those phrases).

## 7. What's proven to work (with evidence, not assumed)

- **Stage A produces genuinely fluent language on its own.** Raw unconditional generation, no memory context, no QA formatting: `"One day a boy named Timmy went to the park. He was so excited that he wanted to play with his toy car..."` — grammatical, consistent characters, real story structure. Verified at multiple temperatures (0.4, 0.6, 0.7), not a sampling fluke.
- **The custom chat format is learned correctly.** `<|user|>`/`<|assistant|>`/`<|end|>` tags (our own invention, `module_AI/model/chat_format.py` — not ChatML, not any library's template) now appear in the right structural positions reliably, which they never did before Section 6 fix #14's predecessor work.
- **Loss numbers track real learning, not memorization.** Held-out validation loss consistently tracks (never diverges wildly from) training loss at every stage — this is the standard signal that the model is generalizing, not just memorizing.
- **The degenerate-repetition failure mode is gone.** Before the freeze fix, output collapsed into repeating one word ("memory memory memory", then "own own own"). After, output is varied (still wrong, but not stuck).

## 8. What's still broken (the honest bottom line)

**The model cannot produce a single coherent, factually correct answer to a question yet — 0% success rate, verified repeatedly, including on near-training-distribution questions** (e.g. "Who is the person seen in the long invalid chair?" — a question style it was directly trained on). Output looks like:
```
<|user|>
What emotion did the little princess feel when the prince entered the room?
<|assistant|>
The context context context. The memory, the context of the old man came from
memory, the valuable. The det of the answer to the recourse to
```
Structurally correct, semantically empty.

**Two most likely causes, neither definitively isolated (see Section 9 for the exact untested experiments):**
1. **Data volume.** 603-607 real examples for a skill this specific (extract an answer from a recalled passage) is very likely still too small — this kind of skill typically needs thousands of examples even in much bigger models.
2. **Adapter capacity.** The LoRA adapter is only 369,408 trainable params (rank 8). It's possible this is simply too small to encode the extraction skill even with enough data. **Never tested** — no run has varied rank while holding data constant.

Do not assume it's "just data" without running experiment 2 below — that conclusion is a reasonable guess, not a proven fact.

## 9. Exact next steps, in priority order

1. **Isolate data vs. capacity** (cheap, ~20-40 min each, do this before anything else):
   - Run A: same 603 real examples, same rank-8 adapter, more epochs (try 20-30 instead of 8) — tests whether it was simply undertrained.
   - Run B: same 603 real examples, bigger adapter rank (try 32 instead of 8, change `lora_rank` in `DocLoomModelConfig`) — tests whether capacity was the ceiling.
   - Change ONE variable at a time. Today's biggest process lesson: confounding variables (changing data AND architecture AND hyperparameters in one run) makes results impossible to interpret. See Section 6's bug list — every fix there was isolated on purpose.
2. **If data is confirmed the bottleneck**: scale `module_AI/scripts/grow_grounded_qa_data.py` up significantly (thousands, not hundreds, of examples) — the pipeline is correct now (bug #14 fixed), it just needs to run longer / with more concurrency. Point it at `assets/.brain_data` (or wherever `_brain_dir()` resolves to — check first, don't assume it hasn't moved).
3. **Run the RAG-vs-physics A/B test** (Section 3's "honest caveat") — zero out the 7 physics fields in `pack_shard_features()` for one training run, compare final quality against a real-physics run. This determines whether Stage 1's core novel idea (physics-aware memory tokens) is actually worth its complexity, which nobody has verified yet.
4. **Parked idea, don't build yet**: separate encoder streams (one for physics, one for temporal/graph history, one for text, late-fused) instead of today's single fused `Linear(135→256)`. Written up with its blockers in `module_AI/ROADMAP.md` under "Future idea, parked" — **do not build this until steps 1-3 above are done**; it adds parameters, which needs more data, and data is already the suspected bottleneck.
5. **If moving to better hardware** (a GPU machine): `pretrain.py` and `train.py` already auto-select CUDA if available (`torch.cuda.is_available()`), no code change needed. `generate_tinystories_data.py` was already fixed to use GPU (bug #13). Before trusting time estimates on new hardware, benchmark for real with a small run (`--count 500` on the generator, or a short `--epochs 1` training run) rather than extrapolating from this session's CPU-only numbers.

## 10. How to run things (exact commands)

All commands run from the DocLoom repo root (`D:\data\persnol\DocLoom`), using the project's venv.

```bash
# Stage A: self-supervised pretraining
.venv/Scripts/python.exe -m module_AI.model.pretrain \
    --brain-dir module_AI/data/stress_brain --limit 90000 --epochs 3 --batch-size 16 --lr 3e-4

# Stage B: distillation (SQuAD pass + real-Loom pass), on top of Stage A checkpoint
.venv/Scripts/python.exe -m module_AI.model.train \
    --squad-epochs 1 --loom-epochs 8 --lr 3e-4 --batch-size 16

# Generate more TinyStories text and ingest into Loom (Stage A corpus growth)
.venv/Scripts/python.exe -m module_AI.scripts.generate_tinystories_data \
    --count 8000 --batch-size 64 --brain-dir module_AI/data/stress_brain

# Grow real Loom/Qwen QA training data (requires LM Studio running locally on 127.0.0.1:1234)
.venv/Scripts/python.exe -m module_AI.scripts.grow_grounded_qa_data \
    --brain-dir assets/.brain_data --sample-size 2000 --workers 8 --time-limit-s 7200

# Rebuild the SQuAD dataset from scratch (downloads train-v1.1.json)
.venv/Scripts/python.exe -m module_AI.scripts.build_squad_dataset

# Test generation quality on the current Stage B checkpoint (uses real Loom recall)
.venv/Scripts/python.exe -m module_AI.scripts.final_gen_check

# Full autonomous overnight pipeline (all of the above, in sequence, fault-tolerant)
.venv/Scripts/python.exe -m module_AI.scripts.run_overnight
```

Live AI server (FastAPI route, real chat pipeline): `module_AI/routes.py` exposes `/ai/ask`, wired into the main app. Requires LM Studio running locally (`127.0.0.1:1234`, OpenAI-compatible endpoint) serving a chat model — this session used `qwen/qwen3-1.7b` with "thinking mode" turned OFF (thinking tokens dominated latency, ~15-30s/response, unrelated to Loom).

## 11. Key files, one line each

| File | Purpose |
|---|---|
| `module_AI/model/config.py` | `DocLoomModelConfig` — all model hyperparameters |
| `module_AI/model/memory_projection.py` | Stage 1 — packs Loom shard + physics into memory tokens |
| `module_AI/model/backbone.py` | Stage 3/5 — transformer blocks + output head |
| `module_AI/model/lora.py` | Stage 4 — growable LoRA adapters, freeze/unfreeze logic |
| `module_AI/model/docloom_model.py` | Assembles all stages into one model; `generate()` |
| `module_AI/model/chat_format.py` | Our own `<|user|>`/`<|assistant|>` tag format |
| `module_AI/model/checkpoint.py` | safetensors save/load, adapter-aware |
| `module_AI/model/batching.py` | Padding, LR schedule, length-bucketed batching |
| `module_AI/model/tokenizer.py` | Thin wrapper around `tiktoken`'s GPT-2 BPE encoding |
| `module_AI/model/pretrain.py` / `pretrain_dataset.py` | Stage A training loop + dataset |
| `module_AI/model/train.py` | Stage B training loop (two-pass) |
| `module_AI/memory_bridge.py` | `LoomMemory` — the bridge between the AI and live Loom |
| `module_AI/pipeline.py` | `CognitivePipeline` — recall → LLM → remember, real chat flow |
| `module_AI/training_logger.py` | Logs every real `/ai/ask` call as free training data |
| `module_AI/llm_client.py` | Thin LM Studio (OpenAI-compatible) HTTP client |
| `module_AI/scripts/generate_tinystories_data.py` | Generates + ingests story text (Stage A corpus growth) |
| `module_AI/scripts/grow_grounded_qa_data.py` | Generates real, memory-grounded QA training examples |
| `module_AI/scripts/build_squad_dataset.py` | Downloads + embeds SQuAD into training-log format |
| `module_AI/scripts/final_gen_check.py` | Loads Stage B checkpoint, tests real generation quality |
| `module_AI/scripts/run_overnight.py` | Fault-tolerant multi-stage autonomous pipeline runner |
| `module_AI/ROADMAP.md` | Original design doc — read this too, it has more context on the 5-stage plan and the parked multi-stream idea |

## 12. Environment notes

- CPU-only development machine used throughout this session: 22 cores, no CUDA GPU. `torch.set_num_threads(os.cpu_count())` set explicitly in `pretrain.py`/`train.py` (default was 16 of 22).
- A second, more powerful machine exists (RTX 3050 4GB / Ryzen 7 8c16t / 16GB RAM) — not yet benchmarked for this project. All CUDA-aware code paths already exist; just needs a CUDA-enabled `torch` install (not the CPU-only pip wheel) to benefit.
- LM Studio must be running locally for any real-data generation (`grow_grounded_qa_data.py`) or live chat (`/ai/ask`) — check `LM_STUDIO_BASE_URL` in `module_AI/config.py` (defaults to `127.0.0.1:1234`).
