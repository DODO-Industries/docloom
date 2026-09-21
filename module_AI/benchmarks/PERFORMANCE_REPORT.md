# DocLoom `module_AI` — Performance & Stress-Test Report

All numbers below are measured, not estimated, unless explicitly marked "extrapolated."
Machine: single dev workstation running the DocLoom server, LM Studio (Qwen3-1.7B), and
the test scripts concurrently unless noted otherwise.

## 1. What was built

`module_AI/` implements the full loop:

```
User prompt → Loom recall (learn=False, read-only) → LM Studio (Qwen3-1.7B)
            → answer (grounded in memory, or model's own knowledge if nothing relevant)
            → Loom insert (new shard)
```

- `llm_client.py` — thin client for LM Studio's OpenAI-compatible `/v1/chat/completions`
- `memory_bridge.py` — embed / recall / remember, wrapping the existing
  `WeaveBrainCoordinator` and embedding service (no new storage layer — reused what exists)
- `pipeline.py` — orchestrates the loop above, returns per-stage timings
- `routes.py` — `POST /ai/ask`, `POST /ai/remember`, `GET /ai/stats`, mounted on the
  running `module_loom` server
- `scripts/bulk_ingest.py` — downloads real public-domain text (15 Project Gutenberg
  books, cached locally), chunks it, and stress-tests ingestion through the real
  production pipeline
- `scripts/peek.py` — zero-overhead read-only snapshot via the existing
  `brain_decoder_service` (no physics/routing computation)

**Correctness verified:** the pipeline correctly answers from memory when relevant
content exists (tested: "Where is the Eiffel Tower?" → grounded answer, memory cited)
and correctly falls back to the model's own knowledge when nothing relevant is stored
(tested: "capital of Australia?" → answered correctly, explicitly noted memory had
nothing on it).

## 2. Root-cause performance fixes

Investigated *why* ingestion was slow rather than guessing — profiled with `cProfile`.

### Fix 1: vectorized the O(m²) repulsion-physics loop
Every single `ingest_shard()` call runs a full "cognitive tick" (the living-memory
physics simulation), which included a nested Python loop computing mutual repulsion
across the working set (capped at 128 shards): ~128×127 ≈ 16,000 individual
`np.linalg.norm()` Python calls per shard ingested. Replaced with one vectorized
numpy broadcast (`weaver_coordinator.py:1171-1198`).

### Fix 2: debounced the tuning-config hot-reload check
`tuning_manager.get_float()`/`get_int()` — called ~200×/shard by the physics code —
did an `os.stat()` filesystem check on *every single call* to support live config
hot-reload. Changed to check wall-clock time at most once/second instead
(`tuning_config.py`), preserving the hot-reload feature while removing ~99% of the
syscalls.

### Measured impact (isolated, `cProfile`, 200 `ingest_shard()` calls, embeddings pre-computed)

| | Before | After Fix 1 | After Fix 1+2 |
|---|---|---|---|
| Wall time (200 calls) | 10.59s | 4.55s | 3.70s |
| Throughput | 18.9/s | 43.9/s | **54.0/s** |
| `np.linalg.norm()` calls | 1,960,006 | 90,766 | 90,766 |

**2.86x** on the isolated cognitive-tick cost. Verified bit-identical output
(same recall scores, same answers) before and after — this is a pure performance
change, not a behavior change.

## 3. Real-world ingestion throughput (end-to-end, clean run)

A clean 5-minute run (no other processes competing for CPU), real text from
*War and Peace* et al., through the full production pipeline (embed → HDC →
atlas routing → physics → append → journal):

| | Before fixes | After fixes (clean) |
|---|---|---|
| Shards ingested | 300 in 36.1s | 5,056 in 302.6s |
| Average throughput | 8.3/s | **16.7/s** |
| Steady-state | ~12.5/s | 15.3 → 16.9/s (rising, then flat) |

**~2x real-world improvement.** Note this is lower than the isolated 54/s figure —
the isolated test pre-computed embeddings and had no corpus-download/disk overhead;
end-to-end throughput also depends on total system load (embedding model, LM Studio
sitting in memory, IDE tooling). 16.7/s is the trustworthy, reproducible number.

**Scaling behavior:** throughput did **not** degrade from shard 512 to shard 5,056 —
flat/stable, as expected, since the cognitive-tick's working set is capped at 128
regardless of total corpus size (by design, see `weaver_coordinator.py:1091-1097`).

**Extrapolated (not measured) ETA for 1,000,000 shards:** ~16.6 hours at 16.7/s,
down from an extrapolated ~33.5 hours pre-fix. A literal 1M-shard run was not
completed in this session (explicit scale-down decision, given ~a day of unattended
runtime was impractical) — the fixes and clean scaling data above are the honest
basis for that projection, not a measured 1M result.

## 4. Recall latency: optimized index vs. naive brute-force

Measured at 5,056 shards (one crystal), comparing the production
`coordinator.recall()` (resonance-jump bucket narrowing) against a naive vectorized
full-corpus scan (`LoomStore.dot_all()` over every shard, no routing/bucketing):

| | Time |
|---|---|
| Optimized (`recall`, `learn=False`) | 9.37ms |
| Naive brute-force scan | 26.75ms |
| **Speedup** | **2.9x** |

This gap is expected to widen substantially at larger scale — the naive scan is
O(n) in total shard count; the optimized path narrows to leader buckets within the
routed crystal and stays roughly flat.

## 5. Memory & storage footprint

Measured continuously during the 5,056-shard clean run:

- **RSS:** stable at 820-910MB throughout — no growth trend, no leak
- **Storage:** ~1KB/shard, linear (5.03MB at 5,056 shards)
- **Crystals:** stayed at 1 (production `MAX_CRYSTAL_SIZE` default is 500,000 — no
  split triggered yet at this scale)

## 6. AI pipeline latency breakdown — the actual overhead question

This is the core hypothesis test: does Loom's memory layer add meaningful overhead
to a prompt→answer round trip? Measured via real `/ai/ask` calls (Qwen3-1.7B, local
LM Studio inference):

| Stage | Measured time |
|---|---|
| Recall (embed query + resonance-jump lookup) | 15-47ms |
| LLM inference (Qwen3-1.7B, local) | 11,846-26,169ms (11.8-26.2s) |
| Memory insert (write new shard) | <1ms |
| **Total round trip** | **~12-26s** |

**Finding (measured, not assumed): Loom's memory layer contributes under 0.5% of
total response latency.** The bottleneck is entirely local LLM inference speed —
recall and insert are already near-instant by comparison. The hypothesis that "a
highly optimized memory connection reduces prompt-to-answer overhead" is true for
the memory-connection component specifically (it was already negligible, and our
fixes made the *storage* side faster too) — but it is not the lever that would
meaningfully change end-to-end latency for this pipeline. The LLM's own inference
speed is.

## 7. Storage architecture note: "Cap'n Proto"

Investigated whether the storage layer should adopt Cap'n Proto (`cotN'proto`) as
requested. Finding: an `atlas.capnp` filename already exists in the codebase, but
it's a legacy naming artifact from an older loose-file boot path — the actual
serialization is a custom struct-based binary format, not real Cap'n Proto. The
current format (`LOM2`, `substrate_layout.py`) already delivers Cap'n Proto's core
benefit — zero-copy mmap reads — via its own append-only log + mmap'd `.idx`
acceleration file. **Recommendation: do not introduce real Cap'n Proto now** — it
would add a schema-compilation dependency to replace something already solved.
Revisit only as part of the Phase 3 `libdocloom` native (C++20/Rust) port on your
roadmap, where a from-scratch storage format is being built anyway.

## 8. What was not done / explicitly out of scope this session

- **Literal 1,000,000-shard ingestion** — descoped to a clean, complete 5,056-shard
  measurement instead, given the real-world time cost (~17 hours at measured
  throughput). The scaling data above supports extrapolation but isn't a substitute
  for actually running it, if you want that number later.
- **Parallel writers** — investigated and explained why the current
  single-writer-per-process design can't safely parallelize `ingest_shard()` without
  a real architecture change (per-crystal locking, or N independent brain partitions
  with fan-out recall). Not implemented — a legitimate future project, not a quick fix.
- **C++/Rust port (`libdocloom`)** — your Phase 3 roadmap item. Confirmed it would
  give a further real speedup (removing Python object/dict overhead, enabling true
  multithreading past the GIL), but is a multi-week rewrite, correctly out of scope
  for this session's fixes.
- **tuning_manager's remaining overhead** beyond the reload-debounce fix (small,
  diminishing-returns items) — not pursued further given time.

## 9. Reproducing these numbers

```bash
# Clean ingestion stress test (adjust --count / --time-limit-s):
python -m module_AI.scripts.bulk_ingest --count 500000 --time-limit-s 300 \
    --batch-size 64 --log-every 500 --checkpoint-every 5000

# Zero-overhead structural snapshot (after a checkpoint or close):
python -m module_AI.scripts.peek --brain-dir module_AI/data/stress_brain

# Live progress while a run is in flight (zero overhead, no file reopening):
tail -f module_AI/benchmarks/ingest_log.jsonl
```
