"""
Large-scale ingestion / stress-test tool for module_loom.

Downloads a handful of large public-domain texts (Project Gutenberg), chunks
them into shard-sized passages, and ingests them through the real
WeaveBrainCoordinator pipeline (embedding -> HDC -> atlas routing -> physics
-> append) via module_AI.memory_bridge.LoomMemory.

Usage:
    python -m module_AI.scripts.bulk_ingest --count 100000 --brain-dir module_AI/data/stress_brain

Writes a per-checkpoint timing log to module_AI/benchmarks/ingest_log.jsonl,
which module_AI/scripts/benchmark.py reads to build the performance report.
"""
import argparse
import gc
import json
import os
import re
import sys
import time
from typing import Iterator, List

import numpy as np
import psutil
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "module_AI", "data", "raw")
BENCH_DIR = os.path.join(PROJECT_ROOT, "module_AI", "benchmarks")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)

# Large public-domain texts (Project Gutenberg). Each entry tries a couple of
# well-known URL patterns since Gutenberg's layout varies by book age.
BOOKS = [
    ("2600", "War and Peace"),
    ("1342", "Pride and Prejudice"),
    ("84", "Frankenstein"),
    ("2701", "Moby Dick"),
    ("100", "Complete Works of Shakespeare"),
    ("8800", "Divine Comedy"),
    ("6130", "The Iliad"),
    ("4300", "Ulysses"),
    ("1661", "Sherlock Holmes"),
    ("11", "Alice in Wonderland"),
    ("345", "Dracula"),
    ("1080", "A Modest Proposal / Swift misc"),
    ("98", "A Tale of Two Cities"),
    ("1080", "Swift"),
    ("46", "A Christmas Carol"),
]

URL_PATTERNS = [
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]


def _download_book(book_id: str, title: str) -> str:
    """Downloads (and caches) one book's plain text. Returns local path, or '' on failure."""
    cache_path = os.path.join(DATA_DIR, f"{book_id}.txt")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        return cache_path
    for pattern in URL_PATTERNS:
        url = pattern.format(id=book_id)
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(cache_path, "wb") as f:
                    f.write(resp.content)
                print(f"  downloaded {title} ({book_id}) — {len(resp.content)/1024:.0f} KB")
                return cache_path
        except requests.RequestException:
            continue
    print(f"  SKIP {title} ({book_id}) — no reachable URL pattern")
    return ""


def _strip_gutenberg_boilerplate(text: str) -> str:
    start = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    end = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start()] if end.start() > 0 else text
    return text


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _chunk_text(text: str, words_per_shard: int = 40) -> Iterator[str]:
    """Splits into sentences, then groups sentences into ~words_per_shard-word passages."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = _SENT_SPLIT.split(text)
    buf: List[str] = []
    word_count = 0
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 3:
            continue
        buf.append(sent)
        word_count += len(sent.split())
        if word_count >= words_per_shard:
            passage = " ".join(buf)
            if len(passage) >= 30:
                yield passage
            buf, word_count = [], 0
    if buf:
        passage = " ".join(buf)
        if len(passage) >= 30:
            yield passage


def gather_corpus(target_count: int) -> Iterator[str]:
    """Yields shard-sized passages from downloaded books, cycling with a suffix
    tag if the real corpus is smaller than target_count (keeps text distinct
    per pass so embeddings/routing aren't degenerate duplicates)."""
    passages: List[str] = []
    print(f"Downloading corpus (target {target_count} shards)...")
    for book_id, title in BOOKS:
        path = _download_book(book_id, title)
        if not path:
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        raw = _strip_gutenberg_boilerplate(raw)
        book_passages = list(_chunk_text(raw))
        passages.extend(book_passages)
        print(f"  {title}: {len(book_passages)} passages (running total {len(passages)})")
        if len(passages) >= target_count:
            break

    if not passages:
        raise RuntimeError("No corpus text could be downloaded — check internet connectivity.")

    n = len(passages)
    print(f"Corpus ready: {n} distinct passages.")
    passnum = 0
    emitted = 0
    while emitted < target_count:
        for p in passages:
            if emitted >= target_count:
                break
            if passnum == 0:
                yield p
            else:
                yield f"{p} [corpus-pass-{passnum}]"
            emitted += 1
        passnum += 1


def _naive_scan_recall(coordinator, query_vec, top_k: int):
    """Brute-force baseline: vectorized dot product against every shard in every
    crystal (LoomStore.dot_all), then a global top-k — no routing, no bucketing.
    Compared against coordinator.recall()'s resonance-jump path to measure what
    the index actually buys at the current corpus size."""
    all_scores = []
    all_ids = []
    for path in list(coordinator.atlas.crystals.keys()):
        store = coordinator._get_store(path)
        if store.n == 0:
            continue
        scores = store.dot_all(query_vec)
        for i in range(store.n):
            all_scores.append(scores[i])
            all_ids.append(store.get_id(i))
    if not all_scores:
        return []
    order = np.argsort(all_scores)[::-1][:top_k]
    return [(all_ids[i], float(all_scores[i])) for i in order]


def _bench_recall(coordinator, memory, ingested: int) -> dict:
    """One in-process, read-only (learn=False) latency comparison: naive
    brute-force scan vs the production resonance-jump recall() path."""
    probe_vec = memory.embed("What happens in this story?")

    t0 = time.perf_counter()
    _ = coordinator.recall(probe_vec, top_k=5, learn=False)
    optimized_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = _naive_scan_recall(coordinator, probe_vec, top_k=5)
    naive_ms = (time.perf_counter() - t0) * 1000

    return {
        "at_shard_count": ingested,
        "optimized_recall_ms": round(optimized_ms, 3),
        "naive_scan_ms": round(naive_ms, 3),
        "speedup_x": round(naive_ms / optimized_ms, 1) if optimized_ms > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000, help="number of shards to ingest")
    ap.add_argument("--time-limit-s", type=float, default=None,
                     help="stop after this many wall-clock seconds instead of a fixed count "
                          "(count is still used as the corpus-gathering upper bound)")
    ap.add_argument("--brain-dir", type=str, default=os.path.join(PROJECT_ROOT, "module_AI", "data", "stress_brain"))
    ap.add_argument("--batch-size", type=int, default=64, help="embedding batch size")
    ap.add_argument("--checkpoint-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--recall-bench-every", type=int, default=20000)
    args = ap.parse_args()

    from module_loom.services.weaver.weaver_coordinator import WeaveBrainCoordinator
    from module_AI.memory_bridge import LoomMemory

    os.makedirs(args.brain_dir, exist_ok=True)
    coordinator = WeaveBrainCoordinator(storage_dir=args.brain_dir)
    memory = LoomMemory(coordinator=coordinator)

    log_path = os.path.join(BENCH_DIR, "ingest_log.jsonl")
    proc = psutil.Process(os.getpid())

    corpus_iter = gather_corpus(args.count)

    batch: List[str] = []
    ingested = 0
    t_run_start = time.perf_counter()
    last_log_t = t_run_start

    def flush_batch(texts: List[str]):
        nonlocal ingested
        if not texts:
            return
        vectors = memory.embed_batch(texts)
        for text, vec in zip(texts, vectors):
            memory.remember(text, metadata={"kind": "corpus", "source": "bulk_ingest"}, vector=vec)
        ingested += len(texts)

    next_log_at = args.log_every
    next_ckpt_at = args.checkpoint_every

    time_up = False
    with open(log_path, "a", encoding="utf-8") as logf:
        for passage in corpus_iter:
            batch.append(passage)
            if len(batch) < args.batch_size:
                continue
            flush_batch(batch)
            batch = []

            if args.time_limit_s is not None and (time.perf_counter() - t_run_start) >= args.time_limit_s:
                time_up = True

            while ingested >= next_log_at:
                now = time.perf_counter()
                elapsed = now - t_run_start
                rate = ingested / elapsed if elapsed > 0 else 0.0
                mem_mb = proc.memory_info().rss / (1024 * 1024)
                storage_bytes = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fs in os.walk(args.brain_dir) for f in fs
                )
                record = {
                    "ts": time.time(),
                    "ingested": ingested,
                    "elapsed_s": round(elapsed, 2),
                    "shards_per_sec": round(rate, 2),
                    "rss_mb": round(mem_mb, 1),
                    "storage_mb": round(storage_bytes / (1024 * 1024), 2),
                    "crystals": len(coordinator.atlas.crystals),
                }
                logf.write(json.dumps(record) + "\n")
                logf.flush()
                print(f"[{ingested}/{args.count}] {rate:.1f} shards/s | "
                      f"RSS {mem_mb:.0f} MB | storage {record['storage_mb']} MB | "
                      f"crystals {record['crystals']} | elapsed {elapsed:.1f}s")
                next_log_at += args.log_every

            while ingested >= next_ckpt_at:
                coordinator.checkpoint()
                gc.collect()
                next_ckpt_at += args.checkpoint_every

            if time_up:
                break

        flush_batch(batch)

    coordinator.checkpoint()
    coordinator.close()

    total_elapsed = time.perf_counter() - t_run_start
    print(f"\nDone. Ingested {ingested} shards in {total_elapsed:.1f}s "
          f"({ingested/total_elapsed:.1f} shards/s average).")


if __name__ == "__main__":
    main()
