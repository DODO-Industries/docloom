"""
Streams the cached real-text corpus into the LIVE running server's testing
sandbox (/loom/testing/ingest_text_batch), so it plots live in the browser at
http://localhost:8000/loom/testing while this script runs. Pure HTTP client —
no direct coordinator access, so there's exactly one writer (the server).

Usage:
    python -m module_AI.scripts.stream_to_server --count 2000 --batch-size 24
"""
import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from module_AI.scripts.bulk_ingest import gather_corpus  # reuses the same corpus/chunking logic

SERVER = "http://localhost:8000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--delay-s", type=float, default=0.0, help="pause between batches (0 = as fast as the server allows)")
    args = ap.parse_args()

    if requests.get(f"{SERVER}/", timeout=5).status_code != 200:
        raise SystemExit("Server not reachable at " + SERVER)

    batch = []
    sent = 0
    t0 = time.perf_counter()
    for passage in gather_corpus(args.count):
        batch.append(passage)
        if len(batch) < args.batch_size:
            continue
        resp = requests.post(f"{SERVER}/loom/testing/ingest_text_batch", json={"texts": batch}, timeout=120)
        resp.raise_for_status()
        sent += len(batch)
        elapsed = time.perf_counter() - t0
        print(f"[{sent}/{args.count}] {sent/elapsed:.1f} shards/s | total_shards={resp.json()['total_shards']}")
        batch = []
        if args.delay_s:
            time.sleep(args.delay_s)

    if batch:
        requests.post(f"{SERVER}/loom/testing/ingest_text_batch", json={"texts": batch}, timeout=120)
        sent += len(batch)

    print(f"\nDone. Streamed {sent} shards in {time.perf_counter()-t0:.1f}s.")


if __name__ == "__main__":
    main()
