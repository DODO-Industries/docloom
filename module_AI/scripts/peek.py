"""
Zero-overhead, read-only snapshot of the stress-test brain — uses the existing
brain_decoder_service (opens .loom files read-only, no physics/routing/3D
computation) instead of the visualizer's per-shard rendering cost. Safe to run
anytime, including while bulk_ingest.py is actively writing.

Usage:
    python -m module_AI.scripts.peek [--brain-dir module_AI/data/stress_brain] [--sample 5]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from module_loom.services.decoder.brain_decoder_service import brain_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain-dir", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "stress_brain"))
    args = ap.parse_args()

    summary = brain_summary(os.path.abspath(args.brain_dir))
    total_shards = sum(c["shard_count"] for c in summary.get("clusters", []))
    print(f"storage_dir : {summary['storage_dir']}")
    print(f"crystals    : {len(summary.get('clusters', []))}")
    print(f"total shards: {total_shards}")
    for c in summary.get("clusters", []):
        print(f"  - {c['name']}: {c['shard_count']} shards, {c['size_bytes']/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
