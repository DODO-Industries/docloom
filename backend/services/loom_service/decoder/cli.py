"""
DocLoom Brain Decoder — interactive terminal browser for .brain_data.

Run directly:   python -m backend.services.loom_service.decoder.cli [path-to-.brain_data]
Or via:          decode_brain.bat   (repo root — no manual invocation needed)

Shows the universe (seed, dimension, cognitive vitals) and every cluster
(crystal) with its shard count and leader buckets. Select one cluster to
browse its shards in plain text, or select "all" to browse/export the whole
brain. Every view can be exported to a JSON file. Read-only — never writes
into the brain itself.
"""
import os
import sys

# Shard text can contain any Unicode (real document content); degrade
# gracefully instead of crashing if the Windows console's codepage can't
# render some of it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.services.loom_service.decoder.brain_decoder_service import (
    find_brain_dir, brain_summary, decode_cluster, decode_all, export_json, DEFAULT_DECODE_LIMIT,
)


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def print_summary(summary):
    print("\n" + "=" * 78)
    print(f" DOCLOOM BRAIN DECODER - {summary['storage_dir']}")
    print("=" * 78)
    print(f" seed: {summary['seed']}    dimension: {summary['dimension']}")
    if summary.get("universe_segments"):
        segs = ", ".join(f"{k}({_fmt_bytes(v)})" for k, v in summary["universe_segments"].items())
        print(f" universe.loom segments: {segs}")
        print(f" journal tail: {_fmt_bytes(summary['universe_journal_bytes'])}   "
              f"total: {_fmt_bytes(summary['universe_total_bytes'])}   "
              f"generation: {summary['universe_generation']}")
    if summary.get("cognitive"):
        c = summary["cognitive"]
        print(f"\n COGNITIVE STATE (Part B, last saved)")
        print(f"   tick={c['tick']}  coherence={c['coherence']}  entropy={c['entropy']}  "
              f"energy_budget={c['energy_budget']}")
        print(f"   assemblies={c['assemblies']}  causal_edges={c['causal_edges']}  "
              f"living_shards_saved={c['living_shards_saved']}")
    if summary.get("legacy_backup_present"):
        print("\n (a legacy_backup/ folder exists - pre-migration files, safe to ignore or inspect separately)")
    print("\n (bucket 'UNBUCKETED (legacy)' means a shard predates leader-bucket tracking, or was written by a\n"
          "  path that never assigned one - it is not live, and is expected for old data. Fresh ingests get a real bucket.)")

    print(f"\n CLUSTERS ({len(summary['clusters'])})")
    print(f" {'#':<4}{'name':<24}{'shards':<10}{'leaders':<10}{'size':<10}{'format'}")
    for i, c in enumerate(summary["clusters"], 1):
        if "error" in c:
            print(f" {i:<4}{c['name']:<24}ERROR: {c['error']}")
            continue
        print(f" {i:<4}{c['name']:<24}{c['shard_count']:<10}{c['leader_count']:<10}"
              f"{_fmt_bytes(c['size_bytes']):<10}v{c['format_version']}")
    print("=" * 78)


def print_cluster(decoded, max_rows=40):
    print(f"\n--- Cluster: {decoded['cluster']} "
          f"({decoded['shard_count']} shards, {decoded['leader_count']} leader buckets, "
          f"dimension {decoded['dimension']}) ---\n")
    header = f"{'#':<6}{'shard_id':<22}{'bucket':<10}{'mass':<8}{'act':<9}{'hits':<6}{'text'}"
    print(header)
    print("-" * len(header))
    shown = decoded["shards"][:max_rows]
    for s in shown:
        text = (s.get("text") or "")[:60]
        act = s.get("activation")
        act_s = f"{act:.2f}" if isinstance(act, (int, float)) else "-"
        act_s += "*" if s.get("awake") else " "
        hits = s.get("hits")
        hits_s = str(hits) if hits is not None else "-"
        print(f"{s['index']:<6}{s['shard_id']:<22}{str(s['bucket']):<10}{s['mass']:<8.2f}{act_s:<9}{hits_s:<6}{text}")
    if shown:
        print("\n  (* = live, currently awake in RAM; no * = frozen at ingest-time, not recalled/checkpointed since)")
    if len(decoded["shards"]) > max_rows:
        print(f"\n  ... {len(decoded['shards']) - max_rows} more shards not shown (export to see all)")
    shard_count = decoded.get("shard_count", len(decoded["shards"]))
    if decoded.get("shown", shard_count) < shard_count:
        print(f"  (decoded {decoded['shown']} of {shard_count} shards in this cluster - "
              f"pass a higher/no limit to decode the rest)")


def main():
    explicit = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        storage_dir = find_brain_dir(explicit)
    except FileNotFoundError as e:
        print(f"\n[!] {e}")
        return

    summary = brain_summary(storage_dir)
    print_summary(summary)

    while True:
        print("\nSelect: [1-N] view a cluster | [a] view ALL | [r] refresh | [q] quit")
        choice = input("> ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("Goodbye.")
            return
        if choice in ("r", "refresh"):
            summary = brain_summary(storage_dir)
            print_summary(summary)
            continue
        if choice in ("a", "all"):
            _browse_all(storage_dir)
            continue

        try:
            idx = int(choice) - 1
            cluster = summary["clusters"][idx]
        except (ValueError, IndexError):
            print("  Not a valid choice - enter a cluster number, 'a', 'r', or 'q'.")
            continue
        if "error" in cluster:
            print(f"  Cluster has an error and cannot be opened: {cluster['error']}")
            continue

        _browse_cluster(storage_dir, cluster["name"])


def _browse_cluster(storage_dir, cluster_name):
    # Capped preview by default (DEFAULT_DECODE_LIMIT) - a single huge cluster
    # no longer eagerly decodes+overlays every shard just to print 40 rows.
    decoded = decode_cluster(storage_dir, cluster_name)
    print_cluster(decoded)
    while True:
        print("\n  [v] show vectors too (same limit as preview) | [e] export this cluster to JSON (no limit) | [b] back")
        sub = input("  > ").strip().lower()
        if sub in ("b", "back", ""):
            return
        if sub in ("v", "vectors"):
            decoded_v = decode_cluster(storage_dir, cluster_name, include_vectors=True)
            print_cluster(decoded_v)
            print("  (vectors are included in the decoded data - export to see the raw numbers)")
            decoded = decoded_v
            continue
        if sub in ("e", "export"):
            print("  Decoding full cluster for export (no row limit)...")
            full = decode_cluster(storage_dir, cluster_name, limit=None)
            out = os.path.join(os.getcwd(), f"decoded_{cluster_name.replace('.loom', '')}.json")
            export_json(full, out)
            print(f"  Exported to: {out}")
            continue
        print("  Not a valid choice.")


def _browse_all(storage_dir):
    print(f"\n  Decoding the whole brain - capped preview (up to {DEFAULT_DECODE_LIMIT} shards/cluster)...")
    data = decode_all(storage_dir)
    for name, cluster in data["clusters"].items():
        if "error" in cluster:
            print(f"\n  [{name}] ERROR: {cluster['error']}")
            continue
        print_cluster(cluster, max_rows=15)
    while True:
        print("\n  [e] export EVERYTHING to JSON (no limit - may be slow/large for a big brain) | [b] back")
        sub = input("  > ").strip().lower()
        if sub in ("b", "back", ""):
            return
        if sub in ("e", "export"):
            print("  Decoding full brain for export (no row limit)...")
            full_data = decode_all(storage_dir, limit=None)
            out = os.path.join(os.getcwd(), "decoded_brain_full.json")
            export_json(full_data, out)
            print(f"  Exported to: {out}")
            continue
        print("  Not a valid choice.")


if __name__ == "__main__":
    main()
