import os
import sys
import json
import struct
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.services.loom_service.weaver.substrate_layout import LoomStore, detect_version, UNBUCKETED
from backend.services.loom_service.weaver.universe_container import UniverseContainer

# Default cap on how many shards a whole-brain decode/browse/export touches.
# Ingest-time metadata + the live-state overlay are decoded per shard (real,
# unavoidable work at scale) — this keeps the routine "browse" path fast and
# the resulting JSON small; pass limit=None explicitly for a full export.
DEFAULT_DECODE_LIMIT = 2000


def hash_shard_id(shard_id: str) -> int:
    """
    Must stay byte-for-byte identical to weaver_coordinator.hash_shard_id —
    it's how the lightweight "snapshot" segment keys its activation records.
    Duplicated (rather than imported) so the decoder's import graph stays
    light: weaver_coordinator.py pulls in networkx/zstandard/the cognitive
    stack just to define this one pure hashlib function, and this is a
    read-only tool people expect to open fast.
    """
    return int.from_bytes(hashlib.sha256(shard_id.encode("utf-8")).digest()[:8], byteorder="big")

# =============================================================================
# DOCLOOM — BRAIN DECODER SERVICE
# =============================================================================
# Turns the binary brain (.brain_data: universe.loom + crystal_*.loom) into
# plain, human-readable structures — dicts/JSON, no msgpack/mmap/binary
# knowledge required to read them. Two audiences:
#   - decoder/cli.py            interactive terminal browser (this module's
#                               primary consumer)
#   - anything else (a route,   calls the functions below directly; every
#     a notebook, a script)     function here returns plain Python data.
#
# A "cluster" in the CLI's vocabulary is one crystal (.loom file) — the
# natural unit DocLoom itself calls a crystal: a self-organized neighborhood
# of shards with its own leader buckets. Nothing here mutates the brain;
# every function opens stores read-only.
# =============================================================================


def find_brain_dir(explicit: Optional[str] = None) -> str:
    """
    Resolves the .brain_data directory: an explicit path, then the two
    locations DocLoom has actually used (assets/.brain_data is current;
    repo-root/.brain_data is the older default some older configs still use).
    """
    if explicit:
        if not os.path.isdir(explicit):
            raise FileNotFoundError(f"Not a directory: {explicit}")
        return os.path.abspath(explicit)
    for candidate in (
        os.path.join(PROJECT_ROOT, "assets", ".brain_data"),
        os.path.join(PROJECT_ROOT, ".brain_data"),
    ):
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "No .brain_data found at assets/.brain_data or ./.brain_data. "
        "Pass an explicit path if the brain lives somewhere else."
    )


def open_universe(storage_dir: str) -> Optional[UniverseContainer]:
    path = os.path.join(storage_dir, "universe.loom")
    if UniverseContainer.exists(path):
        return UniverseContainer(path)
    return None


def brain_summary(storage_dir: str) -> Dict[str, Any]:
    """Top-level overview: seed/dimension, universe.loom segments, and every
    crystal ("cluster") with its shard/leader counts — the CLI's home screen."""
    universe = open_universe(storage_dir)
    summary: Dict[str, Any] = {"storage_dir": storage_dir}

    if universe is not None:
        stats = universe.stats()
        summary["seed"] = stats["seed"]
        summary["dimension"] = stats["dimension"]
        summary["universe_generation"] = stats["generation"]
        summary["universe_segments"] = stats["segments"]
        summary["universe_journal_bytes"] = stats["journal_bytes"]
        summary["universe_total_bytes"] = stats["total_bytes"]

        # Cognitive state (Part B), if a living_state has ever been saved.
        living_blob = universe.get_segment("living_state.bin")
        if living_blob:
            import msgpack
            state = msgpack.unpackb(living_blob, raw=False)
            summary["cognitive"] = {
                "tick": state.get("tick", 0),
                "entropy": round(state.get("entropy", 0.5), 4),
                "coherence": round(state.get("coherence", 0.5), 4),
                "energy_budget": round(state.get("energy_budget", 1.0), 4),
                "assemblies": len(state.get("assemblies", [])),
                "causal_edges": len(state.get("causal_edges", [])),
                "living_shards_saved": len(state.get("living_ledger", {})),
            }

        atlas_blob = universe.get_segment("atlas")
        crystal_paths = _crystal_paths_from_atlas_blob(atlas_blob) if atlas_blob else []
    else:
        summary["seed"] = None
        summary["dimension"] = None
        summary["universe_segments"] = {}
        crystal_paths = _discover_crystals_on_disk(storage_dir)

    clusters = []
    for cpath in crystal_paths:
        abs_path = cpath if os.path.isabs(cpath) else os.path.join(storage_dir, os.path.basename(cpath))
        if not os.path.exists(abs_path):
            abs_path = os.path.join(storage_dir, os.path.basename(cpath))
        info = _cluster_info(abs_path)
        if info:
            clusters.append(info)
    summary["clusters"] = clusters
    summary["legacy_backup_present"] = os.path.isdir(os.path.join(storage_dir, "legacy_backup"))
    return summary


def _crystal_paths_from_atlas_blob(atlas_blob: bytes) -> List[str]:
    from backend.services.loom_service.weaver.atlas_router import GlobalAtlasRouter
    router = GlobalAtlasRouter()
    router.from_bytes(atlas_blob)
    return list(router.crystals.keys())


def _discover_crystals_on_disk(storage_dir: str) -> List[str]:
    return sorted(
        os.path.join(storage_dir, f) for f in os.listdir(storage_dir)
        if f.endswith(".loom") and f != "universe.loom"
    )


def _cluster_info(crystal_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(crystal_path):
        return None
    try:
        version = detect_version(crystal_path)
        with LoomStore(crystal_path, writable=False) as store:
            return {
                "name": os.path.basename(crystal_path),
                "path": crystal_path,
                "format_version": version,
                "shard_count": store.n,
                "leader_count": store.num_leaders,
                "dimension": store.vector_dim,
                "size_bytes": os.path.getsize(crystal_path),
                "idx_present": os.path.exists(crystal_path + ".idx"),
            }
    except Exception as e:
        return {"name": os.path.basename(crystal_path), "path": crystal_path, "error": str(e)}


def _load_live_state(universe: Optional[UniverseContainer]) -> Tuple[Dict[str, Any], Dict[int, Tuple[float, int]]]:
    """
    Reads whatever live cognitive state has actually been checkpointed to
    universe.loom, the same two sources WeaveBrainCoordinator restores its
    RAM ledger from at boot (see _load_persistence_layers) — but read-only,
    no coordinator instantiation needed.

    Returns (living_by_id, snapshot_by_hash):
      - living_by_id: shard_id -> full living_ledger record (activation,
        hits, phase_angle, energy/entropy/..., last_recalled) from the
        rich "living_state.bin" segment, keyed directly by shard_id.
      - snapshot_by_hash: hash_shard_id(sid) -> (activation, timestamp)
        from the lightweight "snapshot" segment (no hits/phase — it only
        exists for fast boot). Used as a fallback for shards not present
        (or not yet present) in living_state.bin.
      - superseded_by_id: shard_id -> the semantic-crystal shard_id it was
        consolidated into by a sleep cycle (see WeaveBrainCoordinator.
        maybe_sleep_cycle) — flattened across all crystals, since shard_ids
        are unique brain-wide. Empty if no sleep cycle has ever run.
    All three are empty if the brain has never been checkpointed/saved (e.g.
    the coordinator process is still running and hasn't hit an autosave or a
    clean shutdown yet) — decode then simply falls back to ingest-time values.
    """
    living_by_id: Dict[str, Any] = {}
    snapshot_by_hash: Dict[int, Tuple[float, int]] = {}
    superseded_by_id: Dict[str, str] = {}
    if universe is None:
        return living_by_id, snapshot_by_hash, superseded_by_id

    living_blob = universe.get_segment("living_state.bin")
    if living_blob:
        try:
            import msgpack
            state = msgpack.unpackb(living_blob, raw=False)
            living_by_id = state.get("living_ledger", {}) or {}
            for _cpath, _sidmap in (state.get("superseded", {}) or {}).items():
                superseded_by_id.update(_sidmap)
        except Exception:
            pass

    snapshot_blob = universe.get_segment("snapshot")
    if snapshot_blob:
        try:
            import zstandard as zstd
            decompressed = zstd.ZstdDecompressor().decompress(snapshot_blob)
            for i in range(len(decompressed) // 16):
                h, act, t = struct.unpack("<QfI", decompressed[i * 16:(i + 1) * 16])
                snapshot_by_hash[h] = (act, t)
        except Exception:
            pass

    return living_by_id, snapshot_by_hash, superseded_by_id


def _overlay_live(
    row: Dict[str, Any],
    shard_id: str,
    living_by_id: Dict[str, Any],
    snapshot_by_hash: Dict[int, Tuple[float, int]],
    superseded_by_id: Optional[Dict[str, str]] = None,
) -> None:
    """
    Mutates `row` in place with live activation/hits/phase when this shard
    is (or recently was) awake in RAM. Sets row["awake"] so it's always
    visible whether a value is live or frozen-since-ingest (the .loom log's
    per-shard metadata is never rewritten after ingest, by design). Also
    tags row["superseded_by"] when a sleep cycle has consolidated this shard
    into a newer semantic crystal — still fully decodable, just deprioritized
    in recall()'s fast path (see §13 of the brain reference doc).
    """
    if superseded_by_id:
        sb = superseded_by_id.get(shard_id)
        if sb:
            row["superseded_by"] = sb

    rec = living_by_id.get(shard_id)
    if rec is not None:
        row["activation"] = rec.get("activation", row.get("activation"))
        row["hits"] = rec.get("hits", row.get("hits"))
        row["phase_angle"] = rec.get("phase_angle")
        row["last_recalled"] = rec.get("last_recalled", row.get("last_recalled"))
        for f in ("energy", "entropy", "stability", "resonance", "momentum", "attention"):
            if f in rec:
                row[f] = rec[f]
        row["awake"] = True
        return

    snap = snapshot_by_hash.get(hash_shard_id(shard_id))
    if snap is not None:
        act, t = snap
        row["activation"] = act
        row["last_recalled"] = float(t)
        row["awake"] = True
        return

    row["awake"] = False


def decode_cluster(
    storage_dir: str,
    cluster_name: str,
    include_vectors: bool = False,
    limit: Optional[int] = DEFAULT_DECODE_LIMIT,
    _live_state: Optional[Tuple[Dict[str, Any], Dict[int, Tuple[float, int]], Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Full human-readable decode of one crystal: every shard's id, text, mass,
    activation, hits, leader bucket, and (Part B) cognitive-physics fields —
    overlaid with LIVE values from universe.loom's checkpointed cognitive
    state when available (see _load_live_state), falling back to the frozen
    ingest-time metadata otherwise. Vectors are omitted by default (a 384-D
    float array per shard is not "human readable"); pass include_vectors=True
    to get them too. `limit` caps how many shards are decoded (not just
    printed) — pass limit=None for a full, uncapped decode of this cluster.
    `_live_state` is an internal cache-sharing hook used by decode_all(); leave
    it unset for a standalone call.
    """
    crystal_path = cluster_name if os.path.isabs(cluster_name) else os.path.join(storage_dir, cluster_name)
    if not os.path.exists(crystal_path):
        raise FileNotFoundError(f"Cluster not found: {crystal_path}")

    if _live_state is None:
        living_by_id, snapshot_by_hash, superseded_by_id = _load_live_state(open_universe(storage_dir))
    else:
        living_by_id, snapshot_by_hash, superseded_by_id = _live_state

    with LoomStore(crystal_path, writable=False) as store:
        n = store.n
        take = n if limit is None else min(n, limit)
        bucket_ids = store.bucket_ids_all()
        mass_all = store.mass_all()
        all_ids = store.get_all_ids()  # one cheap cached read, not per-shard get_id()

        shards = []
        for i in range(take):
            meta = store.get_meta(i)
            bid = int(bucket_ids[i]) if i < len(bucket_ids) else UNBUCKETED
            sid = all_ids[i]
            row = {
                "index": i,
                "shard_id": sid,
                "text": meta.get("text", ""),
                "mass": float(mass_all[i]) if i < len(mass_all) else meta.get("mass", 1.0),
                "activation": meta.get("activation"),
                "hits": meta.get("hits"),
                "last_recalled": meta.get("last_recalled"),
                "bucket": "UNBUCKETED (legacy)" if bid == UNBUCKETED else bid,
            }
            for cog_field in ("energy", "entropy", "stability", "resonance", "momentum", "decay", "attention"):
                if cog_field in meta:
                    row[cog_field] = meta[cog_field]
            # Sleep-cycle consolidation tags (frozen into meta at ingest time
            # for a semantic crystal; superseded_by comes from live state below).
            if meta.get("semantic_crystal"):
                row["semantic_crystal"] = True
                row["source_shard_ids"] = meta.get("source_shard_ids")
                row["consolidated_concept"] = meta.get("consolidated_concept")
            _overlay_live(row, sid, living_by_id, snapshot_by_hash, superseded_by_id)
            if include_vectors:
                row["vector"] = store.get_vector(i).tolist()
            shards.append(row)

        return {
            "cluster": os.path.basename(crystal_path),
            "shard_count": n,
            "shown": take,
            "leader_count": store.num_leaders,
            "dimension": store.vector_dim,
            "shards": shards,
        }


def decode_all(
    storage_dir: str,
    include_vectors: bool = False,
    limit: Optional[int] = DEFAULT_DECODE_LIMIT,
) -> Dict[str, Any]:
    """
    The whole brain, human-readable: summary + every cluster's shards.
    `limit` (per cluster) defaults to DEFAULT_DECODE_LIMIT so a large brain
    stays fast to decode and the exported JSON stays fast to open; pass
    limit=None for a full, uncapped export of every shard in every cluster.
    """
    summary = brain_summary(storage_dir)
    live_state = _load_live_state(open_universe(storage_dir))
    clusters = {}
    for c in summary["clusters"]:
        if "error" in c:
            clusters[c["name"]] = {"error": c["error"]}
            continue
        clusters[c["name"]] = decode_cluster(
            storage_dir, c["name"], include_vectors=include_vectors, limit=limit, _live_state=live_state
        )
    return {"summary": summary, "clusters": clusters}


def export_json(data: Dict[str, Any], out_path: str) -> str:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default, ensure_ascii=False)
    return out_path


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)
