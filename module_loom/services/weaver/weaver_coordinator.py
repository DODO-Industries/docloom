import os
import sys
import time
import heapq
import struct
import numpy as np
import zstandard as zstd
import hashlib
import json
import networkx as nx
from functools import lru_cache
from typing import Dict, List, Any, Optional, Tuple
from module_loom.config.tuning_config import tuning_manager

from module_loom.services.weaver.seed_core import UniverseSeedCore
from module_loom.services.weaver.atlas_router import GlobalAtlasRouter, STATE_MAP_TO_BYTE, STATE_MAP_FROM_BYTE
from module_loom.services.weaver.substrate_layout import LoomSubstrate, LoomStore, detect_version, UNBUCKETED
from module_loom.services.weaver.storage_physics import LatentFieldPhysicsEngine
from module_loom.services.weaver.memory_fluidity import DynamicMemoryFluidity
from module_loom.services.weaver.universe_container import (
    UniverseContainer, pack_txn_record, pack_crystal_record, iter_journal_records,
    JREC_TXN, JREC_CRYSTAL, TXN_PAYLOAD, CRYSTAL_PREFIX,
)
from module_loom.services.cortex.latent_field_cognition.predictive_processing import CausalGraphs
from module_loom.services.cortex.latent_field_cognition.attractor_basin_compilation import (
    AssemblyCompilation, CognitiveAssembly,
)
from module_loom.services.cortex.latent_field_cognition.cognitive_field_substrate import CognitiveMetrics

# Legacy loose files consolidated into universe.loom (moved to legacy_backup/ after migration)
LEGACY_STATE_FILES = [
    "coordinates.bin", "velocities.bin", "physics_tensors.bin", "brain_embeddings.bin",
    "working_memory.bin", "causal_links.bin", "metadata.bin", "replay.bin", "hdc_signatures.bin",
]

# Segments that were written by the pre-Part-B save format but are NEVER read
# back anywhere in the current code (verified 2026-07-15): recall/boot/decode use
# none of these. They were carried forward on every universe.loom rebuild purely
# out of "don't drop unknown state" caution (~39 KB dead weight). We now let them
# age out of the container on the next rebuild. NOTE: coordinates.bin /
# physics_tensors.bin / causal_links.bin / hdc_signatures.bin are NOT here — they
# are still written by the orchestrate_weave() bulk path and must be preserved.
DEAD_CONTAINER_SEGMENTS = frozenset({
    "velocities.bin", "brain_embeddings.bin", "working_memory.bin",
    "metadata.bin", "replay.bin",
})

# Byte-wise popcount lookup table — used for Hamming distance on packed 128-bit
# leader signatures. ~1.8x faster than np.unpackbits(...).sum() at this scale
# (same trick already used in cortex/latent_field_cognition/dynamic_field_substrate.py).
_BIT_COUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming_distances(matrix: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Row-wise Hamming distance between a (L,16) packed-byte matrix and one signature."""
    if len(matrix) == 0:
        return np.empty(0, dtype=np.int64)
    return _BIT_COUNT_LUT[np.bitwise_xor(matrix, sig)].sum(axis=1).astype(np.int64)


@lru_cache(maxsize=1_048_576)
def hash_shard_id(shard_id: str) -> int:
    # Memoized: recall/cognitive-tick hash the same shard_ids thousands of times
    # per query; the SHA-256 was a measurable hot spot (profiled ~6k calls/recall).
    return int.from_bytes(hashlib.sha256(shard_id.encode("utf-8")).digest()[:8], byteorder="big")


def _seed_phase(vector: np.ndarray) -> float:
    """
    Deterministic initial Kuramoto phase derived from the vector itself (the
    same split-sum arctan2 trick SubstrateWeaver.weave() already uses for its
    own phase field) instead of a uniform 0.0. Every ram_ledger entry used to
    be BORN at phase 0.0 with no exception — but the Kuramoto coupling update
    in recall() produces exactly zero drift when the whole population is
    already at the same phase (sin(0)=0 kills the coupling sum), so an
    all-zero population is a degenerate fixed point it can never escape.
    Seeding real per-shard variation here is what makes phase sync a real
    mechanic instead of a permanent no-op.
    """
    half = len(vector) // 2
    if half == 0:
        return 0.0
    a = float(np.sum(vector[:half]))
    b = float(np.sum(vector[half:]))
    return float((np.arctan2(a, b) + 2 * np.pi) % (2 * np.pi))


def _np_msgpack_default(obj):
    """msgpack fallback for numpy scalars/arrays that slip into cognitive state."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Cannot serialize type {type(obj)} into cortex state")


def _default_cognitive_fields() -> Dict[str, float]:
    """
    Part B: the cognitive-physics fields every ram_ledger entry now carries
    alongside the original storage fields (activation/phase_angle/hits/...).
    One record, one source of truth — process_cognitive_tick reads and
    writes these same dict entries, not a separate concept registry.
    """
    return {
        "energy": 1.0, "entropy": 0.0, "stability": 1.0,
        "resonance": 0.0, "momentum": 0.0, "decay": 0.05, "attention": 0.0,
    }

class WeaveBrainCoordinator:
    """
    ============================================================================
    DOCLOOM — WEAVE BRAIN COORDINATOR (The Runtime Orchestrator)
    ============================================================================
    Coordinates ingestion and recall loops across the seed core, atlas router,
    physical substrate mmap layers, physics engines, and temporal memory loops.
    ============================================================================
    """
    def __init__(
        self,
        seed: Optional[int] = None,
        storage_dir: str = "",
        dimension: int = 128,
        lambda_base: float = 0.05
    ):
        from module_loom.config.env_config import BRAIN_STORAGE_DIR, EMBEDDING_DIMENSION
        if dimension == 128:
            dimension = EMBEDDING_DIMENSION
        self.dimension = dimension
        if not storage_dir:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            configured = os.path.join(project_root, BRAIN_STORAGE_DIR)
            preferred = os.path.join(project_root, "assets", ".brain_data")
            storage_dir = configured if os.path.isdir(configured) else (preferred if os.path.isdir(preferred) else os.path.join(project_root, ".brain_data"))
        self.storage_dir = os.path.abspath(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.atlas_path = os.path.join(self.storage_dir, "atlas.capnp")
        self.metric_path = os.path.join(self.storage_dir, "universe.metric")
        self.universe_path = os.path.join(self.storage_dir, "universe.loom")
        self.max_shards_per_crystal = tuning_manager.get_int("MAX_CRYSTAL_SIZE", 500000)
        self.universe: Optional[UniverseContainer] = None
        self._atlas_dirty = False

        # Mechanism 4 (forgetting ledger with provenance): an append-only,
        # compact tombstone log written at every eviction site instead of a
        # bare `del` — so forgetting is auditable (what was known, why it
        # faded, and what replaced it if anything). Loaded lazily; capped
        # in-memory copy for explain_forgotten(), full history stays on disk.
        self._tombstone_path = os.path.join(self.storage_dir, "tombstones.jsonl")
        self.tombstones: List[Dict[str, Any]] = []
        if os.path.exists(self._tombstone_path):
            try:
                with open(self._tombstone_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.tombstones.append(json.loads(line))
                self.tombstones = self.tombstones[-2000:]
            except Exception:
                pass

        if UniverseContainer.exists(self.universe_path):
            # ---- Unified boot: everything comes from universe.loom ----
            self.universe = UniverseContainer(self.universe_path)
            self.seed = self.universe.seed
            self.dimension = self.universe.dimension
            self.atlas = GlobalAtlasRouter(seed=self.seed)
            atlas_blob = self.universe.get_segment("atlas")
            if atlas_blob:
                self.atlas.from_bytes(atlas_blob)
            self._normalize_crystal_paths()
        else:
            # ---- Legacy boot (loose files), then one-time consolidation ----
            # 1. Parse universe.metric (Genesis Read)
            metric_loaded = False
            if os.path.exists(self.metric_path):
                try:
                    with open(self.metric_path, "rb") as f:
                        metric_bytes = f.read(14)
                    if len(metric_bytes) == 14:
                        magic, dim, metric_seed = struct.unpack("<4sHQ", metric_bytes)
                        if magic == b"LOOM":
                            self.seed = metric_seed
                            self.dimension = dim
                            metric_loaded = True
                except Exception:
                    pass

            if not metric_loaded:
                if os.path.exists(self.atlas_path):
                    self.atlas = GlobalAtlasRouter(atlas_path=self.atlas_path)
                    self.seed = self.atlas.seed
                else:
                    if seed is None:
                        from module_loom.config.env_config import MASTER_SEED
                        self.seed = MASTER_SEED
                    else:
                        self.seed = seed & 0xFFFFFFFFFFFFFFFF

            self.atlas = GlobalAtlasRouter(atlas_path=self.atlas_path, seed=self.seed)
            self._normalize_crystal_paths()
            self._consolidate_universe()
        
        self.seed_core = UniverseSeedCore(seed=self.seed, dimension=self.dimension)
        self.physics = LatentFieldPhysicsEngine(dimension=self.dimension)
        if lambda_base == 0.05:
            lambda_base_val = tuning_manager.get_float("MEMORY_DECAY_RATE", lambda_base)
        else:
            lambda_base_val = lambda_base
        self.fluidity = DynamicMemoryFluidity(lambda_base=lambda_base_val)
        
        # Open crystal store handles (append-only .loom v2; v1 auto-migrates on open)
        self._stores: Dict[str, LoomStore] = {}

        # Initialize Active Latent Cortex Ledger and boot-up restore
        self.ram_ledger: Dict[str, Dict[str, Any]] = {}

        # ---- Shared cognition state (Part B: deep merge) ----
        # Real causal graph over actual shard_ids, grown by process_cognitive_tick
        # as ingest/recall actually happen. recall()'s propagation hop
        # (Part A) reads self.causal_graph directly — same object, aliased
        # from the CausalGraphs wrapper below so both sides always agree.
        self.causal = CausalGraphs()
        self.causal_graph: "nx.DiGraph" = self.causal.causal_matrix
        self.compiler = AssemblyCompilation()
        self.active_assemblies: List["CognitiveAssembly"] = []
        self.latent_field: Optional[np.ndarray] = None
        self.working_latent: Optional[np.ndarray] = None
        self.latent_velocity: Optional[np.ndarray] = None
        self.cognitive_entropy: float = 0.5
        self.cognitive_coherence: float = 0.5
        self.cognitive_energy_budget: float = 1.0
        self.cognitive_tick: int = 0
        self._last_cognitive_coherence: float = 0.5
        self._last_ingested_sid: Optional[str] = None
        self._last_adequacy_score: float = 1.0
        self._last_hop_triggered: bool = False
        # Mechanism 7 (usage-pattern "working set" memory): pairs of shards
        # that keep getting recalled TOGETHER across queries/sessions, tracked
        # separately from the content-coherence-gated causal graph (mechanism
        # 5) — this binds purely by co-use, the "keys and coat" association,
        # regardless of topic similarity. No decay (unlike the causal graph's
        # STALE_DECAY_RATE): a working-set link is meant to persist, not fade
        # between sessions the way a single ingest-order edge should.
        self.working_set_pairs: Dict[Tuple[str, str], int] = {}
        self._last_full_save_ts: float = 0.0
        # Set once close() has run so __del__ never re-attempts a save. A save
        # driven from __del__ fires during interpreter/GC shutdown, where
        # `import msgpack` fails ("sys.meta_path is None") and the storage dir
        # may already be gone — producing spurious "Error in save_cortex_state"
        # / "No such file: universe.loom" noise even though the real close()
        # already persisted everything. close() is called explicitly (server
        # lifespan + tools), so __del__ is only a last-resort fallback.
        self._closed: bool = False

        # ---- Sleep cycle: consolidation over the external store (Part D) ----
        # crystal_path -> {shard_id: superseded_by_shard_id}. Living state, not
        # a storage-format change — recall() masks these out of the fast path,
        # nothing is ever deleted from the underlying .loom log.
        self.superseded_by_crystal: Dict[str, Dict[str, str]] = {}
        # crystal_path -> np.ndarray of local indices, resolved from the above
        # via store.index_of() ONCE per change, not per recall() call.
        self._superseded_idx_cache: Dict[str, np.ndarray] = {}
        self._last_message_ts: float = time.time()
        self._shards_since_last_sleep: int = 0
        self._last_sleep_ts: float = 0.0

        # Deferred-learning buffer: serving-mode (learn=False) recalls record a
        # cheap trace of what they surfaced here instead of paying the full
        # write-and-think cost per query. consolidate_traces() drains it in
        # batches (auto-triggered at CONSOLIDATE_EVERY, or on the heartbeat), so
        # the brain STILL reinforces, wires associations, and remembers what it
        # recalled — just consolidated in the background, the way perception +
        # consolidation actually split in a real brain. Nothing is lost.
        self._recall_traces: List[Tuple[List[str], List[float], float]] = []

        self._load_persistence_layers()
        try:
            self.load_cortex_state()  # best-effort; False for a fresh brain
        except Exception:
            pass

    def _get_store(self, path: str) -> LoomStore:
        """Returns (and caches) the writable LoomStore handle for a crystal."""
        path = os.path.abspath(path)
        store = self._stores.get(path)
        if store is None:
            fsync = tuning_manager.get_int("LOOM_FSYNC", 0) == 1
            tail_limit = tuning_manager.get_int("LOOM_CHECKPOINT_TAIL", 25000)
            store = LoomStore(
                path,
                vector_dim=self.dimension,
                seed=self.seed,
                writable=True,
                fsync=fsync,
                checkpoint_tail_limit=tail_limit,
            )
            self._stores[path] = store
        return store

    def _normalize_crystal_paths(self) -> None:
        """
        Brains must be portable: the atlas may hold absolute crystal paths from
        a previous location (e.g. .brain_data moved into assets/). Re-anchor any
        path whose file is missing to storage_dir/<basename> when that exists.
        """
        remapped = {}
        for p, info in self.atlas.crystals.items():
            resolved = p
            if not os.path.exists(p):
                candidate = os.path.join(self.storage_dir, os.path.basename(p))
                if os.path.exists(candidate):
                    resolved = candidate
            remapped[resolved] = info
        self.atlas.crystals = remapped

    # ------------------------------------------------------------------------
    # UNIVERSE CONTAINER (single-file brain state)
    # ------------------------------------------------------------------------

    def _consolidate_universe(self) -> None:
        """
        One-time migration: folds the loose legacy files (universe.metric,
        atlas.capnp, latent_state.snapshot, cortex_journal.bin, and the cortex
        state .bin files) into a single universe.loom container, then moves the
        originals into legacy_backup/ (content preserved, nothing deleted).
        """
        segments: Dict[str, bytes] = {"atlas": self.atlas.to_bytes()}

        snapshot_path = os.path.join(self.storage_dir, "latent_state.snapshot")
        if os.path.exists(snapshot_path):
            try:
                with open(snapshot_path, "rb") as f:
                    segments["snapshot"] = f.read()
            except Exception:
                pass

        for name in LEGACY_STATE_FILES:
            p = os.path.join(self.storage_dir, name)
            if os.path.exists(p):
                try:
                    with open(p, "rb") as f:
                        segments[name] = f.read()
                except Exception:
                    pass

        # Convert the legacy 32-byte TXN journal into typed container records
        journal_blob = bytearray()
        legacy_journal = os.path.join(self.storage_dir, "cortex_journal.bin")
        if os.path.exists(legacy_journal):
            try:
                with open(legacy_journal, "rb") as f:
                    raw = f.read()
                for i in range(len(raw) // 32):
                    chunk = raw[i * 32:(i + 1) * 32]
                    magic, h, act, t, _ = struct.unpack("<4sQfd8s", chunk)
                    if magic == b"TXN\x00":
                        journal_blob.extend(pack_txn_record(h, act, t))
            except Exception:
                pass

        self.universe = UniverseContainer(
            self.universe_path, seed=self.seed, dimension=self.dimension, create=True
        )
        self.universe.rebuild(segments, journal=bytes(journal_blob))

        # Retire legacy files (moved, never deleted)
        backup_dir = os.path.join(self.storage_dir, "legacy_backup")
        legacy_files = [self.metric_path, self.atlas_path, snapshot_path, legacy_journal] + \
                       [os.path.join(self.storage_dir, n) for n in LEGACY_STATE_FILES]
        for p in legacy_files:
            if os.path.exists(p):
                try:
                    os.makedirs(backup_dir, exist_ok=True)
                    os.replace(p, os.path.join(backup_dir, os.path.basename(p)))
                except Exception:
                    pass
        # Atlas persistence now lives inside the container
        self.atlas.atlas_path = None

    def _mark_atlas_dirty(self) -> None:
        """Centroid drift is folded into the container at checkpoint cadence."""
        if self.universe is not None:
            self._atlas_dirty = True
        else:
            self.atlas.save()

    def _journal_crystal_registration(self, crystal_path: str) -> None:
        """Crystal registrations are rare but must survive a crash immediately —
        they are journaled as typed records and replayed on boot."""
        if self.universe is None:
            self.atlas.save()
            return
        info = self.atlas.crystals.get(crystal_path)
        if info is None:
            return
        state_byte = STATE_MAP_TO_BYTE.get(info["state"], 2)
        centroid_bytes = np.asarray(info["centroid"], dtype=np.float32).tobytes()
        self.universe.append_journal(
            pack_crystal_record(state_byte, int(info["num_shards"]), crystal_path, centroid_bytes)
        )
        self._atlas_dirty = True

    def _carryover_segments(self) -> Dict[str, bytes]:
        """Current container segments, for atomic rebuilds that must not drop state."""
        segments: Dict[str, bytes] = {}
        if self.universe is not None:
            for name in self.universe.segment_names():
                if name in DEAD_CONTAINER_SEGMENTS:
                    continue  # #7: let confirmed-dead pre-Part-B segments age out
                blob = self.universe.get_segment(name)
                if blob is not None:
                    segments[name] = blob
        return segments

    def _commit_state_segments(self, new_segments: Dict[str, bytes]) -> None:
        """
        Atomically folds new state segments into universe.loom (preserving all
        other segments and the pending journal tail). Legacy fallback: loose files.
        """
        if self.universe is not None:
            journal = self.universe.read_journal()
            segments = self._carryover_segments()
            segments.update(new_segments)
            segments["atlas"] = self.atlas.to_bytes()
            self.universe.rebuild(segments, journal=journal)
            self._atlas_dirty = False
        else:
            for name, blob in new_segments.items():
                with open(os.path.join(self.storage_dir, name), "wb") as f:
                    f.write(blob)

    def _state_read(self, name: str) -> Optional[bytes]:
        """Reads a state segment from universe.loom (legacy loose-file fallback)."""
        if self.universe is not None:
            blob = self.universe.get_segment(name)
            if blob:
                return blob
        p = os.path.join(self.storage_dir, name)
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return f.read()
            except Exception:
                return None
        return None

    def _load_persistence_layers(self) -> None:
        """
        Sub-Second Boot Sequence:
        Loads the compressed snapshot segment and replays journal records from
        the universe.loom container to reconstruct the RAM ledger.
        """
        # 0. Replay crystal-registration records first so routing knows every
        #    crystal even if the atlas segment predates the registration.
        journal_blob = b""
        if self.universe is not None:
            try:
                journal_blob = self.universe.read_journal()
            except Exception:
                journal_blob = b""
            for rec_type, payload in iter_journal_records(journal_blob):
                if rec_type != JREC_CRYSTAL:
                    continue
                try:
                    state_byte, num_shards, path_len = CRYSTAL_PREFIX.unpack_from(payload, 0)
                    pos = CRYSTAL_PREFIX.size
                    cpath = payload[pos:pos + path_len].decode("utf-8")
                    pos += path_len
                    centroid = np.frombuffer(payload[pos:], dtype=np.float32).copy()
                    if cpath not in self.atlas.crystals:
                        self.atlas.crystals[cpath] = {
                            "state": STATE_MAP_FROM_BYTE.get(state_byte, "warm"),
                            "centroid": centroid,
                            "num_shards": num_shards
                        }
                except Exception:
                    pass

        # 1. Build hash-to-shard mapping. v2 crystals expose their id table
        #    directly from the .idx snapshot — no metadata decode.
        self.hash_to_shard: Dict[int, Tuple[str, str]] = {}
        for crystal_path in list(self.atlas.crystals.keys()):
            if os.path.exists(crystal_path):
                try:
                    if detect_version(crystal_path) == 2:
                        with LoomStore(crystal_path, writable=False) as ro:
                            ids = ro.get_all_ids()
                    else:
                        ids = [e["shard_id"] for e in LoomSubstrate.get_journal(crystal_path)]
                    for sid in ids:
                        self.hash_to_shard[hash_shard_id(sid)] = (sid, crystal_path)
                except Exception:
                    pass

        # 2. Load ledger snapshot (container segment; legacy file fallback)
        snapshot_blob = None
        if self.universe is not None:
            snapshot_blob = self.universe.get_segment("snapshot")
        if snapshot_blob is None:
            snapshot_path = os.path.join(self.storage_dir, "latent_state.snapshot")
            if os.path.exists(snapshot_path):
                try:
                    with open(snapshot_path, "rb") as f:
                        snapshot_blob = f.read()
                except Exception:
                    snapshot_blob = None
        if snapshot_blob:
            try:
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(snapshot_blob)
                num_records = len(decompressed) // 16
                for i in range(num_records):
                    chunk = decompressed[i*16 : (i+1)*16]
                    h, act, t = struct.unpack("<QfI", chunk)
                    if h in self.hash_to_shard:
                        sid, crystal_path = self.hash_to_shard[h]
                        self.ram_ledger[sid] = {
                            "activation": act,
                            "latent_position": np.zeros(self.dimension, dtype=np.float32),
                            "velocity": np.zeros(self.dimension, dtype=np.float32),
                            "phase_angle": 0.0,
                            "hits": 0,
                            "last_recalled": float(t),
                            "crystal_path": crystal_path,
                            **_default_cognitive_fields()
                        }
            except Exception:
                pass

        # 3. Replay ledger TXN records (container journal; legacy file fallback)
        def _apply_txn(h: int, act: float, t: float) -> None:
            if h in self.hash_to_shard:
                sid, crystal_path = self.hash_to_shard[h]
                if sid not in self.ram_ledger:
                    self.ram_ledger[sid] = {
                        "activation": act,
                        "latent_position": np.zeros(self.dimension, dtype=np.float32),
                        "velocity": np.zeros(self.dimension, dtype=np.float32),
                        "phase_angle": 0.0,
                        "hits": 0,
                        "last_recalled": t,
                        "crystal_path": crystal_path,
                        **_default_cognitive_fields()
                    }
                else:
                    self.ram_ledger[sid]["activation"] = act
                    self.ram_ledger[sid]["last_recalled"] = t

        if self.universe is not None:
            for rec_type, payload in iter_journal_records(journal_blob):
                if rec_type == JREC_TXN and len(payload) >= TXN_PAYLOAD.size:
                    try:
                        h, act, t = TXN_PAYLOAD.unpack(payload[:TXN_PAYLOAD.size])
                        _apply_txn(h, act, t)
                    except Exception:
                        pass
        else:
            journal_path = os.path.join(self.storage_dir, "cortex_journal.bin")
            if os.path.exists(journal_path):
                try:
                    with open(journal_path, "rb") as f:
                        journal_bytes = f.read()
                    for i in range(len(journal_bytes) // 32):
                        chunk = journal_bytes[i*32 : (i+1)*32]
                        magic, h, act, t, _ = struct.unpack("<4sQfd8s", chunk)
                        if magic == b"TXN\x00":
                            _apply_txn(h, act, t)
                except Exception:
                    pass

    def _append_journal_entry(self, shard_id: str, activation: float, timestamp: float) -> None:
        """Appends a 32-byte transaction string to the end of cortex_journal.bin."""
        self._append_journal_entries([(shard_id, activation, timestamp)])

    def _append_journal_entries(self, entries: List[Tuple[str, float, float]]) -> None:
        """Appends many ledger transactions with a single O(1) file append."""
        if not entries:
            return
        try:
            if self.universe is not None:
                blob = bytearray()
                for shard_id, activation, timestamp in entries:
                    blob.extend(pack_txn_record(hash_shard_id(shard_id), activation, timestamp))
                self.universe.append_journal(bytes(blob))
            else:
                blob = bytearray()
                for shard_id, activation, timestamp in entries:
                    h = hash_shard_id(shard_id)
                    blob.extend(struct.pack("<4sQfd8s", b"TXN\x00", h, activation, timestamp, b"\x00" * 8))
                journal_path = os.path.join(self.storage_dir, "cortex_journal.bin")
                with open(journal_path, "ab") as f:
                    f.write(blob)
        except Exception:
            pass

    def _ledger_put_many(self, entries: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Places entries into the Active Latent Cortex Ledger under the
        RAM_LEDGER_MAX cap (flat-RAM guarantee at bulk-ingest scale).
        Updates to already-active nodes always apply. When the ledger is full,
        stronger newcomers evict the weakest residents; weaker ones stay
        asleep on disk (they remain fully durable in their crystal).
        Returns the shard_ids actually placed.
        """
        cap = tuning_manager.get_int("RAM_LEDGER_MAX", 50000)
        accepted: List[str] = []
        new_items: List[Tuple[str, Dict[str, Any]]] = []
        for sid, entry in entries.items():
            if sid in self.ram_ledger:
                self.ram_ledger[sid] = entry
                accepted.append(sid)
            else:
                new_items.append((sid, entry))
        if not new_items:
            return accepted

        space = cap - len(self.ram_ledger)
        if space >= len(new_items):
            for sid, entry in new_items:
                self.ram_ledger[sid] = entry
                accepted.append(sid)
            return accepted

        # Strongest newcomers claim free space first, then trade against the
        # weakest residents in a single bounded pass.
        new_items.sort(key=lambda kv: kv[1].get("activation", 0.0), reverse=True)
        head = max(0, space)
        for sid, entry in new_items[:head]:
            self.ram_ledger[sid] = entry
            accepted.append(sid)
        remaining = new_items[head:]
        if remaining:
            weakest = heapq.nsmallest(
                len(remaining), self.ram_ledger.items(),
                key=lambda kv: kv[1].get("activation", 0.0)
            )
            for (sid, entry), (wsid, wstate) in zip(remaining, weakest):
                if entry.get("activation", 0.0) > wstate.get("activation", 0.0):
                    del self.ram_ledger[wsid]
                    self.ram_ledger[sid] = entry
                    accepted.append(sid)
                else:
                    break  # both lists sorted; no later pair can succeed
        return accepted

    def checkpoint(self) -> None:
        """
        Crystallizing Checkpoints:
        Folds the Active RAM ledger snapshot + current atlas into the
        universe.loom container (atomic rebuild) and clears the journal tail.
        """
        try:
            records = bytearray()
            for sid, entry in list(self.ram_ledger.items()):
                act = entry.get("activation", 0.0)
                if act > 0.0:
                    h = hash_shard_id(sid)
                    t = int(entry.get("last_recalled", 0.0))
                    records.extend(struct.pack("<QfI", h, act, t))

            if self.universe is not None:
                cctx = zstd.ZstdCompressor()
                segments = self._carryover_segments()
                segments["atlas"] = self.atlas.to_bytes()
                segments["snapshot"] = cctx.compress(bytes(records)) if records else b""
                self.universe.rebuild(segments, journal=b"")  # journal folded into snapshot
                self._atlas_dirty = False
            else:
                if records:
                    cctx = zstd.ZstdCompressor()
                    compressed = cctx.compress(bytes(records))
                    snapshot_path = os.path.join(self.storage_dir, "latent_state.snapshot")
                    with open(snapshot_path, "wb") as f:
                        f.write(compressed)
                # Truncate journal
                journal_path = os.path.join(self.storage_dir, "cortex_journal.bin")
                with open(journal_path, "wb") as f:
                    f.truncate(0)
        except Exception:
            pass

    def _get_split_path(self, path: str) -> str:
        """Procedurally determine the next sub-sector split path name."""
        dir_name, file_name = os.path.split(path)
        base, ext = os.path.splitext(file_name)
        if "_" in base:
            parts = base.split("_")
            last = parts[-1]
            if len(last) == 1 and 'A' <= last <= 'Z':
                next_char = chr(ord(last) + 1)
                new_base = "_".join(parts[:-1]) + f"_{next_char}"
            else:
                new_base = base + "_A"
        else:
            new_base = base + "_A"
        return os.path.join(dir_name, new_base + ext)

    def _route_for_ingest(self, route_vec: np.ndarray) -> str:
        """
        Atlas Capture (Routing) + Cellular Division. Returns the target crystal
        path, splitting a full crystal into a new hot sub-sector when needed.

        `route_vec` must be the raw SEMANTIC embedding, not the physics-blended
        momentum vector — same convention as _resolve_bucket/leader signatures.
        Crystal centroids are tracked in that same space (atlas.update_centroid)
        specifically so recall()'s query routing (which has no per-shard
        scaffold and so no momentum equivalent to compare) agrees with where a
        shard actually landed. Routing on momentum was tried first and broke
        recall silently the instant more than one crystal existed — invisible
        before mechanism 1 (entropy/cohesion-triggered division) made division
        actually happen; the old MAX_CRYSTAL_SIZE count ceiling essentially
        never fired, so the atlas was always exactly one crystal in practice.
        """
        if not self.atlas.crystals:
            target = os.path.join(self.storage_dir, "crystal_1.loom")
            self.atlas.register_crystal(target, route_vec, state="hot")
            self._journal_crystal_registration(target)
            return target

        # Ingest routes ONLY among crystals with headroom. Full crystals keep
        # serving recall but never receive new shards — otherwise every routing
        # hit on a full crystal spawns another split (division cascade).
        #
        # "Full" is no longer pure item count (mechanism 1, entropy/cohesion-
        # triggered division): a crystal also becomes full once it stops being
        # a coherent unit of knowledge (avg cosine-to-centroid similarity below
        # CRYSTAL_COHESION_MIN), so long as it has at least MIN_CRYSTAL_SPLIT_SIZE
        # members (never split a handful of tightly related facts just because
        # they happen to sit at a wide angle early on). MAX_CRYSTAL_SIZE remains
        # a hard safety ceiling regardless of cohesion.
        min_split_size = tuning_manager.get_int("MIN_CRYSTAL_SPLIT_SIZE", 5000)
        cohesion_min = tuning_manager.get_float("CRYSTAL_COHESION_MIN", 0.27)

        def _is_full(info: Dict[str, Any]) -> bool:
            n = info.get("num_shards", 0)
            if n >= self.max_shards_per_crystal:
                return True
            if cohesion_min > 0.0 and n >= min_split_size and GlobalAtlasRouter.avg_cohesion(info) < cohesion_min:
                return True
            return False

        open_paths = [p for p, info in self.atlas.crystals.items() if not _is_full(info)]
        if open_paths:
            if len(open_paths) == len(self.atlas.crystals):
                return self.atlas.route_vector(route_vec)
            v_norm = np.linalg.norm(route_vec)
            best_path, best_sim = open_paths[0], -2.0
            for p in open_paths:
                centroid = self.atlas.crystals[p]["centroid"]
                c_norm = np.linalg.norm(centroid)
                sim = 0.0 if (c_norm == 0 or v_norm == 0) else float(np.dot(route_vec, centroid) / (v_norm * c_norm))
                if sim > best_sim:
                    best_sim, best_path = sim, p
            return best_path

        # Every crystal is full → cellular division of the nearest one
        target = self.atlas.route_vector(route_vec)
        crystal_info = self.atlas.crystals.get(target)
        if crystal_info:
            # Freeze current crystal
            self.atlas.set_state(target, "warm")

            # Determine split path (skip names already registered)
            new_path = self._get_split_path(target)
            while new_path in self.atlas.crystals:
                new_path = self._get_split_path(new_path)

            # Compute perturbed sub-centroid (Proximity Retention)
            offset = self.seed_core.get_macro_offset(os.path.basename(new_path))
            old_centroid = crystal_info["centroid"]
            new_centroid = old_centroid + 0.05 * offset

            old_norm = np.linalg.norm(old_centroid)
            new_norm = np.linalg.norm(new_centroid)
            if new_norm > 0:
                new_centroid = new_centroid * (old_norm / new_norm)

            self.atlas.register_crystal(new_path, new_centroid, state="hot")
            self._journal_crystal_registration(new_path)
            return new_path
        return target

    @staticmethod
    def _momentum_signature(momentum: np.ndarray) -> bytes:
        """128-bit packed sign signature of a momentum vector (routing leader form)."""
        sig = np.sign(momentum[:128]).astype(np.int8)
        sig[sig == 0] = 1
        if len(sig) < 128:
            sig = np.pad(sig, (0, 128 - len(sig)), constant_values=1)
        return np.packbits(((sig + 1) // 2).astype(np.uint8)).tobytes()

    def _resolve_bucket(self, store: LoomStore, momentum: np.ndarray) -> int:
        """
        Routing-space update AND resonance-jump bucket resolution in one step:
        finds the leader whose 128-bit sign signature is closest to this
        momentum vector's. Within Hamming distance 32/128 → assimilation into
        that existing neighborhood (returns its index). Otherwise, if there's
        room, spawns a new leader — accommodation, a brand-new neighborhood
        (returns the new leader's index). This is the SAME self-organizing
        partition recall() later uses to jump straight to the right
        neighborhood instead of scanning a whole crystal.

        Leaders are capped (LOOM_MAX_LEADERS) so sparse signature spaces
        (worst case: uniform random vectors) can't grow one leader per shard
        and turn every ingest into an O(n) Hamming scan.
        """
        max_leaders = tuning_manager.get_int("LOOM_MAX_LEADERS", 512)
        leader_matrix = store.get_leader_matrix()
        sig_packed = self._momentum_signature(momentum)

        if len(leader_matrix):
            sig_arr = np.frombuffer(sig_packed, dtype=np.uint8)
            dists = _hamming_distances(leader_matrix, sig_arr)
            best_idx = int(np.argmin(dists))
            best_dist = int(dists[best_idx])
            if best_dist <= 32:
                return best_idx  # assimilation

        if len(leader_matrix) >= max_leaders:
            # No room for a new neighborhood — fall back to the nearest
            # existing one rather than leaving the shard unbucketed.
            return best_idx if len(leader_matrix) else 0

        sig = np.sign(momentum[:128]).astype(np.int8)
        sig[sig == 0] = 1
        store.append_leader(sig)
        return len(leader_matrix)  # index of the newly appended leader

    def ingest_shard(
        self,
        shard_id: str,
        true_vector: np.ndarray,
        text: str,
        mass: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Ingestion Pathway (Writing):
        1. Seed Evaluation: get default micro scaffold
        2. Field Interference: calculate Momentum Vector
        3. Atlas Capture: Route to target .loom container (Cellular division fracturing)
        4. Substrate Append Execution: ONE append to the physical binary log —
           the crystal is never read back or rewritten.
        """
        if len(true_vector) != self.dimension:
            raise ValueError(f"Vector dimension must be {self.dimension}")

        # 1. Seed Evaluation
        seed_scaffold = self.seed_core.get_micro_socket(shard_id)

        # 2. Field Interference (Momentum Vector)
        momentum = self.physics.calculate_momentum_vector(
            true_vector=true_vector,
            seed_scaffold=seed_scaffold,
            mass=mass
        )

        # 3. Atlas Capture (Routing + Cellular Division + Border Embassy)
        target_crystal_path = self._route_for_ingest(true_vector)
        embassy_eval = self.atlas.route_vector_with_embassy(true_vector)

        # 4. Substrate Append Execution
        store = self._get_store(target_crystal_path)
        # Resonance-jump bucket resolution must use the SEMANTIC vector, not the
        # scaffold-blended momentum: recall() signatures the raw query (which has
        # no per-id scaffold), so bucketing by momentum here put semantically
        # identical content into different buckets (its scaffold differs per
        # shard_id) and made narrowed recall miss even a shard's own vector.
        # Content-based signatures on both sides = correct random-hyperplane LSH.
        bucket_id = self._resolve_bucket(store, true_vector)

        now = time.time()
        meta = dict(metadata or {})
        meta.update({
            "text": text,
            "mass": mass,
            "last_recalled": now,
            "hits": 0,
            "activation": 1.0,
            "bucket_id": bucket_id,
            "is_embassy": embassy_eval.get("is_embassy", False),
            "secondary_crystal": embassy_eval.get("secondary_crystal"),
        })
        true_vector = np.asarray(true_vector, dtype=np.float32)
        crystal_idx = store.append_shard(shard_id, true_vector, mass, meta)

        # Update centroid in global atlas (folded into container at checkpoint)
        self.atlas.update_centroid(target_crystal_path, true_vector)
        self._mark_atlas_dirty()

        # Update active RAM ledger (cap-guarded) and delta log
        # #6: intern the crystal path so all shards routed to the same crystal
        # share ONE string object in hash_to_shard, instead of one fresh copy per
        # ingest (the dominant repeated-string cost of this O(n)-RAM map at 1M
        # shards — a few unique paths, millions of references). Full mmap-backing
        # of the map is the larger deferred win; this is the safe, free part.
        target_crystal_path = sys.intern(target_crystal_path)
        self.hash_to_shard[hash_shard_id(shard_id)] = (shard_id, target_crystal_path)
        placed = self._ledger_put_many({shard_id: {
            "activation": 1.0,
            "latent_position": true_vector.copy(),
            "velocity": np.zeros(self.dimension, dtype=np.float32),
            "phase_angle": _seed_phase(true_vector),
            "hits": 0,
            "last_recalled": now,
            "crystal_path": target_crystal_path,
            "crystal_idx": crystal_idx,
            **_default_cognitive_fields()
        }})
        if placed:
            self._append_journal_entry(shard_id, 1.0, now)
            # Part B: a fresh perception — let the shared cognitive field feel it.
            self.process_cognitive_tick({shard_id: 0.8})
            # Direct sequential-adjacency causal signal ("what was ingested
            # right before this") — simpler and far more reliable than
            # depending solely on emergent assembly-transition detection,
            # which is sparse/threshold-gated. Complements it, doesn't
            # replace it: record_transition still rejects pure temporal
            # noise below CAUSAL_THRESHOLD.
            if self._last_ingested_sid is not None and self._last_ingested_sid != shard_id:
                if self._ingest_edge_coherent(self._last_ingested_sid, true_vector):
                    self.causal.record_transition(self._last_ingested_sid, shard_id, weight=1.0, prediction_error=0.5)
            self._last_ingested_sid = shard_id
            self._last_message_ts = now
            self._shards_since_last_sleep += 1

        return target_crystal_path

    def _ingest_edge_coherent(self, prev_sid: str, new_pos: np.ndarray) -> bool:
        """
        Mechanism 5 (coherence-seeded causal edges): ingest-order adjacency is
        exactly why CAUSAL_SPREAD_WEIGHT ships disabled by default — sequential
        adjacency on shuffled bulk data is noise. Gate the ingest-order edge by
        the same cosine-coherence primitive _coherence_filter uses (anchor vs.
        candidate), so two shards that just happen to land next to each other
        in ingest order but aren't topically related never get an edge. Fails
        OPEN (returns True) if the previous shard's position isn't resident —
        can't verify, and this is strictly an additional gate, not a new
        requirement for edges that already existed.
        """
        prev_entry = self.ram_ledger.get(prev_sid)
        if prev_entry is None:
            return True
        coherence_min = tuning_manager.get_float("CAUSAL_EDGE_COHERENCE_MIN", 0.3)
        if coherence_min <= 0.0:
            return True
        prev_pos = np.asarray(prev_entry["latent_position"], dtype=np.float32)
        new_pos = np.asarray(new_pos, dtype=np.float32)
        pn, nn = np.linalg.norm(prev_pos), np.linalg.norm(new_pos)
        if pn <= 1e-9 or nn <= 1e-9:
            return True
        sim = float(np.dot(prev_pos / pn, new_pos / nn))
        return sim >= coherence_min

    def ingest_batch(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Bulk Ingestion Pathway — the millions-scale write path.
        items: [{"shard_id", "vector", "text", "mass"?, "metadata"?}, ...]
        Routes every shard, then performs ONE batched append per target crystal,
        ONE atlas save, and ONE journal write. Returns {crystal_path: count}.
        """
        if not items:
            return {}
        now = time.time()
        grouped: Dict[str, List[Tuple[str, np.ndarray, float, Dict[str, Any]]]] = {}
        momenta: Dict[str, List[np.ndarray]] = {}
        journal_entries: List[Tuple[str, float, float]] = []

        for item in items:
            shard_id = item["shard_id"]
            vec = np.asarray(item["vector"], dtype=np.float32)
            if len(vec) != self.dimension:
                raise ValueError(f"Vector dimension must be {self.dimension} (shard {shard_id})")
            mass = float(item.get("mass", 1.0))

            scaffold = self.seed_core.get_micro_socket(shard_id)
            momentum = self.physics.calculate_momentum_vector(vec, scaffold, mass)
            # Routes on the raw semantic vector, not momentum — see
            # _route_for_ingest's docstring.
            target = self._route_for_ingest(vec)
            # In-memory centroid update keeps division-aware routing correct mid-batch.
            self.atlas.update_centroid(target, vec)

            meta = dict(item.get("metadata") or {})
            meta.update({
                "text": item.get("text", ""),
                "mass": mass,
                "last_recalled": now,
                "hits": 0,
                "activation": 1.0
            })
            grouped.setdefault(target, []).append((shard_id, vec, mass, meta))
            momenta.setdefault(target, []).append(momentum)

        counts: Dict[str, int] = {}
        ledger_batch: Dict[str, Dict[str, Any]] = {}
        for target, shard_items in grouped.items():
            target = sys.intern(target)  # #6: canonical path shared across batches
            store = self._get_store(target)

            # Resolve each shard's leader bucket (also grows the routing
            # space itself — assimilation vs. accommodation, same as ingest_shard).
            # Bucket by the SEMANTIC vector, not momentum — see ingest_shard for
            # why (recall signatures the raw query; content signatures must match).
            for (shard_id, vec, mass, meta), momentum in zip(shard_items, momenta[target]):
                meta["bucket_id"] = self._resolve_bucket(store, vec)

            indices = store.append_batch(shard_items)
            counts[target] = len(shard_items)

            for (shard_id, vec, mass, meta), crystal_idx in zip(shard_items, indices):
                self.hash_to_shard[hash_shard_id(shard_id)] = (shard_id, target)
                ledger_batch[shard_id] = {
                    "activation": 1.0,
                    "latent_position": vec.copy(),
                    "velocity": np.zeros(self.dimension, dtype=np.float32),
                    "phase_angle": _seed_phase(vec),
                    "hits": 0,
                    "last_recalled": now,
                    "crystal_path": target,
                    "crystal_idx": crystal_idx,
                    **_default_cognitive_fields()
                }

        placed = set(self._ledger_put_many(ledger_batch))
        journal_entries = [(sid, 1.0, now) for sid in placed]
        self._mark_atlas_dirty()
        self._append_journal_entries(journal_entries)
        if placed:
            # Part B: one cognitive tick for the whole arriving batch — the
            # bounded hot-set cap inside process_cognitive_tick keeps this
            # cheap even for a 5,000-shard batch.
            self.process_cognitive_tick({sid: 0.8 for sid in placed})
            # Direct sequential-adjacency causal signal, chained through the
            # batch in the caller's original order (the narrative order),
            # same as ingest_shard.
            ordered_placed = [item["shard_id"] for item in items if item["shard_id"] in placed]
            prev = self._last_ingested_sid
            for sid in ordered_placed:
                if prev is not None and prev != sid and self._ingest_edge_coherent(prev, ledger_batch[sid]["latent_position"]):
                    self.causal.record_transition(prev, sid, weight=1.0, prediction_error=0.5)
                prev = sid
            if ordered_placed:
                self._last_ingested_sid = ordered_placed[-1]
            self._last_message_ts = now
            self._shards_since_last_sleep += len(placed)
        return counts

    def enforce_spatiotemporal_decay(self, current_time: float) -> None:
        """
        Evaluates memory states against continuous decay curves — vectorized
        (one numpy pass over the whole ledger; a per-entry Python loop here
        was the dominant cost of recall() at scale, unrelated to which crystal
        or how many candidates a query touches).
        Triggers Deep Sleep evictions immediately when activation drops to zero (<= 0.001).
        """
        if not self.ram_ledger:
            return

        shard_ids = list(self.ram_ledger.keys())
        states = list(self.ram_ledger.values())
        init_act = np.array([s["activation"] for s in states], dtype=np.float64)
        last_recalled = np.array([s["last_recalled"] for s in states], dtype=np.float64)
        hits = np.array([s["hits"] for s in states], dtype=np.float64)

        decayed = self.fluidity.calculate_decay_batch(init_act, last_recalled, current_time, hits)

        sleep_eviction_limit = tuning_manager.get_float("SLEEP_EVICTION_LIMIT", 0.001)
        evict_mask = decayed <= sleep_eviction_limit

        for shard_id, state, dec, evict in zip(shard_ids, states, decayed, evict_mask):
            if evict:
                self._tombstone(shard_id, state, reason="decay")
                del self.ram_ledger[shard_id]
            else:
                state["activation"] = float(dec)
                state["last_recalled"] = current_time

    def process_cognitive_tick(self, resonance_input: Dict[str, float]) -> Optional["CognitiveAssembly"]:
        """
        Part B — the merged cognition step. Evolves a shared latent field and
        per-shard cognitive physics (energy/entropy/stability/resonance/
        momentum/attention) directly on the SAME ram_ledger entries storage-
        physics already maintains — no separate concept registry, no second
        copy of a shard's position. Ported from the (until now unwired)
        GlobalCognitiveState.process_tick — same formulas, same tuning keys,
        new home. Called after every ingest/recall with the shard_ids just
        touched, as a real percept.

        Bounded to a small hot working set (COGNITION_WORKING_SET_MAX) so the
        O(m^2) attraction/repulsion physics never scales with the full RAM
        ledger (up to RAM_LEDGER_MAX=50,000) — cognition runs on a small
        active focus, matching the deferred WorkingMemory's own design intent
        (capacity=10), just reused at a slightly larger, still-bounded scale.
        This is additive: it runs after the existing, unchanged storage-
        physics scoring in ingest/recall and never alters what those return.
        """
        if not resonance_input:
            return None

        working_set_max = tuning_manager.get_int("COGNITION_WORKING_SET_MAX", 128)

        # Cap the touched-this-tick set itself first (a bulk ingest can hand
        # us thousands of shard_ids at once), then, if there's room, round out
        # the hot set with the currently-strongest awake shards.
        touched = [sid for sid in resonance_input.keys() if sid in self.ram_ledger]
        if len(touched) > working_set_max:
            touched.sort(key=lambda sid: resonance_input.get(sid, 0.0), reverse=True)
            touched = touched[:working_set_max]
        hot_ids = list(touched)
        if len(self.ram_ledger) > len(hot_ids) and len(hot_ids) < working_set_max:
            hot_set_lookup = set(hot_ids)
            ranked = sorted(self.ram_ledger.items(), key=lambda kv: kv[1].get("activation", 0.0), reverse=True)
            for sid, _ in ranked:
                if len(hot_ids) >= working_set_max:
                    break
                if sid not in hot_set_lookup:
                    hot_ids.append(sid)
                    hot_set_lookup.add(sid)
        if not hot_ids:
            return None

        dim = self.dimension
        if self.latent_field is None:
            self.latent_field = np.zeros(dim, dtype=np.float32)
        if self.working_latent is None:
            self.working_latent = np.zeros(dim, dtype=np.float32)
        if self.latent_velocity is None:
            self.latent_velocity = np.zeros(dim, dtype=np.float32)

        # 1. Input tensor from whatever was just touched this tick.
        input_tensor = np.zeros(dim, dtype=np.float32)
        for sid, intensity in resonance_input.items():
            entry = self.ram_ledger.get(sid)
            if entry is not None:
                input_tensor += np.asarray(entry["latent_position"], dtype=np.float32) * intensity
        inorm = np.linalg.norm(input_tensor)
        if inorm > 0:
            input_tensor /= inorm

        prev_latent = self.latent_field.copy()

        # 2. Second-order mass-spring-damper on the shared latent field —
        #    same formulas/tuning keys as GlobalCognitiveState, reused here so
        #    both share one tuning surface.
        sync_force = self.working_latent - prev_latent
        inhibition = self.working_latent * tuning_manager.get_float("INHIBITION_FACTOR", 0.15)
        drag = tuning_manager.get_float("VELOCITY_DRAG", 0.8)
        input_gain = tuning_manager.get_float("INPUT_FORCE_GAIN", 0.4)
        sync_gain = tuning_manager.get_float("SYNC_FORCE_GAIN", 0.2)
        self.latent_velocity = (drag * self.latent_velocity) + (input_tensor * input_gain) + (sync_force * sync_gain) - inhibition
        new_latent = prev_latent + self.latent_velocity
        lnorm = np.linalg.norm(new_latent)
        self.latent_field = (new_latent / lnorm) if lnorm > 0 else new_latent

        wm_alpha = tuning_manager.get_float("WM_EMA_ALPHA", 0.1)
        self.working_latent = (1.0 - wm_alpha) * self.working_latent + wm_alpha * self.latent_field
        wnorm = np.linalg.norm(self.working_latent)
        if wnorm > 0:
            self.working_latent = self.working_latent / wnorm

        # 3. Per-shard physics over the bounded hot set: attraction to the
        #    shared field, causal-spring pull, mutual repulsion — writing
        #    straight back into the ram_ledger entries storage-physics owns.
        dt = 0.1
        repulsion_radius = tuning_manager.get_float("REPULSION_RADIUS", 0.2)
        repulsion_constant = tuning_manager.get_float("REPULSION_CONSTANT", 0.02)
        drag_friction = tuning_manager.get_float("DRAG_FRICTION", 0.7)

        positions = {sid: np.asarray(self.ram_ledger[sid]["latent_position"], dtype=np.float32) for sid in hot_ids}
        active_vecs = list(positions.values())

        # Vectorized pairwise repulsion (was an O(m^2) Python loop calling
        # np.linalg.norm() per pair — ~16k Python-level calls per tick at the
        # working-set cap of 128. Same physics, computed as a handful of
        # broadcasted array ops instead: one (m,m,dim) diff, one (m,m) norm.
        pos_matrix = np.stack(active_vecs)  # (m, dim), same order as hot_ids
        diffs = pos_matrix[:, None, :] - pos_matrix[None, :, :]  # (m, m, dim)
        dists = np.linalg.norm(diffs, axis=-1)  # (m, m)
        within_radius = dists < repulsion_radius
        np.fill_diagonal(within_radius, False)  # exclude self-pair (dist=0)
        safe_dists = np.where(within_radius, dists + 1e-5, 1.0)[:, :, None]
        contrib = np.where(within_radius[:, :, None], diffs / safe_dists, 0.0)
        repulsion_matrix = contrib.sum(axis=1) * repulsion_constant  # (m, dim)

        for idx, sid in enumerate(hot_ids):
            entry = self.ram_ledger[sid]
            pos = positions[sid]
            vel = np.asarray(entry.get("velocity", np.zeros(dim, dtype=np.float32)), dtype=np.float32)
            activation = float(entry.get("activation", 0.5))

            f_latent = (self.latent_field - pos) * (0.25 * activation)

            f_causal = np.zeros(dim, dtype=np.float32)
            if self.causal_graph.has_node(sid):
                for _, target, data in self.causal_graph.edges(sid, data=True):
                    if target in positions:
                        weight = data.get("weight", 0.5)
                        f_causal += (positions[target] - pos) * (weight * 0.15)

            f_repulsion = repulsion_matrix[idx]

            vel = vel * drag_friction + (f_latent + f_causal + f_repulsion) * dt
            new_pos = pos + vel * dt
            pnorm = np.linalg.norm(new_pos)
            new_pos = (new_pos / pnorm) if pnorm > 1e-9 else new_pos

            entry["latent_position"] = new_pos
            entry["velocity"] = vel
            entry["resonance"] = float(np.clip(np.dot(new_pos, self.latent_field), 0.0, 1.0))
            entry["momentum"] = float(np.linalg.norm(vel))
            entry["entropy"] = float(np.clip(np.linalg.norm(vel), 0.0, 1.0))

            prev_energy = float(entry.get("energy", 1.0))
            max_energy = tuning_manager.get_float("MAX_COGNITIVE_ENERGY", 2.0)
            base_energy = tuning_manager.get_float("BASE_RESTING_ENERGY", 0.15)
            if resonance_input.get(sid, 0.0) > 0:
                entry["energy"] = min(max_energy, prev_energy + 0.2)
            else:
                entry["energy"] = max(base_energy, prev_energy - 0.02)

            stability = float(entry.get("stability", 1.0))
            entry["stability"] = min(1.0, stability + 0.02) if activation > 0.4 else max(0.01, stability - 0.005)

            # "attention" was carried in _default_cognitive_fields()/saved in
            # living_state.bin but never actually computed anywhere — left
            # behind when this tick was ported from the original
            # GlobalCognitiveState.process_tick (see cognitive_field_substrate.py's
            # AttentionDynamics), so it silently stayed 0.0 forever. Rises with
            # how strongly this tick's own resonance_input touched this shard,
            # decays multiplicatively otherwise — same decay convention as the
            # analogous attention_bias in cortex/TemporalCognitiveStream.py.
            prev_attention = float(entry.get("attention", 0.0))
            intensity = resonance_input.get(sid, 0.0)
            entry["attention"] = min(1.0, prev_attention + intensity * 0.5) if intensity > 0 else prev_attention * 0.85

        # 4. Field-level metrics + thermodynamic energy budget.
        self.cognitive_coherence = CognitiveMetrics.compute_coherence(active_vecs)
        self.cognitive_entropy = CognitiveMetrics.compute_entropy(self.latent_field, prev_latent)
        coherence_gain = max(0.0, self.cognitive_coherence - self._last_cognitive_coherence)
        self._last_cognitive_coherence = self.cognitive_coherence
        cognitive_load = self.cognitive_entropy * 0.02
        recovery = coherence_gain * 0.15 + (self.cognitive_coherence * 0.01)
        self.cognitive_energy_budget = float(np.clip(self.cognitive_energy_budget + recovery - cognitive_load, 0.0, 1.0))

        # 5. Attractor crystallization: real shard ids become the dominant
        #    concept — an emergent assembly is a cluster of real, retrievable
        #    shards resonating together right now, not an abstract label.
        projected = {}
        for sid in hot_ids:
            sim = float(np.dot(self.latent_field, self.ram_ledger[sid]["latent_position"]))
            if sim > 0.4:
                projected[sid] = sim

        # Semantic-coherence gate (bug #1): membership above only requires a
        # shard to resonate with the SHARED latent field, which is a blend of
        # everything touched this tick. When two unrelated topics are active at
        # once, both resonate with the blend and get bundled into one assembly
        # -> one topic-mixed semantic crystal downstream. Keep only members that
        # actually cohere with the assembly's dominant (highest-resonance)
        # member, so an assembly is a single concept, not an accidental bundle.
        coherence_min = tuning_manager.get_float("ASSEMBLY_COHERENCE_MIN", 0.5)
        if len(projected) > 1 and coherence_min > 0.0:
            projected = self._coherence_filter(projected, coherence_min)

        actual_vec = None
        if projected:
            core_positions = [self.ram_ledger[sid]["latent_position"] for sid in projected if sid in self.ram_ledger]
            if core_positions:
                actual_vec = np.mean(core_positions, axis=0)

        ca = self.compiler.compile_assembly(
            latent_field=self.latent_field,
            projected_activations=projected,
            entropy=self.cognitive_entropy,
            coherence=self.cognitive_coherence,
            pressure=0.0,
            semantic_field=None,
            actual_vec=actual_vec
        )

        if ca:
            self.active_assemblies.append(ca)
            if len(self.active_assemblies) > 1:
                prev_dominant = self.active_assemblies[-2].dominant_concept
                if prev_dominant != ca.dominant_concept:
                    # Grounds the causal graph in real shard_ids — exactly
                    # what recall()'s propagation hop (Part A) reads from.
                    self.causal.record_transition(prev_dominant, ca.dominant_concept)
            if len(self.active_assemblies) > 200:
                self.active_assemblies = self.active_assemblies[-200:]

        self.cognitive_tick += 1
        return ca

    def _causal_hop_indices(self, store: LoomStore, seed_abs_idx: np.ndarray) -> np.ndarray:
        """
        Propagation hop: for shards found via the resonance jump, pull in
        anything causally linked to them (self.causal_graph, grown by
        cognition-layer ticks) even if it sits in a different bucket — this
        is where a wave "spreads" past the direct hit. Iterates up to
        CAUSAL_HOP_DEPTH levels (default 2) so a genuine multi-hop chain
        (A resonance-matched -> B one hop -> C two hops) is actually
        reachable, not just direct neighbors of the resonance-jump seeds —
        each level only expands from nodes newly added by the previous one,
        capped by CAUSAL_HOP_MAX_NODES so a densely-connected graph can't
        blow the candidate set back up to O(n).
        Empty/no-op until cognition has run over live traffic and the graph
        has real edges.
        """
        if self.causal_graph.number_of_edges() == 0 or len(seed_abs_idx) == 0:
            return np.empty(0, dtype=np.int64)

        hop_depth = tuning_manager.get_int("CAUSAL_HOP_DEPTH", 2)
        max_hop_nodes = tuning_manager.get_int("CAUSAL_HOP_MAX_NODES", 2000)
        all_ids = store.get_all_ids()

        seen_sids = {all_ids[int(abs_i)] for abs_i in seed_abs_idx}
        frontier = set(seen_sids)
        collected = set()

        for _ in range(max(1, hop_depth)):
            next_frontier = set()
            for sid in frontier:
                if not self.causal_graph.has_node(sid):
                    continue
                for neighbor in list(self.causal_graph.successors(sid)) + list(self.causal_graph.predecessors(sid)):
                    if neighbor not in seen_sids:
                        next_frontier.add(neighbor)
            if not next_frontier:
                break
            seen_sids.update(next_frontier)
            collected.update(next_frontier)
            if len(collected) >= max_hop_nodes:
                break
            frontier = next_frontier

        if not collected:
            return np.empty(0, dtype=np.int64)
        hop_idx = [pos for pos in (store.index_of(sid) for sid in collected) if pos is not None]
        return np.asarray(hop_idx, dtype=np.int64)

    def _coherence_filter(self, members, coherence_min: float):
        """
        Semantic-coherence gate (bug #1). Given either a {sid: score} dict or a
        list of sids, drops members whose latent_position does not cohere
        (cosine) with the group's dominant anchor, so an assembly / semantic
        crystal covers ONE concept instead of an accidental co-active bundle.

        The anchor is the highest-scored member (dict) or the member closest to
        the group centroid (list) — the most representative shard, not just the
        first one, which also gives a meaningful dominant concept rather than
        "whatever shard happened to be first" (the old dominant_concept bug).
        Returns the same type it was given (dict subset or filtered list).
        """
        as_dict = isinstance(members, dict)
        sids = list(members.keys()) if as_dict else list(members)
        if len(sids) <= 1:
            return members

        def _pos(sid):
            p = np.asarray(self.ram_ledger[sid]["latent_position"], dtype=np.float32)
            n = np.linalg.norm(p)
            return p / n if n > 1e-9 else p

        pos = {sid: _pos(sid) for sid in sids}
        if as_dict:
            anchor_sid = max(sids, key=lambda s: members[s])
        else:
            centroid = np.mean([pos[s] for s in sids], axis=0)
            cn = np.linalg.norm(centroid)
            centroid = centroid / cn if cn > 1e-9 else centroid
            anchor_sid = max(sids, key=lambda s: float(np.dot(pos[s], centroid)))
        anchor = pos[anchor_sid]

        kept = [s for s in sids if s == anchor_sid or float(np.dot(pos[s], anchor)) >= coherence_min]
        if as_dict:
            return {s: members[s] for s in kept}
        # keep original ordering of the incoming list
        keptset = set(kept)
        return [s for s in members if s in keptset]

    def _mark_superseded(self, shard_id: str, superseded_by: str) -> None:
        """
        Records that a shard has been consolidated into a newer semantic
        crystal — living state (like activation/hits), never a mutation of
        the immutable .loom log. The source shard stays fully readable via
        decode_cluster(); it's only deprioritized in recall()'s fast path
        (see _superseded_indices_for), matching the same safety-net
        philosophy as UNBUCKETED — deprioritized, never unreachable.
        """
        entry = self.ram_ledger.get(shard_id)
        crystal_path = entry.get("crystal_path") if entry else None
        if not crystal_path:
            hit = self.hash_to_shard.get(hash_shard_id(shard_id))
            crystal_path = hit[1] if hit else None
        if not crystal_path:
            return
        self.superseded_by_crystal.setdefault(crystal_path, {})[shard_id] = superseded_by
        self._superseded_idx_cache.pop(crystal_path, None)
        if entry is not None:
            self._tombstone(shard_id, entry, reason="consolidated", superseded_by=superseded_by)

    def _tombstone(self, shard_id: str, entry: Dict[str, Any], reason: str, superseded_by: Optional[str] = None) -> None:
        """
        Mechanism 4 (forgetting ledger with provenance): called at every RAM-
        ledger eviction site instead of a bare `del`. Keeps a cheap trace of
        what a shard was and why it went — never the full content, that would
        defeat the point of forgetting — appended to an on-disk .jsonl log so
        the substrate can explain a gap in its own memory instead of either
        hallucinating an answer or returning nothing.
        """
        gist = ""
        cpath, cidx = entry.get("crystal_path"), entry.get("crystal_idx")
        if cpath and cidx is not None:
            try:
                gist = (self._get_store(cpath).get_meta(cidx).get("text") or "")[:120]
            except Exception:
                pass
        rec = {
            "shard_id": shard_id,
            "gist": gist,
            "reason": reason,
            "last_recalled": entry.get("last_recalled"),
            "superseded_by": superseded_by,
            "ts": time.time(),
        }
        self.tombstones.append(rec)
        if len(self.tombstones) > 2000:
            self.tombstones = self.tombstones[-2000:]
        try:
            with open(self._tombstone_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def explain_forgotten(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Answers "did you once know this, and why did it fade" by cosine-
        matching the query against tombstone gists (re-embedded on the fly —
        the tombstone log is small, this is not a hot path). Returns the
        top_k closest tombstones above a minimal relevance floor, each with
        its reason and, if applicable, what replaced it.
        """
        if not self.tombstones:
            return []
        from module_loom.services.embedding.embedding_manager import get_embedding_model
        model = get_embedding_model()
        qv = np.asarray(query_vector, dtype=np.float32)
        qn = np.linalg.norm(qv)
        if qn <= 1e-9:
            return []
        qv = qv / qn
        scored = []
        for rec in self.tombstones:
            if not rec.get("gist"):
                continue
            gv = np.asarray(model.encode(rec["gist"]), dtype=np.float32)
            gn = np.linalg.norm(gv)
            if gn <= 1e-9:
                continue
            sim = float(np.dot(qv, gv / gn))
            if sim > 0.3:
                scored.append((sim, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**rec, "relevance": sim} for sim, rec in scored[:top_k]]

    def _superseded_indices_for(self, store: LoomStore, crystal_path: str) -> np.ndarray:
        """Local store indices for every shard superseded in this crystal —
        resolved via store.index_of() once per change, cached thereafter."""
        sup_map = self.superseded_by_crystal.get(crystal_path)
        if not sup_map:
            return np.empty(0, dtype=np.int64)
        cached = self._superseded_idx_cache.get(crystal_path)
        if cached is not None:
            return cached
        idx = [pos for pos in (store.index_of(sid) for sid in sup_map) if pos is not None]
        resolved = np.asarray(idx, dtype=np.int64)
        self._superseded_idx_cache[crystal_path] = resolved
        return resolved

    def _apply_causal_spread(self, score: np.ndarray, cos: np.ndarray,
                             abs_idx: np.ndarray, store: LoomStore, spread_w: float) -> np.ndarray:
        """
        One round of spreading activation over the causal graph, HippoRAG-style:
        each candidate pushes a fraction of its own cosine relevance to its
        causal neighbours that are ALSO in the candidate set, and the neighbour
        INHERITS that relevance. This is how an associative / multi-hop answer —
        one that is causally linked to a strong match but is NOT itself similar
        to the query — can rise into the results even though plain cosine can't
        reach it.

        ADDITIVE, not multiplicative (fixed 2026-07-15 after the isolated 2-hop
        benchmark, run_bridge_multihop.py, exposed the bug). The genuine
        multi-hop answer has cosine ~0 to the query, so its base gravity score
        is ~0; a `score *= (1 + boost)` factor cannot lift ~0 — the answer stays
        at the bottom (measured: docloom answer@5 = 0.00 on the hard subset). An
        additive injection lets a spread-only hit land just BELOW genuine direct
        matches (bounded to spread_w * a high percentile of the base scores), so
        it enters top-k without ever swamping a real direct match.

        Only meaningful when the causal edges are informative (narrative or
        co-access order); on a shuffled bulk load the sequential-adjacency edges
        are arbitrary, which is why CAUSAL_SPREAD_WEIGHT defaults to 0 (off) and
        this whole path is opt-in — plain retrieval (spread_w=0) never calls it.
        """
        G = self.causal_graph
        ids = store.get_all_ids()
        local_of: Dict[str, int] = {}
        sid_of: List[Optional[str]] = [None] * len(abs_idx)
        for pos, a in enumerate(abs_idx):
            a = int(a)
            if a < len(ids):
                sid_of[pos] = ids[a]
                local_of[ids[a]] = pos
        rel = np.maximum(cos, 0.0)
        # Only genuinely-relevant candidates seed the spread — otherwise every
        # filler, whose gravity score is nonzero even at cos~0, would leak score
        # to its sequential-chain neighbours and drown the signal. Gate on cosine.
        seed_min = tuning_manager.get_float("CAUSAL_SPREAD_SEED_MIN", 0.2)
        # Propagate in SCORE space: a candidate lets each causal neighbour INHERIT
        # a fraction of its own ranking score (taking the strongest such offer).
        # So a cosine-distant answer inherits ~spread_w * (its strong neighbour's
        # score) — landing just below that direct match, above the distractors,
        # which is exactly the multi-hop lift a cosine-blind index cannot make.
        inherited = np.zeros(len(abs_idx), dtype=np.float32)
        for pos, sid in enumerate(sid_of):
            if sid is None or rel[pos] < seed_min or sid not in G:
                continue
            src_score = float(score[pos])
            if src_score <= 0.0:
                continue
            for nbr in list(G.successors(sid)) + list(G.predecessors(sid)):
                j = local_of.get(nbr)
                if j is None:
                    continue
                if G.has_edge(sid, nbr):
                    w = G[sid][nbr].get("weight", 1.0)
                else:
                    w = G[nbr][sid].get("weight", 1.0)
                # Clamp weight to <=1 so one hop can never make a neighbour
                # outrank its own seed (offer <= src_score); spread_w<1 then keeps
                # the inherited hit strictly below the direct match it came from.
                offer = src_score * min(float(w), 1.0)
                if offer > inherited[j]:
                    inherited[j] = offer
        if len(inherited) and float(inherited.max()) > 0.0:
            score = np.maximum(score, spread_w * inherited)
        return score.astype(np.float32)

    def _apply_working_set_boost(self, score: np.ndarray, cos: np.ndarray,
                                  abs_idx: np.ndarray, store: LoomStore, boost_w: float) -> np.ndarray:
        """
        Mechanism 7 (usage-pattern "working set" memory): same additive-in-
        score-space spread as _apply_causal_spread, but keyed on pure co-recall
        frequency (self.working_set_pairs) instead of content-coherence-gated
        causal edges — this is the ONLY reorganization signal here driven by
        ACCESS pattern rather than content, binding cross-topic pairs that keep
        getting retrieved together (repeated use should warm this pair up the
        way any cache warms, independent of semantic distance).
        """
        ids = store.get_all_ids()
        sid_of: List[Optional[str]] = [None] * len(abs_idx)
        local_of: Dict[str, int] = {}
        for pos, a in enumerate(abs_idx):
            a = int(a)
            if a < len(ids):
                sid_of[pos] = ids[a]
                local_of[ids[a]] = pos
        rel = np.maximum(cos, 0.0)
        seed_min = tuning_manager.get_float("CAUSAL_SPREAD_SEED_MIN", 0.2)

        # Adjacency restricted to sids actually in this candidate set, built
        # once (O(pairs)) rather than re-scanned per candidate (O(m*pairs)).
        candidate_sids = set(local_of.keys())
        neighbors: Dict[str, List[Tuple[str, int]]] = {}
        for (a, b), count in self.working_set_pairs.items():
            if a in candidate_sids and b in candidate_sids:
                neighbors.setdefault(a, []).append((b, count))
                neighbors.setdefault(b, []).append((a, count))

        inherited = np.zeros(len(abs_idx), dtype=np.float32)
        for pos, sid in enumerate(sid_of):
            if sid is None or rel[pos] < seed_min or sid not in neighbors:
                continue
            src_score = float(score[pos])
            if src_score <= 0.0:
                continue
            for nbr, count in neighbors[sid]:
                j = local_of.get(nbr)
                if j is None:
                    continue
                # Diminishing returns on raw count — a pair recalled together
                # 100 times shouldn't dominate 100x more than one recalled
                # together twice; log-scaled strength, capped to <=1.
                strength = min(1.0, np.log1p(count) / np.log1p(20))
                offer = src_score * strength
                if offer > inherited[j]:
                    inherited[j] = offer
        if len(inherited) and float(inherited.max()) > 0.0:
            score = np.maximum(score, boost_w * inherited)
        return score.astype(np.float32)

    def recall(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        current_time: Optional[float] = None,
        query_phase: float = 0.0,
        learn: bool = True
    ) -> List[Dict[str, Any]]:
        """
        `learn` (default True) = "learning mode": recall also runs the living
        dynamics — decay, reinforcement, re-awakening, Kuramoto coupling, the
        cognitive tick, and journaling. That is what makes this a memory rather
        than an index, but it costs a full write-and-think cycle per query.

        `learn=False` = "serving mode": pure, read-only retrieval (route → score
        → top-k), skipping every mutation and the O(ledger) overlay. Ranking is
        relevance-first so results match learning-mode ordering, at ~vector-index
        speed. Use it for high-throughput read-only querying / benchmarking; use
        learn=True when the substrate should actually remember the interaction.

        Recall Pathway (Querying):
        1. Topology Sweep: Find closest target .loom crystal in Atlas
        2. Resonance-Jump Entry: Hamming-jump to the nearest leader buckets
           (the crystal's own self-organized neighborhoods) instead of
           scanning every shard, then propagate one hop along the causal
           graph. Small/legacy crystals fall back to a full scan — identical
           numbers to before this existed.
        3. Wave Resonance + Gravity: batch physics, identical formulas either way
        4. Re-awaken sleeping nodes (capped by RECALL_MAX_WAKE strongest)
        5. Kuramoto Coupling: lock active phase angles
        Metadata is decoded ONLY for the returned top_k shards.
        """
        if not self.atlas.crystals:
            return []

        if current_time is None:
            current_time = time.time()

        # Enforce continuous decay for all active states first (Deep Sleep Protocol
        # check). Learning-mode only — it mutates the whole ledger; serving mode
        # ranks relevance-first and doesn't need live activation.
        if learn:
            self.enforce_spatiotemporal_decay(current_time)

        # 1. Topology Sweep — query_vector is a raw semantic embedding, and
        # crystal centroids are tracked in that same space (atlas.update_centroid
        # is now called with the semantic vector, not momentum; see
        # _route_for_ingest's docstring), so this agrees with where a shard
        # actually landed at ingest time.
        target_path = self.atlas.route_vector(query_vector)
        if not os.path.exists(target_path):
            return []

        # OS mounting emulation: transition Cold crystal to Warm for mmap access
        if self.atlas.crystals[target_path]["state"] == "cold":
            self.atlas.set_state(target_path, "warm")
            self._mark_atlas_dirty()

        store = self._get_store(target_path)
        n = store.n
        k = min(int(top_k), n)
        if n == 0 or k <= 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))

        # 2. Resonance-Jump Entry: narrow to bucket candidates when the crystal
        #    is large enough that scoring everyone is the real cost.
        full_scan_below = tuning_manager.get_int("RECALL_FULL_SCAN_BELOW", 5000)
        seed_leader_k = tuning_manager.get_int("RECALL_SEED_LEADERS", 5)
        narrowed = False
        abs_idx = None

        leader_matrix = store.get_leader_matrix()
        if n > full_scan_below and len(leader_matrix):
            bucket_ids = store.bucket_ids_all()
            sig_packed = self._momentum_signature(q)
            sig_arr = np.frombuffer(sig_packed, dtype=np.uint8)
            leader_dists = _hamming_distances(leader_matrix, sig_arr)
            seed_k = min(seed_leader_k, len(leader_dists))
            if seed_k < len(leader_dists):
                seed_leaders = np.argpartition(leader_dists, seed_k - 1)[:seed_k]
            else:
                seed_leaders = np.arange(len(leader_dists))

            # Safety net: shards from before bucket tracking existed (or
            # written through a path that never assigned one) always stay
            # reachable, never silently dropped by the fast path.
            mask = np.isin(bucket_ids, seed_leaders) | (bucket_ids == UNBUCKETED)
            hit_idx = np.nonzero(mask)[0]

            # Mechanism 6 (retrieval-adequacy self-check): only pay for the
            # causal-hop expansion when the resonance-jump set looks
            # insufficient on its own — an agentic retrieve-judge-retrieve
            # loop pushed into the storage layer instead of an LLM round-trip.
            # Adequacy is the best cosine already in the seed set: a strong
            # direct hit means the hop is very unlikely to change the answer.
            adequacy_min = tuning_manager.get_float("RETRIEVAL_ADEQUACY_MIN", 0.5)
            self._last_adequacy_score = 1.0
            self._last_hop_triggered = False
            always_hop = adequacy_min <= 0.0  # 0.0 = old unconditional-hop behavior
            if not always_hop and len(hit_idx):
                seed_vectors = store.gather_vectors(hit_idx)
                seed_norms = np.linalg.norm(seed_vectors, axis=1)
                seed_cos = (seed_vectors @ q) / np.maximum(seed_norms * q_norm, 1e-12)
                self._last_adequacy_score = float(np.max(seed_cos)) if len(seed_cos) else 0.0
            if always_hop or self._last_adequacy_score < adequacy_min:
                self._last_hop_triggered = True
                hop_idx = self._causal_hop_indices(store, hit_idx)
                if len(hop_idx):
                    hit_idx = np.union1d(hit_idx, hop_idx)

            if 0 < len(hit_idx) < n:
                abs_idx = hit_idx
                narrowed = True

        if not narrowed:
            abs_idx = np.arange(n, dtype=np.int64)

        # Deprioritize consolidated-away shards from the fast path (sleep
        # cycle) — same UNBUCKETED-style philosophy: never unreachable
        # (decode_cluster always sees them), just excluded from ranking here.
        sup_idx = self._superseded_indices_for(store, target_path)
        if len(sup_idx):
            abs_idx = abs_idx[~np.isin(abs_idx, sup_idx)]

        m = len(abs_idx)
        k = min(k, m)
        if m == 0 or k <= 0:
            return []

        # 3a. Gather candidate vectors: the actual payoff of narrowing — an
        #     O(m*dim) fancy-index gather instead of an O(n*dim) full scan.
        if narrowed:
            cand_vectors = store.gather_vectors(abs_idx)
            norms = store.norms_all()[abs_idx]
            mass = store.mass_all()[abs_idx]
        else:
            cand_vectors = None
            norms = store.norms_all()
            mass = store.mass_all()

        denom = norms * q_norm
        safe_denom = np.maximum(denom, 1e-12)
        if narrowed:
            cos = (cand_vectors @ q / safe_denom).astype(np.float32)
        else:
            cos = (store.dot_all(q) / safe_denom).astype(np.float32)
        cos[denom <= 1e-12] = 0.0

        # Wormhole warp for gravity (identity tensor short-circuits to plain cosine)
        if self.physics.is_identity_wormhole():
            cos_warped = cos
        elif narrowed:
            W = self.physics.wormhole_tensor
            qw = (W @ q).astype(np.float32)
            qw_norm = float(np.linalg.norm(qw))
            warped = cand_vectors @ W.T
            wn = np.linalg.norm(warped, axis=1) * qw_norm
            cos_warped = (warped @ qw) / np.maximum(wn, 1e-12)
            cos_warped[wn <= 1e-12] = 0.0
        else:
            W = self.physics.wormhole_tensor
            qw = (W @ q).astype(np.float32)
            qw_norm = float(np.linalg.norm(qw))
            cos_warped = np.zeros(n, dtype=np.float32)
            for start, block in store.iter_vector_blocks():
                warped = block @ W.T
                wn = np.linalg.norm(warped, axis=1) * qw_norm
                cw = (warped @ qw) / np.maximum(wn, 1e-12)
                cw[wn <= 1e-12] = 0.0
                cos_warped[start:start + len(block)] = cw

        if not narrowed and m != n:
            # Superseded-masking (above) can now punch a subset out of the
            # full-scan candidate set too — cos/cos_warped/mass were just
            # computed over all n rows on this branch; slice down to the m
            # survivors, in abs_idx's order, so they align with
            # initial_act/cell_phase (sized m) below. A no-op copy when
            # nothing was masked (m == n already skips this).
            cos = cos[abs_idx]
            cos_warped = cos_warped[abs_idx]
            mass = mass[abs_idx]

        # 3b. Overlay live RAM-ledger states (activation / phase / hits).
        # Local position i corresponds to absolute store index abs_idx[i].
        initial_act = np.zeros(m, dtype=np.float32)
        cell_phase = np.zeros(m, dtype=np.float32)
        prev_hits = np.zeros(m, dtype=np.int64)
        active_by_idx: Dict[int, str] = {}  # keyed by LOCAL position
        target_abs = os.path.abspath(target_path)
        # Always a real lookup, never a bare `pos = idx` shortcut: superseded
        # masking (above) can punch gaps into abs_idx even on the full-scan
        # (not narrowed) path now, so position-in-candidates and
        # absolute-store-index are no longer interchangeable in either branch.
        # Serving mode skips this O(ledger) Python overlay entirely: ranking is
        # relevance-first (initial_act stays 0 -> score == gravity == cosine
        # order), so the live-state overlay isn't needed to rank, and skipping it
        # removes the per-query full-ledger scan that dominates recall latency.
        if learn:
            abs_to_local = {int(a): pos for pos, a in enumerate(abs_idx)}
            for sid, state in self.ram_ledger.items():
                spath = state.get("crystal_path")
                if not spath or os.path.abspath(spath) != target_abs:
                    continue
                idx = state.get("crystal_idx")
                if idx is None or not (0 <= idx < n):
                    idx = store.index_of(sid)
                    if idx is None:
                        continue
                    state["crystal_idx"] = idx
                pos = abs_to_local.get(int(idx))
                if pos is None:
                    continue  # active elsewhere in the crystal, or masked out (superseded) — not among this query's candidates
                initial_act[pos] = state.get("activation", 1.0)
                cell_phase[pos] = state.get("phase_angle", 0.0)
                prev_hits[pos] = state.get("hits", 0)
                active_by_idx[pos] = sid

        # 3c. Wave Resonance + Gravitational scoring (batch physics) — same
        #     formulas whether narrowed or not, over m candidates instead of n.
        gain = self.physics.calculate_excitation_gain_batch(cos, query_phase, cell_phase)
        new_act = np.clip(initial_act + gain, 0.0, 1.0)   # memory DYNAMICS (drives wake/reinforce below)
        grav = self.physics.calculate_gravitational_attraction_batch(cos_warped, mass, mass_a=1.0)

        # --- RANKING score (separated from memory dynamics) ---
        # Relevance-first: gravity is monotonic in similarity, so on plain
        # retrieval this reproduces the cosine ordering (never worse than a
        # vector index). Activation adds a monotonic recency/importance nudge.
        # The Kuramoto PHASE term is deliberately NOT in the ranking: it is
        # memory dynamics, uncorrelated with relevance, and reordering by it is
        # what dragged ranking below plain cosine (measured 0.43 vs 0.50 R@1).
        rank_act_w = tuning_manager.get_float("RANK_ACTIVATION_WEIGHT", 0.25)
        score = (grav * (1.0 + rank_act_w * initial_act)).astype(np.float32)

        # Causal spreading-activation boost (HippoRAG-style, opt-in via tuning):
        # lift candidates reachable through the causal graph from strong-cosine
        # seeds — the associative/multi-hop answers plain similarity misses.
        # Defaults to a no-op when the graph has no edges among candidates, so
        # plain retrieval is unaffected.
        spread_w = tuning_manager.get_float("CAUSAL_SPREAD_WEIGHT", 0.0)
        if spread_w > 0.0 and self.causal_graph.number_of_edges() > 0:
            score = self._apply_causal_spread(score, cos, abs_idx, store, spread_w)

        working_set_w = tuning_manager.get_float("WORKING_SET_BOOST", 0.15)
        if working_set_w > 0.0 and self.working_set_pairs:
            score = self._apply_working_set_boost(score, cos, abs_idx, store, working_set_w)

        if k < m:
            top_idx = np.argpartition(-score, k - 1)[:k]
            top_idx = top_idx[np.argsort(-score[top_idx])]
        else:
            top_idx = np.argsort(-score)
        top_set = {int(i) for i in top_idx}

        # 4. Plasticity & Wave Excitation Reinforcement / Re-awakening.
        # At scale a strong query can excite most of a crystal; only the
        # RECALL_MAX_WAKE strongest sleepers are re-awakened into RAM
        # (already-active nodes and top_k hits are always processed).
        # LEARNING-MODE ONLY: this whole block (reinforcement, re-awakening,
        # journaling, Kuramoto) mutates state and is the bulk of recall's cost.
        final_hits: Dict[int, int] = {}
        final_act: Dict[int, float] = {}
        if learn:
            sleep_limit = tuning_manager.get_float("SLEEP_EVICTION_LIMIT", 0.001)
            max_wake = tuning_manager.get_int("RECALL_MAX_WAKE", 256)
            recall_boost_val = tuning_manager.get_float("RECALL_BOOST_VALUE", 0.15)

            awake_idx = np.nonzero(new_act > sleep_limit)[0]
            if len(awake_idx) > max_wake:
                strongest = awake_idx[np.argsort(-score[awake_idx])[:max_wake]]
                wake_set = {int(i) for i in strongest}
            else:
                wake_set = {int(i) for i in awake_idx}
            wake_set.update(i for i in active_by_idx if new_act[i] > sleep_limit)
            wake_set.update(i for i in top_set if new_act[i] > sleep_limit)

            wake_entries: Dict[str, Dict[str, Any]] = {}
            wake_stamp: Dict[str, Tuple[float, float]] = {}
            for i in wake_set:
                abs_i = int(abs_idx[i])
                sid = active_by_idx[i] if i in active_by_idx else store.get_id(abs_i)
                excited_act = float(new_act[i])
                hits = int(prev_hits[i])
                if i in top_set:
                    last_recalled, hits = self.fluidity.reinforce(current_time, hits)
                    excited_act = min(1.0, excited_act + recall_boost_val)
                else:
                    last_recalled = current_time

                prev_entry = self.ram_ledger.get(sid)
                if prev_entry is not None:
                    latent_position = np.asarray(prev_entry["latent_position"], dtype=np.float32)
                    velocity = np.asarray(prev_entry.get("velocity", np.zeros(self.dimension, dtype=np.float32)), dtype=np.float32)
                    wake_phase = float(cell_phase[i])  # already-tracked: keep its live Kuramoto-evolved phase
                else:
                    latent_position = store.get_vector(abs_i)
                    velocity = np.zeros(self.dimension, dtype=np.float32)
                    # First time this shard enters ram_ledger this session (never
                    # ingested-then-slept before, or booted from the lightweight
                    # snapshot which doesn't preserve phase) — cell_phase[i] would
                    # be a bare 0.0 here (same degenerate-fixed-point issue as at
                    # ingest), so seed real variation from its own vector instead.
                    wake_phase = _seed_phase(latent_position)

                # _ledger_put_many replaces the whole entry — carry forward any
                # existing Part B cognitive-physics fields (energy/entropy/...)
                # instead of resetting them every time a shard is re-woken.
                cognitive_fields = {
                    k: prev_entry[k] for k in ("energy", "entropy", "stability", "resonance", "momentum", "decay", "attention")
                    if prev_entry is not None and k in prev_entry
                } if prev_entry is not None else _default_cognitive_fields()
                if i in top_set:
                    ltp_boost = tuning_manager.get_float("LTP_ENERGY_BOOST", 0.015)
                    cognitive_fields["energy"] = min(2.0, cognitive_fields.get("energy", 1.0) + ltp_boost)

                wake_entries[sid] = {
                    "activation": excited_act,
                    "latent_position": latent_position,
                    "velocity": velocity,
                    "phase_angle": wake_phase,
                    "hits": hits,
                    "last_recalled": last_recalled,
                    "crystal_path": target_path,
                    "crystal_idx": abs_i,
                    **_default_cognitive_fields(),
                    **cognitive_fields
                }
                wake_stamp[sid] = (excited_act, last_recalled)
                final_hits[i] = hits
                final_act[i] = excited_act

            # Deep Sleep evictions: active nodes whose excitation collapsed
            for i, sid in list(active_by_idx.items()):
                if new_act[i] <= sleep_limit and sid in self.ram_ledger:
                    self._tombstone(sid, self.ram_ledger[sid], reason="decay")
                    del self.ram_ledger[sid]

            placed = self._ledger_put_many(wake_entries)
            self._append_journal_entries([(sid,) + wake_stamp[sid] for sid in placed])

            # 5. Kuramoto Coupling phase locking on active ledger concepts.
            # Closed form: sum_j a_j*sin(th_j - th_i) = cos(th_i)*S1 - sin(th_i)*S2
            # with S1 = sum a*sin(th), S2 = sum a*cos(th) — O(L) instead of O(L^2),
            # mathematically identical to per-node compute_kuramoto_coupling.
            if self.ram_ledger:
                phases = np.array([s["phase_angle"] for s in self.ram_ledger.values()], dtype=np.float64)
                acts = np.array([s["activation"] for s in self.ram_ledger.values()], dtype=np.float64)
                coupling_k = tuning_manager.get_float("KURAMOTO_COUPLING", 0.1)
                s1 = float(np.sum(acts * np.sin(phases)))
                s2 = float(np.sum(acts * np.cos(phases)))
                d_theta = (coupling_k / len(phases)) * (np.cos(phases) * s1 - np.sin(phases) * s2)
                new_phases = (phases + d_theta + np.pi) % (2 * np.pi) - np.pi
                for state, ph in zip(self.ram_ledger.values(), new_phases):
                    state["phase_angle"] = float(ph)

        # Deep Sleep: metadata decoded only for the winners
        results = []
        for i in top_idx:
            i = int(i)
            abs_i = int(abs_idx[i])
            sid = active_by_idx[i] if i in active_by_idx else store.get_id(abs_i)
            meta = store.get_meta(abs_i)
            results.append({
                "shard_id": sid,
                "text": meta.get("text", ""),
                "score": float(score[i]),
                "hits": final_hits.get(i, int(prev_hits[i])),
                "activation": float(final_act.get(i, float(new_act[i])))
            })

        # Part B: what was just found is a real percept — let the shared
        # cognitive field feel it too (grounds assemblies/causal graph in
        # what recall actually surfaced, additive, doesn't alter `results`).
        if learn:
            # Learning mode: consolidate synchronously (the expensive tick).
            if results:
                self.process_cognitive_tick({r["shard_id"]: r["activation"] for r in results})
            self._last_message_ts = time.time()
        elif results:
            # Serving mode: DON'T drop the experience — record a cheap trace and
            # consolidate it in batches so the brain still learns from recall
            # without blocking each query with the full physics.
            self._recall_traces.append((
                [r["shard_id"] for r in results],
                [float(r["activation"]) for r in results],
                current_time,
            ))
            if len(self._recall_traces) >= tuning_manager.get_int("CONSOLIDATE_EVERY", 64):
                self.consolidate_traces()

        return results

    def consolidate_traces(self, max_edge_fanout: int = 5) -> int:
        """
        Deferred learning from serving-mode recalls — the background half of the
        perception/consolidation split. Drains the recall-trace buffer and, in
        ONE batched pass, applies exactly the memory updates that learning-mode
        recall would have done per query:
          1. Reinforce every surfaced shard (activation boost + hit count + recency)
             so frequently-recalled memories strengthen and decay slower (ACT-R).
          2. Wire Hebbian co-occurrence edges between shards surfaced together
             ("fire together -> wire together") — a real associative signal that
             grows the causal graph from genuine query co-activation.
          3. One cognitive tick over the union of touched shards (bounded) so the
             latent field, assemblies, and Kuramoto phases evolve.
        This is how serving mode still updates the neurons and remembers. Returns
        the number of traces consolidated. Safe to call anytime (no-op if empty).
        """
        traces = self._recall_traces
        if not traces:
            return 0
        self._recall_traces = []
        now = time.time()
        boost = tuning_manager.get_float("RECALL_BOOST_VALUE", 0.15)

        # 1. Batched reinforcement of surfaced shards.
        touched: Dict[str, float] = {}
        for sids, acts, _t in traces:
            for sid, a in zip(sids, acts):
                touched[sid] = max(touched.get(sid, 0.0), float(a))
                entry = self.ram_ledger.get(sid)
                if entry is not None:
                    entry["activation"] = min(1.0, float(entry.get("activation", 0.0)) + boost)
                    entry["hits"] = int(entry.get("hits", 0)) + 1
                    entry["last_recalled"] = now

        # 2. Hebbian co-occurrence edges (top-N of each recall, batched — one
        #    stale-edge decay for the whole batch, not per edge).
        pairs: List[Tuple[str, str]] = []
        for sids, acts, _t in traces:
            top = sids[:max_edge_fanout]
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    pairs.append((top[i], top[j]))
        if pairs:
            self.causal.record_transitions_batch(pairs, weight=0.6, prediction_error=0.2)
            # Mechanism 7: same co-recalled pairs also feed the undecaying
            # working-set counter — pure co-use, independent of the causal
            # graph's content-adjacency semantics and stale-edge decay.
            for a, b in pairs:
                key = (a, b) if a <= b else (b, a)
                self.working_set_pairs[key] = self.working_set_pairs.get(key, 0) + 1

        # 3. One bounded cognitive tick over everything touched this batch.
        if touched:
            self.process_cognitive_tick(touched)
        self._last_message_ts = now
        return len(traces)

    def _synthesize_semantic_text(self, concept: str, texts: List[str]) -> Tuple[str, str]:
        """
        REM-stage synthesis. Tries the real completion provider (the GenAI
        proxy configured in .env.dev) and falls back to the deterministic
        template on any failure — the substrate must never REQUIRE the LLM
        to function. Returns (text, mode) where mode is "llm" or "template";
        the mode is persisted into the semantic crystal's metadata so the
        consolidation-faithfulness audit can tell the two apart later.
        Gate with tuning key SLEEP_LLM_SYNTHESIS=0 to force the template.
        """
        seen: List[str] = []
        for t in texts:
            t = (t or "").strip()
            if t and t not in seen:
                seen.append(t)

        if seen and tuning_manager.get_int("SLEEP_LLM_SYNTHESIS", 1):
            try:
                # Lazy import: keeps coordinator boot free of any provider cost.
                from module_loom.services.completion.completion_service import get_completion_service
                svc = get_completion_service()
                if svc.available():
                    sources = "\n".join(f"- {t}" for t in seen[:12])
                    out = svc.complete(
                        [
                            {"role": "system", "content": (
                                "You consolidate related memory fragments into one dense semantic memory. "
                                "Write 2-4 sentences that preserve every distinct fact present in the sources. "
                                "Do not invent facts that are not in the sources. "
                                "Do not mention memories, sources, or this instruction. "
                                "Output only the consolidated text.")},
                            {"role": "user", "content": f"Theme: {concept}\nSources:\n{sources}"},
                        ],
                        max_tokens=220,
                        temperature=0.3,
                    )
                    if out and out.strip():
                        return out.strip(), "llm"
            except Exception:
                pass  # provider logs its own failures; template below is the contract

        bullets = "\n".join(f"- {t}" for t in seen[:8])
        return (
            f"Consolidated theme: {concept}\nRecurring observations across {len(seen)} memories:\n{bullets}",
            "template",
        )

    def _get_faithfulness_judge(self):
        """Lazily creates (once) the FaithfulnessJudge used to gate sleep-cycle
        synthesis (mechanism 3) — same lazy-import discipline as the completion
        service: boot never pays for a provider it might not need."""
        judge = getattr(self, "_faithfulness_judge", None)
        if judge is None:
            try:
                from module_loom.services.eval.faithfulness_judge import FaithfulnessJudge
                judge = FaithfulnessJudge()
            except Exception:
                judge = False
            self._faithfulness_judge = judge
        return judge or None

    def maybe_sleep_cycle(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Sleep-cycle consolidation over the external store — the mechanism
        this session's research found nobody has done at this substrate's
        scale with continuous dynamics (see the plan for context). Reuses
        REAL, already-running state instead of inventing a new signal:
        AssemblyCompilation.attractor_memory's frequency/basin_strength IS
        "has this concept kept recurring," and active_assemblies gives the
        real member shard_ids behind it.

        For each concept that has recurred often enough: reinforces causal
        edges + activation among its members (NREM-equivalent), then
        synthesizes and ingests one new "semantic crystal" shard summarizing
        them (REM-equivalent, via the placeholder above) and marks the
        sources superseded. Nothing is ever deleted — see _mark_superseded.

        Returns a summary dict if anything was consolidated, else None (and
        does nothing else) — cheap to call speculatively on every idle check.
        """
        if not tuning_manager.get_int("SLEEP_CYCLE_ENABLED", 1):
            return None

        now = time.time()
        idle_for = now - self._last_message_ts
        # Mechanism 2 (surprise-gated consolidation): a region under active
        # contradiction (attractor_memory[concept]["surprise_accum"], grown in
        # AssemblyCompilation.compile_assembly) wakes sleep early even if the
        # idle timer hasn't elapsed yet — biological sleep-pressure is driven
        # by how much prediction error accumulated while awake, not a clock.
        surprise_threshold = tuning_manager.get_float("SURPRISE_CONSOLIDATE_THRESHOLD", 3.0)
        surprise_ok = force or (surprise_threshold > 0.0 and any(
            m.get("surprise_accum", 0.0) >= surprise_threshold for m in self.compiler.attractor_memory.values()
        ))
        idle_ok = force or surprise_ok or idle_for >= tuning_manager.get_int("SLEEP_IDLE_TRIGGER_SEC", 120)
        activity_ok = force or self._shards_since_last_sleep >= tuning_manager.get_int("SLEEP_MIN_ACTIVITY_SHARDS", 50)
        if not (idle_ok and activity_ok):
            return None

        freq_threshold = tuning_manager.get_int("SLEEP_CONSOLIDATION_FREQUENCY_THRESHOLD", 3)
        max_source = tuning_manager.get_int("SLEEP_MAX_SOURCE_SHARDS", 12)

        consolidated: List[Dict[str, Any]] = []
        settled_concepts: set = set()
        consolidated_faithfulness: Dict[str, Optional[float]] = {}

        for concept, mem in list(self.compiler.attractor_memory.items()):
            if mem.get("frequency", 0) < freq_threshold:
                continue

            member_sids: List[str] = []
            seen_sids: set = set()
            for a in self.active_assemblies:
                if a.dominant_concept != concept:
                    continue
                for sid in a.node_activations.keys():
                    if sid in self.ram_ledger and sid not in seen_sids:
                        seen_sids.add(sid)
                        member_sids.append(sid)
            if len(member_sids) < 2:
                continue
            # Second coherence gate (bug #1), at consolidation time: even a
            # gated assembly can accumulate drift, and members are unioned
            # across multiple assemblies sharing this concept. Drop members
            # that don't cohere with the member centroid so the synthesized
            # crystal is faithful to ONE concept — a topic-mixed crystal
            # cannot be faithful to any of its sources (guards #3's audit too).
            crystal_coherence_min = tuning_manager.get_float("SLEEP_CRYSTAL_COHERENCE_MIN", 0.5)
            if crystal_coherence_min > 0.0:
                member_sids = self._coherence_filter(member_sids, crystal_coherence_min)
            if len(member_sids) < 2:
                continue
            member_sids.sort(key=lambda s: self.ram_ledger[s].get("activation", 0.0), reverse=True)
            member_sids = member_sids[:max_source]

            # 1. NREM-equivalent: reinforce causal edges + activation among members.
            for i in range(len(member_sids) - 1):
                self.causal.record_transition(member_sids[i], member_sids[i + 1], weight=1.0, prediction_error=0.1)
            for sid in member_sids:
                entry = self.ram_ledger[sid]
                entry["activation"] = min(1.0, float(entry.get("activation", 0.0)) + 0.1)

            # 2. REM-equivalent: synthesize + ingest a new semantic crystal.
            # Text lives only in each shard's frozen crystal-log metadata,
            # never in the ram_ledger — look it up via its own crystal_path/idx.
            texts: List[str] = []
            for sid in member_sids:
                entry = self.ram_ledger[sid]
                cpath, cidx = entry.get("crystal_path"), entry.get("crystal_idx")
                text = ""
                if cpath and cidx is not None:
                    try:
                        text = self._get_store(cpath).get_meta(cidx).get("text", "")
                    except Exception:
                        pass
                texts.append(text)
            positions = [np.asarray(self.ram_ledger[sid]["latent_position"], dtype=np.float32) for sid in member_sids]
            new_vec = np.mean(positions, axis=0)
            vnorm = np.linalg.norm(new_vec)
            if vnorm > 1e-9:
                new_vec = new_vec / vnorm
            synthesis_text, synthesis_mode = self._synthesize_semantic_text(concept, texts)

            # Mechanism 3 (faithfulness-gated consolidation): an LLM synthesis
            # that hallucinates a claim not in its sources ships identically to
            # a good one today. Score it inline with the same judge the
            # 2026-07-15 faithfulness audit uses; below threshold, retry once
            # through _synthesize_semantic_text (which itself falls back to the
            # deterministic template on any LLM failure) instead of shipping
            # the ungated synthesis.
            faithfulness_score = None
            if synthesis_mode == "llm":
                min_faithfulness = tuning_manager.get_float("SLEEP_FAITHFULNESS_MIN", 0.7)
                judge = self._get_faithfulness_judge()
                if judge is not None and judge.available():
                    fr = judge.judge(synthesis_text, [t for t in texts if t])
                    if fr.status == "ok":
                        faithfulness_score = fr.faithfulness
                        if fr.faithfulness is not None and fr.faithfulness < min_faithfulness:
                            synthesis_text, synthesis_mode = self._synthesize_semantic_text(concept, texts)
                            if synthesis_mode == "template":
                                faithfulness_score = 1.0  # template only ever quotes sources verbatim
            consolidated_faithfulness[concept] = faithfulness_score

            new_sid = f"crystal_{hashlib.md5((concept + '|' + '|'.join(sorted(member_sids))).encode('utf-8')).hexdigest()[:12]}"
            mass = min(5.0, 1.5 + 0.1 * len(member_sids))
            self.ingest_shard(new_sid, new_vec, synthesis_text, mass=mass, metadata={
                "semantic_crystal": True,
                "source_shard_ids": member_sids,
                "consolidated_concept": concept,
                "synthesis_mode": synthesis_mode,
            })

            # 3. Mark sources superseded — deprioritized, never deleted.
            for sid in member_sids:
                self._mark_superseded(sid, new_sid)

            consolidated.append({"concept": concept, "new_shard": new_sid, "source_shard_ids": member_sids, "synthesis_mode": synthesis_mode, "faithfulness": consolidated_faithfulness.get(concept)})
            settled_concepts.add(concept)
            mem["frequency"] = 0
            mem["surprise_accum"] = 0.0

        if consolidated:
            self.active_assemblies = [a for a in self.active_assemblies if a.dominant_concept not in settled_concepts]

        self._shards_since_last_sleep = 0
        self._last_sleep_ts = now
        if consolidated:
            self.autosave(force_full=True)
            return {"consolidated": consolidated}
        return None

    def orchestrate_weave(self, data: List[Any]) -> None:
        """
        Elevates Weaver as the sole orchestrator.
        Takes raw data, passes it to the in-memory Cortex (SubstrateWeaver) for calculation,
        then handles all physical storage to .brain_data crystals.
        """
        from module_loom.services.cortex.SubstrateWeaver import SubstrateWeaver
        
        weaver = SubstrateWeaver(storage_dir=self.storage_dir)
        # 1. Ask Cortex to calculate the graph in memory
        result = weaver.weave(data, output_path=None)
        
        nodes = result["nodes"]
        edges = result["edges"]
        all_embeddings = result["all_embeddings"]
        
        # 2. Extract and persist the results (Weaver's job)
        for edge in edges:
            f_id = edge["f"]
            if f_id in nodes:
                if "edges" not in nodes[f_id]["m"]:
                    nodes[f_id]["m"]["edges"] = []
                nodes[f_id]["m"]["edges"].append(edge)

        batch_items = []
        for nid, node in nodes.items():
            if node["t"] in ["root", "meta_shard"]:
                continue

            true_vec = None
            if nid in all_embeddings:
                true_vec = np.array(all_embeddings[nid], dtype=np.float32)[:self.dimension]
                if len(true_vec) < self.dimension:
                    true_vec = np.pad(true_vec, (0, self.dimension - len(true_vec)), constant_values=0.0)
            elif "emb" in node["m"]:
                packed_list = node["m"]["emb"]
                packed_arr = np.array(packed_list, dtype=np.uint8)
                unpacked = np.unpackbits(packed_arr)[:self.dimension]
                true_vec = (unpacked.astype(np.float32) * 2.0) - 1.0
                if len(true_vec) < self.dimension:
                    true_vec = np.pad(true_vec, (0, self.dimension - len(true_vec)), constant_values=0.0)
            else:
                true_vec = np.zeros(self.dimension, dtype=np.float32)

            batch_items.append({
                "shard_id": nid,
                "vector": true_vec,
                "text": node.get("c", ""),
                "mass": float(node["m"].get("mass", 1.0)),
                "metadata": node["m"]
            })

        # Persist to dynamic crystals: one batched append per crystal, one atlas save
        self.ingest_batch(batch_items)

        # 3. Persist the raw in-memory physics tensors from the Cortex Engine
        #    as segments of the unified universe.loom container.
        engine = weaver.engine
        n = engine.current_node_count
        self._commit_state_segments({
            "coordinates.bin": engine.coords_map[:n].tobytes(),
            "physics_tensors.bin": engine.physics_map[:n].tobytes(),
            "hdc_signatures.bin": engine.hdc_map[:n].tobytes(),
            "causal_links.bin": engine.causal_map[:n].tobytes(),
        })

        print("Weaver orchestration complete. Pure in-memory Cortex data successfully flushed to universe.loom.")


    def _build_living_state_blob(self) -> bytes:
        """
        Part B — full-fidelity serialization of the shared cognitive state AND
        every ram_ledger entry's complete living record (position, velocity,
        phase, hits, and the seven cognitive-physics fields). Richer than the
        always-on lightweight checkpoint snapshot (activation+timestamp only,
        used for fast boot) — this is the deep, exact restore point. Split out
        from save_cortex_state() so autosave() can fold this into the SAME
        universe.rebuild() call as the lightweight snapshot instead of paying
        a second full-container rewrite back-to-back.

        There is no external cortex_state object anymore: storage and
        cognition are one merged record now (self.ram_ledger), so there is
        nothing separate left to pass in — that IS the deep merge.
        """
        import msgpack

        living: Dict[str, Dict[str, Any]] = {}
        for sid, entry in self.ram_ledger.items():
            living[sid] = {
                "activation": float(entry.get("activation", 0.0)),
                "latent_position": np.asarray(entry["latent_position"], dtype=np.float32).tolist(),
                "velocity": np.asarray(
                    entry.get("velocity", np.zeros(self.dimension, dtype=np.float32)), dtype=np.float32
                ).tolist(),
                "phase_angle": float(entry.get("phase_angle", 0.0)),
                "hits": int(entry.get("hits", 0)),
                "last_recalled": float(entry.get("last_recalled", 0.0)),
                "crystal_path": entry.get("crystal_path", ""),
                "crystal_idx": entry.get("crystal_idx"),
                "energy": float(entry.get("energy", 1.0)),
                "entropy": float(entry.get("entropy", 0.0)),
                "stability": float(entry.get("stability", 1.0)),
                "resonance": float(entry.get("resonance", 0.0)),
                "momentum": float(entry.get("momentum", 0.0)),
                "decay": float(entry.get("decay", 0.05)),
                "attention": float(entry.get("attention", 0.0)),
            }

        assemblies = [{
            "id": a.assembly_id,
            "dominant": a.dominant_concept,
            "activations": a.node_activations,
            "confidence": a.confidence,
            "stability": a.stability,
            "phase_coherence": a.phase_coherence,
            "basin_strength": a.meta_data.get("basin_strength", 0.0),
        } for a in self.active_assemblies]

        causal_edges = [{
            "from": u, "to": v,
            "weight": float(d.get("weight", 0.0)),
            "frequency": int(d.get("frequency", 1)),
        } for u, v, d in self.causal_graph.edges(data=True)]

        state = {
            "tick": self.cognitive_tick,
            "entropy": self.cognitive_entropy,
            "coherence": self.cognitive_coherence,
            "energy_budget": self.cognitive_energy_budget,
            "latent_field": self.latent_field.tolist() if self.latent_field is not None else None,
            "working_latent": self.working_latent.tolist() if self.working_latent is not None else None,
            "latent_velocity": self.latent_velocity.tolist() if self.latent_velocity is not None else None,
            "assemblies": assemblies,
            "causal_edges": causal_edges,
            "living_ledger": living,
            "superseded": self.superseded_by_crystal,
        }
        return msgpack.packb(state, use_bin_type=True, default=_np_msgpack_default)

    def save_cortex_state(self) -> None:
        """
        Writes the full-fidelity living state (see _build_living_state_blob)
        to the "living_state.bin" segment in universe.loom. Used on clean
        shutdown (close()) and by tools that want an explicit, immediate
        deep save; autosave() below is the periodic, cheaper-by-default path.
        """
        try:
            segment = self._build_living_state_blob()
            self._commit_state_segments({"living_state.bin": segment})
        except Exception as e:
            print("Error in Weaver save_cortex_state:", e)
            raise e

    def autosave(self, force_full: bool = False) -> None:
        """
        Periodic durability point for long-running sessions (e.g. the
        neuro-visualizer WS connection): folds the live RAM ledger into
        universe.loom without waiting for a clean shutdown, so a decode
        taken while the process is still running reflects recent
        ingest/recall activity instead of only the last close().

        Always refreshes the lightweight activation snapshot (cheap — same
        records checkpoint() builds). Only rebuilds the full living_state.bin
        (richer, more expensive — scales with ram_ledger size) when
        force_full or VIZ_FULL_SAVE_INTERVAL_SEC has elapsed since the last
        one, and folds whichever segments were built into a SINGLE
        universe.rebuild() call — calling checkpoint() then save_cortex_state()
        back-to-back would pay the full-container rewrite twice for nothing.
        """
        # Fold in any pending serving-mode recall experience before persisting,
        # so deferred learning is never lost to an autosave/shutdown window.
        self.consolidate_traces()
        try:
            now = time.time()
            do_full = force_full or (
                now - self._last_full_save_ts >= tuning_manager.get_int("VIZ_FULL_SAVE_INTERVAL_SEC", 180)
            )

            if self.universe is None:
                # Legacy loose-file brain: checkpoint()/save_cortex_state() already
                # write small individual files, not a monolithic container - no
                # double-rewrite cost to avoid, so just delegate directly.
                self.checkpoint()
                if do_full:
                    self.save_cortex_state()
                    self._last_full_save_ts = now
                return

            records = bytearray()
            for sid, entry in list(self.ram_ledger.items()):
                act = entry.get("activation", 0.0)
                if act > 0.0:
                    h = hash_shard_id(sid)
                    t = int(entry.get("last_recalled", 0.0))
                    records.extend(struct.pack("<QfI", h, act, t))

            segments = self._carryover_segments()
            segments["atlas"] = self.atlas.to_bytes()
            cctx = zstd.ZstdCompressor()
            segments["snapshot"] = cctx.compress(bytes(records)) if records else b""

            if do_full:
                segments["living_state.bin"] = self._build_living_state_blob()
                self._last_full_save_ts = now

            self.universe.rebuild(segments, journal=b"")  # journal folded into snapshot
            self._atlas_dirty = False
        except Exception:
            pass

    def load_cortex_state(self) -> bool:
        """
        Restores the full-fidelity cognitive + per-shard living state saved
        by save_cortex_state(). Returns False if none exists yet (a fresh
        brain, or one that has never called save_cortex_state — the
        always-on lightweight checkpoint snapshot still restores activation/
        timestamp on every normal boot regardless).
        """
        blob = self._state_read("living_state.bin")
        if blob is None:
            return False

        try:
            import msgpack

            state = msgpack.unpackb(blob, raw=False)

            self.cognitive_tick = state.get("tick", 0)
            self.cognitive_entropy = state.get("entropy", 0.5)
            self.cognitive_coherence = state.get("coherence", 0.5)
            self.cognitive_energy_budget = state.get("energy_budget", 1.0)
            self._last_cognitive_coherence = self.cognitive_coherence

            lf = state.get("latent_field")
            self.latent_field = np.array(lf, dtype=np.float32) if lf is not None else None
            wl = state.get("working_latent")
            self.working_latent = np.array(wl, dtype=np.float32) if wl is not None else None
            lv = state.get("latent_velocity")
            self.latent_velocity = np.array(lv, dtype=np.float32) if lv is not None else None

            self.active_assemblies = [
                CognitiveAssembly(
                    assembly_id=a["id"],
                    dominant_concept=a["dominant"],
                    node_activations=a["activations"],
                    confidence=a["confidence"],
                    stability=a["stability"],
                    phase_coherence=a.get("phase_coherence", self.cognitive_coherence),
                    meta_data={"basin_strength": a.get("basin_strength", 0.0)},
                )
                for a in state.get("assemblies", [])
            ]

            self.causal_graph.clear()
            for e in state.get("causal_edges", []):
                self.causal_graph.add_edge(
                    e["from"], e["to"], weight=e["weight"], frequency=e["frequency"], last_seen=time.time()
                )

            self.superseded_by_crystal = {
                cpath: dict(sidmap) for cpath, sidmap in state.get("superseded", {}).items()
            }
            self._superseded_idx_cache.clear()

            for sid, rec in state.get("living_ledger", {}).items():
                entry = self.ram_ledger.get(sid, {})
                entry.update({
                    "activation": rec["activation"],
                    "latent_position": np.array(rec["latent_position"], dtype=np.float32),
                    "velocity": np.array(rec["velocity"], dtype=np.float32),
                    "phase_angle": rec["phase_angle"],
                    "hits": rec["hits"],
                    "last_recalled": rec["last_recalled"],
                    "crystal_path": rec.get("crystal_path") or entry.get("crystal_path", ""),
                    "crystal_idx": rec.get("crystal_idx", entry.get("crystal_idx")),
                    "energy": rec["energy"],
                    "entropy": rec["entropy"],
                    "stability": rec["stability"],
                    "resonance": rec["resonance"],
                    "momentum": rec["momentum"],
                    "decay": rec["decay"],
                    "attention": rec["attention"],
                })
                self.ram_ledger[sid] = entry

            return True
        except Exception as e:
            print("Error loading Cortex state from Weaver:", e)
            return False

    def close(self) -> None:
        """
        Safe shutdown: fold crystal tails into .idx snapshots, save the
        lightweight activation snapshot, and save the full-fidelity Part B
        cognitive/living-ledger state (positions, velocities, assemblies,
        causal graph) — durable by default, not an easily-forgotten manual step.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.consolidate_traces()   # flush any pending serving-mode learning
        except Exception:
            pass
        for store in list(self._stores.values()):
            try:
                store.close()
            except Exception:
                pass
        self._stores = {}
        self.checkpoint()
        try:
            self.save_cortex_state()
        except Exception:
            pass

    def __del__(self) -> None:
        # Best-effort fallback ONLY if close() was never called. During normal
        # interpreter shutdown, module imports are already torn down, so a save
        # here cannot succeed and only produces noise — skip it. A live process
        # that drops its last reference mid-run (import machinery still intact)
        # still gets one clean save.
        try:
            if getattr(self, "_closed", False):
                return
            if sys is None or getattr(sys, "meta_path", None) is None:
                return  # interpreter is shutting down; imports would fail
            self.close()
        except Exception:
            pass
