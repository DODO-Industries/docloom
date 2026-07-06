import os
import time
import heapq
import struct
import numpy as np
import zstandard as zstd
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from backend.config.tunningManagment import tuning_manager

from backend.services.loom_service.weaver.seed_core import UniverseSeedCore
from backend.services.loom_service.weaver.atlas_router import GlobalAtlasRouter, STATE_MAP_TO_BYTE, STATE_MAP_FROM_BYTE
from backend.services.loom_service.weaver.substrate_layout import LoomSubstrate, LoomStore, detect_version
from backend.services.loom_service.weaver.storage_physics import LatentFieldPhysicsEngine
from backend.services.loom_service.weaver.memory_fluidity import DynamicMemoryFluidity
from backend.services.loom_service.weaver.universe_container import (
    UniverseContainer, pack_txn_record, pack_crystal_record, iter_journal_records,
    JREC_TXN, JREC_CRYSTAL, TXN_PAYLOAD, CRYSTAL_PREFIX,
)

# Legacy loose files consolidated into universe.loom (moved to legacy_backup/ after migration)
LEGACY_STATE_FILES = [
    "coordinates.bin", "velocities.bin", "physics_tensors.bin", "brain_embeddings.bin",
    "working_memory.bin", "causal_links.bin", "metadata.bin", "replay.bin", "hdc_signatures.bin",
]

def hash_shard_id(shard_id: str) -> int:
    return int.from_bytes(hashlib.sha256(shard_id.encode("utf-8")).digest()[:8], byteorder="big")

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
        self.dimension = dimension
        if not storage_dir:
            storage_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".brain_data"))
        self.storage_dir = os.path.abspath(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.atlas_path = os.path.join(self.storage_dir, "atlas.capnp")
        self.metric_path = os.path.join(self.storage_dir, "universe.metric")
        self.universe_path = os.path.join(self.storage_dir, "universe.loom")
        self.max_shards_per_crystal = tuning_manager.get_int("MAX_CRYSTAL_SIZE", 500000)
        self.universe: Optional[UniverseContainer] = None
        self._atlas_dirty = False

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
                        from backend.config.envConfig import MASTER_SEED
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
        self._load_persistence_layers()

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
                            "crystal_path": crystal_path
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
                        "crystal_path": crystal_path
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

    def _route_for_ingest(self, momentum: np.ndarray) -> str:
        """
        Atlas Capture (Routing) + Cellular Division. Returns the target crystal
        path, splitting a full crystal into a new hot sub-sector when needed.
        """
        if not self.atlas.crystals:
            target = os.path.join(self.storage_dir, "crystal_1.loom")
            self.atlas.register_crystal(target, momentum, state="hot")
            self._journal_crystal_registration(target)
            return target

        # Ingest routes ONLY among crystals with headroom. Full crystals keep
        # serving recall but never receive new shards — otherwise every routing
        # hit on a full crystal spawns another split (division cascade).
        open_paths = [
            p for p, info in self.atlas.crystals.items()
            if info.get("num_shards", 0) < self.max_shards_per_crystal
        ]
        if open_paths:
            if len(open_paths) == len(self.atlas.crystals):
                return self.atlas.route_vector(momentum)
            v_norm = np.linalg.norm(momentum)
            best_path, best_sim = open_paths[0], -2.0
            for p in open_paths:
                centroid = self.atlas.crystals[p]["centroid"]
                c_norm = np.linalg.norm(centroid)
                sim = 0.0 if (c_norm == 0 or v_norm == 0) else float(np.dot(momentum, centroid) / (v_norm * c_norm))
                if sim > best_sim:
                    best_sim, best_path = sim, p
            return best_path

        # Every crystal is full → cellular division of the nearest one
        target = self.atlas.route_vector(momentum)
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

    def _maybe_add_leader(self, store: LoomStore, momentum: np.ndarray,
                          extra_packed: Optional[List[bytes]] = None) -> Optional[bytes]:
        """
        Routing-space update: registers the momentum signature as a new leader
        when it is Hamming-far (>32/128 bits) from every existing leader.
        Returns the packed signature if it was added.
        """
        # Routing leaders are an LSH acceleration structure — cap them so sparse
        # signature spaces (worst case: uniform random vectors) cannot grow one
        # leader per shard and turn every ingest into an O(n) Hamming scan.
        max_leaders = tuning_manager.get_int("LOOM_MAX_LEADERS", 512)
        existing = store.get_leaders_packed()
        if len(existing) + len(extra_packed or []) >= max_leaders:
            return None

        sig_packed = self._momentum_signature(momentum)
        sig_arr = np.frombuffer(sig_packed, dtype=np.uint8)
        candidates = existing + (extra_packed or [])
        if candidates:
            cand_matrix = np.frombuffer(b"".join(candidates), dtype=np.uint8).reshape(len(candidates), 16)
            xored = np.bitwise_xor(cand_matrix, sig_arr)
            best_dist = int(np.unpackbits(xored, axis=1).sum(axis=1).min())
        else:
            best_dist = 129
        if best_dist > 32 or not candidates:
            sig = np.sign(momentum[:128]).astype(np.int8)
            sig[sig == 0] = 1
            store.append_leader(sig)
            return sig_packed
        return None

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

        # 3. Atlas Capture (Routing + Cellular Division)
        target_crystal_path = self._route_for_ingest(momentum)

        # 4. Substrate Append Execution
        store = self._get_store(target_crystal_path)
        self._maybe_add_leader(store, momentum)

        now = time.time()
        meta = dict(metadata or {})
        meta.update({
            "text": text,
            "mass": mass,
            "last_recalled": now,
            "hits": 0,
            "activation": 1.0
        })
        true_vector = np.asarray(true_vector, dtype=np.float32)
        crystal_idx = store.append_shard(shard_id, true_vector, mass, meta)

        # Update centroid in global atlas (folded into container at checkpoint)
        self.atlas.update_centroid(target_crystal_path, momentum)
        self._mark_atlas_dirty()

        # Update active RAM ledger (cap-guarded) and delta log
        self.hash_to_shard[hash_shard_id(shard_id)] = (shard_id, target_crystal_path)
        placed = self._ledger_put_many({shard_id: {
            "activation": 1.0,
            "latent_position": true_vector.copy(),
            "velocity": np.zeros(self.dimension, dtype=np.float32),
            "phase_angle": 0.0,
            "hits": 0,
            "last_recalled": now,
            "crystal_path": target_crystal_path,
            "crystal_idx": crystal_idx
        }})
        if placed:
            self._append_journal_entry(shard_id, 1.0, now)

        return target_crystal_path

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
            target = self._route_for_ingest(momentum)
            # In-memory centroid update keeps division-aware routing correct mid-batch.
            self.atlas.update_centroid(target, momentum)

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
            store = self._get_store(target)

            # Routing-space leaders: check each momentum against store + freshly added
            added_packed: List[bytes] = []
            for momentum in momenta[target]:
                added = self._maybe_add_leader(store, momentum, extra_packed=added_packed)
                if added is not None:
                    added_packed.append(added)

            indices = store.append_batch(shard_items)
            counts[target] = len(shard_items)

            for (shard_id, vec, mass, meta), crystal_idx in zip(shard_items, indices):
                self.hash_to_shard[hash_shard_id(shard_id)] = (shard_id, target)
                ledger_batch[shard_id] = {
                    "activation": 1.0,
                    "latent_position": vec.copy(),
                    "velocity": np.zeros(self.dimension, dtype=np.float32),
                    "phase_angle": 0.0,
                    "hits": 0,
                    "last_recalled": now,
                    "crystal_path": target,
                    "crystal_idx": crystal_idx
                }

        placed = set(self._ledger_put_many(ledger_batch))
        journal_entries = [(sid, 1.0, now) for sid in placed]
        self._mark_atlas_dirty()
        self._append_journal_entries(journal_entries)
        return counts

    def enforce_spatiotemporal_decay(self, current_time: float) -> None:
        """
        Evaluates memory states against continuous decay curves.
        Triggers Deep Sleep evictions immediately when activation drops to zero (<= 0.001).
        """
        eviction_targets = []

        for shard_id, state in list(self.ram_ledger.items()):
            decayed_amp = self.fluidity.calculate_decay(
                initial_activation=state["activation"],
                last_recalled=state["last_recalled"],
                current_time=current_time,
                hits=state["hits"]
            )
            
            sleep_eviction_limit = tuning_manager.get_float("SLEEP_EVICTION_LIMIT", 0.001)
            if decayed_amp <= sleep_eviction_limit:
                eviction_targets.append(shard_id)
            else:
                state["activation"] = decayed_amp
                state["last_recalled"] = current_time

        # Execute Deep Sleep Protocol (Purge from transient memory structures)
        for shard_id in eviction_targets:
            del self.ram_ledger[shard_id]

    def recall(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        current_time: Optional[float] = None,
        query_phase: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Recall Pathway (Querying) — fully vectorized over the crystal's mmap matrix:
        1. Topology Sweep: Find closest target .loom crystal in Atlas
        2. Zero-Copy Fetch: score EVERY shard via the contiguous .idx matrix
           (OS page cache; dormant shards cost no heap — Deep Sleep preserved)
        3. Wave Resonance + Gravity: batch physics, identical formulas
        4. Re-awaken sleeping nodes (capped by RECALL_MAX_WAKE strongest)
        5. Kuramoto Coupling: lock active phase angles
        Metadata is decoded ONLY for the returned top_k shards.
        """
        if not self.atlas.crystals:
            return []

        if current_time is None:
            current_time = time.time()

        # Enforce continuous decay for all active states first (Deep Sleep Protocol check)
        self.enforce_spatiotemporal_decay(current_time)

        # 1. Topology Sweep
        target_path = self.atlas.route_vector(query_vector)
        if not os.path.exists(target_path):
            return []

        # OS mounting emulation: transition Cold crystal to Warm for mmap access
        if self.atlas.crystals[target_path]["state"] == "cold":
            self.atlas.set_state(target_path, "warm")
            self._mark_atlas_dirty()

        # 2. Zero-Copy Fetch — mmap-backed arrays, no journal decode
        store = self._get_store(target_path)
        n = store.n
        k = min(int(top_k), n)
        if n == 0 or k <= 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        norms = store.norms_all()
        denom = norms * q_norm
        safe_denom = np.maximum(denom, 1e-12)
        cos = (store.dot_all(q) / safe_denom).astype(np.float32)
        cos[denom <= 1e-12] = 0.0

        # Wormhole warp for gravity (identity tensor short-circuits to plain cosine)
        if self.physics.is_identity_wormhole():
            cos_warped = cos
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

        # Overlay live RAM-ledger states (activation / phase / hits) onto arrays
        initial_act = np.zeros(n, dtype=np.float32)
        cell_phase = np.zeros(n, dtype=np.float32)
        prev_hits = np.zeros(n, dtype=np.int64)
        active_by_idx: Dict[int, str] = {}
        target_abs = os.path.abspath(target_path)
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
            initial_act[idx] = state.get("activation", 1.0)
            cell_phase[idx] = state.get("phase_angle", 0.0)
            prev_hits[idx] = state.get("hits", 0)
            active_by_idx[idx] = sid

        # 3. Wave Resonance + Gravitational scoring (batch physics)
        gain = self.physics.calculate_excitation_gain_batch(cos, query_phase, cell_phase)
        new_act = np.clip(initial_act + gain, 0.0, 1.0)
        mass = store.mass_all()
        grav = self.physics.calculate_gravitational_attraction_batch(cos_warped, mass, mass_a=1.0)
        score = (new_act * grav).astype(np.float32)

        if k < n:
            top_idx = np.argpartition(-score, k - 1)[:k]
            top_idx = top_idx[np.argsort(-score[top_idx])]
        else:
            top_idx = np.argsort(-score)
        top_set = {int(i) for i in top_idx}

        # 4. Plasticity & Wave Excitation Reinforcement / Re-awakening.
        # At scale a strong query can excite most of a crystal; only the
        # RECALL_MAX_WAKE strongest sleepers are re-awakened into RAM
        # (already-active nodes and top_k hits are always processed).
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
        final_hits: Dict[int, int] = {}
        final_act: Dict[int, float] = {}
        for i in wake_set:
            sid = active_by_idx[i] if i in active_by_idx else store.get_id(i)
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
            else:
                latent_position = store.get_vector(i)

            wake_entries[sid] = {
                "activation": excited_act,
                "latent_position": latent_position,
                "velocity": np.zeros(self.dimension, dtype=np.float32),
                "phase_angle": float(cell_phase[i]),
                "hits": hits,
                "last_recalled": last_recalled,
                "crystal_path": target_path,
                "crystal_idx": int(i)
            }
            wake_stamp[sid] = (excited_act, last_recalled)
            final_hits[i] = hits
            final_act[i] = excited_act

        # Deep Sleep evictions: active nodes whose excitation collapsed
        for i, sid in list(active_by_idx.items()):
            if new_act[i] <= sleep_limit and sid in self.ram_ledger:
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
            sid = active_by_idx[i] if i in active_by_idx else store.get_id(i)
            meta = store.get_meta(i)
            results.append({
                "shard_id": sid,
                "text": meta.get("text", ""),
                "score": float(score[i]),
                "hits": final_hits.get(i, int(prev_hits[i])),
                "activation": float(final_act.get(i, float(new_act[i])))
            })
        return results

    def orchestrate_weave(self, data: List[Any]) -> None:
        """
        Elevates Weaver as the sole orchestrator.
        Takes raw data, passes it to the in-memory Cortex (SubstrateWeaver) for calculation,
        then handles all physical storage to .brain_data crystals.
        """
        from backend.services.loom_service.cortex.SubstrateWeaver import SubstrateWeaver
        
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


    def save_cortex_state(self, cortex_state) -> None:
        """
        Saves the complete transient Cortex state (embeddings, coordinates, velocities, 
        assemblies, working memory graph, causal graphs, physics tensors, replay logs)
        to the Weaver's binary files.
        """
        try:
            import msgpack
            
            segments: Dict[str, bytes] = {}

            # 1. Serialize concept embeddings
            emb_dict = {}
            for k, v in cortex_state.semantic_field.concept_embeddings.items():
                emb_dict[k] = v.tolist()
            segments["brain_embeddings.bin"] = msgpack.packb(emb_dict, use_bin_type=True)
                
            # 2. Coordinates & Velocities: raw float32 binary arrays
            concept_keys = sorted(list(cortex_state.concept_coords.keys()))
            
            coords_list = []
            for k in concept_keys:
                coords_list.append(cortex_state.concept_coords[k].astype(np.float32))
            if coords_list:
                coords_array = np.vstack(coords_list)
                coords_bytes = coords_array.tobytes()
            else:
                coords_bytes = b""
            segments["coordinates.bin"] = coords_bytes
                
            vel_list = []
            for k in concept_keys:
                vel_list.append(cortex_state.concept_velocities[k].astype(np.float32))
            if vel_list:
                vel_array = np.vstack(vel_list)
                vel_bytes = vel_array.tobytes()
            else:
                vel_bytes = b""
            segments["velocities.bin"] = vel_bytes
                
            # 3. Serialize active assemblies
            assemblies = []
            for a in cortex_state.active_assemblies:
                assemblies.append({
                    "id": a.assembly_id,
                    "dominant": a.dominant_concept,
                    "activations": a.node_activations,
                    "confidence": a.confidence,
                    "stability": a.stability,
                    "basin_strength": a.meta_data.get("basin_strength", 0.0)
                })
                
            # 4. Serialize working memory graph
            wm_nodes = []
            for node, data in cortex_state.memory.graph.nodes(data=True):
                wm_nodes.append({
                    "id": node,
                    "activation": float(data.get("activation", 0.0)),
                    "stability": float(data.get("stability", 0.0)),
                    "temporal_depth": int(data.get("temporal_depth", 0))
                })
            wm_edges = []
            for u, v, data in cortex_state.memory.graph.edges(data=True):
                wm_edges.append({
                    "from": u,
                    "to": v,
                    "weight": float(data.get("weight", 0.0))
                })
            wm_data = {
                "nodes": wm_nodes,
                "edges": wm_edges
            }
            segments["working_memory.bin"] = msgpack.packb(wm_data, use_bin_type=True)
                
            # 5. Serialize causal links
            causal_edges = []
            for u, v, data in cortex_state.causal.causal_matrix.edges(data=True):
                causal_edges.append({
                    "from": u,
                    "to": v,
                    "weight": float(data.get("weight", 0.0)),
                    "frequency": int(data.get("frequency", 1))
                })
            segments["causal_links.bin"] = msgpack.packb(causal_edges, use_bin_type=True)
                
            # 6. Serialize meta shards
            meta_shards = []
            for meta_id, shard in cortex_state.meta_shards.meta_store.items():
                meta_shards.append({
                    "id": meta_id,
                    "label": shard.emergent_label,
                    "children": shard.children,
                    "stability": float(shard.stability),
                    "abstraction_level": int(shard.abstraction_level)
                })
                
            # 7. Persist concept-to-thought mapping
            concept_thoughts = getattr(cortex_state, '_concept_thoughts', {})
            
            # 8. Serialize physics tensors
            segments["physics_tensors.bin"] = msgpack.packb(cortex_state.physics_tensors, use_bin_type=True)

            # 9. Serialize replay logs (episodic tensors) Zstd + MsgPack
            replay_list = [v.tolist() for v in cortex_state.episodic_tensors]
            cctx = zstd.ZstdCompressor()
            segments["replay.bin"] = cctx.compress(msgpack.packb(replay_list, use_bin_type=True))
                
            # 10. Save metadata
            metadata = {
                "tick": cortex_state.tick,
                "entropy": cortex_state.entropy,
                "coherence": cortex_state.coherence,
                "pressure": cortex_state.pressure,
                "surprise": cortex_state.surprise,
                "energy_budget": cortex_state.energy_budget,
                "latent_field": cortex_state.latent_field.tolist() if cortex_state.latent_field is not None else None,
                "working_latent": cortex_state.working_latent.tolist() if cortex_state.working_latent is not None else None,
                "latent_velocity": cortex_state.latent_velocity.tolist() if getattr(cortex_state, 'latent_velocity', None) is not None else None,
                "concept_keys": concept_keys,
                "active_assemblies": assemblies,
                "attention_focus": list(cortex_state.attention_focus),
                "meta_shards": meta_shards,
                "concept_thoughts": concept_thoughts
            }
            
            segments["metadata.bin"] = msgpack.packb(metadata, use_bin_type=True)

            # Single atomic commit of every state segment into universe.loom
            self._commit_state_segments(segments)
        except Exception as e:
            print("Error in Weaver save_cortex_state:", e)
            raise e

    def load_cortex_state(self, cortex_state) -> bool:
        """
        Loads and restores the transient Cortex state from Weaver's binary files.
        """
        metadata_blob = self._state_read("metadata.bin")
        if metadata_blob is None:
            return False

        try:
            import msgpack

            metadata = msgpack.unpackb(metadata_blob, raw=False)
                
            cortex_state.tick = metadata.get("tick", 0)
            cortex_state.entropy = metadata.get("entropy", 0.5)
            cortex_state.coherence = metadata.get("coherence", 0.5)
            cortex_state.pressure = metadata.get("pressure", 0.0)
            cortex_state.surprise = metadata.get("surprise", 0.0)
            cortex_state.energy_budget = metadata.get("energy_budget", 1.0)
            
            lf = metadata.get("latent_field")
            if lf is not None:
                cortex_state.latent_field = np.array(lf)
            wl = metadata.get("working_latent")
            if wl is not None:
                cortex_state.working_latent = np.array(wl)
            lv = metadata.get("latent_velocity")
            if lv is not None:
                cortex_state.latent_velocity = np.array(lv)
                
            concept_keys = metadata.get("concept_keys", [])
            cortex_state._concept_thoughts = metadata.get("concept_thoughts", {})
            cortex_state.attention_focus = metadata.get("attention_focus", [])
            
            # Load active assemblies
            from backend.services.loom_service.cortex.latent_field_cognition.attractor_basin_compilation import CognitiveAssembly
            cortex_state.active_assemblies = []
            for a in metadata.get("active_assemblies", []):
                cortex_state.active_assemblies.append(CognitiveAssembly(
                    assembly_id=a["id"],
                    dominant_concept=a["dominant"],
                    node_activations=a["activations"],
                    confidence=a["confidence"],
                    stability=a["stability"],
                    phase_coherence=cortex_state.coherence,
                    meta_data={"basin_strength": a["basin_strength"]}
                ))
                
            # Load meta shards
            from backend.services.loom_service.cortex.latent_field_cognition.attractor_basin_compilation import MetaShard
            cortex_state.meta_shards.meta_store.clear()
            for ms in metadata.get("meta_shards", []):
                cortex_state.meta_shards.meta_store[ms["id"]] = MetaShard(
                    meta_id=ms["id"],
                    children=ms["children"],
                    field_vec=np.zeros(1),
                    abstraction_level=ms["abstraction_level"],
                    stability=ms["stability"],
                    emergent_label=ms["label"]
                )
                
            # Load concept coordinates
            coords_bytes = self._state_read("coordinates.bin")
            if coords_bytes and concept_keys:
                coords_array = np.frombuffer(coords_bytes, dtype=np.float32).reshape(len(concept_keys), -1)
                cortex_state.concept_coords = {k: coords_array[idx] for idx, k in enumerate(concept_keys)}
            else:
                cortex_state.concept_coords = {}

            # Load concept velocities
            vel_bytes = self._state_read("velocities.bin")
            if vel_bytes and concept_keys:
                vel_array = np.frombuffer(vel_bytes, dtype=np.float32).reshape(len(concept_keys), -1)
                cortex_state.concept_velocities = {k: vel_array[idx] for idx, k in enumerate(concept_keys)}
            else:
                cortex_state.concept_velocities = {}

            # Load causal links
            causal_blob = self._state_read("causal_links.bin")
            cortex_state.causal.causal_matrix.clear()
            if causal_blob:
                causal_edges = msgpack.unpackb(causal_blob, raw=False)
                for l in causal_edges:
                    cortex_state.causal.causal_matrix.add_edge(
                        l["from"],
                        l["to"],
                        weight=l["weight"],
                        frequency=l["frequency"],
                        last_seen=time.time()
                    )
                    
            # Load working memory graph
            wm_blob = self._state_read("working_memory.bin")
            cortex_state.memory.graph.clear()
            if wm_blob:
                wm_data = msgpack.unpackb(wm_blob, raw=False)
                for n in wm_data.get("nodes", []):
                    cortex_state.memory.graph.add_node(
                        n["id"],
                        activation=n["activation"],
                        stability=n["stability"],
                        temporal_depth=n["temporal_depth"]
                    )
                for e in wm_data.get("edges", []):
                    cortex_state.memory.graph.add_edge(
                        e["from"],
                        e["to"],
                        weight=e["weight"]
                    )
                    
            # Load physics tensors
            physics_blob = self._state_read("physics_tensors.bin")
            if physics_blob:
                cortex_state.physics_tensors = msgpack.unpackb(physics_blob, raw=False)
            else:
                cortex_state.physics_tensors = {}

            # Load replay logs
            compressed_replay = self._state_read("replay.bin")
            cortex_state.episodic_tensors = []
            if compressed_replay:
                dctx = zstd.ZstdDecompressor()
                replay_list = msgpack.unpackb(dctx.decompress(compressed_replay), raw=False)
                cortex_state.episodic_tensors = [np.array(v) for v in replay_list]

            # Load concept embeddings
            cortex_state.semantic_field.concept_embeddings.clear()
            emb_blob = self._state_read("brain_embeddings.bin")
            if emb_blob:
                emb_dict = msgpack.unpackb(emb_blob, raw=False)
                for k, v in emb_dict.items():
                    cortex_state.semantic_field.concept_embeddings[k] = np.array(v)
                    
            return True
        except Exception as e:
            print("Error loading Cortex state from Weaver:", e)
            return False

    def close(self) -> None:
        """Safe shutdown: fold crystal tails into .idx snapshots + save state snapshot."""
        for store in list(self._stores.values()):
            try:
                store.close()
            except Exception:
                pass
        self._stores = {}
        self.checkpoint()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
