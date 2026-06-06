import os
import time
import struct
import numpy as np
import zstandard as zstd
import hashlib
from typing import Dict, List, Any, Optional, Tuple

from backend.services.loom_service.weaver.seed_core import UniverseSeedCore
from backend.services.loom_service.weaver.atlas_router import GlobalAtlasRouter
from backend.services.loom_service.weaver.substrate_layout import LoomSubstrate
from backend.services.loom_service.weaver.physics_engine import LatentFieldPhysicsEngine
from backend.services.loom_service.weaver.memory_fluidity import DynamicMemoryFluidity

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
        self.max_shards_per_crystal = 500000
        
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
            # Fallback to loading seed from existing atlas or parameters
            if os.path.exists(self.atlas_path):
                self.atlas = GlobalAtlasRouter(atlas_path=self.atlas_path)
                self.seed = self.atlas.seed
            else:
                if seed is None:
                    from backend.config.envConfig import MASTER_SEED
                    self.seed = MASTER_SEED
                else:
                    self.seed = seed & 0xFFFFFFFFFFFFFFFF
            
            # Save the metric file
            try:
                with open(self.metric_path, "wb") as f:
                    f.write(struct.pack("<4sHQ", b"LOOM", self.dimension, self.seed))
            except Exception:
                pass
                
        # Initialize Atlas Router using atlas.capnp
        self.atlas = GlobalAtlasRouter(atlas_path=self.atlas_path, seed=self.seed)
        
        self.seed_core = UniverseSeedCore(seed=self.seed, dimension=self.dimension)
        self.physics = LatentFieldPhysicsEngine(dimension=self.dimension)
        self.fluidity = DynamicMemoryFluidity(lambda_base=lambda_base)
        
        # Initialize Active Latent Cortex Ledger and boot-up restore
        self.ram_ledger: Dict[str, Dict[str, Any]] = {}
        self._load_persistence_layers()

    def _load_persistence_layers(self) -> None:
        """
        Sub-Second Boot Sequence:
        Loads the compressed snapshot and replays remaining journal entries to reconstruct RAM ledger.
        """
        # Build hash-to-shard mapping from physical journals
        self.hash_to_shard: Dict[int, Tuple[str, str]] = {}
        for crystal_path in list(self.atlas.crystals.keys()):
            if os.path.exists(crystal_path):
                try:
                    journal = LoomSubstrate.get_journal(crystal_path)
                    for entry in journal:
                        sid = entry["shard_id"]
                        self.hash_to_shard[hash_shard_id(sid)] = (sid, crystal_path)
                except Exception:
                    pass
                    
        # Load snapshot (latent_state.snapshot)
        snapshot_path = os.path.join(self.storage_dir, "latent_state.snapshot")
        if os.path.exists(snapshot_path):
            try:
                with open(snapshot_path, "rb") as f:
                    compressed = f.read()
                if compressed:
                    dctx = zstd.ZstdDecompressor()
                    decompressed = dctx.decompress(compressed)
                    num_records = len(decompressed) // 16
                    for i in range(num_records):
                        chunk = decompressed[i*16 : (i+1)*16]
                        h, act, t = struct.unpack("<QfI", chunk)
                        if h in self.hash_to_shard:
                            sid, crystal_path = self.hash_to_shard[h]
                            self.ram_ledger[sid] = {
                                "activation": act,
                                "last_recalled": float(t),
                                "hits": 0,
                                "crystal_path": crystal_path
                            }
            except Exception:
                pass
                
        # Replay journal (cortex_journal.bin)
        journal_path = os.path.join(self.storage_dir, "cortex_journal.bin")
        if os.path.exists(journal_path):
            try:
                with open(journal_path, "rb") as f:
                    journal_bytes = f.read()
                num_txns = len(journal_bytes) // 32
                for i in range(num_txns):
                    chunk = journal_bytes[i*32 : (i+1)*32]
                    magic, h, act, t, _ = struct.unpack("<4sQfd8s", chunk)
                    if magic == b"TXN\x00" and h in self.hash_to_shard:
                        sid, crystal_path = self.hash_to_shard[h]
                        if sid not in self.ram_ledger:
                            self.ram_ledger[sid] = {
                                "hits": 0,
                                "crystal_path": crystal_path
                            }
                        self.ram_ledger[sid]["activation"] = act
                        self.ram_ledger[sid]["last_recalled"] = t
            except Exception:
                pass

    def _append_journal_entry(self, shard_id: str, activation: float, timestamp: float) -> None:
        """Appends a 32-byte transaction string to the end of cortex_journal.bin."""
        try:
            h = hash_shard_id(shard_id)
            txn = struct.pack("<4sQfd8s", b"TXN\x00", h, activation, timestamp, b"\x00" * 8)
            journal_path = os.path.join(self.storage_dir, "cortex_journal.bin")
            with open(journal_path, "ab") as f:
                f.write(txn)
        except Exception:
            pass

    def checkpoint(self) -> None:
        """
        Crystallizing Checkpoints:
        Dumps compressed snapshot of Active RAM ledger and wipes the cortex journal.
        """
        try:
            records = bytearray()
            for sid, entry in list(self.ram_ledger.items()):
                act = entry.get("activation", 0.0)
                if act > 0.0:
                    h = hash_shard_id(sid)
                    t = int(entry.get("last_recalled", 0.0))
                    records.extend(struct.pack("<QfI", h, act, t))
                    
            if records:
                cctx = zstd.ZstdCompressor()
                compressed = cctx.compress(records)
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
        3. Atlas Capture: Route to target .loom container (handles Cellular division fracturing)
        4. Substrate Append Execution: Write to physical binary file
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

        # 3. Atlas Capture (Routing)
        if not self.atlas.crystals:
            target_crystal_path = os.path.join(self.storage_dir, "crystal_1.loom")
            self.atlas.register_crystal(target_crystal_path, momentum, state="hot")
        else:
            target_crystal_path = self.atlas.route_vector(momentum)
            
        # Cellular Division Logic
        if os.path.exists(target_crystal_path):
            crystal_info = self.atlas.crystals.get(target_crystal_path)
            if crystal_info and crystal_info.get("num_shards", 0) >= self.max_shards_per_crystal:
                # Freeze current crystal path
                self.atlas.set_state(target_crystal_path, "warm")
                
                # Determine split path
                new_path = self._get_split_path(target_crystal_path)
                
                # Compute perturbed sub-centroid (Proximity Retention)
                offset = self.seed_core.get_macro_offset(os.path.basename(new_path))
                old_centroid = crystal_info["centroid"]
                new_centroid = old_centroid + 0.05 * offset
                
                # Maintain norm scale
                old_norm = np.linalg.norm(old_centroid)
                new_norm = np.linalg.norm(new_centroid)
                if new_norm > 0:
                    new_centroid = new_centroid * (old_norm / new_norm)
                    
                # Register new sub-centroid
                self.atlas.register_crystal(new_path, new_centroid, state="hot")
                target_crystal_path = new_path

        # 4. Substrate Append Execution
        leaders: List[np.ndarray] = []
        shards: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []

        if os.path.exists(target_crystal_path):
            leaders = LoomSubstrate.get_routing_space(target_crystal_path)
            journal = LoomSubstrate.get_journal(target_crystal_path)
            for idx, entry in enumerate(journal):
                sid = entry["shard_id"]
                vec = LoomSubstrate.get_vector_mmap(target_crystal_path, idx)
                shards.append((sid, vec, entry["meta"]))

        # Initialize metadata
        meta = metadata or {}
        meta.update({
            "text": text,
            "mass": mass,
            "last_recalled": time.time(),
            "hits": 0,
            "activation": 1.0
        })

        # Append shard
        shards.append((shard_id, true_vector, meta))

        # Check routing space leaders update
        sig = np.sign(momentum[:128]).astype(np.int8)
        sig[sig == 0] = 1
        sig_bin = ((sig + 1) // 2).astype(np.uint8)

        best_dist = 129
        for leader in leaders:
            leader_bin = ((leader + 1) // 2).astype(np.uint8)
            dist = np.bitwise_xor(sig_bin, leader_bin).sum()
            if dist < best_dist:
                best_dist = dist

        if best_dist > 32 or not leaders:
            leaders.append(sig)

        # Write back to Loom container
        LoomSubstrate.write(
            path=target_crystal_path,
            leaders=leaders,
            shards=shards,
            vector_dim=self.dimension,
            seed=self.seed
        )

        # Update centroid in global atlas
        self.atlas.update_centroid(target_crystal_path, momentum)
        self.atlas.save()

        # Update active RAM ledger and delta log
        self.ram_ledger[shard_id] = {
            "activation": 1.0,
            "last_recalled": meta["last_recalled"],
            "hits": 0,
            "crystal_path": target_crystal_path
        }
        self._append_journal_entry(shard_id, 1.0, meta["last_recalled"])

        return target_crystal_path

    def recall(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        current_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Recall Pathway (Querying):
        1. Topology Sweep: Find closest target .loom crystal in Atlas
        2. Zero-Copy Fetch: memory map target .loom file to read shards
        3. Physics Correction & Fluidity updates
        4. Reinforce matched thoughts
        """
        if not self.atlas.crystals:
            return []

        if current_time is None:
            current_time = time.time()

        # 1. Topology Sweep
        target_path = self.atlas.route_vector(query_vector)
        if not os.path.exists(target_path):
            return []

        # OS mounting emulation: transition Cold crystal to Warm for mmap access
        if self.atlas.crystals[target_path]["state"] == "cold":
            self.atlas.set_state(target_path, "warm")
            self.atlas.save()

        # 2. Zero-Copy Fetch
        journal = LoomSubstrate.get_journal(target_path)
        leaders = LoomSubstrate.get_routing_space(target_path)
        
        results = []
        for idx, entry in enumerate(journal):
            sid = entry["shard_id"]
            meta = entry["meta"]
            vec = LoomSubstrate.get_vector_mmap(target_path, idx)

            # Retrieve dynamic active states from RAM ledger if available
            last_recalled = meta.get("last_recalled", current_time)
            hits = meta.get("hits", 0)
            initial_act = meta.get("activation", 1.0)
            
            if sid in self.ram_ledger:
                ledger_entry = self.ram_ledger[sid]
                last_recalled = ledger_entry.get("last_recalled", last_recalled)
                hits = ledger_entry.get("hits", hits)
                initial_act = ledger_entry.get("activation", initial_act)
                
            mass = meta.get("mass", 1.0)

            # 3. Physics Correction
            decayed_act = self.fluidity.calculate_decay(
                initial_activation=initial_act,
                last_recalled=last_recalled,
                current_time=current_time,
                hits=hits
            )

            # Calculate gravitational pull inside warped latent field
            grav_pull = self.physics.calculate_gravitational_attraction(
                vector_a=query_vector,
                mass_a=1.0,
                vector_b=vec,
                mass_b=mass,
                apply_wormhole=True
            )

            score = decayed_act * grav_pull
            results.append({
                "shard_id": sid,
                "text": meta.get("text", ""),
                "score": score,
                "vector": vec,
                "meta": meta,
                "decayed_activation": decayed_act
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]

        # 4. Plasticity Reinforcement
        recalled_ids = {r["shard_id"] for r in top_results}
        all_shards_reconstructed = []
        
        for idx, entry in enumerate(journal):
            sid = entry["shard_id"]
            meta = entry["meta"]
            vec = LoomSubstrate.get_vector_mmap(target_path, idx)

            current_hits = meta.get("hits", 0)
            if sid in self.ram_ledger:
                current_hits = self.ram_ledger[sid].get("hits", current_hits)

            if sid in recalled_ids:
                # Reinforce
                last_recalled, hits = self.fluidity.reinforce(current_time, current_hits)
                meta["last_recalled"] = last_recalled
                meta["hits"] = hits
                meta["activation"] = 1.0
            else:
                # Decay
                last_recalled = meta.get("last_recalled", current_time)
                hits = meta.get("hits", 0)
                initial_act = meta.get("activation", 1.0)
                
                if sid in self.ram_ledger:
                    ledger_entry = self.ram_ledger[sid]
                    last_recalled = ledger_entry.get("last_recalled", last_recalled)
                    hits = ledger_entry.get("hits", hits)
                    initial_act = ledger_entry.get("activation", initial_act)
                    
                decayed_act = self.fluidity.calculate_decay(
                    initial_activation=initial_act,
                    last_recalled=last_recalled,
                    current_time=current_time,
                    hits=hits
                )
                meta["activation"] = decayed_act
                meta["last_recalled"] = current_time
                meta["hits"] = hits

            all_shards_reconstructed.append((sid, vec, meta))
            
            # Update RAM ledger and append to delta journal
            self.ram_ledger[sid] = {
                "activation": meta["activation"],
                "last_recalled": meta["last_recalled"],
                "hits": meta["hits"],
                "crystal_path": target_path
            }
            self._append_journal_entry(sid, meta["activation"], meta["last_recalled"])

        # Persist updated temporal/fluidity values to substrate container
        LoomSubstrate.write(
            path=target_path,
            leaders=leaders,
            shards=all_shards_reconstructed,
            vector_dim=self.dimension,
            seed=self.seed
        )

        return [{
            "shard_id": r["shard_id"],
            "text": r["text"],
            "score": r["score"],
            "hits": r["meta"]["hits"],
            "activation": r["decayed_activation"]
        } for r in top_results]

    def close(self) -> None:
        """Safe shutdown sequence saving state snapshot."""
        self.checkpoint()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
