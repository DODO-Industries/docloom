import os
import time
import struct
import numpy as np
import zstandard as zstd
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from backend.config.tunningManagment import tuning_manager

from backend.services.loom_service.weaver.seed_core import UniverseSeedCore
from backend.services.loom_service.weaver.atlas_router import GlobalAtlasRouter
from backend.services.loom_service.weaver.substrate_layout import LoomSubstrate
from backend.services.loom_service.weaver.storage_physics import LatentFieldPhysicsEngine
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
        self.max_shards_per_crystal = tuning_manager.get_int("MAX_CRYSTAL_SIZE", 500000)
        
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
        if lambda_base == 0.05:
            lambda_base_val = tuning_manager.get_float("MEMORY_DECAY_RATE", lambda_base)
        else:
            lambda_base_val = lambda_base
        self.fluidity = DynamicMemoryFluidity(lambda_base=lambda_base_val)
        
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
                                "latent_position": np.zeros(self.dimension, dtype=np.float32),
                                "velocity": np.zeros(self.dimension, dtype=np.float32),
                                "phase_angle": 0.0,
                                "hits": 0,
                                "last_recalled": float(t),
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
            "latent_position": true_vector.copy().astype(np.float32),
            "velocity": np.zeros(self.dimension, dtype=np.float32),
            "phase_angle": 0.0,
            "hits": 0,
            "last_recalled": meta["last_recalled"],
            "crystal_path": target_crystal_path
        }
        self._append_journal_entry(shard_id, 1.0, meta["last_recalled"])

        return target_crystal_path

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
        Recall Pathway (Querying):
        1. Topology Sweep: Find closest target .loom crystal in Atlas
        2. Zero-Copy Fetch: memory map target .loom file to read shards
        3. Wave Resonance: calculate wave interference and excitation gain
        4. Re-awaken sleeping nodes if activation threshold is crossed
        5. Kuramoto Coupling: lock active phase angles
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
            self.atlas.save()

        # 2. Zero-Copy Fetch
        journal = LoomSubstrate.get_journal(target_path)
        
        results = []
        for idx, entry in enumerate(journal):
            sid = entry["shard_id"]
            meta = entry["meta"]
            vec = LoomSubstrate.get_vector_mmap(target_path, idx)

            # Retrieve dynamic active states from RAM ledger if available
            is_active = sid in self.ram_ledger
            
            if is_active:
                ledger_entry = self.ram_ledger[sid]
                last_recalled = ledger_entry.get("last_recalled", current_time)
                hits = ledger_entry.get("hits", 0)
                initial_act = ledger_entry.get("activation", 1.0)
                cell_phase = ledger_entry.get("phase_angle", 0.0)
            else:
                last_recalled = current_time
                hits = 0
                initial_act = 0.0  # Asleep
                cell_phase = 0.0

            mass = meta.get("mass", 1.0)

            # Calculate Hyperdimensional Overlap Similarity (cosine similarity)
            q_norm = np.linalg.norm(query_vector)
            v_norm = np.linalg.norm(vec)
            if q_norm == 0 or v_norm == 0:
                hdc_sim = 0.0
            else:
                hdc_sim = float(np.dot(query_vector, vec) / (q_norm * v_norm))

            # 3. Wave Resonance
            gain = self.physics.calculate_excitation_gain(
                hdc_similarity=hdc_sim,
                query_phase=query_phase,
                cell_phase=cell_phase
            )

            # Continuous wave excitation updates
            new_act = initial_act + gain
            new_act_clipped = min(1.0, max(0.0, new_act))

            # Calculate gravitational score
            grav_pull = self.physics.calculate_gravitational_attraction(
                vector_a=query_vector,
                mass_a=1.0,
                vector_b=vec,
                mass_b=mass,
                apply_wormhole=True
            )
            score = new_act_clipped * grav_pull

            results.append({
                "shard_id": sid,
                "text": meta.get("text", ""),
                "score": score,
                "vector": vec,
                "meta": meta,
                "decayed_activation": new_act_clipped,
                "hits": hits,
                "is_active": is_active,
                "hdc_sim": hdc_sim,
                "cell_phase": cell_phase
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]

        # 4. Plasticity & Wave Excitation Reinforcement / Re-awakening
        recalled_ids = {r["shard_id"] for r in top_results}
        
        for r in results:
            sid = r["shard_id"]
            excited_act = r["decayed_activation"]
            hits = r["hits"]
            cell_phase = r["cell_phase"]
            is_active = r["is_active"]
            vec = r["vector"]

            if excited_act > 0.001:
                # Node is awake or re-awakening
                if sid in recalled_ids:
                    # Reinforce hits
                    last_recalled, hits = self.fluidity.reinforce(current_time, hits)
                    # For recalled nodes, we ensure their activation is boosted/reinforced
                    recall_boost_val = tuning_manager.get_float("RECALL_BOOST_VALUE", 0.15)
                    excited_act = min(1.0, excited_act + recall_boost_val)
                else:
                    last_recalled = current_time

                self.ram_ledger[sid] = {
                    "activation": excited_act,
                    "latent_position": vec.copy().astype(np.float32),
                    "velocity": np.zeros(self.dimension, dtype=np.float32),
                    "phase_angle": cell_phase,
                    "hits": hits,
                    "last_recalled": last_recalled,
                    "crystal_path": target_path
                }
                self._append_journal_entry(sid, excited_act, last_recalled)
                r["hits"] = hits
                r["decayed_activation"] = excited_act
            else:
                # Node is in Deep Sleep
                if is_active:
                    del self.ram_ledger[sid]

        # 5. Kuramoto Coupling phase locking updates on active ledger concepts
        if self.ram_ledger:
            active_phases = np.array([state["phase_angle"] for state in self.ram_ledger.values()])
            active_activations = np.array([state["activation"] for state in self.ram_ledger.values()])
            
            for sid, state in self.ram_ledger.items():
                d_theta = self.physics.compute_kuramoto_coupling(
                    active_phases=active_phases,
                    active_activations=active_activations,
                    current_phase=state["phase_angle"]
                )
                new_phase = state["phase_angle"] + d_theta
                # Map to [-pi, pi]
                new_phase = (new_phase + np.pi) % (2 * np.pi) - np.pi
                state["phase_angle"] = float(new_phase)

        return [{
            "shard_id": r["shard_id"],
            "text": r["text"],
            "score": r["score"],
            "hits": r["hits"],
            "activation": r["decayed_activation"]
        } for r in top_results]

    def save_cortex_state(self, cortex_state) -> None:
        """
        Saves the complete transient Cortex state (embeddings, coordinates, velocities, 
        assemblies, working memory graph, causal graphs, physics tensors, replay logs)
        to the Weaver's binary files.
        """
        try:
            import msgpack
            
            # 1. Serialize concept embeddings
            emb_dict = {}
            for k, v in cortex_state.semantic_field.concept_embeddings.items():
                emb_dict[k] = v.tolist()
            emb_file_path = os.path.join(self.storage_dir, "brain_embeddings.bin")
            with open(emb_file_path, "wb") as f:
                f.write(msgpack.packb(emb_dict, use_bin_type=True))
                
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
            coords_file_path = os.path.join(self.storage_dir, "coordinates.bin")
            with open(coords_file_path, "wb") as f:
                f.write(coords_bytes)
                
            vel_list = []
            for k in concept_keys:
                vel_list.append(cortex_state.concept_velocities[k].astype(np.float32))
            if vel_list:
                vel_array = np.vstack(vel_list)
                vel_bytes = vel_array.tobytes()
            else:
                vel_bytes = b""
            vel_file_path = os.path.join(self.storage_dir, "velocities.bin")
            with open(vel_file_path, "wb") as f:
                f.write(vel_bytes)
                
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
            wm_file_path = os.path.join(self.storage_dir, "working_memory.bin")
            with open(wm_file_path, "wb") as f:
                f.write(msgpack.packb(wm_data, use_bin_type=True))
                
            # 5. Serialize causal links
            causal_edges = []
            for u, v, data in cortex_state.causal.causal_matrix.edges(data=True):
                causal_edges.append({
                    "from": u,
                    "to": v,
                    "weight": float(data.get("weight", 0.0)),
                    "frequency": int(data.get("frequency", 1))
                })
            causal_file_path = os.path.join(self.storage_dir, "causal_links.bin")
            with open(causal_file_path, "wb") as f:
                f.write(msgpack.packb(causal_edges, use_bin_type=True))
                
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
            physics_file_path = os.path.join(self.storage_dir, "physics_tensors.bin")
            with open(physics_file_path, "wb") as f:
                f.write(msgpack.packb(cortex_state.physics_tensors, use_bin_type=True))
                
            # 9. Serialize replay logs (episodic tensors) Zstd + MsgPack
            replay_list = [v.tolist() for v in cortex_state.episodic_tensors]
            cctx = zstd.ZstdCompressor()
            compressed_replay = cctx.compress(msgpack.packb(replay_list, use_bin_type=True))
            replay_file_path = os.path.join(self.storage_dir, "replay.bin")
            with open(replay_file_path, "wb") as f:
                f.write(compressed_replay)
                
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
            
            metadata_file_path = os.path.join(self.storage_dir, "metadata.bin")
            with open(metadata_file_path, "wb") as f:
                f.write(msgpack.packb(metadata, use_bin_type=True))
        except Exception as e:
            print("Error in Weaver save_cortex_state:", e)
            raise e

    def load_cortex_state(self, cortex_state) -> bool:
        """
        Loads and restores the transient Cortex state from Weaver's binary files.
        """
        metadata_file_path = os.path.join(self.storage_dir, "metadata.bin")
        if not os.path.exists(metadata_file_path):
            return False
            
        try:
            import msgpack
            
            with open(metadata_file_path, "rb") as f:
                metadata = msgpack.unpackb(f.read(), raw=False)
                
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
            coords_file_path = os.path.join(self.storage_dir, "coordinates.bin")
            if os.path.exists(coords_file_path) and concept_keys:
                with open(coords_file_path, "rb") as f:
                    coords_bytes = f.read()
                if coords_bytes:
                    coords_array = np.frombuffer(coords_bytes, dtype=np.float32).reshape(len(concept_keys), -1)
                    cortex_state.concept_coords = {k: coords_array[idx] for idx, k in enumerate(concept_keys)}
            else:
                cortex_state.concept_coords = {}
                
            # Load concept velocities
            vel_file_path = os.path.join(self.storage_dir, "velocities.bin")
            if os.path.exists(vel_file_path) and concept_keys:
                with open(vel_file_path, "rb") as f:
                    vel_bytes = f.read()
                if vel_bytes:
                    vel_array = np.frombuffer(vel_bytes, dtype=np.float32).reshape(len(concept_keys), -1)
                    cortex_state.concept_velocities = {k: vel_array[idx] for idx, k in enumerate(concept_keys)}
            else:
                cortex_state.concept_velocities = {}
                
            # Load causal links
            causal_file_path = os.path.join(self.storage_dir, "causal_links.bin")
            cortex_state.causal.causal_matrix.clear()
            if os.path.exists(causal_file_path):
                with open(causal_file_path, "rb") as f:
                    causal_edges = msgpack.unpackb(f.read(), raw=False)
                for l in causal_edges:
                    cortex_state.causal.causal_matrix.add_edge(
                        l["from"],
                        l["to"],
                        weight=l["weight"],
                        frequency=l["frequency"],
                        last_seen=time.time()
                    )
                    
            # Load working memory graph
            wm_file_path = os.path.join(self.storage_dir, "working_memory.bin")
            cortex_state.memory.graph.clear()
            if os.path.exists(wm_file_path):
                with open(wm_file_path, "rb") as f:
                    wm_data = msgpack.unpackb(f.read(), raw=False)
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
            physics_file_path = os.path.join(self.storage_dir, "physics_tensors.bin")
            if os.path.exists(physics_file_path):
                with open(physics_file_path, "rb") as f:
                    cortex_state.physics_tensors = msgpack.unpackb(f.read(), raw=False)
            else:
                cortex_state.physics_tensors = {}
                
            # Load replay logs
            replay_file_path = os.path.join(self.storage_dir, "replay.bin")
            cortex_state.episodic_tensors = []
            if os.path.exists(replay_file_path):
                dctx = zstd.ZstdDecompressor()
                with open(replay_file_path, "rb") as f:
                    compressed_replay = f.read()
                if compressed_replay:
                    replay_list = msgpack.unpackb(dctx.decompress(compressed_replay), raw=False)
                    cortex_state.episodic_tensors = [np.array(v) for v in replay_list]
                    
            # Load concept embeddings
            cortex_state.semantic_field.concept_embeddings.clear()
            emb_file_path_bin = os.path.join(self.storage_dir, "brain_embeddings.bin")
            if os.path.exists(emb_file_path_bin):
                with open(emb_file_path_bin, "rb") as f:
                    emb_dict = msgpack.unpackb(f.read(), raw=False)
                for k, v in emb_dict.items():
                    cortex_state.semantic_field.concept_embeddings[k] = np.array(v)
                    
            return True
        except Exception as e:
            print("Error loading Cortex state from Weaver:", e)
            return False

    def close(self) -> None:
        """Safe shutdown sequence saving state snapshot."""
        self.checkpoint()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
