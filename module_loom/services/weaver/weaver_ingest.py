"""
DocLoom Weave Ingestion Mixin
Handles single shard ingestion, bulk ingestion, cellular crystal splitting,
128-bit leader sign signature bucketing, and orchestrate_weave pipelines.
"""
import os
import sys
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from module_loom.config.tuning_config import tuning_manager
from module_loom.services.weaver.weaver_common import (
    hash_shard_id, seed_phase, default_cognitive_fields, hamming_distances
)
from module_loom.services.weaver.atlas_router import GlobalAtlasRouter
from module_loom.services.weaver.substrate_layout import LoomStore


class WeaveIngestMixin:
    """Ingestion, routing, and cellular division methods for WeaveBrainCoordinator."""

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
        Atlas Capture (Routing) + Cellular Division. Returns target crystal path,
        splitting a full or decoherent crystal into a new hot sub-sector when needed.
        """
        if not self.atlas.crystals:
            target = os.path.join(self.storage_dir, "crystal_1.loom")
            self.atlas.register_crystal(target, route_vec, state="hot")
            self._journal_crystal_registration(target)
            return target

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
            if v_norm == 0.0:
                return open_paths[0]
            v_unit = route_vec / v_norm
            best_p, best_sim = None, -1.0
            for p in open_paths:
                c = self.atlas.crystals[p]["centroid"]
                c_norm = np.linalg.norm(c)
                sim = float(np.dot(v_unit, c / c_norm)) if c_norm > 0 else 0.0
                if sim > best_sim:
                    best_sim, best_p = sim, p
            return best_p or open_paths[0]

        # Cellular Division: all crystals are full/decoherent.
        target = self.atlas.route_vector(route_vec)
        if _is_full(self.atlas.crystals[target]):
            self.atlas.set_state(target, "cold")
            self._mark_atlas_dirty()

            new_path = self._get_split_path(target)
            seed_offset = self.seed_core.get_split_offset(new_path)
            new_centroid = route_vec + seed_offset
            old_norm = np.linalg.norm(self.atlas.crystals[target]["centroid"])
            new_norm = np.linalg.norm(new_centroid)
            if new_norm > 0 and old_norm > 0:
                new_centroid = new_centroid * (old_norm / new_norm)

            self.atlas.register_crystal(new_path, new_centroid, state="hot")
            self._journal_crystal_registration(new_path)
            return new_path
        return target

    @staticmethod
    def _momentum_signature(momentum: np.ndarray) -> bytes:
        """128-bit packed sign signature of a vector (routing leader form)."""
        sig = np.sign(momentum[:128]).astype(np.int8)
        sig[sig == 0] = 1
        if len(sig) < 128:
            sig = np.pad(sig, (0, 128 - len(sig)), constant_values=1)
        return np.packbits(((sig + 1) // 2).astype(np.uint8)).tobytes()

    def _resolve_bucket(self, store: LoomStore, momentum: np.ndarray) -> int:
        """Resonance-jump bucket resolution via 128-bit Hamming leader distance."""
        max_leaders = tuning_manager.get_int("LOOM_MAX_LEADERS", 512)
        leader_matrix = store.get_leader_matrix()
        sig_packed = self._momentum_signature(momentum)

        if len(leader_matrix):
            sig_arr = np.frombuffer(sig_packed, dtype=np.uint8)
            dists = hamming_distances(leader_matrix, sig_arr)
            best_idx = int(np.argmin(dists))
            best_dist = int(dists[best_idx])
            if best_dist <= 32:
                return best_idx

        if len(leader_matrix) >= max_leaders:
            return best_idx if len(leader_matrix) else 0

        sig = np.sign(momentum[:128]).astype(np.int8)
        sig[sig == 0] = 1
        store.append_leader(sig)
        return len(leader_matrix)

    def ingest_shard(
        self,
        shard_id: str,
        true_vector: np.ndarray,
        text: str,
        mass: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        target_crystal: Optional[str] = None
    ) -> str:
        """Physical Ingestion Pathway: Seed -> Momentum -> Atlas Route -> Substrate Append."""
        if len(true_vector) != self.dimension:
            raise ValueError(f"Vector dimension must be {self.dimension}")

        seed_scaffold = self.seed_core.get_micro_socket(shard_id)
        momentum = self.physics.calculate_momentum_vector(
            true_vector=true_vector,
            seed_scaffold=seed_scaffold,
            mass=mass
        )

        target_crystal_path = target_crystal if target_crystal is not None else self._route_for_ingest(true_vector)
        embassy_eval = self.atlas.route_vector_with_embassy(true_vector)

        store = self._get_store(target_crystal_path)
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

        self.atlas.update_centroid(target_crystal_path, true_vector)
        self._mark_atlas_dirty()

        target_crystal_path = sys.intern(target_crystal_path)
        self.hash_to_shard[hash_shard_id(shard_id)] = (shard_id, target_crystal_path)
        placed = self._ledger_put_many({shard_id: {
            "activation": 1.0,
            "latent_position": true_vector.copy(),
            "velocity": np.zeros(self.dimension, dtype=np.float32),
            "phase_angle": seed_phase(true_vector),
            "hits": 0,
            "last_recalled": now,
            "crystal_path": target_crystal_path,
            "crystal_idx": crystal_idx,
            **default_cognitive_fields()
        }})
        if placed:
            self._append_journal_entry(shard_id, 1.0, now)
            self.process_cognitive_tick({shard_id: 0.8})
            if self._last_ingested_sid is not None and self._last_ingested_sid != shard_id:
                if self._ingest_edge_coherent(self._last_ingested_sid, true_vector):
                    self.causal.record_transition(self._last_ingested_sid, shard_id, weight=1.0, prediction_error=0.5)
            self._last_ingested_sid = shard_id
            self._last_message_ts = now
            self._shards_since_last_sleep += 1

        return target_crystal_path

    def _ingest_edge_coherent(self, prev_sid: str, new_pos: np.ndarray) -> bool:
        """Coherence gate preventing accidental adjacency noise on bulk data."""
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
        """Bulk Ingestion Pathway — batched binary writes and journal entries."""
        if not items:
            return {}
        now = time.time()
        grouped: Dict[str, List[Tuple[str, np.ndarray, float, Dict[str, Any]]]] = {}
        momenta: Dict[str, List[np.ndarray]] = {}

        for item in items:
            shard_id = item["shard_id"]
            vec = np.asarray(item["vector"], dtype=np.float32)
            if len(vec) != self.dimension:
                raise ValueError(f"Vector dimension must be {self.dimension} (shard {shard_id})")
            mass = float(item.get("mass", 1.0))

            scaffold = self.seed_core.get_micro_socket(shard_id)
            momentum = self.physics.calculate_momentum_vector(vec, scaffold, mass)
            target = self._route_for_ingest(vec)
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
            target = sys.intern(target)
            store = self._get_store(target)

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
                    "phase_angle": seed_phase(vec),
                    "hits": 0,
                    "last_recalled": now,
                    "crystal_path": target,
                    "crystal_idx": crystal_idx,
                    **default_cognitive_fields()
                }

        placed = set(self._ledger_put_many(ledger_batch))
        journal_entries = [(sid, 1.0, now) for sid in placed]
        self._mark_atlas_dirty()
        self._append_journal_entries(journal_entries)
        if placed:
            self.process_cognitive_tick({sid: 0.8 for sid in placed})
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

    def orchestrate_weave(self, data: List[Any]) -> None:
        """Takes raw data, calculates graph in memory via SubstrateWeaver, and persists."""
        from module_loom.services.cortex.SubstrateWeaver import SubstrateWeaver

        weaver = SubstrateWeaver(storage_dir=self.storage_dir)
        result = weaver.weave(data, output_path=None)

        nodes = result["nodes"]
        edges = result["edges"]
        all_embeddings = result["all_embeddings"]

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

        self.ingest_batch(batch_items)
        engine = weaver.engine
        n = engine.current_node_count
        self._commit_state_segments({
            "coordinates.bin": engine.coords_map[:n].tobytes(),
            "physics_tensors.bin": engine.physics_map[:n].tobytes(),
            "hdc_signatures.bin": engine.hdc_map[:n].tobytes(),
            "causal_links.bin": engine.causal_map[:n].tobytes(),
        })
        print("Weaver orchestration complete. Pure in-memory Cortex data successfully flushed to universe.loom.")
