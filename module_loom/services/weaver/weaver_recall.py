"""
DocLoom Weave Recall Mixin
Handles single-crystal recall, multi-cluster border embassy fusion,
deep fallback sweeps, causal spread, and multi-turn attractor working memory.
"""
import os
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from module_loom.config.tuning_config import tuning_manager
from module_loom.services.weaver.weaver_common import (
    hash_shard_id, seed_phase, default_cognitive_fields, hamming_distances
)
from module_loom.services.weaver.substrate_layout import LoomStore, UNBUCKETED


class WeaveRecallMixin:
    """Recall, multi-pass distillation, and cross-cluster fusion for WeaveBrainCoordinator."""

    def _causal_hop_indices(self, store: LoomStore, seed_abs_idx: np.ndarray) -> np.ndarray:
        """Propagation hop expanding along causal graph edges."""
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
        """Drops members whose latent position does not cohere with dominant anchor."""
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
        keptset = set(kept)
        return [s for s in members if s in keptset]

    def _mark_superseded(self, shard_id: str, superseded_by: str) -> None:
        """Records consolidation into newer semantic crystal in RAM living state."""
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
        """Appends an eviction record to tombstones.jsonl log for provenance."""
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
        """Answers why forgotten shards faded by matching query against tombstones."""
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
        """Local store indices for superseded shards in this crystal."""
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
        """Spreading activation over causal graph edges (additive in score space)."""
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
        seed_min = tuning_manager.get_float("CAUSAL_SPREAD_SEED_MIN", 0.2)
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
                w = G[sid][nbr].get("weight", 1.0) if G.has_edge(sid, nbr) else G[nbr][sid].get("weight", 1.0)
                offer = src_score * min(float(w), 1.0)
                if offer > inherited[j]:
                    inherited[j] = offer
        if len(inherited) and float(inherited.max()) > 0.0:
            score = np.maximum(score, spread_w * inherited)
        return score.astype(np.float32)

    def _apply_working_set_boost(self, score: np.ndarray, cos: np.ndarray,
                                 abs_idx: np.ndarray, store: LoomStore, boost_w: float) -> np.ndarray:
        """Access-driven working set co-recall boost."""
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
        learn: bool = True,
        target_crystal: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Physical recall pathway: Topology sweep -> Leader jump -> Physics resonance -> Deep Sleep metadata decode."""
        if not self.atlas.crystals:
            return []

        if current_time is None:
            current_time = time.time()

        if learn:
            self.enforce_spatiotemporal_decay(current_time)

        target_path = target_crystal if (target_crystal and os.path.exists(target_crystal)) else self.atlas.route_vector(query_vector)
        if not os.path.exists(target_path):
            return []

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

        full_scan_below = tuning_manager.get_int("RECALL_FULL_SCAN_BELOW", 5000)
        seed_leader_k = tuning_manager.get_int("RECALL_SEED_LEADERS", 5)
        narrowed = False
        abs_idx = None

        leader_matrix = store.get_leader_matrix()
        if n > full_scan_below and len(leader_matrix):
            bucket_ids = store.bucket_ids_all()
            sig_packed = self._momentum_signature(q)
            sig_arr = np.frombuffer(sig_packed, dtype=np.uint8)
            leader_dists = hamming_distances(leader_matrix, sig_arr)
            seed_k = min(seed_leader_k, len(leader_dists))
            if seed_k < len(leader_dists):
                seed_leaders = np.argpartition(leader_dists, seed_k - 1)[:seed_k]
            else:
                seed_leaders = np.arange(len(leader_dists))

            mask = np.isin(bucket_ids, seed_leaders) | (bucket_ids == UNBUCKETED)
            hit_idx = np.nonzero(mask)[0]

            adequacy_min = tuning_manager.get_float("RETRIEVAL_ADEQUACY_MIN", 0.5)
            self._last_adequacy_score = 1.0
            self._last_hop_triggered = False
            always_hop = adequacy_min <= 0.0
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

        sup_idx = self._superseded_indices_for(store, target_path)
        if len(sup_idx):
            abs_idx = abs_idx[~np.isin(abs_idx, sup_idx)]

        m = len(abs_idx)
        k = min(k, m)
        if m == 0 or k <= 0:
            return []

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
            cos = cos[abs_idx]
            cos_warped = cos_warped[abs_idx]
            mass = mass[abs_idx]

        initial_act = np.zeros(m, dtype=np.float32)
        cell_phase = np.zeros(m, dtype=np.float32)
        prev_hits = np.zeros(m, dtype=np.int64)
        active_by_idx: Dict[int, str] = {}
        target_abs = os.path.abspath(target_path)

        if learn:
            abs_to_local = {int(a): pos for pos, a in enumerate(abs_idx)}
            for sid, state in list(self.ram_ledger.items()):
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
                    continue
                initial_act[pos] = state.get("activation", 1.0)
                cell_phase[pos] = state.get("phase_angle", 0.0)
                prev_hits[pos] = state.get("hits", 0)
                active_by_idx[pos] = sid

        gain = self.physics.calculate_excitation_gain_batch(cos, query_phase, cell_phase)
        new_act = np.clip(initial_act + gain, 0.0, 1.0)
        grav = self.physics.calculate_gravitational_attraction_batch(cos_warped, mass, mass_a=1.0)

        rank_act_w = tuning_manager.get_float("RANK_ACTIVATION_WEIGHT", 0.25)
        score = (grav * (1.0 + rank_act_w * initial_act)).astype(np.float32)

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
                    wake_phase = float(cell_phase[i])
                else:
                    latent_position = store.get_vector(abs_i)
                    velocity = np.zeros(self.dimension, dtype=np.float32)
                    wake_phase = seed_phase(latent_position)

                cognitive_fields = {
                    k: prev_entry[k] for k in ("energy", "entropy", "stability", "resonance", "momentum", "decay", "attention")
                    if prev_entry is not None and k in prev_entry
                } if prev_entry is not None else default_cognitive_fields()
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
                    **default_cognitive_fields(),
                    **cognitive_fields
                }
                wake_stamp[sid] = (excited_act, last_recalled)
                final_hits[i] = hits
                final_act[i] = excited_act

            for i, sid in list(active_by_idx.items()):
                if new_act[i] <= sleep_limit and sid in self.ram_ledger:
                    self._tombstone(sid, self.ram_ledger[sid], reason="decay")
                    del self.ram_ledger[sid]

            placed = self._ledger_put_many(wake_entries)
            self._append_journal_entries([(sid,) + wake_stamp[sid] for sid in placed])

            if self.ram_ledger:
                ledger_items = list(self.ram_ledger.values())
                phases = np.array([s["phase_angle"] for s in ledger_items], dtype=np.float64)
                acts = np.array([s["activation"] for s in ledger_items], dtype=np.float64)
                coupling_k = tuning_manager.get_float("KURAMOTO_COUPLING", 0.1)
                s1 = float(np.sum(acts * np.sin(phases)))
                s2 = float(np.sum(acts * np.cos(phases)))
                d_theta = (coupling_k / len(phases)) * (np.cos(phases) * s1 - np.sin(phases) * s2)
                new_phases = (phases + d_theta + np.pi) % (2 * np.pi) - np.pi
                for state, ph in zip(ledger_items, new_phases):
                    state["phase_angle"] = float(ph)

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

        if learn:
            if results:
                self.process_cognitive_tick({r["shard_id"]: r["activation"] for r in results})
            self._last_message_ts = time.time()
        elif results:
            self._recall_traces.append((
                [r["shard_id"] for r in results],
                [float(r["activation"]) for r in results],
                current_time,
            ))
            if len(self._recall_traces) >= tuning_manager.get_int("CONSOLIDATE_EVERY", 64):
                self.consolidate_traces()

        return results

    def recall_multi_pass(self, query_vec: np.ndarray, candidate_pool: int = 16, top_k: int = 8,
                          learn: bool = False, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Multi-Pass Living Memory Distillation + Multi-Turn Working Memory + Cross-Cluster Fusion."""
        effective_query_vec = query_vec
        session = None
        if session_id:
            session = self.sessions.setdefault(session_id, {
                "working_memory": {},
                "attractor_vec": None,
                "turn": 0
            })
            session["turn"] += 1
            prune_sids = []
            for sid, wstate in session["working_memory"].items():
                wstate["activation"] *= 0.7788
                wstate["energy"] *= 0.7788
                if wstate["activation"] < 0.1:
                    prune_sids.append(sid)
            for sid in prune_sids:
                del session["working_memory"][sid]

            is_topic_shift = False
            if session["attractor_vec"] is not None:
                sim = float(np.dot(query_vec, session["attractor_vec"]))
                if sim >= 0.30:
                    blended = 0.60 * query_vec + 0.40 * session["attractor_vec"]
                    norm = np.linalg.norm(blended)
                    if norm > 1e-6:
                        effective_query_vec = (blended / norm).astype(np.float32)
                else:
                    session["attractor_vec"] = query_vec.copy()
                    is_topic_shift = True

        embassy_info = self.atlas.route_vector_with_embassy(effective_query_vec)
        primary_path = embassy_info.get("primary_crystal")
        secondary_path = embassy_info.get("secondary_crystal")
        is_embassy = embassy_info.get("is_embassy", False)

        broad_candidates = []
        if primary_path and os.path.exists(primary_path):
            broad_candidates.extend(self.recall(effective_query_vec, top_k=candidate_pool, learn=learn, target_crystal=primary_path))

        primary_max_score = broad_candidates[0]["score"] if broad_candidates else 0.0

        if (is_embassy or primary_max_score < 5.0) and secondary_path and os.path.exists(secondary_path) and secondary_path != primary_path:
            sec_pool = max(4, candidate_pool // 2)
            sec_cands = self.recall(effective_query_vec, top_k=sec_pool, learn=learn, target_crystal=secondary_path)
            for sc in sec_cands:
                sc["is_cross_cluster"] = True
            broad_candidates.extend(sec_cands)

        if (primary_max_score < 3.5 or len(broad_candidates) < top_k) and len(self.atlas.crystals) > 1:
            for cpath in list(self.atlas.crystals.keys()):
                if cpath not in (primary_path, secondary_path) and os.path.exists(cpath):
                    fallback_cands = self.recall(effective_query_vec, top_k=max(2, candidate_pool // 4), learn=False, target_crystal=cpath)
                    for fc in fallback_cands:
                        fc["is_fallback_deep"] = True
                    broad_candidates.extend(fallback_cands)

        if broad_candidates and self.working_set_pairs:
            cand_sids = {c["shard_id"] for c in broad_candidates}
            cross_sids = set()
            for c in broad_candidates[:4]:
                csid = c["shard_id"]
                for (a, b), cnt in self.working_set_pairs.items():
                    if cnt >= 2:
                        if a == csid and b not in cand_sids:
                            cross_sids.add(b)
                        elif b == csid and a not in cand_sids:
                            cross_sids.add(a)
            for xsid in list(cross_sids)[:4]:
                if xsid in self.ram_ledger:
                    xentry = self.ram_ledger[xsid]
                    broad_candidates.append({
                        "shard_id": xsid,
                        "text": xentry.get("text", ""),
                        "score": float(primary_max_score * 0.90),
                        "activation": float(xentry.get("activation", 0.9)),
                        "energy": float(xentry.get("energy", 1.0)),
                        "phase_angle": float(xentry.get("phase_angle", 0.0)),
                        "is_associative_bridge": True
                    })

        if not broad_candidates:
            return []

        if session and session["working_memory"] and not is_topic_shift:
            cand_sids = {c["shard_id"] for c in broad_candidates}
            lead_ref_score = broad_candidates[0].get("score", 10.0) if broad_candidates else 10.0
            for sid, wstate in session["working_memory"].items():
                if sid not in cand_sids and wstate.get("text"):
                    broad_candidates.append({
                        "shard_id": sid,
                        "text": wstate["text"],
                        "score": float(lead_ref_score * min(1.0, wstate.get("activation", 0.8))),
                        "activation": wstate.get("activation", 0.8),
                        "energy": wstate.get("energy", 0.8)
                    })

        augmented = []
        for cand in broad_candidates:
            sid = cand["shard_id"]
            state = self.ram_ledger.get(sid, {})
            session_boost = 0.0
            if session and sid in session["working_memory"] and not is_topic_shift:
                session_boost = session["working_memory"][sid].get("activation", 0.0)

            cand_act = float(state.get("activation", cand.get("activation", 1.0))) + session_boost
            cand_energy = float(state.get("energy", 1.0))
            cand_phase = float(state.get("phase_angle", 0.0))
            cand_res = float(state.get("resonance", 1.0))
            cand_mom = float(state.get("momentum", 0.0))
            cand_stab = float(state.get("stability", 1.0))
            augmented.append({
                **cand,
                "activation": cand_act,
                "energy": cand_energy,
                "phase_angle": cand_phase,
                "resonance": cand_res,
                "momentum": cand_mom,
                "stability": cand_stab,
            })

        lead_phase = augmented[0]["phase_angle"]
        for cand in augmented:
            phase_diff = cand["phase_angle"] - lead_phase
            coherence = max(0.0, float(np.cos(phase_diff)))
            wm_mult = 1.0
            if session and not is_topic_shift and cand["shard_id"] in session.get("working_memory", {}):
                wm_state = session["working_memory"][cand["shard_id"]]
                turn_age = session["turn"] - wm_state.get("turn", session["turn"])
                if turn_age <= 1:
                    wm_mult = 1.6

            cand["salience_score"] = float(
                cand["score"] * (0.4 + 0.3 * min(2.0, cand["activation"]) + 0.3 * coherence) * wm_mult
            )

        augmented.sort(key=lambda c: c["salience_score"], reverse=True)
        distilled = augmented[:top_k]

        if session and distilled:
            for cand in distilled:
                csid = cand["shard_id"]
                session["working_memory"][csid] = {
                    "shard_id": csid,
                    "text": cand.get("text", ""),
                    "activation": min(1.0, cand.get("activation", 1.0)),
                    "energy": cand.get("energy", 1.0) + 1.0,
                    "turn": session["turn"]
                }
            if session["attractor_vec"] is None:
                session["attractor_vec"] = query_vec.copy()
            else:
                session["attractor_vec"] = 0.70 * session["attractor_vec"] + 0.30 * query_vec

        return distilled
