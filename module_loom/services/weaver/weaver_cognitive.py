"""
DocLoom Weave Cognitive Mixin
Handles spatiotemporal biological decay, merged cognitive tick dynamics,
serving-mode trace consolidation, and sleep-cycle semantic crystal consolidation.
"""
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from module_loom.config.tuning_config import tuning_manager
from module_loom.services.cortex.latent_field_cognition.cognitive_field_substrate import CognitiveMetrics
from module_loom.services.cortex.latent_field_cognition.attractor_basin_compilation import CognitiveAssembly


class WeaveCognitiveMixin:
    """Cognitive physics, spatiotemporal decay, and REM sleep consolidation for WeaveBrainCoordinator."""

    def enforce_spatiotemporal_decay(self, current_time: float) -> None:
        """Evaluates memory states against continuous decay curves via vectorized batch."""
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
        """Evolves shared latent field, assemblies, and per-shard cognitive physics over a bounded hot set."""
        if not resonance_input:
            return None

        working_set_max = tuning_manager.get_int("COGNITION_WORKING_SET_MAX", 128)

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

        input_tensor = np.zeros(dim, dtype=np.float32)
        for sid, intensity in resonance_input.items():
            entry = self.ram_ledger.get(sid)
            if entry is not None:
                input_tensor += np.asarray(entry["latent_position"], dtype=np.float32) * intensity
        inorm = np.linalg.norm(input_tensor)
        if inorm > 0:
            input_tensor /= inorm

        prev_latent = self.latent_field.copy()

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

        dt = 0.1
        repulsion_radius = tuning_manager.get_float("REPULSION_RADIUS", 0.2)
        repulsion_constant = tuning_manager.get_float("REPULSION_CONSTANT", 0.02)
        drag_friction = tuning_manager.get_float("DRAG_FRICTION", 0.7)

        positions = {sid: np.asarray(self.ram_ledger[sid]["latent_position"], dtype=np.float32) for sid in hot_ids}
        active_vecs = list(positions.values())

        pos_matrix = np.stack(active_vecs)
        diffs = pos_matrix[:, None, :] - pos_matrix[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        within_radius = dists < repulsion_radius
        np.fill_diagonal(within_radius, False)
        safe_dists = np.where(within_radius, dists + 1e-5, 1.0)[:, :, None]
        contrib = np.where(within_radius[:, :, None], diffs / safe_dists, 0.0)
        repulsion_matrix = contrib.sum(axis=1) * repulsion_constant

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

            prev_attention = float(entry.get("attention", 0.0))
            intensity = resonance_input.get(sid, 0.0)
            entry["attention"] = min(1.0, prev_attention + intensity * 0.5) if intensity > 0 else prev_attention * 0.85

        self.cognitive_coherence = CognitiveMetrics.compute_coherence(active_vecs)
        self.cognitive_entropy = CognitiveMetrics.compute_entropy(self.latent_field, prev_latent)
        coherence_gain = max(0.0, self.cognitive_coherence - self._last_cognitive_coherence)
        self._last_cognitive_coherence = self.cognitive_coherence
        cognitive_load = self.cognitive_entropy * 0.02
        recovery = coherence_gain * 0.15 + (self.cognitive_coherence * 0.01)
        self.cognitive_energy_budget = float(np.clip(self.cognitive_energy_budget + recovery - cognitive_load, 0.0, 1.0))

        projected = {}
        for sid in hot_ids:
            sim = float(np.dot(self.latent_field, self.ram_ledger[sid]["latent_position"]))
            if sim > 0.4:
                projected[sid] = sim

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
                    self.causal.record_transition(prev_dominant, ca.dominant_concept)
            if len(self.active_assemblies) > 200:
                self.active_assemblies = self.active_assemblies[-200:]

        self.cognitive_tick += 1
        return ca

    def consolidate_traces(self, max_edge_fanout: int = 5) -> int:
        """Applies batched memory updates and Hebbian co-occurrence edges from serving-mode recalls."""
        traces = self._recall_traces
        if not traces:
            return 0
        self._recall_traces = []
        now = time.time()
        boost = tuning_manager.get_float("RECALL_BOOST_VALUE", 0.15)

        touched: Dict[str, float] = {}
        for sids, acts, _t in traces:
            for sid, a in zip(sids, acts):
                touched[sid] = max(touched.get(sid, 0.0), float(a))
                entry = self.ram_ledger.get(sid)
                if entry is not None:
                    entry["activation"] = min(1.0, float(entry.get("activation", 0.0)) + boost)
                    entry["hits"] = int(entry.get("hits", 0)) + 1
                    entry["last_recalled"] = now

        pairs: List[Tuple[str, str]] = []
        for sids, acts, _t in traces:
            top = sids[:max_edge_fanout]
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    pairs.append((top[i], top[j]))
        if pairs:
            self.causal.record_transitions_batch(pairs, weight=0.6, prediction_error=0.2)
            for a, b in pairs:
                key = (a, b) if a <= b else (b, a)
                self.working_set_pairs[key] = self.working_set_pairs.get(key, 0) + 1

        if touched:
            self.process_cognitive_tick(touched)
        self._last_message_ts = now
        return len(traces)

    def _synthesize_semantic_text(self, concept: str, texts: List[str]) -> Tuple[str, str]:
        """REM-stage synthesis using LLM service with fallback to deterministic template."""
        seen: List[str] = []
        for t in texts:
            t = (t or "").strip()
            if t and t not in seen:
                seen.append(t)

        if seen and tuning_manager.get_int("SLEEP_LLM_SYNTHESIS", 1):
            try:
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
                pass

        bullets = "\n".join(f"- {t}" for t in seen[:8])
        return (
            f"Consolidated theme: {concept}\nRecurring observations across {len(seen)} memories:\n{bullets}",
            "template",
        )

    def _get_faithfulness_judge(self):
        """Lazily creates FaithfulnessJudge for gating sleep-cycle consolidation."""
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
        """Consolidates recurring assemblies into permanent semantic crystal shards."""
        if not tuning_manager.get_int("SLEEP_CYCLE_ENABLED", 1):
            return None

        now = time.time()
        idle_for = now - self._last_message_ts
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

            crystal_coherence_min = tuning_manager.get_float("SLEEP_CRYSTAL_COHERENCE_MIN", 0.5)
            if crystal_coherence_min > 0.0:
                member_sids = self._coherence_filter(member_sids, crystal_coherence_min)
            if len(member_sids) < 2:
                continue
            member_sids.sort(key=lambda s: self.ram_ledger[s].get("activation", 0.0), reverse=True)
            member_sids = member_sids[:max_source]

            for i in range(len(member_sids) - 1):
                self.causal.record_transition(member_sids[i], member_sids[i + 1], weight=1.0, prediction_error=0.1)
            for sid in member_sids:
                entry = self.ram_ledger[sid]
                entry["activation"] = min(1.0, float(entry.get("activation", 0.0)) + 0.1)

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
                                faithfulness_score = 1.0
            consolidated_faithfulness[concept] = faithfulness_score

            new_sid = f"crystal_{hashlib.md5((concept + '|' + '|'.join(sorted(member_sids))).encode('utf-8')).hexdigest()[:12]}"
            mass = min(5.0, 1.5 + 0.1 * len(member_sids))
            self.ingest_shard(new_sid, new_vec, synthesis_text, mass=mass, metadata={
                "semantic_crystal": True,
                "source_shard_ids": member_sids,
                "consolidated_concept": concept,
                "synthesis_mode": synthesis_mode,
            })

            for sid in member_sids:
                self._mark_superseded(sid, new_sid)

            consolidated.append({
                "concept": concept, "new_shard": new_sid,
                "source_shard_ids": member_sids, "synthesis_mode": synthesis_mode,
                "faithfulness": consolidated_faithfulness.get(concept)
            })
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
