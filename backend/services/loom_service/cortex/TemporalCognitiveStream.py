import sys
import os
import time
import math
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any, Tuple

# Ensure project root is in sys.path to allow backend imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.config.envConfig import setup_logger, log_service
from backend.services.loom_service.cortex.ResonanceField import ResonanceFieldEngine

logger = setup_logger("TemporalCognitiveStream")


class TemporalCognitiveStream:
    """
    ============================================================================
    DOCLOOM — TEMPORAL COGNITIVE STREAM
    ============================================================================
    Tracks continuous cognitive stream activity, managing attention momentum, 
    attractor persistence, and temporal motif extraction.
    ============================================================================
    """

    EPSILON = 1e-8

    def __init__(
        self,
        field_engine: ResonanceFieldEngine,
        history_size: int = 1000,
        stream_decay: float = 0.90,
        momentum_decay: float = 0.85,
        attractor_decay: float = 0.95,
        attractor_threshold: float = 0.35,
        entropy_ceiling: float = 0.85,
        energy_ceiling: float = 10.0,
    ):
        self.field_engine = field_engine
        self.stream_decay = stream_decay
        self.momentum_decay = momentum_decay
        self.attractor_decay = attractor_decay
        self.attractor_threshold = attractor_threshold
        self.entropy_ceiling = entropy_ceiling
        self.energy_ceiling = energy_ceiling

        self.current_assembly: List[Dict[str, Any]] = []
        self.cognitive_history: deque = deque(maxlen=history_size)
        self.attractor_memory: Dict[str, Dict[str, float]] = {}
        self.attention_bias: Dict[str, float] = {}

        self._cycle_count: int = 0
        self._consolidation_interval: int = 10

        self.fatigue: Dict[str, float] = {}
        self._fatigue_decay: float = 0.92
        self._fatigue_rate: float = 0.08

        self.transition_memory: Dict[tuple, int] = {}
        self.transition_momentum_strength: float = 0.12

        log_service(
            logger,
            f"TemporalCognitiveStream initialized. Stream_λ={stream_decay}",
            "info",
        )

    def inject_query(self, shard_id: str, energy: float = 1.0) -> None:
        self.field_engine.excite(shard_id, energy)
        bias = self.attention_bias.get(shard_id, 0.0)
        if bias > self.EPSILON:
            bonus = bias * 0.15
            self.field_engine.excite(shard_id, bonus)
        log_service(logger, f"Injected query '{shard_id}' with energy={energy:.4f}", "debug")

    def cognitive_cycle(self, top_k: int = 10) -> Dict[str, Any]:
        self._cycle_count += 1
        self.field_engine.tick()
        self.field_engine.apply_global_decay(self.stream_decay)
        self._update_fatigue()
        self._transition_momentum_step()

        snapshot = self.capture_assembly(top_k=top_k)
        self.update_attention_bias(snapshot["assembly"])
        self.update_attractors(snapshot["assembly"])

        self.cognitive_history.append(snapshot)
        self.current_assembly = snapshot["assembly"]
        self.stabilize_stream()

        if self._cycle_count % self._consolidation_interval == 0:
            self.deep_consolidation_cycle()

        log_service(
            logger,
            f"Cycle #{self._cycle_count}: peak='{snapshot['peak_shard']}' H={snapshot['entropy']:.3f}",
            "debug",
        )
        return snapshot

    def capture_assembly(self, top_k: int = 10) -> Dict[str, Any]:
        assembly = self.field_engine.get_active_assembly(top_k=top_k)
        entropy = self.compute_entropy()
        coherence = self.compute_assembly_coherence(assembly)
        phase_r = self.compute_phase_coherence(assembly)
        total_e = sum(self.field_engine.activation_state.values())
        peak_shard = assembly[0]["shard_id"] if assembly else None

        return {
            "timestamp": time.time(),
            "assembly": assembly,
            "peak_shard": peak_shard,
            "entropy": entropy,
            "coherence": coherence,
            "phase_r": phase_r,
            "field_energy": total_e,
            "cycle": self._cycle_count,
        }

    def compute_entropy(self) -> float:
        activations = [
            v for v in self.field_engine.activation_state.values()
            if v > self.EPSILON
        ]
        N = len(activations)
        if N < 2:
            return 0.0

        total = sum(activations)
        if total < self.EPSILON:
            return 0.0

        raw_entropy = 0.0
        for a in activations:
            p = a / total
            raw_entropy -= p * math.log(p + self.EPSILON)

        return float(raw_entropy / math.log(N))

    def update_attention_bias(self, assembly: List[Dict[str, Any]]) -> None:
        for sid in list(self.attention_bias):
            self.attention_bias[sid] *= self.momentum_decay
            if abs(self.attention_bias[sid]) < self.EPSILON:
                del self.attention_bias[sid]

        for node in assembly:
            sid = node["shard_id"]
            act = node["activation"]
            prev = self.attention_bias.get(sid, 0.0)
            self.attention_bias[sid] = math.tanh(prev + act)

    def update_attractors(self, assembly: List[Dict[str, Any]]) -> None:
        EMA_ALPHA = 0.15
        active_ids = {node["shard_id"]: node["activation"] for node in assembly}

        prev_peak = self.current_assembly[0]["shard_id"] if self.current_assembly else None
        curr_peak = assembly[0]["shard_id"] if assembly else None
        if prev_peak and curr_peak and prev_peak != curr_peak:
            key = (prev_peak, curr_peak)
            self.transition_memory[key] = self.transition_memory.get(key, 0) + 1

        for sid, act in active_ids.items():
            fatigue = self.fatigue.get(sid, 0.0)
            effective_act = act * max(0.0, 1.0 - fatigue)

            if sid not in self.attractor_memory:
                self.attractor_memory[sid] = {
                    "persistence": 0.0,
                    "last_seen": time.time(),
                    "peak_activation": 0.0,
                }
            mem = self.attractor_memory[sid]
            mem["persistence"] += EMA_ALPHA * (effective_act - mem["persistence"])
            mem["last_seen"] = time.time()
            mem["peak_activation"] = max(mem["peak_activation"], effective_act)

        for sid in list(self.attractor_memory):
            if sid not in active_ids:
                self.attractor_memory[sid]["persistence"] *= self.attractor_decay
                if self.attractor_memory[sid]["persistence"] < self.EPSILON:
                    del self.attractor_memory[sid]

    def _update_fatigue(self) -> None:
        active_ids = {node["shard_id"]: node["activation"] for node in self.current_assembly}
        for sid, act in active_ids.items():
            self.fatigue[sid] = min(0.9, self.fatigue.get(sid, 0.0) + self._fatigue_rate * act)

        for sid in list(self.fatigue):
            if sid not in active_ids:
                self.fatigue[sid] *= self._fatigue_decay
                if self.fatigue[sid] < self.EPSILON:
                    del self.fatigue[sid]

    def get_top_transitions(self, top_k: int = 10) -> List[Dict[str, Any]]:
        sorted_t = sorted(self.transition_memory.items(), key=lambda x: x[1], reverse=True)
        return [{"from": k[0], "to": k[1], "count": v} for k, v in sorted_t[:top_k]]

    ATTRACTOR_TIERS = [
        (0.95, "fixation"),
        (0.80, "dominant"),
        (0.50, "stable"),
        (0.20, "transient"),
    ]

    def classify_attractor(self, persistence: float) -> str:
        for threshold, label in self.ATTRACTOR_TIERS:
            if persistence >= threshold:
                return label
        return "noise"

    def detect_stable_attractors(self, top_k: int = 5, min_tier: str = "transient") -> List[Dict[str, Any]]:
        tier_order = {"fixation": 4, "dominant": 3, "stable": 2, "transient": 1, "noise": 0}
        min_rank = tier_order.get(min_tier, 1)

        attractors = []
        for sid, mem in self.attractor_memory.items():
            p = mem["persistence"]
            tier = self.classify_attractor(p)
            rank = tier_order.get(tier, 0)
            if rank >= min_rank:
                attractors.append({
                    "shard_id": sid,
                    "persistence": p,
                    "tier": tier,
                    "peak_activation": mem["peak_activation"],
                    "last_seen": mem["last_seen"],
                })

        attractors.sort(key=lambda x: x["persistence"], reverse=True)
        return attractors[:top_k]

    def episodic_path(self, window: int = 20) -> List[Dict[str, Any]]:
        recent = list(self.cognitive_history)[-window:]
        return [
            {
                "cycle": snap["cycle"],
                "peak_shard": snap["peak_shard"],
                "entropy": snap["entropy"],
                "energy": snap["field_energy"],
                "timestamp": snap["timestamp"],
            }
            for snap in recent
            if snap["peak_shard"] is not None
        ]

    def stabilize_stream(self) -> None:
        entropy = self.compute_entropy()

        if entropy > self.entropy_ceiling:
            cool_factor = self.entropy_ceiling / (entropy + self.EPSILON)
            self.field_engine.apply_global_scaling(cool_factor)

        total_e = sum(v for v in self.field_engine.activation_state.values() if v > 0)
        if total_e > self.energy_ceiling:
            smooth_scale = 1.0 / (1.0 + total_e / self.energy_ceiling)
            self.field_engine.apply_global_scaling(smooth_scale)

        active_vals = [v for v in self.field_engine.activation_state.values() if v > self.EPSILON]
        if active_vals:
            adaptive_floor = (sum(active_vals) / len(active_vals)) * self.EPSILON * 10
            for sid in self.field_engine.activation_state:
                if 0 < self.field_engine.activation_state[sid] < adaptive_floor:
                    self.field_engine.activation_state[sid] = 0.0
                    self.field_engine.field_nodes[sid]["state"]["activation"]["amplitude"] = 0.0

        for sid in list(self.attention_bias):
            if abs(self.attention_bias[sid]) < self.EPSILON:
                del self.attention_bias[sid]

        bias_pressure = 0.0
        top_bias = sorted(self.attention_bias.values(), reverse=True)[:5]
        if top_bias:
            bias_pressure = sum(top_bias) / len(top_bias)

        mem_size = len(self.field_engine.resonance_memory)
        mem_pressure = min(1.0, mem_size / max(1, len(self.field_engine.field_nodes) * 4))

        momentum_pressure = 0.0
        if self.transition_memory:
            max_t = max(self.transition_memory.values())
            momentum_pressure = min(1.0, max_t / 20.0)

        combined = (bias_pressure + mem_pressure + momentum_pressure) / 3.0
        TRIPLE_PRESSURE_THRESHOLD = 0.65
        if combined > TRIPLE_PRESSURE_THRESHOLD:
            brake = 1.0 - 0.15 * (combined - TRIPLE_PRESSURE_THRESHOLD)
            self.field_engine.apply_global_scaling(max(0.70, brake))
            for sid in self.attention_bias:
                self.attention_bias[sid] *= 0.92

    def stream_state(self) -> Dict[str, Any]:
        entropy = self.compute_entropy()
        total_e = sum(self.field_engine.activation_state.values())
        attractors = self.detect_stable_attractors(top_k=5)
        path = self.episodic_path(window=10)

        return {
            "cycle": self._cycle_count,
            "entropy": entropy,
            "field_energy": total_e,
            "active_nodes": sum(1 for v in self.field_engine.activation_state.values() if v > self.EPSILON),
            "stable_attractors": attractors,
            "episodic_path": [p["peak_shard"] for p in path],
            "attention_peaks": sorted(self.attention_bias.items(), key=lambda x: x[1], reverse=True)[:5],
            "resonance_memory_size": len(self.field_engine.resonance_memory),
        }

    def run(
        self,
        cycles: int,
        inject: Optional[Dict[str, float]] = None,
        top_k: int = 10,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        if inject:
            for sid, energy in inject.items():
                self.inject_query(sid, energy)

        snapshots = []
        for _ in range(cycles):
            snap = self.cognitive_cycle(top_k=top_k)
            snapshots.append(snap)
            if verbose:
                print(f"  Cycle #{snap['cycle']:4d}  peak={str(snap['peak_shard']):<20}  H={snap['entropy']:.3f}  E={snap['field_energy']:.3f}")
        return snapshots

    def compute_assembly_coherence(self, assembly: List[Dict[str, Any]]) -> float:
        ids = [node["shard_id"] for node in assembly]
        if len(ids) < 2:
            return 1.0
        total, count = 0.0, 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                total += abs(self.field_engine.resonance(ids[i], ids[j]))
                count += 1
        return float(total / count) if count > 0 else 0.0

    def compute_phase_coherence(self, assembly: List[Dict[str, Any]]) -> float:
        ids = [node["shard_id"] for node in assembly]
        N = len(ids)
        if N == 0:
            return 0.0
        real_sum, imag_sum = 0.0, 0.0
        for sid in ids:
            phi = self.field_engine.field_nodes[sid]["state"]["activation"]["phase"]
            real_sum += math.cos(phi)
            imag_sum += math.sin(phi)
        return float(math.sqrt(real_sum ** 2 + imag_sum ** 2) / N)

    def deep_consolidation_cycle(self) -> None:
        self.field_engine.consolidate_field_memory()
        PRUNE = 1e-4
        for sid in list(self.attractor_memory):
            if self.attractor_memory[sid]["persistence"] < PRUNE:
                del self.attractor_memory[sid]
        for sid in list(self.attention_bias):
            if abs(self.attention_bias[sid]) < self.EPSILON:
                del self.attention_bias[sid]

        motifs = self.extract_temporal_motifs(min_length=2, max_length=3, min_occurrences=3)
        for motif_info in motifs[:3]:
            peak_of_motif = motif_info["motif"][0]
            mem = self.attractor_memory.get(peak_of_motif, {})
            persistence = mem.get("persistence", 0.0)
            if persistence >= 0.50:
                log_service(
                    logger,
                    f"Meta-shard candidate: {motif_info['motif']} (count={motif_info['count']}) [ELIGIBLE]",
                    "info"
                )
            else:
                log_service(
                    logger,
                    f"Meta-shard blocked: {motif_info['motif']} persistence={persistence:.3f} < 0.50 [NOT STABLE]",
                    "debug"
                )

        log_service(logger, f"Deep consolidation #{self._cycle_count}: attractors={len(self.attractor_memory)}", "debug")

    def trajectory_divergence(self, window: int = 10) -> float:
        path = self.episodic_path(window=window)
        if len(path) < 2:
            return 0.0
        changes = sum(1 for i in range(1, len(path)) if path[i]["peak_shard"] != path[i - 1]["peak_shard"])
        return float(changes / (len(path) - 1))

    def trajectory_recurrence(self, window: int = 20) -> Dict[str, int]:
        path = self.episodic_path(window=window)
        counts: Dict[str, int] = {}
        for p in path:
            sid = p["peak_shard"]
            if sid:
                counts[sid] = counts.get(sid, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def trajectory_stability(self, window: int = 10) -> float:
        path = self.episodic_path(window=window)
        if not path:
            return 0.0
        return float(sum(p["entropy"] for p in path) / len(path))

    def reset(self, hard: bool = False) -> None:
        self.cognitive_history.clear()
        self.attractor_memory.clear()
        self.attention_bias.clear()
        self.fatigue.clear()
        self.transition_memory.clear()
        self.current_assembly = []
        self._cycle_count = 0
        if hard:
            self.field_engine.reset_activations()
            log_service(logger, "TemporalCognitiveStream HARD reset.", "info")
        else:
            log_service(logger, "TemporalCognitiveStream soft reset.", "info")

    def _transition_momentum_step(self) -> None:
        if not self.current_assembly or not self.transition_memory:
            return

        peak_sid = self.current_assembly[0]["shard_id"]
        max_count = max(self.transition_memory.values()) if self.transition_memory else 1

        for (src, dst), count in self.transition_memory.items():
            if src != peak_sid:
                continue
            if dst not in self.field_engine.field_nodes:
                continue

            dst_fatigue = self.fatigue.get(dst, 0.0)
            fatigue_gate = math.exp(-dst_fatigue * 3.0)

            weight = (count / max_count) * self.transition_momentum_strength * fatigue_gate
            if weight > self.EPSILON:
                self.field_engine.excite(dst, weight)

    def extract_temporal_motifs(
        self, min_length: int = 2, max_length: int = 4, min_occurrences: int = 2
    ) -> List[Dict[str, Any]]:
        history_list = list(self.cognitive_history)
        peak_sequence = [snap["peak_shard"] for snap in history_list if snap["peak_shard"] is not None]

        if len(peak_sequence) < min_length:
            return []

        motif_counts: Dict[tuple, int] = {}
        for n in range(min_length, max_length + 1):
            for i in range(len(peak_sequence) - n + 1):
                gram = tuple(peak_sequence[i:i + n])
                if len(set(gram)) == 1:
                    continue
                motif_counts[gram] = motif_counts.get(gram, 0) + 1

        total = len(peak_sequence)
        motifs = [
            {
                "motif": gram,
                "count": cnt,
                "density": round(cnt / max(1, total - len(gram) + 1), 3),
                "length": len(gram),
            }
            for gram, cnt in motif_counts.items()
            if cnt >= min_occurrences
        ]
        motifs.sort(key=lambda x: (x["count"], x["length"]), reverse=True)
        return motifs


if __name__ == "__main__":
    from backend.services.loom_service.cortex.HyperVectorCreation import HyperVectorEngine

    print("=" * 65)
    print("STAGE 6 — TemporalCognitiveStream Validation")
    print("=" * 65)

    DIM = 1000
    hdc = HyperVectorEngine(dimension=DIM)
    rfe = ResonanceFieldEngine(decay_rate=0.95, inhibition_strength=0.1, propagation_depth=5)

    concepts = {
        "god": ["god", "creation", "divine"],
        "creation": ["creation", "universe", "origin"],
        "universe": ["universe", "cosmos", "space"],
        "entropy": ["entropy", "disorder", "chaos"],
        "thermodynamics": ["thermodynamics", "energy", "heat"],
        "earth": ["earth", "ground", "world"],
        "life": ["life", "living", "biology"],
    }

    print("\nRegistering shards into field...")
    for concept_id, seeds in concepts.items():
        basis_vecs = [hdc.get_basis_vector(s).astype(np.float32) for s in seeds]
        field_vec, _bv = hdc.bundle(basis_vecs)
        rfe.register_shard(f"s_{concept_id}", field_vec, meta={"label": concept_id})

    stream = TemporalCognitiveStream(
        field_engine=rfe,
        history_size=100,
        stream_decay=0.90,
        entropy_ceiling=0.85,
    )

    print("\n--- Injecting 'god' and running 30 cognitive cycles ---")
    snapshots = stream.run(
        cycles=30,
        inject={"s_god": 1.0},
        top_k=5,
        verbose=True,
    )
