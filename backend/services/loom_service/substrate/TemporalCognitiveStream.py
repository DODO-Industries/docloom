import sys
import os
import time
import math
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any, Tuple

def setup_python_path():
    """Dynamically finds the project root and adds it to sys.path."""
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr != os.path.dirname(curr):
        if os.path.isdir(os.path.join(curr, "backend")):
            if curr not in sys.path:
                sys.path.insert(0, curr)
            return
        curr = os.path.dirname(curr)

if __name__ == "__main__":
    setup_python_path()

from backend.config.envConfig import setup_logger, log_service
from backend.services.loom_service.substrate.ResonanceFieldEngine import ResonanceFieldEngine

logger = setup_logger("TemporalCognitiveStream")


class TemporalCognitiveStream:
    """
    ============================================================================
    DOCLOOM — TEMPORAL COGNITIVE STREAM  (Stage 6 — Full Hardening)
    ============================================================================

    Version: 1.1  —  Full Cognitive Physics

    This is NOT a memory manager, cache, or conversation history.
    This is the temporal dynamical consciousness layer.

    It transforms:
        static resonance -> continuous cognitive flow

    Wraps ResonanceFieldEngine (the cortex) with:
    - Episodic trace capture (cognitive history deque)
    - Multi-tier attractor classification (transient/stable/dominant/fixation)
    - Inhibitory fatigue (prevents attractor lock-in)
    - Temporal attention momentum (working memory / recency)
    - Semantic transition momentum (A->B momentum biases future activation)
    - Field entropy measurement (focus vs diffusion, normalized [0,1])
    - Kuramoto phase coherence (real oscillatory synchronization via RFE)
    - Stream stabilization (prevents semantic seizure)
    - Episodic path reconstruction + temporal motif extraction

    Biological Analogy:
        Brain Region          | DocLoom Equivalent
        ----------------------|-----------------------------
        Cortex                | ResonanceFieldEngine
        Working Memory        | TemporalCognitiveStream
        Oscillatory Synchrony | Kuramoto Phase Coupling
        Attention Persistence | Multi-Tier Attractor Memory
        Inhibitory Interneuron| Fatigue Dynamics
        Associative Memory    | Transition Momentum
        Episodic Memory       | Cognitive History + Motifs
        Concept Formation     | Meta-Shards (Stage 8)

    Attractor Tier Table:
        Persistence | Cognitive Meaning
        ------------|-------------------
        > 0.20      | transient concept
        > 0.50      | stable attractor
        > 0.80      | dominant belief
        > 0.95      | fixation / obsession

    Core Cognitive Equations:
        H_norm   = (-Sigma p_i log p_i) / log(N)   (normalized field entropy)
        M_t      = tanh(lambda*M_(t-1) + A_t)       (bounded attention momentum)
        P_i(t+1) = P_i(t) + alpha*(A_i - P_i(t))   (EMA attractor persistence)
        phi_i(t) = phi_i + omega + K*mean(sin(phi_j-phi_i))  (Kuramoto phase)
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
        attractor_threshold: float = 0.35,   # FIX #1: EMA range ~[0,1], not [0,inf)
        entropy_ceiling: float = 0.85,        # FIX #2: normalized H in [0,1], 0.85 = chaotic
        energy_ceiling: float = 10.0,
    ):
        """
        Parameters
        ----------
        field_engine        : The living ResonanceFieldEngine (cortex).
        history_size        : Max cognitive snapshots to retain (episodic memory).
        stream_decay        : λ for stream-level energy fade per cycle.
        momentum_decay      : λ for attention momentum fade (recency weighting).
        attractor_decay     : λ for attractor persistence decay when shard is inactive.
        attractor_threshold : Min persistence score to report as stable attractor.
        entropy_ceiling     : Max entropy before stabilization kicks in.
        energy_ceiling      : Max total field energy before normalization.
        """
        self.field_engine         = field_engine
        self.stream_decay         = stream_decay
        self.momentum_decay       = momentum_decay
        self.attractor_decay      = attractor_decay
        self.attractor_threshold  = attractor_threshold
        self.entropy_ceiling      = entropy_ceiling
        self.energy_ceiling       = energy_ceiling

        # ── Current Active Assembly ─────────────────────────────────────
        # Represents: current thought, dominant attractor, semantic focus
        self.current_assembly: List[Dict[str, Any]] = []

        # ── Episodic History ────────────────────────────────────────────
        # Bounded deque: O(1) append, automatic overflow eviction
        self.cognitive_history: deque = deque(maxlen=history_size)

        # ── Attractor Memory ────────────────────────────────────────────
        # Tracks which shards survive across time → stable concepts
        # { shard_id: { "persistence", "last_seen", "peak_activation" } }
        self.attractor_memory: Dict[str, Dict[str, float]] = {}

        # ── Attention Momentum ──────────────────────────────────────────
        # Exponentially weighted recent activation → working memory / inertia
        # { shard_id: float }
        self.attention_bias: Dict[str, float] = {}

        # ── Cycle Counter ───────────────────────────────────────────────
        self._cycle_count:            int  = 0
        self._consolidation_interval: int  = 10

        # ── Inhibitory Fatigue (#4) ─────────────────────────────────────
        # Prevents infinite attractor lock-in (god god god god...)
        # fatigue[sid] grows with repeated activation, decays slowly
        self.fatigue: Dict[str, float] = {}
        self._fatigue_decay: float     = 0.92   # slow decay
        self._fatigue_rate:  float     = 0.08   # how fast fatigue accumulates

        # ── Transition Memory (#5) ──────────────────────────────────────
        # Records how often cognition transitions A->B
        # Becomes causal flow / reasoning bridge
        self.transition_memory: Dict[tuple, int] = {}

        # ── Transition Momentum strength ─────────────────────────────────
        # How strongly past A->B transitions bias future activation
        # 0 = disabled, 0.1 = soft nudge, 0.3 = strong momentum
        self.transition_momentum_strength: float = 0.12

        log_service(
            logger,
            f"TemporalCognitiveStream v1.0 initialized. "
            f"history={history_size}, stream_λ={stream_decay}, "
            f"momentum_λ={momentum_decay}",
            "info",
        )

    # =================================================================
    # 1.  INJECT QUERY — inject_query()
    # =================================================================

    def inject_query(self, shard_id: str, energy: float = 1.0) -> None:
        """
        Inject excitation into the cognitive stream.

        CRITICAL: Does NOT reset the field.
        Cognition continues — this is a stimulus INTO an ongoing field,
        not a restart of a static query system.

        Parameters
        ----------
        shard_id : the shard to excite
        energy   : activation energy to inject (default 1.0)
        """
        self.field_engine.excite(shard_id, energy)

        # Apply attention bias: already-attended shards get a small bonus
        bias = self.attention_bias.get(shard_id, 0.0)
        if bias > self.EPSILON:
            bonus = bias * 0.15    # semantics first, attention second
            self.field_engine.excite(shard_id, bonus)

        log_service(logger, f"Injected query '{shard_id}' with energy={energy:.4f}", "debug")

    # =================================================================
    # 2.  COGNITIVE CYCLE — cognitive_cycle()
    # =================================================================

    def cognitive_cycle(self, top_k: int = 10) -> Dict[str, Any]:
        """
        The heartbeat of continuous cognition.

            1. field.tick()              — cortical physics evolve
            2. stream_decay applied      — P0: macro temporal fading
            3. capture_assembly()        — snapshot current thought
            4. update_attention_bias()   — bounded momentum update
            5. update_attractors()       — bounded persistence tracking
            6. store_episode()           — episodic memory
            7. stabilize_stream()        — prevent runaway
            8. deep_consolidation (N)    — P2: periodic deep cycle
        """
        self._cycle_count += 1

        # 1. Physics tick
        self.field_engine.tick()

        # 2. FIX #3: Stream macro decay via RFE method (no direct mutation)
        self.field_engine.apply_global_decay(self.stream_decay)

        # 2b. FIX #4: Inhibitory fatigue BEFORE snapshot
        self._update_fatigue()

        # 2c. Semantic transition momentum — bias field toward likely next concept
        self._transition_momentum_step()

        # 3. Snapshot
        snapshot = self.capture_assembly(top_k=top_k)

        # 4. Bounded attention momentum
        self.update_attention_bias(snapshot["assembly"])

        # 5. Bounded attractor persistence
        self.update_attractors(snapshot["assembly"])

        # 6. Store episode
        self.cognitive_history.append(snapshot)
        self.current_assembly = snapshot["assembly"]

        # 7. Stabilize
        self.stabilize_stream()

        # 8. P2: Periodic deep consolidation
        if self._cycle_count % self._consolidation_interval == 0:
            self.deep_consolidation_cycle()

        log_service(
            logger,
            f"Cycle #{self._cycle_count}: peak='{snapshot['peak_shard']}' "
            f"H={snapshot['entropy']:.3f} E={snapshot['field_energy']:.3f} "
            f"coherence={snapshot.get('coherence', 0.0):.3f}",
            "debug",
        )
        return snapshot

    # =================================================================
    # 3.  CAPTURE ASSEMBLY — capture_assembly()
    # =================================================================

    def capture_assembly(self, top_k: int = 10) -> Dict[str, Any]:
        """
        Capture the current emergent cognitive state.

        Returns a snapshot with entropy, coherence, and phase coherence.
        """
        assembly    = self.field_engine.get_active_assembly(top_k=top_k)
        entropy     = self.compute_entropy()
        coherence   = self.compute_assembly_coherence(assembly)
        phase_r     = self.compute_phase_coherence(assembly)
        total_e     = sum(self.field_engine.activation_state.values())
        peak_shard  = assembly[0]["shard_id"] if assembly else None

        return {
            "timestamp":    time.time(),
            "assembly":     assembly,
            "peak_shard":   peak_shard,
            "entropy":      entropy,
            "coherence":    coherence,
            "phase_r":      phase_r,
            "field_energy": total_e,
            "cycle":        self._cycle_count,
        }

    # =================================================================
    # 4.  FIELD ENTROPY — compute_entropy()
    # =================================================================

    def compute_entropy(self) -> float:
        """
        Compute NORMALIZED Shannon entropy of the activation field.

            H_norm = (-Σ p_i·log(p_i)) / log(N)

        P1 FIX: Normalized by log(N) so entropy is scale-independent.
        Now always in [0, 1] regardless of node count:
            0 = perfect focus
            1 = maximal diffusion
        """
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

        # Normalize by maximum possible entropy (uniform distribution)
        return float(raw_entropy / math.log(N))

    # =================================================================
    # 5.  ATTENTION MOMENTUM — update_attention_bias()
    # =================================================================

    def update_attention_bias(self, assembly: List[Dict[str, Any]]) -> None:
        """
        Bounded attention momentum via tanh saturation (P1 FIX).

            M_i <- tanh(λ·M_(t-1) + A_t)

        tanh bounds M_i to [-1, 1], preventing attention explosion
        and cognitive fixation on dominant concepts.
        """
        # Decay all existing biases
        for sid in list(self.attention_bias):
            self.attention_bias[sid] *= self.momentum_decay
            if abs(self.attention_bias[sid]) < self.EPSILON:
                del self.attention_bias[sid]

        # Bounded accumulation: tanh saturates to prevent monopoly
        for node in assembly:
            sid = node["shard_id"]
            act = node["activation"]
            prev = self.attention_bias.get(sid, 0.0)
            self.attention_bias[sid] = math.tanh(prev + act)

    # =================================================================
    # 6.  ATTRACTOR TRACKING — update_attractors()
    # =================================================================

    def update_attractors(self, assembly: List[Dict[str, Any]]) -> None:
        """
        Bounded attractor persistence via EMA.
        Also records cognitive transitions (A->B) for transition memory (#5).
        """
        EMA_ALPHA = 0.15
        active_ids = {node["shard_id"]: node["activation"] for node in assembly}

        # FIX #5: Record transition from previous peak to current peak
        prev_peak = self.current_assembly[0]["shard_id"] if self.current_assembly else None
        curr_peak = assembly[0]["shard_id"] if assembly else None
        if prev_peak and curr_peak and prev_peak != curr_peak:
            key = (prev_peak, curr_peak)
            self.transition_memory[key] = self.transition_memory.get(key, 0) + 1

        # EMA update for active shards
        for sid, act in active_ids.items():
            # FIX #4: Apply fatigue to effective activation
            fatigue = self.fatigue.get(sid, 0.0)
            effective_act = act * max(0.0, 1.0 - fatigue)

            if sid not in self.attractor_memory:
                self.attractor_memory[sid] = {
                    "persistence":      0.0,
                    "last_seen":        time.time(),
                    "peak_activation":  0.0,
                }
            mem = self.attractor_memory[sid]
            mem["persistence"]    += EMA_ALPHA * (effective_act - mem["persistence"])
            mem["last_seen"]       = time.time()
            mem["peak_activation"] = max(mem["peak_activation"], effective_act)

        # Decay inactive shards
        for sid in list(self.attractor_memory):
            if sid not in active_ids:
                self.attractor_memory[sid]["persistence"] *= self.attractor_decay
                if self.attractor_memory[sid]["persistence"] < self.EPSILON:
                    del self.attractor_memory[sid]

    def _update_fatigue(self) -> None:
        """
        Inhibitory fatigue: shards that dominate too long get suppressed.

            fatigue[sid] += rate * activation   (accumulates during dominance)
            fatigue[sid] *= decay               (fades slowly when inactive)

        Prevents infinite lock-in: effective_activation = activation*(1-fatigue)
        """
        active_ids = {node["shard_id"]: node["activation"]
                      for node in self.current_assembly}

        for sid, act in active_ids.items():
            self.fatigue[sid] = min(
                0.9,  # hard cap: never fully silence
                self.fatigue.get(sid, 0.0) + self._fatigue_rate * act
            )

        for sid in list(self.fatigue):
            if sid not in active_ids:
                self.fatigue[sid] *= self._fatigue_decay
                if self.fatigue[sid] < self.EPSILON:
                    del self.fatigue[sid]

    def get_top_transitions(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return most frequent cognitive transitions A->B as causal flow graph."""
        sorted_t = sorted(
            self.transition_memory.items(), key=lambda x: x[1], reverse=True
        )
        return [
            {"from": k[0], "to": k[1], "count": v}
            for k, v in sorted_t[:top_k]
        ]

    # =================================================================
    # 7.  DETECT STABLE ATTRACTORS — Multi-Tier Classification
    # =================================================================

    # Attractor tier thresholds (OPTION 3 from to_fix.md)
    ATTRACTOR_TIERS = [
        (0.95, "fixation"),    # pathological loop / obsession
        (0.80, "dominant"),    # long-term belief / goal
        (0.50, "stable"),      # stable attractor / working concept
        (0.20, "transient"),   # fleeting concept
    ]

    def classify_attractor(self, persistence: float) -> str:
        """Return cognitive tier label for a given persistence value."""
        for threshold, label in self.ATTRACTOR_TIERS:
            if persistence >= threshold:
                return label
        return "noise"

    def detect_stable_attractors(
        self, top_k: int = 5, min_tier: str = "transient"
    ) -> List[Dict[str, Any]]:
        """
        Return all attractors at or above min_tier, ranked by persistence.

        Tiers (from to_fix.md OPTION 3):
            fixation  > 0.95  — pathological loop / obsession
            dominant  > 0.80  — long-term belief / goal anchor
            stable    > 0.50  — working stable concept
            transient > 0.20  — fleeting concept
            noise     <= 0.20 — excluded

        Parameters
        ----------
        top_k    : max results to return
        min_tier : lowest tier to include (default: transient)
        """
        tier_order = {"fixation": 4, "dominant": 3, "stable": 2, "transient": 1, "noise": 0}
        min_rank   = tier_order.get(min_tier, 1)

        attractors = []
        for sid, mem in self.attractor_memory.items():
            p    = mem["persistence"]
            tier = self.classify_attractor(p)
            rank = tier_order.get(tier, 0)
            if rank >= min_rank:
                attractors.append({
                    "shard_id":        sid,
                    "persistence":     p,
                    "tier":            tier,
                    "peak_activation": mem["peak_activation"],
                    "last_seen":       mem["last_seen"],
                })

        attractors.sort(key=lambda x: x["persistence"], reverse=True)
        return attractors[:top_k]

    # =================================================================
    # 8.  EPISODIC PATH — episodic_path()
    # =================================================================

    def episodic_path(self, window: int = 20) -> List[Dict[str, Any]]:
        """
        Return the temporal cognitive trajectory — the sequence of
        dominant shards across recent cognitive history.

        Example output:
            god → creation → universe → entropy → thermodynamics

        This is proto-reasoning: semantic navigation through time.

        Parameters
        ----------
        window : how many recent snapshots to include

        Returns ordered list of { cycle, peak_shard, entropy, energy }
        """
        recent = list(self.cognitive_history)[-window:]
        return [
            {
                "cycle":       snap["cycle"],
                "peak_shard":  snap["peak_shard"],
                "entropy":     snap["entropy"],
                "energy":      snap["field_energy"],
                "timestamp":   snap["timestamp"],
            }
            for snap in recent
            if snap["peak_shard"] is not None
        ]

    # =================================================================
    # 9.  STREAM STABILIZATION — stabilize_stream()
    # =================================================================

    def stabilize_stream(self) -> None:
        """
        Prevent runaway resonance, infinite fixation, semantic seizure.

        1. Entropy ceiling: cool field if H_norm > ceiling
        2. Smooth energy normalization A/(1 + E/Emax)
        3. Adaptive activation floor = mean(A)*epsilon
        4. Prune dead attention_bias entries
        5. FIX #3 (to_fix.md): Triple feedback damper
           resonance_memory + attention_bias + transition_momentum are all
           positive feedback systems. When they jointly over-reinforce,
           apply a soft global scaling cap to prevent obsession basins.
        """
        entropy = self.compute_entropy()

        # 1. Entropy ceiling
        if entropy > self.entropy_ceiling:
            cool_factor = self.entropy_ceiling / (entropy + self.EPSILON)
            self.field_engine.apply_global_scaling(cool_factor)

        # 2. Smooth energy regulation
        total_e = sum(
            v for v in self.field_engine.activation_state.values() if v > 0
        )
        if total_e > self.energy_ceiling:
            smooth_scale = 1.0 / (1.0 + total_e / self.energy_ceiling)
            self.field_engine.apply_global_scaling(smooth_scale)

        # 3. Adaptive activation floor
        active_vals = [
            v for v in self.field_engine.activation_state.values() if v > self.EPSILON
        ]
        if active_vals:
            adaptive_floor = (sum(active_vals) / len(active_vals)) * self.EPSILON * 10
            for sid in self.field_engine.activation_state:
                if 0 < self.field_engine.activation_state[sid] < adaptive_floor:
                    self.field_engine.activation_state[sid] = 0.0
                    self.field_engine.field_nodes[sid]["state"]["activation"]["amplitude"] = 0.0

        # 4. Prune decayed attention_bias entries
        for sid in list(self.attention_bias):
            if abs(self.attention_bias[sid]) < self.EPSILON:
                del self.attention_bias[sid]

        # 5. TRIPLE FEEDBACK DAMPER (to_fix.md issue #3)
        # Measure combined positive pressure:
        #   - bias pressure = mean top-5 attention bias
        #   - memory pressure = resonance_memory size ratio
        #   - momentum pressure = max transition count normalized
        bias_pressure  = 0.0
        top_bias = sorted(self.attention_bias.values(), reverse=True)[:5]
        if top_bias:
            bias_pressure = sum(top_bias) / len(top_bias)

        mem_size      = len(self.field_engine.resonance_memory)
        mem_pressure  = min(1.0, mem_size / max(1, len(self.field_engine.field_nodes) * 4))

        momentum_pressure = 0.0
        if self.transition_memory:
            max_t = max(self.transition_memory.values())
            momentum_pressure = min(1.0, max_t / 20.0)  # saturates at 20 repetitions

        combined = (bias_pressure + mem_pressure + momentum_pressure) / 3.0
        # If all 3 systems are pushing hard simultaneously, apply soft brake
        TRIPLE_PRESSURE_THRESHOLD = 0.65
        if combined > TRIPLE_PRESSURE_THRESHOLD:
            brake = 1.0 - 0.15 * (combined - TRIPLE_PRESSURE_THRESHOLD)  # gentle, not cliff
            self.field_engine.apply_global_scaling(max(0.70, brake))
            # Also partially decay strongest attention biases
            for sid in self.attention_bias:
                self.attention_bias[sid] *= 0.92

    # =================================================================
    # 10. STREAM STATE SUMMARY — stream_state()
    # =================================================================

    def stream_state(self) -> Dict[str, Any]:
        """
        Return a full diagnostic snapshot of the cognitive stream state.
        Useful for monitoring, debugging, and integration.
        """
        entropy     = self.compute_entropy()
        total_e     = sum(self.field_engine.activation_state.values())
        attractors  = self.detect_stable_attractors(top_k=5)
        path        = self.episodic_path(window=10)

        return {
            "cycle":            self._cycle_count,
            "entropy":          entropy,
            "field_energy":     total_e,
            "active_nodes":     sum(
                1 for v in self.field_engine.activation_state.values()
                if v > self.EPSILON
            ),
            "stable_attractors": attractors,
            "episodic_path":    [p["peak_shard"] for p in path],
            "attention_peaks":  sorted(
                self.attention_bias.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "resonance_memory_size": len(self.field_engine.resonance_memory),
        }

    # =================================================================
    # 11. MULTI-CYCLE RUN — run()
    # =================================================================

    def run(
        self,
        cycles: int,
        inject: Optional[Dict[str, float]] = None,
        top_k: int = 10,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run N continuous cognitive cycles.

        Parameters
        ----------
        cycles : number of cycles to run
        inject : optional dict of { shard_id: energy } to inject before cycle 1
        top_k  : assembly size per capture
        verbose: print cycle diagnostics if True

        Returns list of cognitive snapshots.
        """
        if inject:
            for sid, energy in inject.items():
                self.inject_query(sid, energy)

        snapshots = []
        for _ in range(cycles):
            snap = self.cognitive_cycle(top_k=top_k)
            snapshots.append(snap)
            if verbose:
                print(
                    f"  Cycle #{snap['cycle']:4d}  peak={str(snap['peak_shard']):<20}  "
                    f"H={snap['entropy']:.3f}  E={snap['field_energy']:.3f}"
                )
        return snapshots

    # =================================================================
    # P1 — ASSEMBLY COHERENCE
    # =================================================================

    def compute_assembly_coherence(self, assembly: List[Dict[str, Any]]) -> float:
        """Mean pairwise resonance within the active assembly.

            C = mean(|R(i,j)|) for all active pairs

        High   -> focused cognition (semantically aligned)
        Medium -> associative exploration
        Low    -> semantic chaos
        """
        ids = [node["shard_id"] for node in assembly]
        if len(ids) < 2:
            return 1.0
        total, count = 0.0, 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                total += abs(self.field_engine.resonance(ids[i], ids[j]))
                count += 1
        return float(total / count) if count > 0 else 0.0

    # =================================================================
    # P1 — PHASE COHERENCE (Kuramoto Order Parameter)
    # =================================================================

    def compute_phase_coherence(self, assembly: List[Dict[str, Any]]) -> float:
        """Kuramoto order parameter R across active assembly nodes.

            R = |sum(exp(i*phi_k))| / N

        R=1 -> synchronized cognition
        R=0 -> fragmented cognition
        """
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

    # =================================================================
    # P2 — DEEP CONSOLIDATION CYCLE
    # =================================================================

    def deep_consolidation_cycle(self) -> None:
        """Periodic slow consolidation every N cognitive cycles.

        1. RFE resonance memory pruning + topology rebucketing
        2. Prune weak attractor_memory entries
        3. Prune dead attention_bias entries
        4. FIX #5 (to_fix.md): Meta-shard stabilization guard
           Only promote assemblies to meta-shards if:
           - The pattern has recurred >= 3 times in episodic motifs
           - The peak shard has attractor persistence >= stable tier (0.50)
           This prevents transient assemblies becoming abstract concepts too early.
        """
        self.field_engine.consolidate_field_memory()
        PRUNE = 1e-4
        for sid in list(self.attractor_memory):
            if self.attractor_memory[sid]["persistence"] < PRUNE:
                del self.attractor_memory[sid]
        for sid in list(self.attention_bias):
            if abs(self.attention_bias[sid]) < self.EPSILON:
                del self.attention_bias[sid]

        # META-SHARD BIRTH GUARD: only form concepts from durable motifs
        motifs = self.extract_temporal_motifs(min_length=2, max_length=3, min_occurrences=3)
        for motif_info in motifs[:3]:  # top 3 most-recurring motifs
            peak_of_motif = motif_info["motif"][0]  # first shard in sequence
            mem = self.attractor_memory.get(peak_of_motif, {})
            persistence = mem.get("persistence", 0.0)
            # GUARD: require stable tier (>=0.50) persistence before abstracting
            if persistence >= 0.50:
                log_service(
                    logger,
                    f"Meta-shard candidate: {motif_info['motif']} "
                    f"(count={motif_info['count']}, persistence={persistence:.3f}) "
                    f"[ELIGIBLE for form_meta_shard]",
                    "info"
                )
            else:
                log_service(
                    logger,
                    f"Meta-shard blocked: {motif_info['motif']} "
                    f"persistence={persistence:.3f} < 0.50 [NOT YET STABLE]",
                    "debug"
                )

        log_service(logger, f"Deep consolidation #{self._cycle_count}: "
                    f"attractors={len(self.attractor_memory)} "
                    f"bias_entries={len(self.attention_bias)}", "debug")

    # =================================================================
    # P2 — TRAJECTORY ANALYTICS
    # =================================================================

    def trajectory_divergence(self, window: int = 10) -> float:
        """Fraction of steps where the dominant shard changes.
        High = rapidly shifting / exploratory.
        Low  = stable attractor / focused reasoning."""
        path = self.episodic_path(window=window)
        if len(path) < 2:
            return 0.0
        changes = sum(
            1 for i in range(1, len(path))
            if path[i]["peak_shard"] != path[i - 1]["peak_shard"]
        )
        return float(changes / (len(path) - 1))

    def trajectory_recurrence(self, window: int = 20) -> Dict[str, int]:
        """Count peak-shard recurrences. High = cognitive loop / concept anchoring."""
        path = self.episodic_path(window=window)
        counts: Dict[str, int] = {}
        for p in path:
            sid = p["peak_shard"]
            if sid:
                counts[sid] = counts.get(sid, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def trajectory_stability(self, window: int = 10) -> float:
        """Mean entropy over recent trajectory. Low = consistently focused."""
        path = self.episodic_path(window=window)
        if not path:
            return 0.0
        return float(sum(p["entropy"] for p in path) / len(path))

    # =================================================================
    # 12. RESET STREAM — reset()
    # =================================================================

    def reset(self, hard: bool = False) -> None:
        """
        Reset the temporal stream.

        Parameters
        ----------
        hard : if True, also reset the underlying field activations.
               if False, only clear stream-level state.
        """
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

    # =================================================================
    # SEMANTIC TRANSITION MOMENTUM
    # =================================================================

    def _transition_momentum_step(self) -> None:
        """
        Use learned transition frequencies to softly bias the field
        toward the next likely concept.

        For each current peak shard P with known transitions P->X:
            excite(X, strength * frequency_weight)

        This creates semantic momentum: thoughts naturally flow toward
        their historically associated next concepts.
        This is a proto-causal reasoning bridge.
        """
        if not self.current_assembly or not self.transition_memory:
            return

        peak_sid  = self.current_assembly[0]["shard_id"]
        max_count = max(self.transition_memory.values()) if self.transition_memory else 1

        for (src, dst), count in self.transition_memory.items():
            if src != peak_sid:
                continue
            if dst not in self.field_engine.field_nodes:
                continue

            # FIX #2 (to_fix.md): Gate by destination fatigue
            # Prevents A->B->A self-reinforcing loops.
            # As dst accumulates fatigue, the momentum injection shrinks.
            dst_fatigue = self.fatigue.get(dst, 0.0)
            fatigue_gate = math.exp(-dst_fatigue * 3.0)   # aggressive: 0 fatigue->1.0, full->~0.05

            weight = (count / max_count) * self.transition_momentum_strength * fatigue_gate
            if weight > self.EPSILON:
                self.field_engine.excite(dst, weight)

    # =================================================================
    # EPISODIC MOTIF EXTRACTION
    # =================================================================

    def extract_temporal_motifs(
        self, min_length: int = 2, max_length: int = 4, min_occurrences: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Extract repeated sequential thought patterns (temporal motifs)
        from the full episodic history.

        A motif is a sequence of peak shards that recurs >= min_occurrences times.
        These are the proto-reasoning chains from which meta-shards emerge.

        Example:
            god -> creation -> universe  (occurred 5x)
            entropy -> chaos -> disorder (occurred 3x)

        Parameters
        ----------
        min_length     : minimum motif length (n-gram size)
        max_length     : maximum motif length to scan
        min_occurrences: minimum repetitions to qualify as a motif

        Returns list of { motif: tuple, count: int, density: float }
        """
        # Extract peak sequence from full history
        history_list = list(self.cognitive_history)
        peak_sequence = [
            snap["peak_shard"]
            for snap in history_list
            if snap["peak_shard"] is not None
        ]

        if len(peak_sequence) < min_length:
            return []

        # Count all n-grams
        motif_counts: Dict[tuple, int] = {}
        for n in range(min_length, max_length + 1):
            for i in range(len(peak_sequence) - n + 1):
                gram = tuple(peak_sequence[i:i + n])
                # Skip trivial constant motifs (god, god, god)
                if len(set(gram)) == 1:
                    continue
                motif_counts[gram] = motif_counts.get(gram, 0) + 1

        # Filter by minimum occurrences
        total = len(peak_sequence)
        motifs = [
            {
                "motif":   gram,
                "count":   cnt,
                "density": round(cnt / max(1, total - len(gram) + 1), 3),
                "length":  len(gram),
            }
            for gram, cnt in motif_counts.items()
            if cnt >= min_occurrences
        ]
        motifs.sort(key=lambda x: (x["count"], x["length"]), reverse=True)
        return motifs


# =============================================================================
# VALIDATION — __main__
# =============================================================================

if __name__ == "__main__":
    from backend.services.loom_service.substrate.HyperVectorCreation import HyperVectorEngine

    print("=" * 65)
    print("STAGE 6 — TemporalCognitiveStream Validation")
    print("=" * 65)

    DIM = 1000
    hdc = HyperVectorEngine(dimension=DIM)
    rfe = ResonanceFieldEngine(decay_rate=0.95, inhibition_strength=0.1, propagation_depth=5)

    # Build semantic concepts
    concepts = {
        "god":            ["god", "creation", "divine"],
        "creation":       ["creation", "universe", "origin"],
        "universe":       ["universe", "cosmos", "space"],
        "entropy":        ["entropy", "disorder", "chaos"],
        "thermodynamics": ["thermodynamics", "energy", "heat"],
        "earth":          ["earth", "ground", "world"],
        "life":           ["life", "living", "biology"],
    }

    print("\nRegistering shards into field...")
    for concept_id, seeds in concepts.items():
        # get_basis_vector: deterministic random HDC basis (no transformer needed)
        # bundle() returns (field_vec, binary_vec) — use field_vec for the field engine
        basis_vecs       = [hdc.get_basis_vector(s).astype(np.float32) for s in seeds]
        field_vec, _bv   = hdc.bundle(basis_vecs)
        rfe.register_shard(f"s_{concept_id}", field_vec, meta={"label": concept_id})

    # Wrap in temporal stream
    stream = TemporalCognitiveStream(
        field_engine=rfe,
        history_size=100,
        stream_decay=0.90,
        entropy_ceiling=0.85,    # FIX: normalized H is [0,1], not [0,4]
    )

    print("\n--- Injecting 'god' and running 30 cognitive cycles ---")
    snapshots = stream.run(
        cycles=30,
        inject={"s_god": 1.0},
        top_k=5,
        verbose=True,
    )

    print("\n--- Multi-Tier Attractor Classification ---")
    for att in stream.detect_stable_attractors(top_k=7, min_tier="transient"):
        print(f"  [{att['tier']:>9}]  {att['shard_id']:<22}  P={att['persistence']:.3f}  peak={att['peak_activation']:.4f}")

    print("\n--- Episodic Path (thought trajectory) ---")
    path = stream.episodic_path(window=30)
    trajectory = " > ".join(p["peak_shard"] for p in path if p["peak_shard"])
    print(f"  {trajectory}")

    print("\n--- Stream State Diagnostic ---")
    state = stream.stream_state()
    print(f"  Cycles run:        {state['cycle']}")
    print(f"  Field entropy:     {state['entropy']:.4f}  (0=focused, 1=chaotic)")
    print(f"  Total field energy:{state['field_energy']:.4f}")
    print(f"  Active nodes:      {state['active_nodes']}")
    print(f"  Memory pairs:      {state['resonance_memory_size']}")
    print(f"  Episodic path:     {' > '.join(state['episodic_path'])}")

    print("\n--- Phase Coherence ---")
    assembly = stream.current_assembly
    phase_r  = stream.compute_phase_coherence(assembly)
    coherence = stream.compute_assembly_coherence(assembly)
    print(f"  Kuramoto R:        {phase_r:.3f}   (1=synchronized, 0=fragmented)")
    print(f"  Assembly coherence:{coherence:.3f}  (1=semantically aligned)")

    print("\n--- Trajectory Analytics ---")
    print(f"  Divergence:        {stream.trajectory_divergence(window=20):.3f}  (0=locked, 1=shifting)")
    print(f"  Stability (H):     {stream.trajectory_stability(window=20):.3f}  (low=focused)")
    recur = stream.trajectory_recurrence(window=20)
    top3  = dict(list(recur.items())[:3])
    print(f"  Recurrence top-3:  {top3}")

    transitions = stream.get_top_transitions(top_k=5)
    if transitions:
        print("  Top Causal Transitions:")
        for t in transitions:
            print(f"    {t['from']} -> {t['to']}  (x{t['count']})")

    print("\n--- Temporal Motifs (proto-reasoning chains) ---")
    motifs = stream.extract_temporal_motifs(min_length=2, max_length=3, min_occurrences=2)
    if motifs:
        for m in motifs[:5]:
            chain = " -> ".join(m["motif"])
            print(f"  [{m['count']}x d={m['density']}]  {chain}")
    else:
        print("  (no repeated motifs yet — run more cycles)")

    print("\nValidation complete. Stage 6 v1.1 — Full Cognitive Physics [OK]")
