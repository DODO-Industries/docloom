import sys
import os
import time
import math
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

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

logger = setup_logger("ResonanceFieldEngine")


class ResonanceFieldEngine:
    """
    ============================================================================
    DOCLOOM — RESONANCE FIELD ENGINE  (Stage 3 → Stage 5)
    ============================================================================

    Version: 2.3  —  Phase-Coupled Cortical Dynamics

    Version: 2.4  —  Grounded Multiplicative Memory + Cognitive Scaffolding

    Upgrades over v2.3:
    - #1  Semantic rebucketing: blends vector sig + learned resonance sig
    - #2  Multiplicative memory modulation: soft_r * (1 + mem_boost)
            Prevents hallucinated attractors from overpowering semantics
    - #3  attention_gate() stub: salience-weighted amplification scaffold
    - #4  form_meta_shard() stub: hierarchical assembly → concept recursion

    Cognitive Dynamics:
        soft_R(i,j)  = tanh(max(0, R(i,j)) * 3.0) * cos(φ_i - φ_j)  (phase-coupled)
        A_j(t+1)    += A_i(t) * (soft_R + mem_boost) * λ              (local sparse wave)
        A_i(t+1)     = A_i(t) * λ                                      (decay)
        φ_i(t+1)     = (φ_i + 2π·f_i·dt) mod 2π                       (phase advance)
        A_i          = A_i - α * Σ R(i,j)² * A_j                      (quad inhibition)
    ============================================================================
    """

    EPSILON = 1e-8    # Numerical floor — prevents eternal field ghosts

    # -----------------------------------------------------------------
    # FIELD NODE STATE SCHEMA (per-shard neuron state)
    # -----------------------------------------------------------------
    # {
    #   "activation" : { "amplitude": float, "phase": float, "frequency": float }
    #   "energy"     : float  — cumulative excitation history
    #   "stability"  : float  — how resistant to perturbation [0.0 – 1.0]
    #   "decay_rate" : float  — λ per tick  [0.90 – 0.99]
    #   "inhibition" : float  — lateral suppression amount
    #   "last_excited": float — unix timestamp of last excitation
    # }

    def __init__(
        self,
        decay_rate: float = 0.95,
        inhibition_strength: float = 0.1,
        propagation_depth: int = 3,
        neighborhood_size: int = 32,
        target_energy: float = 1.0,
        phase_dt: float = 0.1,
        rebucket_interval: int = 50,
    ):
        """
        Parameters
        ----------
        decay_rate          : λ — energy fade per propagation tick [0.90–0.99]
        inhibition_strength : how strongly active nodes suppress each other
        propagation_depth   : how many cognitive ticks per run_query() call
        neighborhood_size   : K — max local neighbors per shard in sparse routing
        target_energy       : energy normalization ceiling after propagation
        phase_dt            : time step for phase advance per decay tick
        rebucket_interval   : consolidation cycles between topology rebucketing
        """
        self.decay_rate          = decay_rate
        self.inhibition_strength = inhibition_strength
        self.propagation_depth   = propagation_depth
        self.neighborhood_size   = neighborhood_size
        self.target_energy       = target_energy
        self.phase_dt            = phase_dt
        self.rebucket_interval   = rebucket_interval
        self._consolidation_tick = 0          # counter for rebucket scheduling

        # ── max resonance_memory entries per shard  (synaptic pruning budget)
        self._max_memory_per_shard = neighborhood_size * 4

        # ── Field Registry ─────────────────────────────────────────────
        self.field_nodes: Dict[str, Dict[str, Any]] = {}

        # ── Runtime Activation State ────────────────────────────────────
        # shard_id → float (current activation energy amplitude)
        self.activation_state: Dict[str, float] = {}

        # ── Persistent Resonance Memory  ──────────────────────────────
        # Sparse: only experienced neighbor pairs stored, NOT all N²
        self.resonance_memory: Dict[Tuple[str, str], Dict[str, float]] = {}

        # ── Locality Signature Neighborhood Index ────────────────────
        # bucket_id → set of shard_ids in that locality bucket
        # Transforms propagate/inhibit from O(N²) to O(N·K)
        self.neighborhood_index: Dict[int, List[str]] = {}
        # shard_id → bucket_id (reverse lookup)
        self._shard_bucket: Dict[str, int] = {}

        log_service(
            logger,
            f"ResonanceFieldEngine v2.3 initialized. "
            f"λ={decay_rate}, inhibition={inhibition_strength}, "
            f"depth={propagation_depth}, K={neighborhood_size}, dt={phase_dt}",
            "info",
        )

    # =================================================================
    # 1.  FIELD REGISTRY — register_shard()
    # =================================================================

    def register_shard(self, shard_id: str, field_vec: np.ndarray, meta: Optional[Dict] = None) -> None:
        """
        Insert a semantic neuron into the living cognitive field.

        Parameters
        ----------
        shard_id  : unique shard identifier (e.g. shard UUID)
        field_vec : the ANALOG field vector from HyperVectorEngine.bundle()
                    (float32, shape = [D])
        meta      : optional metadata dict (text, phi, etc.)
        """
        if shard_id in self.field_nodes:
            log_service(logger, f"Shard '{shard_id}' already registered — skipping.", "warning")
            return

        self.field_nodes[shard_id] = {
            "field_vec": field_vec.astype(np.float32),
            "meta": meta or {},
            "state": {
                # Phase-ready activation schema (Step 6)
                "activation":   {"amplitude": 0.0, "phase": 0.0, "frequency": 1.0},
                "energy":       0.0,
                "stability":    1.0,
                "decay_rate":   self.decay_rate,
                "inhibition":   0.0,
                "last_excited": None,
            },
        }
        self.activation_state[shard_id] = 0.0

        # ── Deterministic bucket hash (fix #2: stable across process restarts) ─
        # Python hash() is randomized per-process — use md5 for reproducibility
        sig_bits  = np.sign(field_vec[:128]).astype(np.int8)
        bucket_id = int(hashlib.md5(sig_bits.tobytes()).hexdigest()[:8], 16)
        self._shard_bucket[shard_id] = bucket_id
        if bucket_id not in self.neighborhood_index:
            self.neighborhood_index[bucket_id] = []
        self.neighborhood_index[bucket_id].append(shard_id)

        log_service(logger, f"Shard '{shard_id}' registered into field. Total nodes: {len(self.field_nodes)}", "info")

    def deregister_shard(self, shard_id: str) -> None:
        """Remove a shard from the field entirely."""
        # Remove from neighborhood index first
        bucket_id = self._shard_bucket.pop(shard_id, None)
        if bucket_id is not None and bucket_id in self.neighborhood_index:
            try:
                self.neighborhood_index[bucket_id].remove(shard_id)
            except ValueError:
                pass
        self.field_nodes.pop(shard_id, None)
        self.activation_state.pop(shard_id, None)
        log_service(logger, f"Shard '{shard_id}' removed from field.", "info")

    def reset_activations(self) -> None:
        """
        Zero all activation energies while preserving field topology.

        Keeps intact:
          - registered shards (field_nodes)
          - resonance_memory (learned topology)
          - neighborhood_index (spatial routing)
          - field_vec (semantic content)

        Resets:
          - activation_state → 0.0
          - phase, energy per neuron → 0.0

        Used by TemporalCognitiveStream.reset(hard=True) to give the
        field a clean energetic slate without losing what it learned.
        """
        for sid, node in self.field_nodes.items():
            self.activation_state[sid] = 0.0
            state = node["state"]
            state["activation"]["amplitude"] = 0.0
            state["activation"]["phase"]     = 0.0
            state["energy"]                  = 0.0
            state["inhibition"]              = 0.0
            state["last_excited"]            = None
        log_service(logger, f"Field activations reset. Topology preserved ({len(self.field_nodes)} shards).", "info")

    def apply_global_decay(self, factor: float) -> None:
        """
        Apply a multiplicative decay factor to ALL activation amplitudes.

        Called by TemporalCognitiveStream for stream-level macro fading.
        Keeps amplitude and activation_state in sync (single source of truth).

        Parameters
        ----------
        factor : decay multiplier in (0, 1], e.g. 0.90
        """
        for sid in self.field_nodes:
            self.activation_state[sid] *= factor
            self.field_nodes[sid]["state"]["activation"]["amplitude"] *= factor

    def apply_global_scaling(self, factor: float) -> None:
        """
        Scale ALL activations by a factor (used for smooth energy normalization).

        Parameters
        ----------
        factor : scale multiplier, e.g. 1/(1 + E/Emax)
        """
        for sid in self.field_nodes:
            self.activation_state[sid] *= factor
            self.field_nodes[sid]["state"]["activation"]["amplitude"] *= factor

    # =================================================================
    # 2.  RESONANCE FUNCTION — R(i, j)
    # =================================================================

    def resonance(self, shard_i: str, shard_j: str) -> float:
        """
        Continuous cosine resonance between two field nodes.

        R(i,j) = (v_i · v_j) / (|v_i| |v_j|)

        This is the field coupling coefficient — NOT a threshold.
        It is a continuous analog value in [-1.0, +1.0].
        """
        vi = self.field_nodes[shard_i]["field_vec"]
        vj = self.field_nodes[shard_j]["field_vec"]

        norm_i = np.linalg.norm(vi)
        norm_j = np.linalg.norm(vj)

        if norm_i == 0.0 or norm_j == 0.0:
            return 0.0

        return float(np.dot(vi, vj) / (norm_i * norm_j))

    def resonance_matrix(self) -> Dict[Tuple[str, str], float]:
        """
        Compute full pairwise resonance for all registered shards.
        Returns upper-triangle only (symmetric).
        WARNING: O(N²) — use only for analysis/debugging, not live propagation.
        """
        ids = list(self.field_nodes.keys())
        matrix = {}
        for i, id_i in enumerate(ids):
            for id_j in ids[i + 1:]:
                matrix[(id_i, id_j)] = self.resonance(id_i, id_j)
        return matrix

    def _bucket_hamming(self, a: int, b: int) -> int:
        """Hamming distance between two bucket IDs (XOR popcount)."""
        return bin(a ^ b).count("1")

    def _get_local_neighbors(self, shard_id: str) -> List[str]:
        """
        Sparse Cortical Routing (#3 fix): returns the local neighborhood.
        Bucket expansion ordered by Hamming distance — semantically nearest
        buckets first, NOT arbitrary dict iteration order.
        """
        bucket_id = self._shard_bucket.get(shard_id)
        if bucket_id is None:
            return [s for s in self.field_nodes if s != shard_id]

        candidates = [
            s for s in self.neighborhood_index.get(bucket_id, [])
            if s != shard_id
        ]

        # Expand via Hamming-distance ordering if needed
        if len(candidates) < self.neighborhood_size:
            other_buckets = sorted(
                [(bid, self._bucket_hamming(bucket_id, bid))
                 for bid in self.neighborhood_index if bid != bucket_id],
                key=lambda x: x[1],
            )
            for bid, _ in other_buckets:
                for s in self.neighborhood_index[bid]:
                    if s != shard_id and s not in candidates:
                        candidates.append(s)
                if len(candidates) >= self.neighborhood_size:
                    break

        return candidates[:self.neighborhood_size]

    # =================================================================
    # 3.  EXCITATION — excite()
    # =================================================================

    def excite(self, shard_id: str, energy: float = 1.0) -> None:
        """
        Inject activation energy into a shard (the query stimulus).

        Parameters
        ----------
        shard_id : the shard to excite
        energy   : initial activation amplitude (default 1.0)
        """
        if shard_id not in self.field_nodes:
            raise KeyError(f"Shard '{shard_id}' not in field. Call register_shard() first.")

        self.activation_state[shard_id] += energy
        state = self.field_nodes[shard_id]["state"]
        state["activation"]["amplitude"] += energy
        state["energy"]                  += energy
        state["last_excited"]             = time.time()
        # Reset phase on excitation — excited neuron re-anchors its oscillation
        state["activation"]["phase"]      = 0.0

        log_service(logger, f"Excited shard '{shard_id}' with energy={energy:.4f}", "debug")

    # =================================================================
    # NEW — SOFT RESONANCE TRANSFER FUNCTION
    # =================================================================

    def soft_resonance(self, resonance: float) -> float:
        """
        Smooth analog resonance transfer function (Step 1).

        Replaces hard thresholds with biological-style saturation:
        - Weak resonance still contributes weakly
        - Strong resonance dominates naturally
        - tanh prevents explosive propagation
        """
        return float(np.tanh(max(0.0, resonance) * 3.0))

    # =================================================================
    # 4.  PROPAGATION ENGINE — propagate()  (Step 4 rewrite)
    # =================================================================

    def propagate(self) -> None:
        """
        TRUE CONSERVATIVE semantic wave propagation (energy transport, not creation).

        Transport dynamics (to_fix.md issue #1):
            transfer = effective_ai * effective_r
            delta[j] += transfer      (receiver gains)
            delta[i] -= transfer      (source loses — energy is MOVED not cloned)

        Then decay is applied separately in decay().
        This makes the field a conservative redistribution system,
        eliminating the stable energy plateau artifact (E ~ 1.42).

        Additional fixes:
        - MICRO-NOISE: Gaussian(0, 0.002) enables attractor hopping.
        - ANTI-RECURRENCE: prior inhibition reduces outgoing contribution.
        """
        ids     = list(self.field_nodes.keys())
        current = dict(self.activation_state)       # snapshot — no mid-tick mutation
        delta: Dict[str, float] = {sid: 0.0 for sid in ids}

        for id_i in ids:
            ai = current[id_i]
            if abs(ai) < self.EPSILON:
                continue

            # ANTI-RECURRENCE FATIGUE: inhibited nodes spread less
            prior_inhibition = self.field_nodes[id_i]["state"]["inhibition"]
            fatigue_factor   = math.exp(-prior_inhibition * 2.0)
            effective_ai     = ai * fatigue_factor

            phi_i     = self.field_nodes[id_i]["state"]["activation"]["phase"]
            neighbors = self._get_local_neighbors(id_i)

            total_transferred = 0.0
            transfers: Dict[str, float] = {}

            for id_j in neighbors:
                r_ij   = self.resonance(id_i, id_j)
                if r_ij <= 0.0:
                    continue
                soft_r = self.soft_resonance(r_ij)

                # Phase alignment: cos(delta_phi) in [0, 1]
                phi_j       = self.field_nodes[id_j]["state"]["activation"]["phase"]
                phase_align = (math.cos(phi_i - phi_j) + 1.0) * 0.5
                soft_r      = soft_r * phase_align

                # Memory modulation — multiplicative, semantics grounded
                memory_key   = tuple(sorted((id_i, id_j)))
                memory_boost = 0.0
                if memory_key in self.resonance_memory:
                    memory_boost = self.resonance_memory[memory_key]["stability"] * 0.25

                effective_r        = soft_r * (1.0 + memory_boost)
                transfer           = effective_ai * effective_r * self.decay_rate
                transfers[id_j]    = transfer
                total_transferred += transfer

            # Clamp: source cannot give more than it has
            if total_transferred > effective_ai and total_transferred > self.EPSILON:
                clamp = effective_ai / total_transferred
                transfers = {k: v * clamp for k, v in transfers.items()}
                total_transferred = effective_ai

            # TRUE TRANSPORT: subtract from source, add to destinations
            for id_j, transfer in transfers.items():
                delta[id_j] += transfer
            delta[id_i] -= total_transferred

        # Apply deltas + MICRO-NOISE (prevents frozen attractor basins)
        noise_std = 0.002
        for sid in ids:
            new_val = self.activation_state[sid] + delta[sid]
            noise   = float(np.random.normal(0.0, noise_std))
            new_val = max(0.0, new_val + noise)
            self.activation_state[sid] = new_val
            self.field_nodes[sid]["state"]["activation"]["amplitude"] = new_val

        log_service(
            logger,
            f"Propagation tick. Active nodes: "
            f"{sum(1 for v in self.activation_state.values() if abs(v) >= self.EPSILON)}/{len(ids)}",
            "debug",
        )

    # =================================================================
    # 5.  DECAY ENGINE — decay()
    # =================================================================

    def decay(self) -> None:
        """
        Apply per-node energy decay + Kuramoto phase coupling per cognitive tick.

        Energy:
            A_i(t+1) = A_i(t) * lambda_i

        Phase (Kuramoto synchronization dynamics — Issue #5 fix):
            dφ/dt = omega_i + K * mean(sin(phi_j - phi_i))   for active neighbors j

            K = coupling strength (proportional to neighbor amplitude)
            This drives real oscillatory synchronization, not decorative phase advance.
        """
        tau            = self.phase_dt * 2 * math.pi
        KURAMOTO_K     = 0.3   # coupling strength — how fast phases sync
        activation_snapshot = dict(self.activation_state)  # freeze before mutation

        for sid, node in self.field_nodes.items():
            state = node["state"]
            lam   = state["decay_rate"]

            # 1. Energy decay
            self.activation_state[sid] *= lam
            state["activation"]["amplitude"] = max(0.0, self.activation_state[sid])
            if self.activation_state[sid] < 0.0:
                self.activation_state[sid] = 0.0

            # 2. RESONANCE-WEIGHTED Kuramoto phase evolution (to_fix.md issue #4)
            #    dphi = omega + K * sum(R(i,j) * sin(phi_j - phi_i))
            #    Only semantically related nodes (R > 0) pull phases together.
            #    Unrelated nodes remain phase-separated -> semantic decoupling.
            #    Fixes: Kuramoto R=0.97 with coherence=0.09 disagreement.
            freq  = state["activation"]["frequency"]
            phi_i = state["activation"]["phase"]
            dphi  = tau * freq   # natural oscillation

            ai = activation_snapshot[sid]
            if ai > self.EPSILON:
                neighbors    = self._get_local_neighbors(sid)
                coupling_sum = 0.0
                weight_sum   = 0.0
                for nb in neighbors:
                    a_nb = activation_snapshot.get(nb, 0.0)
                    if a_nb > self.EPSILON:
                        r_ij  = max(0.0, self.resonance(sid, nb))  # only positive resonance couples
                        phi_j = self.field_nodes[nb]["state"]["activation"]["phase"]
                        coupling_sum += r_ij * math.sin(phi_j - phi_i)
                        weight_sum   += r_ij
                if weight_sum > self.EPSILON:
                    dphi += KURAMOTO_K * (coupling_sum / weight_sum)

            state["activation"]["phase"] = (phi_i + dphi) % (2 * math.pi)

    # =================================================================
    # 6.  INHIBITION ENGINE — inhibit()
    # =================================================================

    def inhibit(self) -> None:
        """
        Sparse QUADRATIC lateral inhibition with PHASE-DEPENDENT coupling.

        Standard:   A_i -= inhibition_strength * sum(R(i,j)^2 * A_j)
        Phase fix:  suppression *= (cos(phi_i - phi_j) + 1) / 2

        Only phase-synchronized nodes inhibit strongly.
        Desynchronized nodes can coexist — preserving parallel thought streams.
        This prevents synchronized attractors from silencing ALL other concepts.
        """
        ids = list(self.field_nodes.keys())
        current = dict(self.activation_state)
        suppression: Dict[str, float] = {sid: 0.0 for sid in ids}

        for id_i in ids:
            phi_i = self.field_nodes[id_i]["state"]["activation"]["phase"]
            for id_j in self._get_local_neighbors(id_i):
                r_ij = self.resonance(id_i, id_j)
                if r_ij > 0.0:
                    # Phase-dependent suppression: synchronized = strong, async = weak
                    phi_j         = self.field_nodes[id_j]["state"]["activation"]["phase"]
                    phase_sync    = (math.cos(phi_i - phi_j) + 1.0) * 0.5   # [0, 1]
                    suppression[id_i] += (r_ij ** 2) * current[id_j] * phase_sync

        for sid in ids:
            overlap_energy = self.inhibition_strength * suppression[sid]
            self.activation_state[sid] = max(0.0, self.activation_state[sid] - overlap_energy)
            self.field_nodes[sid]["state"]["inhibition"]              = overlap_energy
            self.field_nodes[sid]["state"]["activation"]["amplitude"] = self.activation_state[sid]

    # =================================================================
    # NEW — CO-ACTIVATION MEMORY  (Step 3)
    # =================================================================

    def update_resonance_memory(self) -> None:
        """
        Strengthen persistent semantic relationships through co-activation.
        Sparse: only experienced NEIGHBOR pairs are stored, preventing N² explosion.
        """
        ids = list(self.field_nodes.keys())
        for id_i in ids:
            ai = self.activation_state[id_i]
            if abs(ai) < self.EPSILON:
                continue
            for id_j in self._get_local_neighbors(id_i):
                aj = self.activation_state[id_j]
                if abs(aj) < self.EPSILON:
                    continue
                key = tuple(sorted((id_i, id_j)))
                if key not in self.resonance_memory:
                    self.resonance_memory[key] = {
                        "coactivation": 0.0,
                        "stability":    0.0,
                        "last_update":  time.time(),
                    }
                memory = self.resonance_memory[key]
                memory["coactivation"] += ai * aj * 0.01
                memory["stability"]     = float(np.tanh(memory["coactivation"]))
                memory["last_update"]   = time.time()

    # =================================================================
    # NEW — FIELD CONSOLIDATION  (Step 8)
    # =================================================================

    def _rebucket_all(self) -> None:
        """
        Semantic rebucketing (#1 partial fix):
        Blends the raw vector signature with a learned resonance signature
        so that topology slowly drifts toward experienced co-activation geometry.

            effective_sig = α * vector_sig + β * learned_resonance_sig

        α=0.8, β=0.2 by default: semantics dominate, experience modulates.
        """
        self.neighborhood_index.clear()
        self._shard_bucket.clear()

        ALPHA = 0.8   # weight of raw vector signature
        BETA  = 0.2   # weight of learned resonance drift

        for sid, node in self.field_nodes.items():
            fv       = node["field_vec"]
            vec_sig  = np.sign(fv[:128]).astype(np.float32)

            # Build learned resonance signature: mean of co-active neighbors' sigs
            learned_sig = np.zeros(128, dtype=np.float32)
            neighbor_count = 0
            for (a, b), mem in self.resonance_memory.items():
                if sid not in (a, b) or mem["stability"] < 0.01:
                    continue
                other = b if a == sid else a
                if other in self.field_nodes:
                    learned_sig += np.sign(
                        self.field_nodes[other]["field_vec"][:128]
                    ).astype(np.float32) * mem["stability"]
                    neighbor_count += 1

            if neighbor_count > 0:
                learned_sig /= neighbor_count

            effective_sig = (ALPHA * vec_sig + BETA * learned_sig).astype(np.int8)
            bucket_id = int(hashlib.md5(effective_sig.tobytes()).hexdigest()[:8], 16)
            self._shard_bucket[sid] = bucket_id
            if bucket_id not in self.neighborhood_index:
                self.neighborhood_index[bucket_id] = []
            self.neighborhood_index[bucket_id].append(sid)

        log_service(logger, f"Semantic rebucketing complete. Buckets: {len(self.neighborhood_index)}", "debug")

    def consolidate_field_memory(self) -> None:
        """
        Slow long-term stabilization + SYNAPTIC PRUNING (fix #4) + REBUCKETING (fix #6).

        Operations:
        1. Decay all coactivation scores slowly (semantic plasticity / forgetting).
        2. Prune pairs below stability threshold — biological synapse elimination.
        3. Every rebucket_interval cycles: recompute locality signatures.
        """
        PRUNE_THRESHOLD = 0.001
        to_delete = []
        for key, memory in self.resonance_memory.items():
            memory["coactivation"] *= 0.999
            memory["stability"]     = float(np.tanh(memory["coactivation"]))
            if memory["stability"] < PRUNE_THRESHOLD:
                to_delete.append(key)

        for key in to_delete:
            del self.resonance_memory[key]

        if to_delete:
            log_service(logger, f"Synaptic pruning: removed {len(to_delete)} weak resonance pairs.", "debug")

        # Periodic rebucketing (#6)
        self._consolidation_tick += 1
        if self._consolidation_tick % self.rebucket_interval == 0:
            self._rebucket_all()

    # =================================================================
    # NEW — SEMANTIC NEIGHBORHOOD  (Step 5)
    # =================================================================

    def get_semantic_neighborhood(self, shard_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Return the strongest learned co-activation neighbors of a shard.
        This is emergent semantic geography — NOT vector similarity.
        NOT embeddings. REAL learned topology from experience.
        """
        neighbors = []
        for (a, b), memory in self.resonance_memory.items():
            if shard_id not in (a, b):
                continue
            other = b if a == shard_id else a
            neighbors.append({
                "neighbor":     other,
                "stability":    memory["stability"],
                "coactivation": memory["coactivation"],
            })
        neighbors.sort(key=lambda x: x["stability"], reverse=True)
        return neighbors[:top_k]

    # =================================================================
    # STAGE 6 SCAFFOLD — COMPETITIVE ATTENTION GATE  (#3)
    # =================================================================

    def attention_gate(
        self,
        query_vec: np.ndarray,
        salience_k: int = 10,
    ) -> Dict[str, float]:
        """
        Competitive Attention Gate (Stage 6 scaffold).

        Selectively amplifies shards whose field_vec best aligns with a
        task-specific query vector. Acts as a top-down attentional prior
        that shapes which resonance assemblies form.

        Returns shard_id → salience_weight map.
        This DOES NOT propagate; call excite() with scaled energy instead.
        """
        if not self.field_nodes:
            return {}

        q_norm = np.linalg.norm(query_vec)
        if q_norm < self.EPSILON:
            return {}

        salience: Dict[str, float] = {}
        for sid, node in self.field_nodes.items():
            fv   = node["field_vec"]
            norm = np.linalg.norm(fv)
            if norm < self.EPSILON:
                continue
            salience[sid] = float(np.dot(query_vec, fv) / (q_norm * norm))

        top = sorted(salience.items(), key=lambda x: x[1], reverse=True)[:salience_k]
        return dict(top)

    # =================================================================
    # STAGE 6 SCAFFOLD — HIERARCHICAL META-SHARD  (#4)
    # =================================================================

    def form_meta_shard(
        self,
        assembly: List[Dict[str, Any]],
        meta_id: str,
    ) -> Optional[np.ndarray]:
        """
        Hierarchical Assembly Formation (Stage 6 scaffold).

        Bundles an emergent resonance assembly into a single new meta-shard
        vector. The assembly becomes a concept — recursive abstraction.

            meta_vec = normalize(sum(activation_i * field_vec_i))

        Returns the meta field_vec (caller must call register_shard to add it).
        """
        if not assembly:
            return None

        meta_vec = np.zeros_like(
            next(iter(self.field_nodes.values()))["field_vec"]
        )
        total_weight = 0.0
        for node in assembly:
            sid = node["shard_id"]
            if sid not in self.field_nodes:
                continue
            w         = node["activation"]
            meta_vec += w * self.field_nodes[sid]["field_vec"]
            total_weight += w

        if total_weight < self.EPSILON:
            return None

        meta_vec /= total_weight
        norm = np.linalg.norm(meta_vec)
        if norm > self.EPSILON:
            meta_vec /= norm

        log_service(logger, f"Meta-shard '{meta_id}' formed from {len(assembly)} assembly nodes.", "info")
        return meta_vec.astype(np.float32)

    # =================================================================
    # 7.  FULL COGNITIVE TICK — tick()  (Step 9 reordering)
    # =================================================================

    def tick(self) -> None:
        """
        Run one complete cognitive cycle.

        ORDER IS CRITICAL (fixes attractor monopolization):
            1. inhibit()                  — suppress before spreading (not after!)
            2. propagate()                — spread semantic waves on pre-inhibited state
            3. update_resonance_memory()  — learn from what survived
            4. consolidate_field_memory() — slow plasticity / forgetting
            5. decay()                    — fade energy + Kuramoto phase coupling

        Original order (propagate->inhibit) let dominant basins amplify first,
        then inhibition acted too late. Now inhibition shapes what can spread.
        """
        self.inhibit()
        self.propagate()
        self.update_resonance_memory()
        self.consolidate_field_memory()
        self.decay()

    # =================================================================
    # 8.  EMERGENT RETRIEVAL — get_active_assembly()
    # =================================================================

    def get_active_assembly(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Return the emergent cognitive resonance assembly —
        the set of shards currently most activated.

        Instead of nearest-neighbor retrieval, this returns what the
        field has organically converged on through wave dynamics.

        Parameters
        ----------
        top_k : maximum assembly members to return

        Returns
        -------
        List of dicts sorted by activation descending:
            [{ "shard_id", "activation", "resonance_to_peak", "meta" }]
        """
        # Continuous field — include ALL nodes with any energy (no hard threshold)
        active = [
            (sid, act)
            for sid, act in self.activation_state.items()
            if act > 0.0
        ]

        if not active:
            return []

        # Sort by activation (highest first)
        active.sort(key=lambda x: x[1], reverse=True)
        assembly_ids = active[:top_k]

        # Peak shard is the dominant anchor of the assembly
        peak_id = assembly_ids[0][0]

        result = []
        for sid, act in assembly_ids:
            r_to_peak = self.resonance(sid, peak_id) if sid != peak_id else 1.0
            result.append({
                "shard_id":         sid,
                "activation":       round(act, 6),
                "resonance_to_peak": round(r_to_peak, 6),
                "meta":             self.field_nodes[sid]["meta"],
            })

        return result

    # =================================================================
    # 9.  FULL EXCITE + PROPAGATE PIPELINE — run_query()
    # =================================================================

    def run_query(
        self,
        query_shard_id: str,
        energy: float = 1.0,
        ticks: Optional[int] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        High-level API: excite a shard, run cognitive propagation,
        and return the emergent assembly.

        Parameters
        ----------
        query_shard_id : the shard representing the query stimulus
        energy         : initial excitation amplitude
        ticks          : number of cognitive cycles (default: propagation_depth)
        top_k          : assembly size to return

        Returns
        -------
        Emergent resonance assembly (see get_active_assembly)
        """
        ticks = ticks or self.propagation_depth

        # Reset all activations for a clean query run
        self.reset_activations()

        # Inject stimulus
        self.excite(query_shard_id, energy=energy)

        # Run cognitive dynamics
        for _ in range(ticks):
            self.tick()

        return self.get_active_assembly(top_k=top_k)

    # =================================================================
    # 10.  UTILITY
    # =================================================================

    def field_snapshot(self) -> Dict[str, float]:
        """Return a snapshot of all current activation levels."""
        return dict(self.activation_state)

    def node_count(self) -> int:
        return len(self.field_nodes)


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    import sys
    setup_python_path()

    from backend.services.loom_service.substrate.HyperVectorCreation import HyperVectorEngine

    print("=" * 60)
    print("STAGE 3→4 — ResonanceFieldEngine v2.0 Validation")
    print("=" * 60)

    hdc = HyperVectorEngine(dimension=1000)
    rfe = ResonanceFieldEngine(decay_rate=0.95, inhibition_strength=0.1, propagation_depth=5)

    # --- Create synthetic shards ---
    concepts_set = {
        "s_god":        {"god": 0.6, "creation": 0.4},
        "s_earth":      {"earth": 0.5, "nature": 0.3, "ground": 0.2},
        "s_understand": {"understand": 0.5, "knowledge": 0.3, "wisdom": 0.2},
        "s_creation":   {"creation": 0.6, "god": 0.2, "origin": 0.2},
        "s_physics":    {"physics": 0.5, "energy": 0.3, "matter": 0.2},
    }

    for shard_id, phis in concepts_set.items():
        concepts = list(phis.keys())
        field_vec, _ = hdc.reconstruct_from_recipe(phis)
        rfe.register_shard(
            shard_id,
            field_vec,
            meta={"concepts": concepts, "phi": phis}
        )

    print(f"\nField populated: {rfe.node_count()} nodes registered.")

    # --- Run query ---
    print("\nQuery: excite 's_god' and observe resonance propagation...")
    assembly = rfe.run_query("s_god", energy=1.0, ticks=5, top_k=5)

    print("\n--- Emergent Resonance Assembly ---")
    for rank, node in enumerate(assembly, 1):
        print(
            f"  #{rank}  {node['shard_id']:<20}  "
            f"activation={node['activation']:.6f}  "
            f"resonance_to_peak={node['resonance_to_peak']:.6f}"
        )

    print("\n--- Semantic Neighborhood of 's_god' (learned topology) ---")
    neighborhood = rfe.get_semantic_neighborhood("s_god", top_k=5)
    for n in neighborhood:
        print(f"  neighbor={n['neighbor']:<20}  stability={n['stability']:.6f}")

    print("\nValidation complete. Stage 5 operational 🔥")
