import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from backend.config.envConfig import setup_logger, log_service
from backend.config.tunningManagment import tuning_manager
from backend.services.loom_service.cortex.ResonanceField.CorticalRouter import CorticalRouter
from backend.services.loom_service.cortex.ResonanceField.FieldDynamics import (
    compute_wave_step,
    compute_stdp_update,
    divisive_normalization,
    compute_hamiltonian,
    compute_relational_geometry_modulation,
    compute_prediction_error_gradient
)

logger = setup_logger("ResonanceFieldEngine")

class ResonanceFieldEngine:
    """
    ============================================================================
    DOCLOOM — RESONANCE FIELD ENGINE (Upgraded Stage 5 Core)
    ============================================================================
    Main cortical processor that models neural graph activations using physical 
    and cognitive dynamics: wave equations, STDP plasticity, and phase delays.
    ============================================================================
    """

    EPSILON = 1e-8
    MAX_HISTORY = 10

    def __init__(
        self,
        decay_rate: Optional[float] = None,
        inhibition_strength: Optional[float] = None,
        propagation_depth: Optional[int] = None,
        neighborhood_size: Optional[int] = None,
        target_energy: Optional[float] = None,
        phase_dt: Optional[float] = None,
        rebucket_interval: Optional[int] = None,
        wave_gamma: Optional[float] = None,
        wave_c: Optional[float] = None,
        kuramoto_k: Optional[float] = None,
        noise_std: Optional[float] = None,
        divisive_sigma: Optional[float] = None,
        coactivation_factor: Optional[float] = None,
        stdp_tau_plus: Optional[float] = None,
        stdp_tau_minus: Optional[float] = None,
        stdp_a_plus: Optional[float] = None,
        stdp_a_minus: Optional[float] = None,
        prediction_lr: Optional[float] = None,
        max_history: Optional[int] = None,
        max_activation: Optional[float] = None,
        max_prediction_step: Optional[float] = None,
        attention_decay: Optional[float] = None,
        homeostatic_decay: Optional[float] = None,
        max_synaptic_weight: Optional[float] = None,
        spike_threshold: Optional[float] = None
    ):
        self.decay_rate = decay_rate if decay_rate is not None else tuning_manager.get_float("DECAY_RATE", 0.95)
        self.inhibition_strength = inhibition_strength if inhibition_strength is not None else tuning_manager.get_float("INHIBITION_STRENGTH", 0.1)
        self.propagation_depth = propagation_depth if propagation_depth is not None else tuning_manager.get_int("PROPAGATION_DEPTH", 3)
        self.neighborhood_size = neighborhood_size if neighborhood_size is not None else tuning_manager.get_int("NEIGHBORHOOD_SIZE", 32)
        self.target_energy = target_energy if target_energy is not None else tuning_manager.get_float("TARGET_ENERGY", 1.0)
        self.phase_dt = phase_dt if phase_dt is not None else tuning_manager.get_float("PHASE_DT", 0.1)
        self.rebucket_interval = rebucket_interval if rebucket_interval is not None else tuning_manager.get_int("REBUCKET_INTERVAL", 50)
        self.wave_gamma = wave_gamma if wave_gamma is not None else tuning_manager.get_float("WAVE_GAMMA", 0.2)
        self.wave_c = wave_c if wave_c is not None else tuning_manager.get_float("WAVE_C", 0.1)
        self.kuramoto_k = kuramoto_k if kuramoto_k is not None else tuning_manager.get_float("KURAMOTO_K", 0.3)
        self.noise_std = noise_std if noise_std is not None else tuning_manager.get_float("NOISE_STD", 0.002)
        self.divisive_sigma = divisive_sigma if divisive_sigma is not None else tuning_manager.get_float("DIVISIVE_SIGMA", 0.5)
        self.coactivation_factor = coactivation_factor if coactivation_factor is not None else tuning_manager.get_float("COACTIVATION_FACTOR", 0.01)
        self.stdp_tau_plus = stdp_tau_plus if stdp_tau_plus is not None else tuning_manager.get_float("STDP_TAU_PLUS", 5.0)
        self.stdp_tau_minus = stdp_tau_minus if stdp_tau_minus is not None else tuning_manager.get_float("STDP_TAU_MINUS", 5.0)
        self.stdp_a_plus = stdp_a_plus if stdp_a_plus is not None else tuning_manager.get_float("STDP_A_PLUS", 0.05)
        self.stdp_a_minus = stdp_a_minus if stdp_a_minus is not None else tuning_manager.get_float("STDP_A_MINUS", 0.06)
        self.prediction_lr = prediction_lr if prediction_lr is not None else tuning_manager.get_float("PREDICTION_LR", 0.05)
        self.max_history = max_history if max_history is not None else tuning_manager.get_int("MAX_HISTORY", 10)
        self.max_activation = max_activation if max_activation is not None else tuning_manager.get_float("MAX_ACTIVATION", 10.0)
        self.max_prediction_step = max_prediction_step if max_prediction_step is not None else tuning_manager.get_float("MAX_PREDICTION_STEP", 2.0)
        self.attention_decay = attention_decay if attention_decay is not None else tuning_manager.get_float("ATTENTION_DECAY", 0.9)
        self.homeostatic_decay = homeostatic_decay if homeostatic_decay is not None else tuning_manager.get_float("HOMEOSTATIC_DECAY", 0.995)
        self.max_synaptic_weight = max_synaptic_weight if max_synaptic_weight is not None else tuning_manager.get_float("MAX_SYNAPTIC_WEIGHT", 2.0)
        self.spike_threshold = spike_threshold if spike_threshold is not None else tuning_manager.get_float("SPIKE_THRESHOLD", 0.5)
        
        self.global_tick = 0
        self._consolidation_tick = 0
        self._max_memory_per_shard = self.neighborhood_size * 4
        self.dimension = 8000

        self.field_nodes: Dict[str, Dict[str, Any]] = {}
        self.activation_state: Dict[str, float] = {}
        self.resonance_memory: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.cached_resonance: Dict[Tuple[str, str], float] = {}
        self._neighbor_cache: Dict[str, List[str]] = {}
        self.active_attention: Dict[str, float] = {}
        
        self.router = CorticalRouter(neighborhood_size=self.neighborhood_size)

        log_service(
            logger,
            f"ResonanceFieldEngine initialized. λ={self.decay_rate}, K={self.neighborhood_size}",
            "info"
        )

    @property
    def neighborhood_index(self):
        return self.router.neighborhood_index

    @property
    def _shard_bucket(self):
        return self.router.shard_bucket

    def register_shard(self, shard_id: str, field_vec: np.ndarray, meta: Optional[Dict] = None) -> None:
        if shard_id in self.field_nodes:
            log_service(logger, f"Shard '{shard_id}' already registered — skipping.", "warning")
            return

        self.dimension = len(field_vec)

        self.field_nodes[shard_id] = {
            "field_vec": field_vec.astype(np.float32),
            "meta": meta or {},
            "state": {
                "activation": {
                    "amplitude": 0.0,
                    "prev_amplitude": 0.0,
                    "phase": 0.0,
                    "frequency": 1.0,
                    "amp_history": [0.0],
                    "phase_history": [0.0],
                    "phases": {
                        "gamma": 0.0,
                        "theta": 0.0,
                        "alpha": 0.0
                    },
                    "frequencies": {
                        "gamma": 40.0,
                        "theta": 6.0,
                        "alpha": 10.0
                    },
                    "phase_histories": {
                        "gamma": [0.0],
                        "theta": [0.0],
                        "alpha": [0.0]
                    }
                },
                "energy": 0.0,
                "stability": 1.0,
                "decay_rate": self.decay_rate,
                "inhibition": 0.0,
                "last_excited": None,
                "last_spike_tick": None,
            },
        }
        self.activation_state[shard_id] = 0.0
        self.router.add_shard(shard_id, field_vec)
        log_service(logger, f"Shard '{shard_id}' registered into field. Total nodes: {len(self.field_nodes)}", "info")

    def deregister_shard(self, shard_id: str) -> None:
        keys_to_remove = [k for k in self.cached_resonance if shard_id in k]
        for k in keys_to_remove:
            self.cached_resonance.pop(k, None)
        self.router.remove_shard(shard_id)
        self.field_nodes.pop(shard_id, None)
        self.activation_state.pop(shard_id, None)
        log_service(logger, f"Shard '{shard_id}' removed from field.", "info")

    def reset_activations(self) -> None:
        self.active_attention.clear()
        for sid, node in self.field_nodes.items():
            self.activation_state[sid] = 0.0
            state = node["state"]
            state["activation"]["amplitude"] = 0.0
            state["activation"]["prev_amplitude"] = 0.0
            state["activation"]["phase"] = 0.0
            state["activation"]["frequency"] = 1.0
            state["activation"]["amp_history"] = [0.0]
            state["activation"]["phase_history"] = [0.0]
            state["activation"]["phases"] = {"gamma": 0.0, "theta": 0.0, "alpha": 0.0}
            state["activation"]["frequencies"] = {"gamma": 40.0, "theta": 6.0, "alpha": 10.0}
            state["activation"]["phase_histories"] = {"gamma": [0.0], "theta": [0.0], "alpha": [0.0]}
            state["energy"] = 0.0
            state["inhibition"] = 0.0
            state["last_excited"] = None
            state["last_spike_tick"] = None
        log_service(logger, f"Field activations reset. Topology preserved ({len(self.field_nodes)} shards).", "info")

    def apply_global_decay(self, factor: float) -> None:
        for sid in self.field_nodes:
            self.activation_state[sid] *= factor
            self.field_nodes[sid]["state"]["activation"]["amplitude"] *= factor

    def apply_global_scaling(self, factor: float) -> None:
        for sid in self.field_nodes:
            self.activation_state[sid] *= factor
            self.field_nodes[sid]["state"]["activation"]["amplitude"] *= factor

    def resonance(self, shard_i: str, shard_j: str) -> float:
        key = tuple(sorted((shard_i, shard_j)))
        if key in self.cached_resonance:
            return self.cached_resonance[key]

        vi = self.field_nodes[shard_i]["field_vec"]
        vj = self.field_nodes[shard_j]["field_vec"]
        # Normalized Bipolar Dot Product: (vi · vj) / D
        val = float(np.dot(vi.astype(np.int32), vj.astype(np.int32)) / self.dimension)
        self.cached_resonance[key] = val
        return val

    def resonance_matrix(self) -> Dict[Tuple[str, str], float]:
        ids = list(self.field_nodes.keys())
        matrix = {}
        for i, id_i in enumerate(ids):
            for id_j in ids[i + 1:]:
                matrix[(id_i, id_j)] = self.resonance(id_i, id_j)
        return matrix

    def _get_local_neighbors(self, shard_id: str) -> List[str]:
        if shard_id in self._neighbor_cache:
            return self._neighbor_cache[shard_id]
        
        # Ephemeral node handling
        if shard_id == "_temp_query_":
            # For query node, neighbors must be fetched by query signature from router
            # self.field_nodes["_temp_query_"]["field_vec"] stores query_vector
            fv = self.field_nodes["_temp_query_"]["field_vec"]
            sig = np.sign(fv[:128]).astype(np.int8)
            sig[sig == 0] = 1
            return self.router.get_local_neighbors_by_sig(sig, [sid for sid in self.field_nodes if sid != "_temp_query_"])

        neighbors = self.router.get_local_neighbors(shard_id, list(self.field_nodes.keys()))
        
        # Neighborhood bidirectional injection: if _temp_query_ is in field_nodes, inject it into neighbors
        if "_temp_query_" in self.field_nodes:
            # Query's neighborhood
            query_fv = self.field_nodes["_temp_query_"]["field_vec"]
            query_sig = np.sign(query_fv[:128]).astype(np.int8)
            query_sig[query_sig == 0] = 1
            query_neighbors = self.router.get_local_neighbors_by_sig(query_sig, [sid for sid in self.field_nodes if sid != "_temp_query_"])
            if shard_id in query_neighbors:
                if "_temp_query_" not in neighbors:
                    neighbors = list(neighbors) + ["_temp_query_"]

        return neighbors

    def excite(self, shard_id: str, energy: float = 1.0) -> None:
        if shard_id not in self.field_nodes:
            raise KeyError(f"Shard '{shard_id}' not in field. Call register_shard() first.")

        self.activation_state[shard_id] += energy
        state = self.field_nodes[shard_id]["state"]
        prev_amp = state["activation"]["amplitude"]
        state["activation"]["amplitude"] += energy
        state["energy"] += energy
        state["last_excited"] = self.global_tick
        state["activation"]["phase"] = 0.0
        
        # Spike threshold crossing detection
        new_amp = state["activation"]["amplitude"]
        if prev_amp < self.spike_threshold and new_amp >= self.spike_threshold:
            state["last_spike_tick"] = self.global_tick
            
        log_service(logger, f"Excited shard '{shard_id}' with energy={energy:.4f}", "debug")

    def soft_resonance(self, resonance: float) -> float:
        return float(np.tanh(max(0.0, resonance) * 3.0))

    def get_hamiltonian(self) -> float:
        """
        H = \\sum_i \\frac{1}{2}A_i^2 + \\sum_{ij} W_{ij} A_i A_j \\cos(\\phi_i - \\phi_j)
        """
        phases = {sid: n["state"]["activation"]["phase"] for sid, n in self.field_nodes.items()}
        resonances = {}
        for (i, j) in self.resonance_memory:
            if i in self.field_nodes and j in self.field_nodes:
                resonances[tuple(sorted((i, j)))] = self.resonance(i, j)
        return compute_hamiltonian(self.activation_state, self.resonance_memory, phases, resonances)

    def propagate(self) -> None:
        """
        Upgraded wave propagation using:
        \\frac{\\partial^2 \\Psi}{\\partial t^2} = c^2\\nabla^2\\Psi - \\gamma\\frac{\\partial \\Psi}{\\partial t} + F(\\Psi)
        """
        ids = list(self.field_nodes.keys())
        current = dict(self.activation_state)
        delta: Dict[str, float] = {sid: 0.0 for sid in ids}

        for id_i in ids:
            state_i = self.field_nodes[id_i]["state"]
            meta_i = self.field_nodes[id_i]["meta"]
            act_i = state_i["activation"]
            ai = current[id_i]
            prev_ai = act_i.get("prev_amplitude", 0.0)

            # Prior inhibition fatigue
            prior_inhibition = state_i["inhibition"]
            fatigue_factor = math.exp(-prior_inhibition * 2.0)
            effective_ai = ai * fatigue_factor

            neighbors = self._get_local_neighbors(id_i)
            laplacian = 0.0
            predicted_consensus = 0.0
            coupling_weight_sum = 0.0
            
            for id_j in neighbors:
                r_ij = self.resonance(id_i, id_j)
                if r_ij <= 0.0:
                    continue
                
                # Relational Geometry Modulation
                geom_mod = compute_relational_geometry_modulation(meta_i, self.field_nodes[id_j]["meta"])
                
                # Dynamic delay proportional to semantic distance
                dist_ij = 1.0 - r_ij
                delay = max(1, min(self.max_history - 1, int(dist_ij * 5)))
                
                # Fetch delayed neighbor state
                state_j = self.field_nodes[id_j]["state"]
                j_history = state_j["activation"]["amp_history"]
                aj_delayed = j_history[-delay] if len(j_history) >= delay else state_j["activation"]["amplitude"]

                # Phase alignment
                phi_i = act_i["phase"]
                phi_j_history = state_j["activation"]["phase_history"]
                phi_j_delayed = phi_j_history[-delay] if len(phi_j_history) >= delay else state_j["activation"]["phase"]
                
                # Memory boost (directional: neighbor j to current i)
                memory_key = (id_j, id_i)
                memory_boost = 0.0
                if memory_key in self.resonance_memory:
                    memory_boost = self.resonance_memory[memory_key]["stability"] * 0.25
                
                att_i = self.active_attention.get(id_i, 0.0)
                att_j = self.active_attention.get(id_j, 0.0)
                attention_mod = 1.0 + 0.5 * (att_i + att_j)
                
                phase_align = (math.cos(phi_i - phi_j_delayed) + 1.0) * 0.5
                coupling_weight = r_ij * (1.0 + memory_boost) * phase_align * geom_mod * attention_mod
                laplacian += coupling_weight * (aj_delayed - effective_ai)
                
                predicted_consensus += coupling_weight * aj_delayed
                coupling_weight_sum += coupling_weight

            if coupling_weight_sum > self.EPSILON:
                predicted_consensus /= coupling_weight_sum

            # Compute wave propagation
            new_val = compute_wave_step(
                activation=ai,
                prev_activation=prev_ai,
                laplacian=laplacian,
                gamma=self.wave_gamma,
                c=self.wave_c,
                forcing=0.0,
                dt=self.phase_dt
            )
            
            # Surprise / Prediction Error minimization step
            if coupling_weight_sum > self.EPSILON:
                surprise_step = compute_prediction_error_gradient(
                    activation=ai,
                    predicted_consensus=predicted_consensus,
                    learning_rate=self.prediction_lr
                )
                surprise_step *= state_i.get("stability", 1.0)
                surprise_step = float(np.clip(surprise_step, -self.max_prediction_step, self.max_prediction_step))
                new_val = max(0.0, new_val + surprise_step)

            delta[id_i] = new_val - ai

        for sid in ids:
            state = self.field_nodes[sid]["state"]["activation"]
            state["prev_amplitude"] = self.activation_state[sid]
            
            new_act = max(0.0, self.activation_state[sid] + delta[sid] + float(np.random.normal(0.0, self.noise_std)))
            
            # Apply local Divisive Normalization to keep it stable
            neighbor_sum = sum(self.activation_state.get(n, 0.0) for n in self._get_local_neighbors(sid))
            normalized_act = divisive_normalization(new_act, neighbor_sum, sigma=self.divisive_sigma)

            normalized_act = float(np.clip(normalized_act, 0.0, self.max_activation))
            self.activation_state[sid] = normalized_act

        # Global energy conservation
        total_energy = sum(self.activation_state.values())
        if total_energy > self.EPSILON:
            scale = self.target_energy / total_energy
            for sid in ids:
                self.activation_state[sid] *= scale

        for sid in ids:
            node_state = self.field_nodes[sid]["state"]
            state = node_state["activation"]
            prev_act = state["prev_amplitude"]
            final_act = self.activation_state[sid]
            state["amplitude"] = final_act
            
            # Spike threshold crossing detection
            if prev_act < self.spike_threshold and final_act >= self.spike_threshold:
                node_state["last_spike_tick"] = self.global_tick
            
            # Append history
            state["amp_history"].append(final_act)
            if len(state["amp_history"]) > self.max_history:
                state["amp_history"].pop(0)

    def decay(self) -> None:
        """
        Kuramoto synchronization dynamics with propagation delays and multi-frequency bands.
        \\frac{d\\phi_i}{dt} = \\omega_i + \\sum_j K_{ij} \\sin(\\phi_j(t-\\tau_{ij}) - \\phi_i(t))
        """
        tau = self.phase_dt * 2 * math.pi
        
        for sid, node in self.field_nodes.items():
            state = node["state"]
            freq = state["activation"]["frequency"]
            phi_i = state["activation"]["phase"]
            dphi = tau * freq

            ai = self.activation_state[sid]
            neighbors = self._get_local_neighbors(sid)
            
            # Main phase evolution
            if ai > self.EPSILON:
                coupling_sum = 0.0
                weight_sum = 0.0
                
                for nb in neighbors:
                    r_ij = max(0.0, self.resonance(sid, nb))
                    if r_ij > self.EPSILON:
                        dist_ij = 1.0 - r_ij
                        delay = max(1, min(self.max_history - 1, int(dist_ij * 5)))
                        
                        nb_state = self.field_nodes[nb]["state"]
                        phi_j_history = nb_state["activation"]["phase_history"]
                        phi_j_delayed = phi_j_history[-delay] if len(phi_j_history) >= delay else nb_state["activation"]["phase"]
                        
                        coupling_sum += r_ij * math.sin(phi_j_delayed - phi_i)
                        weight_sum += r_ij
                
                att_i = self.active_attention.get(sid, 0.0)
                attention_mod = 1.0 + 0.5 * att_i
                if weight_sum > self.EPSILON:
                    dphi += self.kuramoto_k * attention_mod * (coupling_sum / weight_sum)

            new_phase = (phi_i + dphi) % (2 * math.pi)
            state["activation"]["phase"] = new_phase
            state["activation"]["phase_history"].append(new_phase)
            if len(state["activation"]["phase_history"]) > self.max_history:
                state["activation"]["phase_history"].pop(0)

            # Multi-frequency bands evolution
            if "phases" not in state["activation"]:
                state["activation"]["phases"] = {"gamma": 0.0, "theta": 0.0, "alpha": 0.0}
                state["activation"]["frequencies"] = {"gamma": 40.0, "theta": 6.0, "alpha": 10.0}
                state["activation"]["phase_histories"] = {"gamma": [0.0], "theta": [0.0], "alpha": [0.0]}

            if ai <= self.EPSILON:
                # Option A — Sparse bands: If node is not active, do baseline phase evolution without coupling
                for band in ["gamma", "theta", "alpha"]:
                    band_freq = state["activation"]["frequencies"][band]
                    band_phi = state["activation"]["phases"][band]
                    band_dphi = self.phase_dt * 0.1 * band_freq * 2 * math.pi
                    
                    if band == "gamma":
                        theta_phi = state["activation"]["phases"]["theta"]
                        band_dphi *= (1.0 + 0.3 * math.sin(theta_phi))
                    
                    new_band_phi = (band_phi + band_dphi) % (2 * math.pi)
                    state["activation"]["phases"][band] = new_band_phi
                    state["activation"]["phase_histories"][band].append(new_band_phi)
                    if len(state["activation"]["phase_histories"][band]) > self.max_history:
                        state["activation"]["phase_histories"][band].pop(0)
            else:
                for band in ["gamma", "theta", "alpha"]:
                    band_freq = state["activation"]["frequencies"][band]
                    band_phi = state["activation"]["phases"][band]
                    band_dphi = self.phase_dt * 0.1 * band_freq * 2 * math.pi
                    
                    if band == "gamma":
                        theta_phi = state["activation"]["phases"]["theta"]
                        band_dphi *= (1.0 + 0.3 * math.sin(theta_phi))
                    
                    band_coupling = 0.0
                    band_weight = 0.0
                    for nb in neighbors:
                        r_ij = max(0.0, self.resonance(sid, nb))
                        if r_ij > self.EPSILON:
                            dist_ij = 1.0 - r_ij
                            delay = max(1, min(self.max_history - 1, int(dist_ij * 5)))
                            
                            nb_state = self.field_nodes[nb]["state"]
                            if "phases" in nb_state["activation"] and band in nb_state["activation"]["phases"]:
                                nb_band_phi_hist = nb_state["activation"]["phase_histories"][band]
                                nb_band_phi_delayed = nb_band_phi_hist[-delay] if len(nb_band_phi_hist) >= delay else nb_state["activation"]["phases"][band]
                                
                                band_coupling += r_ij * math.sin(nb_band_phi_delayed - band_phi)
                                band_weight += r_ij
                    
                    att_i = self.active_attention.get(sid, 0.0)
                    attention_mod = 1.0 + 0.5 * att_i
                    if band_weight > self.EPSILON:
                        band_dphi += self.kuramoto_k * attention_mod * (band_coupling / band_weight)
                        
                    new_band_phi = (band_phi + band_dphi) % (2 * math.pi)
                    state["activation"]["phases"][band] = new_band_phi
                    state["activation"]["phase_histories"][band].append(new_band_phi)
                    if len(state["activation"]["phase_histories"][band]) > self.max_history:
                        state["activation"]["phase_histories"][band].pop(0)

    def inhibit(self) -> None:
        ids = list(self.field_nodes.keys())
        current = dict(self.activation_state)
        suppression: Dict[str, float] = {sid: 0.0 for sid in ids}

        for id_i in ids:
            phi_i = self.field_nodes[id_i]["state"]["activation"]["phase"]
            for id_j in self._get_local_neighbors(id_i):
                r_ij = self.resonance(id_i, id_j)
                if r_ij > 0.0:
                    phi_j = self.field_nodes[id_j]["state"]["activation"]["phase"]
                    phase_sync = (math.cos(phi_i - phi_j) + 1.0) * 0.5
                    suppression[id_i] += (r_ij ** 2) * current[id_j] * phase_sync

        for sid in ids:
            overlap_energy = self.inhibition_strength * suppression[sid]
            att_sid = self.active_attention.get(sid, 0.0)
            resistance = math.exp(-att_sid * 2.0)
            overlap_energy *= resistance
            
            self.activation_state[sid] = max(0.0, self.activation_state[sid] - overlap_energy)
            self.field_nodes[sid]["state"]["inhibition"] = overlap_energy
            self.field_nodes[sid]["state"]["activation"]["amplitude"] = self.activation_state[sid]

    def update_resonance_memory(self) -> None:
        """
        STDP (Spike-timing-dependent plasticity) connection learning:
        Updates co-activation memory using the timing difference between node peaks.
        """
        ids = list(self.field_nodes.keys())
        for id_i in ids:
            ai = self.activation_state[id_i]
            if abs(ai) < self.EPSILON:
                continue
            
            last_spike_i = self.field_nodes[id_i]["state"].get("last_spike_tick")
            if last_spike_i is None:
                continue
            t_i = last_spike_i
            
            for id_j in self._get_local_neighbors(id_i):
                aj = self.activation_state[id_j]
                if abs(aj) < self.EPSILON:
                    continue
                
                last_spike_j = self.field_nodes[id_j]["state"].get("last_spike_tick")
                if last_spike_j is None:
                    continue
                t_j = last_spike_j
                
                # Directional link
                key = (id_i, id_j)
                if key not in self.resonance_memory:
                    self.resonance_memory[key] = {
                        "coactivation": 0.0,
                        "stability": 0.0,
                        "last_update": self.global_tick,
                    }
                
                memory = self.resonance_memory[key]
                
                # Apply STDP weight change
                delta_t = float(t_j - t_i)
                stdp_mod = compute_stdp_update(
                    delta_t,
                    tau_plus=self.stdp_tau_plus,
                    tau_minus=self.stdp_tau_minus,
                    a_plus=self.stdp_a_plus,
                    a_minus=self.stdp_a_minus
                )
                
                new_coact = memory["coactivation"] + (ai * aj * self.coactivation_factor) + stdp_mod
                memory["coactivation"] = float(np.clip(new_coact, -10.0, 10.0))
                memory["stability"] = float(np.tanh(memory["coactivation"]))
                memory["last_update"] = self.global_tick

    def consolidate_field_memory(self) -> None:
        PRUNE_THRESHOLD = 0.001
        to_delete = []
        
        # 1. Homeostatic decay
        for key, memory in self.resonance_memory.items():
            new_coact = memory["coactivation"] * self.homeostatic_decay
            memory["coactivation"] = float(np.clip(new_coact, -10.0, 10.0))
            memory["stability"] = float(np.tanh(memory["coactivation"]))

        # 2. Hebbian Normalization & Synaptic Competition
        outgoing_weights: Dict[str, float] = {}
        for (src, dst), memory in self.resonance_memory.items():
            outgoing_weights[src] = outgoing_weights.get(src, 0.0) + abs(memory["coactivation"])

        for (src, dst), memory in self.resonance_memory.items():
            total_budget = outgoing_weights.get(src, 0.0)
            if total_budget > self.max_synaptic_weight:
                scale = self.max_synaptic_weight / total_budget
                memory["coactivation"] *= scale
                memory["stability"] = float(np.tanh(memory["coactivation"]))

        # 3. Magnitude pruning
        for key, memory in self.resonance_memory.items():
            if abs(memory["stability"]) < PRUNE_THRESHOLD:
                to_delete.append(key)

        for key in to_delete:
            del self.resonance_memory[key]

        self._consolidation_tick += 1
        if self._consolidation_tick % self.rebucket_interval == 0:
            self.router.rebucket_all(self.field_nodes, self.resonance_memory)
            self._neighbor_cache.clear()

    def get_semantic_neighborhood(self, shard_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        neighbors = []
        for (a, b), memory in self.resonance_memory.items():
            if shard_id not in (a, b):
                continue
            other = b if a == shard_id else a
            neighbors.append({
                "neighbor": other,
                "stability": memory["stability"],
                "coactivation": memory["coactivation"],
            })
        neighbors.sort(key=lambda x: x["stability"], reverse=True)
        return neighbors[:top_k]

    def attention_gate(self, query_vec: np.ndarray, salience_k: int = 10) -> Dict[str, float]:
        if not self.field_nodes:
            return {}

        q_norm = np.linalg.norm(query_vec)
        if q_norm < self.EPSILON:
            return {}

        salience: Dict[str, float] = {}
        for sid, node in self.field_nodes.items():
            fv = node["field_vec"]
            norm = np.linalg.norm(fv)
            if norm < self.EPSILON:
                continue
            salience[sid] = float(np.dot(query_vec, fv) / (q_norm * norm))

        top = sorted(salience.items(), key=lambda x: x[1], reverse=True)[:salience_k]
        self.active_attention = dict(top)
        return dict(top)

    def form_meta_shard(self, assembly: List[Dict[str, Any]], meta_id: str) -> Optional[np.ndarray]:
        if not assembly:
            return None

        meta_vec = np.zeros_like(next(iter(self.field_nodes.values()))["field_vec"])
        total_weight = 0.0
        for node in assembly:
            sid = node["shard_id"]
            if sid not in self.field_nodes:
                continue
            w = node["activation"]
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

    def tick(self) -> None:
        self.global_tick += 1
        ids = list(self.field_nodes.keys())
        self._neighbor_cache = {
            sid: self._get_local_neighbors(sid)
            for sid in ids
        }
        self.propagate()
        self.decay()
        self.inhibit()
        self.update_resonance_memory()
        self.consolidate_field_memory()
        
        # Decay active attention state
        for sid in list(self.active_attention.keys()):
            self.active_attention[sid] *= self.attention_decay
            if self.active_attention[sid] < self.EPSILON:
                self.active_attention.pop(sid, None)
                
        self._neighbor_cache.clear()

    def get_active_assembly(self, top_k: int = 10) -> List[Dict[str, Any]]:
        active = [(sid, act) for sid, act in self.activation_state.items() if act > 0.0]
        if not active:
            return []

        active.sort(key=lambda x: x[1], reverse=True)
        assembly_ids = active[:top_k]
        peak_id = assembly_ids[0][0]

        result = []
        for sid, act in assembly_ids:
            r_to_peak = self.resonance(sid, peak_id) if sid != peak_id else 1.0
            result.append({
                "shard_id": sid,
                "activation": round(act, 6),
                "resonance_to_peak": round(r_to_peak, 6),
                "meta": self.field_nodes[sid]["meta"],
            })

        return result

    def run_query(
        self,
        query_vector: np.ndarray,
        energy: float = 1.0,
        ticks: Optional[int] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        ticks = ticks or self.propagation_depth
        self.reset_activations()

        # 1. Immediately find neighborhood nodes for the query vector using its signature
        query_sig = np.sign(query_vector[:128]).astype(np.int8)
        query_sig[query_sig == 0] = 1
        local_neighbors = self.router.get_local_neighbors_by_sig(query_sig, list(self.field_nodes.keys()))

        if not local_neighbors:
            # Force a physics collapse
            return []

        # 2. Orthogonal Deflection Check: Calculate normalized bipolar dot product against local neighborhood nodes
        above_threshold = False
        for nb in local_neighbors:
            nb_vec = self.field_nodes[nb]["field_vec"]
            res_val = float(np.dot(query_vector.astype(np.int32), nb_vec.astype(np.int32)) / self.dimension)
            if res_val >= 0.05:
                above_threshold = True

        if not above_threshold:
            # Force a physics collapse: wave dies instantly
            self.reset_activations()
            return []

        # 3. Ephemeral Query Injection: dynamically register query node ID _temp_query_
        self.field_nodes["_temp_query_"] = {
            "field_vec": query_vector.astype(np.float32),
            "meta": {"type": "query"},
            "state": {
                "activation": {
                    "amplitude": 0.0,
                    "prev_amplitude": 0.0,
                    "phase": 0.0,
                    "frequency": 1.0,
                    "amp_history": [0.0],
                    "phase_history": [0.0],
                    "phases": {"gamma": 0.0, "theta": 0.0, "alpha": 0.0},
                    "frequencies": {"gamma": 40.0, "theta": 6.0, "alpha": 10.0},
                    "phase_histories": {"gamma": [0.0], "theta": [0.0], "alpha": [0.0]}
                },
                "energy": 0.0,
                "stability": 1.0,
                "decay_rate": self.decay_rate,
                "inhibition": 0.0,
                "last_excited": None,
                "last_spike_tick": None,
            }
        }
        self.activation_state["_temp_query_"] = 0.0
        self.router.add_shard("_temp_query_", query_vector)

        # Excite query node with energy
        self.excite("_temp_query_", energy=energy)

        # Run the clock tick loops
        for _ in range(ticks):
            self.tick()

        # Get active assembly (excluding the temp query itself from output)
        assembly = self.get_active_assembly(top_k=top_k + 1)
        assembly = [item for item in assembly if item["shard_id"] != "_temp_query_"][:top_k]

        # Clean up ephemeral query node
        self.deregister_shard("_temp_query_")

        return assembly

    def field_snapshot(self) -> Dict[str, float]:
        return dict(self.activation_state)

    def node_count(self) -> int:
        return len(self.field_nodes)
