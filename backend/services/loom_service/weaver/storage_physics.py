import numpy as np
import math
from backend.config.tunningManagment import tuning_manager

class LatentFieldPhysicsEngine:
    """
    ============================================================================
    DOCLOOM — LATENT FIELD PHYSICS ENGINE (The Active Connections Layer)
    ============================================================================
    Computes Momentum Vectors by warping seed scaffolds with conceptual mass,
    calculates pointerless gravitational attraction between shards, and models
    causal wormhole matrix transformations to distort memory space.
    ============================================================================
    """
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        # Causal wormhole tensor overlay (Identity by default)
        self.wormhole_tensor = np.eye(self.dimension, dtype=np.float32)

    def set_wormhole_tensor(self, tensor: np.ndarray) -> None:
        """
        Sets a transformation matrix that bends semantic space during recall.
        """
        if tensor.shape != (self.dimension, self.dimension):
            raise ValueError(f"Wormhole tensor must have shape ({self.dimension}, {self.dimension})")
        self.wormhole_tensor = tensor.copy().astype(np.float32)

    def reset_wormhole(self) -> None:
        """
        Resets the wormhole tensor to the Identity matrix.
        """
        self.wormhole_tensor = np.eye(self.dimension, dtype=np.float32)

    def calculate_momentum_vector(
        self,
        true_vector: np.ndarray,
        seed_scaffold: np.ndarray,
        mass: float
    ) -> np.ndarray:
        """
        Mass vs Seed Scaffold warping:
        Mass warps the seed scaffold position towards the true content vector.
        """
        scaffold_warp = tuning_manager.get_float("SCAFFOLD_WARP", 1.0)
        warp_factor = mass / (scaffold_warp + max(0.0, mass))
        momentum = seed_scaffold + warp_factor * (true_vector - seed_scaffold)
        
        # Normalize to keep vector representation stable
        norm = np.linalg.norm(momentum)
        if norm > 1e-8:
            momentum /= norm
        else:
            # Fallback to normalized true_vector or seed_scaffold
            norm_true = np.linalg.norm(true_vector)
            if norm_true > 1e-8:
                momentum = true_vector / norm_true
            else:
                norm_scaffold = np.linalg.norm(seed_scaffold)
                if norm_scaffold > 1e-8:
                    momentum = seed_scaffold / norm_scaffold
        return momentum.astype(np.float32)

    def calculate_gravitational_attraction(
        self,
        vector_a: np.ndarray,
        mass_a: float,
        vector_b: np.ndarray,
        mass_b: float,
        apply_wormhole: bool = True
    ) -> float:
        """
        Pointerless Field Edges:
        Calculates the gravitational pull between two shards based on their masses and distance.
        """
        v_a = vector_a
        v_b = vector_b
        
        if apply_wormhole:
            # Warp vectors through the causal wormhole tensor
            v_a = np.dot(self.wormhole_tensor, v_a)
            v_b = np.dot(self.wormhole_tensor, v_b)

        # Distance in warped space
        norm_a = np.linalg.norm(v_a)
        norm_b = np.linalg.norm(v_b)
        
        if norm_a == 0 or norm_b == 0:
            cosine_dist = 1.0
        else:
            cosine_sim = float(np.dot(v_a, v_b) / (norm_a * norm_b))
            cosine_dist = 1.0 - cosine_sim

        # Gravitational force formula: G = (m_A * m_B) / (distance^2 + epsilon)
        epsilon = tuning_manager.get_float("GRAVITY_DENOM_OFFSET", 1e-4)
        distance_sq = max(cosine_dist ** 2, epsilon)
        grav_pull = (mass_a * mass_b) / distance_sq
        return float(grav_pull)

    def calculate_excitation_gain(
        self, 
        hdc_similarity: float, 
        query_phase: float, 
        cell_phase: float,
        eta: float = 0.4
    ) -> float:
        """
        Calculates localized energy spikes using wave mechanics.
        Gain scales based on phase proximity.
        """
        phase_delta = query_phase - cell_phase
        wave_excitation_gain = tuning_manager.get_float("WAVE_EXCITATION_GAIN", eta)
        return float(wave_excitation_gain * hdc_similarity * math.cos(phase_delta))

    def compute_kuramoto_coupling(
        self, 
        active_phases: np.ndarray, 
        active_activations: np.ndarray, 
        current_phase: float,
        coupling_k: float = 0.1
    ) -> float:
        """
        Models phase interactions across active concepts on the fly.
        """
        if len(active_phases) == 0:
            return 0.0
        phase_deltas = active_phases - current_phase
        attraction_forces = active_activations * np.sin(phase_deltas)
        kuramoto_coupling = tuning_manager.get_float("KURAMOTO_COUPLING", coupling_k)
        return float((kuramoto_coupling / len(active_phases)) * np.sum(attraction_forces))
