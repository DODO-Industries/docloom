import math
from typing import Dict, Tuple

# =============================================================================
# COGNITIVE FIELD DYNAMICS FORMULAS
# =============================================================================

def compute_wave_step(
    activation: float,
    prev_activation: float,
    laplacian: float,
    gamma: float,
    c: float,
    forcing: float,
    dt: float = 0.1
) -> float:
    """
    Damped Semantic Wave Equation:
    \\frac{\\partial^2 \\Psi}{\\partial t^2} = c^2\\nabla^2\\Psi - \\gamma\\frac{\\partial \\Psi}{\\partial t} + F(\\Psi)
    """
    damping_term = gamma * dt * (activation - prev_activation)
    propagation_term = (dt ** 2) * ((c ** 2) * laplacian + forcing)
    new_activation = 2.0 * activation - prev_activation + propagation_term - damping_term
    return float(max(0.0, new_activation))


def compute_stdp_update(
    delta_t: float,
    tau_plus: float = 5.0,
    tau_minus: float = 5.0,
    a_plus: float = 0.05,
    a_minus: float = 0.06
) -> float:
    """
    Spike Timing Dependent Plasticity (STDP):
    \\Delta w_{ij} = A_+ e^{-\\Delta t/\\tau_+} \\quad (\\Delta t > 0)
    \\Delta w_{ij} = -A_- e^{\\Delta t/\\tau_-} \\quad (\\Delta t < 0)
    """
    if delta_t > 0:
        return a_plus * math.exp(-delta_t / tau_plus)
    elif delta_t < 0:
        return -a_minus * math.exp(delta_t / tau_minus)
    return 0.0


def divisive_normalization(
    activation: float,
    total_neighbor_activity: float,
    sigma: float = 0.5
) -> float:
    """
    Divisive Normalization:
    A_i' = \\frac{A_i}{\\sigma + \\sum_j W_{ij}A_j}
    """
    denom = max(sigma + total_neighbor_activity, 1e-6)
    return activation / denom


def compute_hamiltonian(
    activations: Dict[str, float],
    weights: Dict[Tuple[str, str], Dict[str, float]],
    phases: Dict[str, float],
    resonances: Dict[Tuple[str, str], float]
) -> float:
    """
    Cognitive Hamiltonian:
    H = \\sum_i \\frac{1}{2}A_i^2 + \\sum_{ij} W_{ij} A_i A_j \\cos(\\phi_i - \\phi_j)
    Uses pre-computed Normalized Bipolar Dot Product Resonance values (R_ij).
    """
    h_self = 0.5 * sum(a ** 2 for a in activations.values())
    h_int = 0.0
    for (i, j), mem in weights.items():
        if i in activations and j in activations:
            a_i = activations[i]
            a_j = activations[j]
            phi_i = phases.get(i, 0.0)
            phi_j = phases.get(j, 0.0)
            
            # Lookup pre-computed resonance
            key = tuple(sorted((i, j)))
            r_ij = resonances.get(key, 0.0)
            
            w_ij = mem.get("stability", 0.0)
            # Combine stability learning weight and base resonance
            h_int += w_ij * r_ij * a_i * a_j * math.cos(phi_i - phi_j)
    return h_self + h_int


def compute_relational_geometry_modulation(meta_i: dict, meta_j: dict) -> float:
    """
    Bypassed Document Artifact: Returns a flat scale.
    Proximity is handled by Vector Field Hamming distances, not page coordinates.
    """
    return 1.0


def compute_prediction_error_gradient(
    activation: float,
    predicted_consensus: float,
    learning_rate: float
) -> float:
    """
    Free Energy / Surprise Minimization:
    E = |Prediction - Reality|^2
    \\frac{d\\Psi}{dt} = -\\nabla E
    """
    # Gradient of E = (activation - predicted_consensus)^2 w.r.t activation
    grad = 2.0 * (activation - predicted_consensus)
    return float(-learning_rate * grad)

