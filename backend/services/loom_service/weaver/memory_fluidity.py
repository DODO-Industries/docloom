import math
from typing import Tuple
from backend.config.tunningManagment import tuning_manager

class DynamicMemoryFluidity:
    """
    ============================================================================
    DOCLOOM — DYNAMIC MEMORY FLUIDITY SUBSTRATE (Calculus Decay Framework)
    ============================================================================
    Processes continuous entropy decay utilizing log-buffered friction shields.
    ============================================================================
    """
    def __init__(self, lambda_base: float = 0.05):
        self.lambda_base = lambda_base
        self._is_default = (lambda_base == 0.05)

    def calculate_decay(
        self,
        initial_activation: float,
        last_recalled: float,
        current_time: float,
        hits: int
    ) -> float:
        """
        Calculates exponential continuous decay using real-time deltas.
        Dynamic memory counters slow the decay constant via logarithmic dampening.
        """
        elapsed_time = max(0.0, current_time - last_recalled)
        if elapsed_time == 0.0:
            return float(initial_activation)
        
        # Implement log-buffered field friction to mitigate decay
        if self._is_default:
            memory_decay_rate = tuning_manager.get_float("MEMORY_DECAY_RATE", self.lambda_base)
        else:
            memory_decay_rate = self.lambda_base
        lambda_effective = memory_decay_rate / (1.0 + math.log(1.0 + max(0, hits)))
        
        # Continuum Equation: A(t) = A0 * e^(-λ_eff * Δt)
        decayed_activation = initial_activation * math.exp(-lambda_effective * elapsed_time)
        return float(decayed_activation)

    def reinforce(self, current_time: float, hits: int) -> Tuple[float, int]:
        """
        Updates tracking properties natively within the RAM ledger workspace.
        """
        return float(current_time), hits + 1

