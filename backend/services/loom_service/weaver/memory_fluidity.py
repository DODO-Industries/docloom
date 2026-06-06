import math
from typing import Tuple

class DynamicMemoryFluidity:
    """
    ============================================================================
    DOCLOOM — DYNAMIC MEMORY FLUIDITY LOOP (Runtime Decay & Recall)
    ============================================================================
    Manages spatiotemporal decay (entropy) of shard activations and retrieval 
    reinforcement (plasticity) utilizing a hit-based immune shield.
    ============================================================================
    """
    def __init__(self, lambda_base: float = 0.05):
        self.lambda_base = lambda_base

    def calculate_decay(
        self,
        initial_activation: float,
        last_recalled: float,
        current_time: float,
        hits: int
    ) -> float:
        """
        Spatiotemporal Degradation:
        Decays activation level exponentially over elapsed time.
        Retrieval Reinforcement:
        Frequent hits grant an immune shield that slows down future decay rate.
        """
        elapsed_time = max(0.0, current_time - last_recalled)
        
        # Effective decay rate decreases as hits increase
        lambda_effective = self.lambda_base / (1.0 + max(0, hits))
        
        # Exponential decay: A(t) = A0 * e^(-lambda * t)
        decayed_activation = initial_activation * math.exp(-lambda_effective * elapsed_time)
        return float(decayed_activation)

    def reinforce(self, current_time: float, hits: int) -> Tuple[float, int]:
        """
        When a thought is recalled, reset its last_recalled timestamp and increment hits.
        Returns the updated (last_recalled, hits) tuple.
        """
        return float(current_time), hits + 1
