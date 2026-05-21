from typing import List, Dict, Any
import random

class StrategyEvolution:
    """
    Simulates genetic parameter tuning of decision heuristics over system evaluations.
    """
    def __init__(self):
        pass

    def mutate_parameters(self, parameters: Dict[str, float], mutation_rate: float = 0.1) -> Dict[str, float]:
        """Mutates float attributes by adding small random offsets to search better settings."""
        mutated = {}
        for key, val in parameters.items():
            if isinstance(val, (int, float)):
                change = random.gauss(0, mutation_rate)
                mutated[key] = max(0.01, min(val + change, 1.0))
            else:
                mutated[key] = val
        return mutated
