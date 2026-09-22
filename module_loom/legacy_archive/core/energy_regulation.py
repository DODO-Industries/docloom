import numpy as np
from typing import Dict, Any

class EnergyRegulation:
    """
    Implements thermodynamic energy regulation. Updates energy level
    and calculates current reasoning budget based on cognitive pressure,
    surprise, and task complexity.
    """
    def __init__(self, base_energy: float = 1.0, pressure_decay: float = 0.85):
        self.energy = base_energy
        self.pressure = 0.0
        self.pressure_decay = pressure_decay
        self.surprise_history = []
        
    def regulate(self, metrics: Dict[str, float]) -> Dict[str, float]:
        entropy = metrics.get("entropy", 0.5)
        coherence = metrics.get("coherence", 0.5)
        surprise = metrics.get("surprise", 0.0)
        
        # Calculate raw cognitive pressure
        raw_pressure = (0.4 * entropy) + (0.3 * (1.0 - coherence)) + (0.3 * surprise)
        self.pressure = (self.pressure * self.pressure_decay) + (raw_pressure * (1.0 - self.pressure_decay))
        
        # Calculate energy drain
        drain = (entropy * 0.15) + (surprise * 0.1) + (self.pressure * 0.05)
        
        # Calculate energy recovery based on coherence
        recovery = (coherence * 0.2)
        
        self.energy = float(np.clip(self.energy + recovery - drain, 0.1, 1.0))
        
        return {
            "energy": self.energy,
            "pressure": self.pressure,
            "drain": drain,
            "recovery": recovery
        }
        
    def get_thinking_budget(self) -> float:
        """Determines the computational/time depth budget allowed for planning/reasoning."""
        # Lower energy or higher pressure restricts thinking budget (focused, greedy searches)
        # Higher energy and moderate pressure allows deep exploration (MCTS width/depth)
        return float(self.energy * (1.0 - self.pressure * 0.5))
