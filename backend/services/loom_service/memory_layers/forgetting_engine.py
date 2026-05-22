from typing import List, Dict, Any
import math

class ForgettingEngine:
    """
    Decays weights/strengths of memory link connections over elapsed time steps.
    """
    def __init__(self):
        pass

    def apply_decay(self, memory_links: List[Dict[str, Any]], elapsed_ticks: int, decay_rate: float = 0.05) -> List[Dict[str, Any]]:
        """
        Applies exponential decay: strength = strength * e^(-decay_rate * elapsed_ticks).
        """
        decay_factor = math.exp(-decay_rate * elapsed_ticks)
        decayed_links = []
        
        for link in memory_links:
            new_link = dict(link)
            old_strength = link.get("strength", 1.0)
            new_link["strength"] = old_strength * decay_factor
            decayed_links.append(new_link)
            
        return decayed_links
