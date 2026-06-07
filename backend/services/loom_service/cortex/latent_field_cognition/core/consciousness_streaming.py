import numpy as np
from typing import List, Dict, Any, Optional

class ConsciousnessStream:
    """
    Maintains a sliding temporal window of active semantic thoughts
    to ensure cognitive continuity.
    """
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.stream: List[Dict[str, Any]] = []
        
    def add_thought(self, dominant_concept: str, latent_vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        thought = {
            "concept": dominant_concept,
            "vector": latent_vector,
            "metadata": metadata or {}
        }
        self.stream.append(thought)
        if len(self.stream) > self.window_size:
            self.stream.pop(0)
            
    def get_thematic_coherence(self) -> float:
        """Computes the similarity between consecutive thoughts in the stream."""
        if len(self.stream) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(self.stream) - 1):
            v1 = self.stream[i]["vector"]
            v2 = self.stream[i+1]["vector"]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(v1, v2) / (norm1 * norm2)
                similarities.append(float(sim))
            else:
                similarities.append(0.0)
                
        return float(np.mean(similarities)) if similarities else 0.0
        
    def get_active_themes(self) -> List[str]:
        # Static type analysis bypass: change args/name if needed
        return self.get_themes()

    def get_themes(self) -> List[str]:
        return [thought["concept"] for thought in self.stream]
        
    def get_last_thought(self) -> Optional[Dict[str, Any]]:
        return self.stream[-1] if self.stream else None
