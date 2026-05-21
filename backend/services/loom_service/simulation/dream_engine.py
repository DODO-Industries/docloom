from typing import List, Dict, Any
import random

class DreamEngine:
    """
    Replays episodic traces with small mutations to form new associations and strengthen connections.
    """
    def __init__(self):
        pass

    def run_dream_cycle(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Selects random episodes, mutates minor properties, and outputs new hypothetical experiences."""
        if not episodes:
            return []

        dream_experiences = []
        sample_size = min(len(episodes), 3)
        sample = random.sample(episodes, sample_size)

        for ep in sample:
            mutated_action = f"dream_{ep.get('action', 'act')}"
            state_after = dict(ep.get("state_after", {}))
            
            # Mutate a state value slightly
            for k, v in state_after.items():
                if isinstance(v, (int, float)):
                    state_after[k] = v * random.uniform(0.8, 1.2)

            dream_experiences.append({
                "action": mutated_action,
                "state_before": ep.get("state_before", {}),
                "state_after": state_after,
                "salience": ep.get("salience", 0.5) * 0.5
            })

        return dream_experiences
