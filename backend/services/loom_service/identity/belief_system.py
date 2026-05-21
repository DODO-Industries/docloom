from typing import Dict, Any, List

class BeliefSystem:
    """
    Manages a dictionary of subjective beliefs and expectations.
    Updates beliefs dynamically using Bayes-like updating or reinforcement.
    """
    def __init__(self):
        # Key -> {"description": str, "probability": float, "reliability": float}
        self.beliefs: Dict[str, Dict[str, Any]] = {
            "gravity_is_active": {"description": "Spacetime is warped by mass, attracting objects.", "probability": 1.0, "reliability": 1.0},
            "user_intent_is_constructive": {"description": "The user seeks optimized code and architectures.", "probability": 0.9, "reliability": 0.8}
        }
        
    def add_or_update_belief(self, key: str, description: str, observation_probability: float, evidence_weight: float = 0.2):
        if key not in self.beliefs:
            self.beliefs[key] = {"description": description, "probability": observation_probability, "reliability": evidence_weight}
        else:
            # Bayes/EMA-like update: posterior = (1 - w) * prior + w * observed
            prior = self.beliefs[key]["probability"]
            posterior = (1.0 - evidence_weight) * prior + evidence_weight * observation_probability
            self.beliefs[key]["probability"] = float(posterior)
            self.beliefs[key]["reliability"] = float(min(1.0, self.beliefs[key]["reliability"] + evidence_weight * 0.1))
            
    def query_belief(self, key: str) -> float:
        """Returns subjective probability of the belief. Returns 0.5 (uncertainty) if unknown."""
        if key in self.beliefs:
            return self.beliefs[key]["probability"]
        return 0.5
        
    def detect_inconsistencies(self) -> List[str]:
        inconsistencies = []
        # Logical check for opposing beliefs
        # For simplicity, if we have "not_X" and "X" both with high probability, alert conflict
        keys = list(self.beliefs.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                k1, k2 = keys[i], keys[j]
                if k1 == f"not_{k2}" or k2 == f"not_{k1}":
                    p1 = self.beliefs[k1]["probability"]
                    p2 = self.beliefs[k2]["probability"]
                    if p1 > 0.6 and p2 > 0.6:
                        inconsistencies.append(f"Contradictory beliefs detected: {k1} (p={p1}) vs {k2} (p={p2})")
        return inconsistencies
