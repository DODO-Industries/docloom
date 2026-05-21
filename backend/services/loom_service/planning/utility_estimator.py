from typing import Dict, Any

class UtilityEstimator:
    """
    Computes mathematical expectations of actions based on reward, risk,
    energy cost, and probability of success.
    
    Formula:
      Expected Utility = P(Success) * Reward - P(Failure) * Risk - EnergyCost
    """
    def __init__(self):
        pass
        
    def estimate_utility(self, parameters: Dict[str, float]) -> float:
        p_success = parameters.get("p_success", 0.8)
        reward = parameters.get("reward", 1.0)
        risk = parameters.get("risk", 0.2)
        energy_cost = parameters.get("energy_cost", 0.1)
        
        p_failure = 1.0 - p_success
        expected_utility = (p_success * reward) - (p_failure * risk) - energy_cost
        return float(expected_utility)
