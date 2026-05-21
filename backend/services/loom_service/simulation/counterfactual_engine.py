from typing import Dict, Any

class CounterfactualEngine:
    """
    Computes regret and compares actual reward results to hypothetical alternative choices.
    """
    def __init__(self):
        pass

    def evaluate_regret(self, actual_action: str, actual_reward: float, alternatives: Dict[str, float]) -> Dict[str, Any]:
        """
        Determines if an alternative decision path would yield higher expected returns.
        Regret = Max(Alternative Rewards) - Actual Reward.
        """
        if not alternatives:
            return {"best_alternative": None, "regret": 0.0}

        best_alt = max(alternatives, key=alternatives.get)
        best_reward = alternatives[best_alt]

        regret = max(0.0, best_reward - actual_reward)

        return {
            "best_alternative": best_alt,
            "best_alternative_reward": best_reward,
            "regret": regret,
            "recommended_shift": regret > 0.2
        }
