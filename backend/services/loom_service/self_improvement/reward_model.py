from typing import Dict, Any

class RewardModel:
    """
    Computes performance/utility reward returns for completed goals.
    """
    def __init__(self):
        pass

    def evaluate_reward(self, goal_accomplished: bool, resources_spent: Dict[str, Any]) -> float:
        """
        Calculates a reward score (higher is better).
        Formula: Reward = 1.0 (if goal met) - penalty (based on steps/energy).
        """
        if not goal_accomplished:
            return 0.0

        ticks = resources_spent.get("ticks", 1.0)
        energy_spent = resources_spent.get("energy_spent", 10.0)

        # Penalize excessive time and energy usage
        time_penalty = 0.05 * ticks
        energy_penalty = 0.005 * energy_spent

        reward = 1.0 - time_penalty - energy_penalty
        return max(reward, 0.1)
