from typing import Dict, Any, List

class AdaptationEngine:
    """
    Dynamically adjusts memory decay rate, search depth, and focus threshold parameters.
    """
    def __init__(self):
        self.reward_history: List[float] = []

    def adapt_hyperparameters(self, last_reward: float, parameters: Dict[str, float]) -> Dict[str, float]:
        """
        Adjusts parameters like 'decay_rate' or 'focus_threshold' based on reward trend.
        If reward is declining, we scale focus and adjust decay rates.
        """
        self.reward_history.append(last_reward)
        if len(self.reward_history) > 10:
            self.reward_history.pop(0)

        adapted_params = dict(parameters)
        if len(self.reward_history) >= 2:
            avg_past = sum(self.reward_history[:-1]) / len(self.reward_history[:-1])
            # If performance dropped, adjust focus/exploration settings
            if last_reward < avg_past:
                # Performance dropped -> increase search depth and focus threshold to think more carefully
                if "search_depth" in adapted_params:
                    adapted_params["search_depth"] = min(adapted_params["search_depth"] * 1.5, 100.0)
                if "focus_threshold" in adapted_params:
                    adapted_params["focus_threshold"] = min(adapted_params["focus_threshold"] * 1.1, 1.0)
            else:
                # Performance improved -> slightly lower focus threshold for faster routing
                if "focus_threshold" in adapted_params:
                    adapted_params["focus_threshold"] = max(adapted_params["focus_threshold"] * 0.95, 0.1)

        return adapted_params
