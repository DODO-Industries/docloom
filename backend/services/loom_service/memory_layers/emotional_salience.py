from typing import Dict, Any

class EmotionalSalience:
    """
    Computes salience/excitement values for memory snapshots using execution utility, surprise, or failure signals.
    """
    def __init__(self):
        pass

    def compute_salience(self, outcome: Dict[str, Any]) -> float:
        """
        Determines relative importance score [0.0 - 1.0].
        Factors: error presence (failure surprise), prediction error, reward score.
        """
        base_salience = 0.5

        # Failures/Errors generate high cognitive surprise (salience)
        if outcome.get("error") or not outcome.get("success", True):
            base_salience += 0.4
        
        # High prediction error indicates learning opportunity
        pred_error = outcome.get("prediction_error", 0.0)
        base_salience += 0.1 * min(pred_error, 3.0)

        # High reward/success shifts impact
        reward = outcome.get("reward", 0.0)
        if reward > 0.8:
            base_salience += 0.1

        return min(max(base_salience, 0.0), 1.0)
