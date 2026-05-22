from typing import Dict, Any

class RealityStateTracker:
    """
    Tracks state drift and performs active inference to minimize prediction error
    (free energy) between internal expectations and sensory observations.
    """
    def __init__(self):
        self.prediction_error_score = 0.0

    def compute_prediction_error(self, predicted_state: Dict[str, Any], observed_state: Dict[str, Any]) -> float:
        """
        Computes the discrepancy/distance between the expected state and actual observed state.
        Returns a score representing the total prediction error (free energy).
        """
        error = 0.0
        # Check all predicted attributes
        for key, pred_val in predicted_state.items():
            if key not in observed_state:
                error += 1.0  # Key missing in observation
            else:
                obs_val = observed_state[key]
                if pred_val != obs_val:
                    if isinstance(pred_val, dict) and isinstance(obs_val, dict):
                        # Nested dict differences
                        sub_errors = sum(1.0 for k in pred_val if pred_val.get(k) != obs_val.get(k))
                        error += sub_errors / max(len(pred_val), 1)
                    else:
                        error += 0.5  # Mismatched value
                        
        # Check for unexpected observations
        for key in observed_state:
            if key not in predicted_state:
                error += 0.5

        self.prediction_error_score = error
        return error

    def update_model(self, current_internal_state: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        """Adjusts internal state toward observations to minimize future error."""
        updated_state = dict(current_internal_state)
        for key, obs_val in observation.items():
            # Blend prediction and reality
            updated_state[key] = obs_val
        return updated_state
