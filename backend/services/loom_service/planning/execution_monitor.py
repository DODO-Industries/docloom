from typing import Dict, Any, List

class ExecutionMonitor:
    """
    Compares the predicted outcome of plan steps against actual feedback
    observations. Alerts the system if drift exceeds acceptable tolerances.
    """
    def __init__(self, drift_tolerance: float = 0.3):
        self.drift_tolerance = drift_tolerance
        
    def check_drift(self, predicted_state: Dict[str, Any], observed_state: Dict[str, Any]) -> Dict[str, Any]:
        discrepancies = []
        drift_score = 0.0
        
        for key, pred_val in predicted_state.items():
            if key not in observed_state:
                discrepancies.append(f"Missing state attribute: {key}")
                drift_score += 0.5
            else:
                obs_val = observed_state[key]
                if isinstance(pred_val, (int, float)) and isinstance(obs_val, (int, float)):
                    diff = abs(pred_val - obs_val)
                    if diff > self.drift_tolerance:
                        discrepancies.append(f"Value drift for {key}: expected {pred_val}, got {obs_val}")
                        drift_score += diff
                elif pred_val != obs_val:
                    discrepancies.append(f"Categorical drift for {key}: expected {pred_val}, got {obs_val}")
                    drift_score += 0.4
                    
        return {
            "drift_detected": drift_score > self.drift_tolerance,
            "drift_score": float(drift_score),
            "discrepancies": discrepancies
        }
