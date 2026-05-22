from typing import List, Dict, Any

class AnomalyDetector:
    """
    Scans internal variables (fatigue, pressure, CPU ticks) to identify spikes or infinite execution loops.
    """
    def __init__(self):
        pass

    def check_anomalies(self, state_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Scans variables for threshold crossings or repetitive state dynamics."""
        if not state_history:
            return {"anomaly_detected": False, "reason": None}

        # Check latest state variables
        latest = state_history[-1]
        fatigue = latest.get("fatigue", 0.0)
        cognitive_pressure = latest.get("cognitive_pressure", 0.0)

        if fatigue > 0.9:
            return {
                "anomaly_detected": True,
                "reason": f"Critical fatigue level detected: {fatigue}"
            }

        if cognitive_pressure > 0.95:
            return {
                "anomaly_detected": True,
                "reason": f"Cognitive overload: pressure {cognitive_pressure}"
            }

        # Check for repetitive tick states
        if len(state_history) >= 4:
            recent_states = [s.get("current_mode") for s in state_history[-4:] if s.get("current_mode")]
            if len(recent_states) == 4 and len(set(recent_states)) == 1:
                return {
                    "anomaly_detected": True,
                    "reason": f"System mode locked in loop: {recent_states[0]}"
                }

        return {"anomaly_detected": False, "reason": None}
