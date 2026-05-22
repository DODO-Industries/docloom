from typing import Dict, Any

class PersonalityField:
    """
    Simulates a multi-dimensional personality vector that influences
    cognitive thresholds, exploratory bias, and risk tolerance.
    """
    def __init__(self, curiosity: float = 0.8, caution: float = 0.5, diligence: float = 0.9):
        self.curiosity = curiosity  # Modulates exploration mode triggers
        self.caution = caution      # Modulates planning depth and safety threshold
        self.diligence = diligence  # Modulates retry counts and search width
        
    def get_behavioral_biases(self) -> Dict[str, float]:
        """Translates personality traits into specific numerical biases."""
        return {
            "exploration_threshold_offset": float((self.curiosity - 0.5) * 0.2),
            "safety_strictness_multiplier": float(self.caution * 1.5),
            "max_planning_rollouts": int(self.diligence * 15),
            "retry_limit": int(self.diligence * 5)
        }
        
    def adapt_personality(self, history_metrics: Dict[str, Any]):
        """Slightly shifts traits based on success/failure history (slow learning)."""
        failure_rate = 1.0 - history_metrics.get("success_rate", 1.0)
        
        # If failure rate is high, increase caution
        if failure_rate > 0.4:
            self.caution = min(1.0, self.caution + 0.05)
            self.curiosity = max(0.1, self.curiosity - 0.05)
        else:
            self.curiosity = min(1.0, self.curiosity + 0.01)
            self.caution = max(0.1, self.caution - 0.01)
