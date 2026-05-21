from typing import Dict, Any

class HypotheticalReasoner:
    """
    Evaluates conditional hypothesis scenarios ("If X was true instead of Y, does goal Z succeed?").
    """
    def __init__(self):
        pass

    def evaluate_hypothesis(self, base_state: Dict[str, Any], hyp_override: Dict[str, Any], test_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Runs validation of conditions over a modified hypothetical state base."""
        hypothetical_state = dict(base_state)
        hypothetical_state.update(hyp_override)

        success = True
        failures = []

        for key, val in test_conditions.items():
            if hypothetical_state.get(key) != val:
                success = False
                failures.append(key)

        return {
            "hypothesis_valid": success,
            "failed_variables": failures,
            "result_state": hypothetical_state
        }
