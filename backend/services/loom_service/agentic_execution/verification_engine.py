from typing import Dict, Any, List

class VerificationEngine:
    """
    Evaluates action execution reports against expected post-conditions to verify success.
    """
    def __init__(self):
        pass

    def verify_postconditions(self, execution_result: Dict[str, Any], expected_postconditions: Dict[str, Any]) -> bool:
        """
        Determines if the result of an action satisfies the goal's requirements.
        Expected postconditions: e.g. {"file_exists": "results.csv", "status_code": 200}
        """
        if not expected_postconditions:
            return True  # No verification constraints

        # Check execution report fields
        for cond_key, cond_val in expected_postconditions.items():
            if cond_key == "file_exists":
                # Mock result check
                result_val = execution_result.get("result", {})
                if isinstance(result_val, dict) and cond_val in result_val:
                    continue
                return False
            elif cond_key == "status_code":
                if execution_result.get("status_code", execution_result.get("status")) != cond_val:
                    return False
            else:
                # Direct lookup checking
                if execution_result.get(cond_key) != cond_val and execution_result.get("result", {}).get(cond_key) != cond_val:
                    return False

        return True
