from typing import List, Dict, Any

class FailureLearner:
    """
    Analyzes historical runtime errors to formulate general rules for avoiding future failures.
    """
    def __init__(self):
        # List of rules: e.g. [{"pattern": "timeout", "action": "use larger timeout parameter"}]
        self.learned_rules: List[Dict[str, Any]] = []

    def derive_rule_from_failure(self, failure_report: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts a preventive guideline from an active failure trace."""
        error = failure_report.get("error", "").lower()
        tool = failure_report.get("tool", "")

        new_rule = {}
        if "permission" in error:
            new_rule = {
                "pattern": f"tool_fail_{tool}_permission",
                "condition": f"tool == '{tool}' and 'permission' in error",
                "prevention": "Check permissions before calling the tool or use sandboxed targets.",
                "confidence": 0.8
            }
        elif "timeout" in error:
            new_rule = {
                "pattern": f"tool_fail_{tool}_timeout",
                "condition": f"tool == '{tool}' and 'timeout' in error",
                "prevention": "Double timeout limits or optimize arguments.",
                "confidence": 0.9
            }
        else:
            new_rule = {
                "pattern": f"tool_fail_{tool}_generic",
                "condition": f"tool == '{tool}'",
                "prevention": "Add safety parameters check to parameters mapping.",
                "confidence": 0.5
            }

        if new_rule and new_rule not in self.learned_rules:
            self.learned_rules.append(new_rule)
        return new_rule
