from typing import Dict, Any, List

class RetryRecoverySystem:
    """
    Diagnoses tool execution errors and suggests/implements recovery adjustments.
    """
    def __init__(self):
        self.error_history: List[Dict[str, Any]] = []

    def log_failure(self, tool_name: str, error_msg: str, args: Dict[str, Any]) -> None:
        """Logs an action failure for context tracking."""
        self.error_history.append({
            "tool": tool_name,
            "error": error_msg,
            "arguments": args
        })

    def formulate_recovery_strategy(self, tool_name: str, error_msg: str) -> Dict[str, Any]:
        """Analyzes the error message and formulates a diagnostic fix strategy."""
        strategy = {"action": "retry", "args_override": {}, "explanation": "Simple retry attempt"}

        err_lower = error_msg.lower()
        if "permission" in err_lower or "access denied" in err_lower:
            strategy = {
                "action": "elevate_permissions",
                "args_override": {},
                "explanation": "Permission issue. Requesting privilege elevation."
            }
        elif "not found" in err_lower or "no such file" in err_lower:
            strategy = {
                "action": "create_missing_resource",
                "args_override": {"create_dirs": True},
                "explanation": "Missing path resource. Re-attempting after folder initialization."
            }
        elif "timeout" in err_lower:
            strategy = {
                "action": "retry",
                "args_override": {"timeout": 15},
                "explanation": "Timeout occurred. Re-attempting execution with relaxed limits."
            }

        return strategy
