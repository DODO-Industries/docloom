from typing import List, Dict, Any

class EthicalConstraints:
    """
    Validates execution plans and actions against alignment, compliance, and privacy rules.
    """
    def __init__(self):
        # Set of rules
        self.rules = [
            "Never bypass user approval for file writes outside sandbox.",
            "Do not execute arbitrary uninspected network code.",
            "Always respect user privacy and sensitive files."
        ]

    def validate_action(self, action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Checks if a specific action and arguments violate any alignment constraints."""
        violations = []
        
        # Check params for credential leaks
        for k, v in params.items():
            if any(term in str(k).lower() for term in ["key", "password", "token", "secret"]):
                if "credential" not in str(v).lower():
                    violations.append(f"Potential credential leak parameter: {k}")

        if action_name == "network_send" and not params.get("encrypted", False):
            violations.append("Action network_send must use encryption parameters.")

        return {
            "compliant": len(violations) == 0,
            "violations": violations
        }
