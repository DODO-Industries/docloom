from typing import List, Dict, Any

class ValueAlignment:
    """
    Acts as the agent's 'superego' or alignment layer.
    Ensures planning actions and goals comply with basic ethical, safety, and operational rules.
    """
    def __init__(self):
        # Operational limits and forbidden actions
        self.forbidden_patterns = ["rm -rf /", "format c:", "delete_system_critical", "override_safety_constraints"]
        
    def assess_alignment(self, proposed_action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Checks if a planned action aligns with safety goals. Returns alignment flag and risk score."""
        violations = []
        action_lower = proposed_action.lower()
        
        # Check direct forbidden command execution patterns
        command = str(payload.get("command", "")).lower()
        for pattern in self.forbidden_patterns:
            if pattern in action_lower or (command and pattern in command):
                violations.append(f"Forbidden action signature detected: {pattern}")
                
        # Check path writes outside project root if target write occurs
        target_path = str(payload.get("path", "")).lower()
        if target_path and "persnol" not in target_path and "Users" not in target_path:
            # We want to keep file writes inside the user workspace
            violations.append(f"Suspicious write target outside user environment: {target_path}")
            
        risk_score = 1.0 if violations else 0.0
        
        return {
            "aligned": len(violations) == 0,
            "risk_score": risk_score,
            "violations": violations
        }
