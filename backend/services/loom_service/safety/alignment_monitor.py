from typing import List, Dict, Any

class AlignmentMonitor:
    """
    Monitors active plans and subgoals to ensure they align with core system intents,
    detecting goal drift or hijacked subtasks.
    """
    def __init__(self, core_objectives: List[str] = None):
        self.core_objectives = core_objectives if core_objectives is not None else ["assist_user", "preserve_integrity"]

    def check_goal_alignment(self, current_goals: List[str]) -> Dict[str, Any]:
        """Calculates drift and flags any goal that directly conflicts with core principles."""
        flagged_goals = []
        
        for goal in current_goals:
            g_lower = goal.lower()
            # If subtasks contain dangerous keywords or conflict patterns
            if "delete_system" in g_lower or "bypass_security" in g_lower or "leak_data" in g_lower:
                flagged_goals.append(goal)

        drift_ratio = len(flagged_goals) / max(len(current_goals), 1)

        return {
            "is_aligned": len(flagged_goals) == 0,
            "flagged_goals": flagged_goals,
            "drift_ratio": drift_ratio
        }
