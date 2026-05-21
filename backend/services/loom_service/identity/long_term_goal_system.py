import time
from typing import Dict, Any, List

class LongTermGoalSystem:
    """
    Manages high-level long-horizon objectives, updates their progression,
    and determines which goals are currently active.
    """
    def __init__(self):
        # GoalID -> GoalData
        self.goals: Dict[str, Dict[str, Any]] = {}
        
    def add_goal(self, goal_id: str, description: str, target_metric: str, target_value: float):
        self.goals[goal_id] = {
            "description": description,
            "target_metric": target_metric,
            "target_value": target_value,
            "current_value": 0.0,
            "status": "active",
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
    def update_goal_progress(self, goal_id: str, current_value: float):
        if goal_id in self.goals:
            self.goals[goal_id]["current_value"] = current_value
            self.goals[goal_id]["updated_at"] = time.time()
            
            # Check completion
            target = self.goals[goal_id]["target_value"]
            if current_value >= target:
                self.goals[goal_id]["status"] = "completed"
                
    def get_active_goals(self) -> List[Dict[str, Any]]:
        return [g for g in self.goals.values() if g["status"] == "active"]
