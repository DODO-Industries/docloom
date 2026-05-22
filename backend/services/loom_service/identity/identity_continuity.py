import json
import os
from typing import Dict, Any

class IdentityContinuity:
    """
    Serializes and deserializes the entire persistent self model,
    autobiographical log, values, and belief systems to disk.
    """
    def __init__(self, persistence_path: str):
        self.persistence_path = persistence_path
        
    def save_identity(self, self_data: Dict[str, Any], belief_data: Dict[str, Any], goal_data: Dict[str, Any]) -> bool:
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            bundle = {
                "self": self_data,
                "beliefs": belief_data,
                "goals": goal_data
            }
            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(bundle, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[IdentityContinuity] Save failed: {e}")
            return False
            
    def load_identity(self) -> Dict[str, Any]:
        if not os.path.exists(self.persistence_path):
            return {}
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[IdentityContinuity] Load failed: {e}")
            return {}
