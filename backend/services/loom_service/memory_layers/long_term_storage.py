from typing import Dict, Any
import json
import os

class LongTermStorage:
    """
    Handles file-system serialization/deserialization for long-term memory backup.
    """
    def __init__(self, storage_path: str = "memory_store.json"):
        self.storage_path = storage_path

    def save_state(self, state_data: Dict[str, Any]) -> bool:
        """Serializes and saves state dict to disk."""
        try:
            with open(self.storage_path, "w") as f:
                json.dump(state_data, f, indent=4)
            return True
        except Exception:
            return False

    def load_state(self) -> Dict[str, Any]:
        """Loads state data from disk if it exists."""
        if not os.path.exists(self.storage_path):
            return {}
        try:
            with open(self.storage_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
