from typing import List, Dict, Any

class MemoryConsolidator:
    """
    Compresses short-term episodic traces into long-term structured semantic concepts.
    """
    def __init__(self):
        pass

    def consolidate_episodes(self, recent_episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Groups high-salience episodes to output generalized rule/theme nodes."""
        high_salience_eps = [ep for ep in recent_episodes if ep.get("salience", 0.0) > 0.6]
        
        consolidated = {}
        for ep in high_salience_eps:
            action = ep.get("action", "")
            if action:
                if action not in consolidated:
                    consolidated[action] = {
                        "count": 0,
                        "success_rate": 0.0,
                        "key_outcomes": []
                    }
                
                consolidated[action]["count"] += 1
                state_after = ep.get("state_after", {})
                success = state_after.get("success", True)
                if success:
                    consolidated[action]["success_rate"] += 1.0

                # Extract keys as outcomes
                consolidated[action]["key_outcomes"].extend(list(state_after.keys()))

        # Normalize success rates
        for action, data in consolidated.items():
            if data["count"] > 0:
                data["success_rate"] /= data["count"]
                data["key_outcomes"] = list(set(data["key_outcomes"]))

        return consolidated
