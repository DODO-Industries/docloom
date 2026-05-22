from typing import List, Dict, Any

class EpisodicMemory:
    """
    Manages episodic memory buffers, recording serial logs of events, actions, and states per tick.
    """
    def __init__(self):
        self.episodes: List[Dict[str, Any]] = []

    def record_episode(self, tick_id: int, action: str, state_before: Dict[str, Any], state_after: Dict[str, Any], salience: float) -> None:
        """Stores a new episode snapshot."""
        self.episodes.append({
            "tick_id": tick_id,
            "action": action,
            "state_before": state_before,
            "state_after": state_after,
            "salience": salience
        })

    def search_episodes(self, action_query: str) -> List[Dict[str, Any]]:
        """Finds all recorded episodes matching the query action name."""
        return [ep for ep in self.episodes if action_query.lower() in ep.get("action", "").lower()]

    def get_all_episodes(self) -> List[Dict[str, Any]]:
        """Retrieves the complete episode log history."""
        return self.episodes
