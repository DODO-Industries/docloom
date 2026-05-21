from typing import List, Dict, Any

class AssociativeRetrieval:
    """
    Implements associative memory lookup matching query tokens across semantic concepts and episodes.
    """
    def __init__(self):
        pass

    def retrieve_associated_nodes(self, query: str, concepts: Dict[str, Dict[str, Any]], episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Performs token checks to pull similar concept keys and actions."""
        tokens = query.lower().split()
        
        matched_concepts = []
        for name, data in concepts.items():
            if any(t in name.lower() for t in tokens):
                matched_concepts.append({"concept": name, "data": data})

        matched_episodes = []
        for ep in episodes:
            action = ep.get("action", "")
            if any(t in action.lower() for t in tokens):
                matched_episodes.append(ep)

        return {
            "concepts": matched_concepts,
            "episodes": matched_episodes
        }
