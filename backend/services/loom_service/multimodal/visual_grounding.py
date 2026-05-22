from typing import List, Dict, Any

class VisualGrounder:
    """
    Ties textual noun phrases and concepts to localized areas of document visual interfaces.
    """
    def __init__(self):
        pass

    def ground_query_to_image(self, query: str, scene_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Finds matching nodes in the scene graph corresponding to the linguistic search."""
        matches = []
        words = query.lower().split()
        for node in scene_graph.get("nodes", []):
            label = node.get("label", "").lower()
            if any(w in label for w in words):
                matches.append(node)
        return matches
