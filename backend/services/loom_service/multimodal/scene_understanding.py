from typing import Dict, Any, List

class SceneUnderstanding:
    """
    Parses complex visual layouts and diagrams, converting spatial relationships
    into semantic scene graphs.
    """
    def __init__(self):
        pass

    def build_scene_graph(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Creates nodes and directional edges reflecting layout or visual scene hierarchy."""
        nodes = []
        edges = []

        for idx, obj in enumerate(objects):
            node_id = f"node_{idx}"
            nodes.append({
                "id": node_id,
                "label": obj.get("type", "object"),
                "bbox": obj.get("bbox", [0, 0, 0, 0])
            })

            # Form adjacency relationships if objects are vertically stacked or nested
            for other_idx, other_obj in enumerate(objects):
                if idx == other_idx:
                    continue
                # If this object is above the other
                if obj.get("bbox", [0,0,0,0])[3] <= other_obj.get("bbox", [0,0,0,0])[1]:
                    edges.append({
                        "from": node_id,
                        "to": f"node_{other_idx}",
                        "relation": "above"
                    })
        
        return {"nodes": nodes, "relationships": edges}
