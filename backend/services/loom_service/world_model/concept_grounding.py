from typing import Dict, Any, List, Optional

class ConceptGrounder:
    """
    Binds abstract symbols and words to concrete object nodes or physical coordinates
    within the world model.
    """
    def __init__(self):
        # Maps concept name -> list of matching node IDs or coordinates
        self.grounding_map: Dict[str, List[Dict[str, Any]]] = {}

    def register_grounding(self, concept: str, target: Dict[str, Any]) -> None:
        """Grounds a concept keyword or entity description to a node or coordinate."""
        if concept not in self.grounding_map:
            self.grounding_map[concept] = []
        self.grounding_map[concept].append(target)

    def ground_concept(self, concept: str) -> List[Dict[str, Any]]:
        """Resolves an abstract concept to its physical or object graph groundings."""
        return self.grounding_map.get(concept, [])

    def clear_grounding(self, concept: str) -> None:
        """Removes all grounding instances for a concept."""
        if concept in self.grounding_map:
            del self.grounding_map[concept]
