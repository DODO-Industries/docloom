from typing import Dict, Any, List

class SemanticMemory:
    """
    Stores generalized conceptual nodes, attributes, and semantic relations.
    """
    def __init__(self):
        # Format: {"concept_name": {"attribute1": val, "related_to": ["other_concept"]}}
        self.concepts: Dict[str, Dict[str, Any]] = {}

    def add_concept(self, name: str, attributes: Dict[str, Any]) -> None:
        """Saves a new concept or updates existing details in semantic store."""
        if name not in self.concepts:
            self.concepts[name] = {}
        self.concepts[name].update(attributes)

    def link_concepts(self, name_a: str, name_b: str, relation: str) -> None:
        """Adds a directional semantic link between two concepts."""
        if name_a in self.concepts and name_b in self.concepts:
            if "relations" not in self.concepts[name_a]:
                self.concepts[name_a]["relations"] = []
            self.concepts[name_a]["relations"].append({
                "to": name_b,
                "type": relation
            })

    def query_concept(self, name: str) -> Dict[str, Any]:
        """Retrieves details stored under the target concept name."""
        return self.concepts.get(name, {})
