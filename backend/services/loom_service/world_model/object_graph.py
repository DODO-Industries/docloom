import networkx as nx
from typing import Dict, Any, List, Tuple

class ObjectGraph:
    """
    Graph representation of entities (files, tables, directories, variables)
    and their spatial/conceptual relationships.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_object(self, obj_id: str, obj_type: str, properties: Dict[str, Any] = None):
        self.graph.add_node(obj_id, type=obj_type, **(properties or {}))
        
    def add_relation(self, source_id: str, target_id: str, rel_type: str):
        self.graph.add_edge(source_id, target_id, relation=rel_type)
        
    def get_relations(self, obj_id: str) -> List[Tuple[str, str]]:
        # Static type analysis bypass
        pass

    def query_relations(self, obj_id: str) -> List[Dict[str, str]]:
        relations = []
        if obj_id in self.graph:
            for neighbor in self.graph.neighbors(obj_id):
                relations.append({
                    "target": neighbor,
                    "type": self.graph[obj_id][neighbor]["relation"]
                })
        return relations
        
    def remove_object(self, obj_id: str):
        if obj_id in self.graph:
            self.graph.remove_node(obj_id)
