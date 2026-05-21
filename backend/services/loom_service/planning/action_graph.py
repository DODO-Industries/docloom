import networkx as nx
from typing import List, Tuple, Dict, Any

class ActionGraph:
    """
    Constructs and manages a directed acyclic graph (DAG) representing
    dependencies between planning operators and execution steps.
    """
    def __init__(self):
        self.dag = nx.DiGraph()
        
    def add_action(self, action_id: str, metadata: Dict[str, Any] = None):
        self.dag.add_node(action_id, **(metadata or {}))
        
    def add_dependency(self, parent_action: str, child_action: str):
        self.dag.add_edge(parent_action, child_action)
        
    def get_execution_order(self) -> List[str]:
        """Returns topological sort of the action graph for valid execution sequence."""
        try:
            return list(nx.topological_sort(self.dag))
        except nx.NetworkXUnfeasible:
            # Contains cycle, return nodes as-is
            return list(self.dag.nodes())
            
    def get_ready_actions(self, completed_actions: List[str]) -> List[str]:
        """Returns actions whose dependencies are fully completed."""
        ready = []
        completed = set(completed_actions)
        for node in self.dag.nodes():
            if node in completed:
                continue
            preds = list(self.dag.predecessors(node))
            if all(p in completed for p in preds):
                ready.append(node)
        return ready
