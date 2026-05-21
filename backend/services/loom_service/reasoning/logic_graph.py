from typing import Dict, List, Set

class LogicGraph:
    """
    Graph representation of logical implications, dependencies, and constraints.
    """
    def __init__(self):
        # Maps node -> set of nodes it implies/points to
        self.adj_list: Dict[str, Set[str]] = {}

    def add_implication(self, source: str, target: str) -> None:
        """Adds a directional implication: source -> target (if source is True, target is True)."""
        if source not in self.adj_list:
            self.adj_list[source] = set()
        self.adj_list[source].add(target)

    def find_all_consequences(self, active_nodes: List[str]) -> Set[str]:
        """Performs a BFS/DFS traversal to find all consequences from a starting set."""
        visited = set(active_nodes)
        queue = list(active_nodes)

        while queue:
            node = queue.pop(0)
            for neighbor in self.adj_list.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited
