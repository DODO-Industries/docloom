import numpy as np
from collections import deque
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("LoomNavigator")

class LoomNavigator:
    def __init__(self, loom_data, node_resolver=None):
        """
        Step 2: Partial Loading support.
        'loom_data' typically contains only structural nodes. 
        'node_resolver' is used to lazy-load neuron shards.
        """
        self.nodes = loom_data.get("g", {}).get("n", {})
        self.edges = loom_data.get("g", {}).get("e", [])
        self.bridge = loom_data.get("bridge", {})
        self.node_resolver = node_resolver
        self.adj = {}
        self._build_adjacency()

    def _build_adjacency(self):
        # We clear and rebuild because edges can grow during lazy-loading
        self.adj = {}
        for e in self.edges:
            f, t = e["f"], e["t"]
            if f not in self.adj: self.adj[f] = []
            self.adj[f].append(e)

    def _get_node(self, node_id):
        """Helper to get node, potentially triggering lazy load."""
        if node_id in self.nodes:
            return self.nodes[node_id]
        
        if self.node_resolver:
            node = self.node_resolver(node_id)
            if node:
                # If new edges were loaded into self.edges during resolving,
                # we should update adjacency.
                self._build_adjacency()
                return node
        return None

    def bfs_traversal(self, start_node_id, max_depth=3):
        """Breadth-First Search for broad knowledge coverage (loom.md Strategy)."""
        log_service(logger, f"Starting BFS Traversal from {start_node_id}...", "debug")
        visited = set()
        queue = deque([(start_node_id, 0)]) # (node_id, depth)
        results = []
        
        # Ensure start node is at least considered
        log_service(logger, f"BFS Entry: {start_node_id}", "debug")
        
        while queue:
            node_id, depth = queue.popleft()
            if depth > max_depth or node_id in visited:
                continue
            
            visited.add(node_id)
            node = self._get_node(node_id)
            if node:
                results.append({"id": node_id, "node": node, "depth": depth})

            for edge in self.adj.get(node_id, []):
                queue.append((edge["t"], depth + 1))
        
        return results

    def beam_search(self, start_node_id, beam_width=3, max_depth=5):
        """
        Optimized Beam Search for focused reasoning (loom.md Strategy).
        Uses truth_score and semantic weight as heuristics.
        """
        log_service(logger, f"Starting Beam Search reasoning (width={beam_width})...", "debug")
        current_beam = [(start_node_id, 0, 1.0)] # (id, depth, accumulated_score)
        all_visited_paths = [(start_node_id, 0, 1.0)] # Include start node by default

        for depth in range(max_depth):
            next_candidates = []
            for node_id, d, score in current_beam:
                if node_id not in self.adj: continue
                
                for edge in self.adj[node_id]:
                    target_id = edge["t"]
                    target_node = self._get_node(target_id) or {}
                    
                    # Heuristic: TruthScore + Structural Weight
                    h_score = target_node.get("m", {}).get("truth_score", 0.5)
                    # Penalize non-essential edges slightly to keep focus
                    if edge["r"] == "relates_to": h_score *= 0.8
                    
                    new_score = score * h_score
                    next_candidates.append((target_id, depth + 1, new_score))

            if not next_candidates:
                break
            
            # Select top-K candidates for next beam
            next_candidates.sort(key=lambda x: x[2], reverse=True)
            current_beam = next_candidates[:beam_width]
            all_visited_paths.extend(current_beam)

        return all_visited_paths

    def concept_jump(self, concept_hash):
        """Step 3: Direct jump to a specific concept in the bridge."""
        nodes = self.bridge.get(concept_hash, [])
        # Return as list of dicts with 'id' and 'score'
        return [{"id": n["n"], "score": n["s"]} if isinstance(n, dict) else {"id": n, "score": 1.0} for n in nodes]

    def heuristic_jump(self, query_embedding, level_centroids):
        """
        Jump to the most relevant hub center (Constellation or Atlas).
        FIX: Support multi-level hierarchy.
        """
        best_id = None
        max_sim = -1
        
        for hub_id, centroid in level_centroids.items():
            # Cosine similarity with centroid
            sim = np.dot(query_embedding, centroid) / (np.linalg.norm(query_embedding) * np.linalg.norm(centroid))
            if sim > max_sim:
                max_sim = sim
                best_id = hub_id
        
        return best_id, max_sim
