import msgpack
import os
import numpy as np
from collections import deque
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("NeuralViewer")

class NeuralViewer:
    def __init__(self, loom_path=None):
        self.data = None
        self.loom_path = loom_path
        self.shard_cache = {}
        self.adj = {}
        if loom_path:
            self.load(loom_path)

    def load(self, loom_path):
        self.loom_path = loom_path
        if not os.path.exists(loom_path):
            print(f"Error: {loom_path} not found.")
            return
            
        with open(loom_path, "rb") as f:
            raw_data = msgpack.unpackb(f.read(), raw=False)
            
            if raw_data.get("type") == "substrate_master":
                print(f"[*] Detected Master Substrate: {os.path.basename(loom_path)}")
                self.data = raw_data
                self._discover_and_load_shards(loom_path)
            else:
                print(f"[*] Detected Standalone Shard: {os.path.basename(loom_path)}")
                self.data = {
                    "v": raw_data.get("v", "1.2"),
                    "g": raw_data.get("g", {"n": {}, "e": []}),
                    "type": "shard"
                }
        self._build_adjacency()

    def _discover_and_load_shards(self, master_path):
        log_service(logger, "Master Substrate loaded. Shards will be lazy-loaded as needed.", "info")

    def get_node(self, node_id):
        if self.data and "g" in self.data and "n" in self.data["g"] and node_id in self.data["g"]["n"]:
            return self.data["g"]["n"][node_id]
        
        shard_map = self.data.get("map", {}) if self.data else {}
        if node_id not in shard_map:
            if hasattr(self, 'node_resolver') and self.node_resolver:
                return self.node_resolver(node_id)
            return None
            
        shard_id = shard_map[node_id]
        
        if shard_id not in self.shard_cache:
            self.load_shard(shard_id)
            
        node = self.shard_cache[shard_id].get(node_id)
        if node and "truth_score" not in node["m"]:
            node["m"]["truth_score"] = 0.1
            
        return node

    def load_shard(self, shard_id):
        if not self.loom_path: return
        
        master_dir = os.path.dirname(self.loom_path)
        base_name = os.path.basename(self.loom_path).replace("substrate_", "").replace(".loom", "")
        s_path = os.path.join(master_dir, f"{base_name}_shard_{shard_id}.loom")
        
        if os.path.exists(s_path):
            with open(s_path, "rb") as f:
                s_data = msgpack.unpackb(f.read(), raw=False)
                shard_nodes = s_data.get("n", {})
                shard_edges = s_data.get("e", [])
                
                self.shard_cache[shard_id] = shard_nodes
                if self.data and "g" in self.data and "e" in self.data["g"]:
                    self.data["g"]["e"].extend(shard_edges)
                self._build_adjacency()
                log_service(logger, f"Lazy-loaded Neuron Shard {shard_id} ({len(shard_nodes)} nodes)", "debug")
        else:
            log_service(logger, f"Shard {shard_id} missing at {s_path}", "warning")

    def _build_adjacency(self):
        if not self.data:
            return
        self.adj = {}
        for e in self.data.get("g", {}).get("e", []):
            f, t = e["f"], e["t"]
            if f not in self.adj: self.adj[f] = []
            self.adj[f].append(e)

    def bfs_traversal(self, start_node_id, max_depth=3):
        log_service(logger, f"Starting BFS Traversal from {start_node_id}...", "debug")
        visited = set()
        queue = deque([(start_node_id, 0)])
        results = []
        
        while queue:
            node_id, depth = queue.popleft()
            if depth > max_depth or node_id in visited:
                continue
            
            visited.add(node_id)
            node = self.get_node(node_id)
            if node:
                results.append({"id": node_id, "node": node, "depth": depth})

            for edge in self.adj.get(node_id, []):
                queue.append((edge["t"], depth + 1))
        
        return results

    def beam_search(self, start_node_id, beam_width=3, max_depth=5):
        log_service(logger, f"Starting Beam Search reasoning (width={beam_width})...", "debug")
        current_beam = [(start_node_id, 0, 1.0)]
        all_visited_paths = [(start_node_id, 0, 1.0)]

        for depth in range(max_depth):
            next_candidates = []
            for node_id, d, score in current_beam:
                if node_id not in self.adj: continue
                
                for edge in self.adj[node_id]:
                    target_id = edge["t"]
                    target_node = self.get_node(target_id) or {}
                    
                    h_score = target_node.get("m", {}).get("truth_score", 0.5)
                    if edge["r"] == "relates_to": h_score *= 0.8
                    
                    new_score = score * h_score
                    next_candidates.append((target_id, depth + 1, new_score))

            if not next_candidates:
                break
            
            next_candidates.sort(key=lambda x: x[2], reverse=True)
            current_beam = next_candidates[:beam_width]
            all_visited_paths.extend(current_beam)

        return all_visited_paths

    def concept_jump(self, concept_hash):
        if not self.data:
            return []
        nodes = self.data.get("bridge", {}).get(concept_hash, [])
        return [{"id": n["n"], "score": n["s"]} if isinstance(n, dict) else {"id": n, "score": 1.0} for n in nodes]

    def heuristic_jump(self, query_embedding, level_centroids):
        best_id = None
        max_sim = -1
        
        for hub_id, centroid in level_centroids.items():
            sim = np.dot(query_embedding, centroid) / (np.linalg.norm(query_embedding) * np.linalg.norm(centroid))
            if sim > max_sim:
                max_sim = sim
                best_id = hub_id
        
        return best_id, max_sim

    def explore(self):
        from backend.services.loom_service.server import launch
        launch()

    def audit(self):
        if not self.data:
            print("No data loaded.")
            return

        if self.loom_path and self.data.get("type") == "substrate_master":
            print(f"\n[*] COGNITIVE SUBSTRATE ARCHITECTURE READY")
            print(f"[*] To activate the neural graph, run: .venv\\Scripts\\python.exe -m backend.services.loom_service.server")
            print(f"[*] Path to load: {os.path.abspath(self.loom_path)}\n")

        nodes = self.data["g"]["n"]
        edges = self.data["g"]["e"]

        adj = {}
        incoming_count = {nid: 0 for nid in nodes}
        for e in edges:
            f, t = e["f"], e["t"]
            if f not in adj: adj[f] = []
            adj[f].append(e)
            if t in incoming_count:
                incoming_count[t] += 1
            else:
                incoming_count[t] = 1

        print("\n" + "="*50)
        print(" [AUDIT] DOCLOOM NEURAL AUDIT: .loom COGNITION ")
        print("="*50)

        root_ids = [nid for nid, node in nodes.items() if node["t"] == "root"]
        
        for rid in root_ids:
            self._print_tree(rid, nodes, adj, level=0)

        print("\n" + "!"*50)
        print(" [WARNING] NEURAL GAP DETECTION (Extraction Holes) ")
        print("!"*50)
        
        orphans_found = False
        for nid, node in nodes.items():
            if incoming_count[nid] == 0 and node["t"] != "root":
                orphans_found = True
        
        if not orphans_found:
            print("[OK] All knowledge units (shards) are integrated into the graph.")
        
        self.reasoning_audit()
        print("="*50 + "\n")

    def reasoning_audit(self):
        print("\n" + "*"*50)
        print(" [COGNITIVE] NEURAL ACTIVATION DEMONSTRATION ")
        print("*"*50)
        
        shard_ids = [nid for nid, n in self.data["g"]["n"].items() if n["t"] == "shard"]
        if not shard_ids: 
            print("[SKIP] No shards found for activation test.")
            return

        start_id = shard_ids[0]
        print(f"[*] Activating Graph from Neuron: {start_id}")
        reasoning_path = self.beam_search(start_id, beam_width=2, max_depth=3)
        
        for nid, depth, score in reasoning_path:
            node = self.data["g"]["n"].get(nid, {})
            content = node.get("c", "")[:100].replace("\n", " ")
            print(f"  [Hop {depth}] Neuron {nid[-6:]} | Score: {score:.2f} | Context: {content}...")

    def _print_tree(self, node_id, nodes, adj, level, visited=None):
        if visited is None: visited = set()
        if node_id in visited: return
        visited.add(node_id)

        node = self.get_node(node_id)
        if not node: return
        indent = "  " * level
        marker = "[HUB]" if node["t"] in ["root", "meta_shard", "macro_shard"] else "[NEURON]"
        
        clean_content = node["c"].replace("\n", " ").strip()
        display_text = (clean_content[:80] + "...") if len(clean_content) > 80 else clean_content
        
        try:
            print(f"{indent}{marker} [{node['t'].upper()}] {display_text}")
        except UnicodeEncodeError:
            print(f"{indent}{marker} [{node['t'].upper()}] {display_text.encode('ascii', 'ignore').decode('ascii')}")

        if node_id in adj:
            children_edges = adj[node_id]
            for e in children_edges:
                if e["r"] in ["contains", "primary_ingestion"]:
                    self._print_tree(e["t"], nodes, adj, level + 1, visited)


class LoomNavigator:
    """Backward-compatible wrapper routing requests to internal NeuralViewer implementation."""
    def __init__(self, loom_data, node_resolver=None):
        self.viewer = NeuralViewer()
        self.viewer.data = loom_data
        self.viewer.node_resolver = node_resolver
        self.viewer._build_adjacency()

    def bfs_traversal(self, start_node_id, max_depth=3):
        return self.viewer.bfs_traversal(start_node_id, max_depth)

    def beam_search(self, start_node_id, beam_width=3, max_depth=5):
        return self.viewer.beam_search(start_node_id, beam_width, max_depth)

    def concept_jump(self, concept_hash):
        return self.viewer.concept_jump(concept_hash)

    def heuristic_jump(self, query_embedding, level_centroids):
        return self.viewer.heuristic_jump(query_embedding, level_centroids)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        viewer = NeuralViewer(sys.argv[1])
        viewer.audit()
