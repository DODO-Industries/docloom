import msgpack
import os
from backend.services.loom_service.substrate.visualizer import LoomVisualizer
from backend.services.loom_service.substrate.navigator import LoomNavigator
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("LoomViewer")

class LoomViewer:
    def __init__(self, loom_path=None):
        self.data = None
        self.loom_path = loom_path
        self.shard_cache = {} # Step 4: Activation Cache
        if loom_path:
            self.load(loom_path)

    def load(self, loom_path):
        self.loom_path = loom_path
        if not os.path.exists(loom_path):
            print(f"Error: {loom_path} not found.")
            return
            
        with open(loom_path, "rb") as f:
            raw_data = msgpack.unpackb(f.read(), raw=False)
            
            # Check if this is a Master Atlas or a single Shard
            if raw_data.get("type") == "atlas_master":
                print(f"[*] Detected Master Atlas: {os.path.basename(loom_path)}")
                self.data = raw_data
                self._discover_and_load_shards(loom_path)
            else:
                print(f"[*] Detected Standalone Shard: {os.path.basename(loom_path)}")
                # Normalize format
                self.data = {
                    "v": raw_data.get("v", "1.2"),
                    "g": raw_data.get("g", {"n": {}, "e": []}),
                    "type": "shard"
                }

    def _discover_and_load_shards(self, master_path):
        """
        Step 2: Partial Loading.
        We no longer load all shards at startup. They are loaded on-demand.
        """
        log_service(logger, "Master Atlas loaded. Shards will be lazy-loaded as needed.", "info")

    def get_node(self, node_id):
        """
        Step 4: Activation Cache.
        Retrieves a node, loading its shard if necessary.
        """
        # 1. Check structural nodes (already in master)
        if node_id in self.data["g"]["n"]:
            return self.data["g"]["n"][node_id]
        
        # 2. Check which shard contains this node
        shard_map = self.data.get("map", {})
        if node_id not in shard_map:
            return None
            
        shard_id = shard_map[node_id]
        
        # 3. Load shard if not in cache
        if shard_id not in self.shard_cache:
            self.load_shard(shard_id)
            
        node = self.shard_cache[shard_id].get(node_id)
        if node and "truth_score" not in node["m"]:
            node["m"]["truth_score"] = 0.1 # Fallback for old files
            
        return node

    def load_shard(self, shard_id):
        """Step 2: Shard-level lazy loading."""
        if not self.loom_path: return
        
        master_dir = os.path.dirname(self.loom_path)
        base_name = os.path.basename(self.loom_path).replace("atlas_", "").replace(".loom", "")
        s_path = os.path.join(master_dir, f"{base_name}_shard_{shard_id}.loom")
        
        if os.path.exists(s_path):
            with open(s_path, "rb") as f:
                s_data = msgpack.unpackb(f.read(), raw=False)
                shard_nodes = s_data.get("n", {})
                shard_edges = s_data.get("e", [])
                
                self.shard_cache[shard_id] = shard_nodes
                # Merge edges into master graph for traversal (Edges are small)
                self.data["g"]["e"].extend(shard_edges)
                log_service(logger, f"Lazy-loaded Neuron Shard {shard_id} ({len(shard_nodes)} nodes)", "debug")
        else:
            log_service(logger, f"Shard {shard_id} missing at {s_path}", "warning")

    def explore(self):
        """Launches the dynamic Loom Explorer server."""
        from backend.services.loom_service.server import launch
        launch()

    def audit(self):
        """Prints structure and generates HTML visualization."""
        if not self.data:
            print("No data loaded.")
            return

        # 1. Output Informational Message
        if self.loom_path and self.data.get("type") == "atlas_master":
            print(f"\n[*] COGNITIVE ATLAS ARCHITECTURE READY")
            print(f"[*] To activate the neural graph, run: .venv\\Scripts\\python.exe -m backend.services.loom_service.server")
            print(f"[*] Path to load: {os.path.abspath(self.loom_path)}\n")

        nodes = self.data["g"]["n"]
        edges = self.data["g"]["e"]

        # Build adjacency for children
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

        # 1. Start from Root nodes
        root_ids = [nid for nid, node in nodes.items() if node["t"] == "root"]
        
        for rid in root_ids:
            self._print_tree(rid, nodes, adj, level=0)

        # 2. Orphan Detection
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
        """Demonstrates multi-hop reasoning using the Beam Search navigator."""
        print("\n" + "*"*50)
        print(" [COGNITIVE] NEURAL ACTIVATION DEMONSTRATION ")
        print("*"*50)
        
        navigator = LoomNavigator(self.data)
        shard_ids = [nid for nid, n in self.data["g"]["n"].items() if n["t"] == "shard"]
        if not shard_ids: 
            print("[SKIP] No shards found for activation test.")
            return

        start_id = shard_ids[0]
        print(f"[*] Activating Graph from Neuron: {start_id}")
        reasoning_path = navigator.beam_search(start_id, beam_width=2, max_depth=3)
        
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
        marker = "[HUB]" if node["t"] in ["root", "atlas", "constellation"] else "[NEURON]"
        
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        viewer = LoomViewer(sys.argv[1])
        viewer.audit()
