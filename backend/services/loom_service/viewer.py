import msgpack
import os
from backend.services.loom_service.visualizer import LoomVisualizer

class LoomViewer:
    def __init__(self, loom_path=None):
        self.data = None
        self.loom_path = loom_path
        if loom_path:
            self.load(loom_path)

    def load(self, loom_path):
        self.loom_path = loom_path
        if not os.path.exists(loom_path):
            print(f"Error: {loom_path} not found.")
            return
        with open(loom_path, "rb") as f:
            self.data = msgpack.unpackb(f.read(), raw=False)

    def audit(self):
        """Prints structure and generates HTML visualization."""
        if not self.data:
            print("No data loaded.")
            return

        # 1. Output HTML Visualization
        if self.loom_path:
            html_path = self.loom_path.replace(".loom", ".html")
            print(f"[*] Generating Intelligence Map Visualizer at: {html_path}")
            LoomVisualizer.generate_html(self.data, html_path)

        nodes = self.data["g"]["n"]
        edges = self.data["g"]["e"]

        # Build adjacency for children
        adj = {}
        incoming_count = {nid: 0 for nid in nodes}
        for e in edges:
            f, t = e["f"], e["t"]
            if f not in adj: adj[f] = []
            adj[f].append(e)
            incoming_count[t] += 1

        print("\n" + "="*50)
        print(" [AUDIT] DOCLOOM HUMAN AUDIT: .loom STRUCTURE ")
        print("="*50)

        # 1. Start from Root nodes (usually type 'root')
        root_ids = [nid for nid, node in nodes.items() if node["t"] == "root"]
        
        for rid in root_ids:
            self._print_tree(rid, nodes, adj, level=0)

        # 2. Orphan Detection
        print("\n" + "!"*50)
        print(" [WARNING] ORPHAN NODE DETECTION (Extraction Gaps) ")
        print("!"*50)
        
        orphans_found = False
        for nid, node in nodes.items():
            if incoming_count[nid] == 0 and node["t"] != "root":
                print(f"[ORPHAN] ID: {nid} | Type: {node['t']} | Content: {node['c'][:100]}...")
                orphans_found = True
        
        if not orphans_found:
            print("[OK] No orphan nodes detected. Graph is fully connected.")
        
        print("="*50 + "\n")

    def _print_tree(self, node_id, nodes, adj, level, visited=None):
        if visited is None: visited = set()
        if node_id in visited: return
        visited.add(node_id)

        node = nodes[node_id]
        indent = "  " * level
        marker = "[FOLDER]" if node["t"] in ["root", "page", "heading"] else "[FILE]"
        
        # Format content for display
        clean_content = node["c"].replace("\n", " ").strip()
        display_text = (clean_content[:80] + "...") if len(clean_content) > 80 else clean_content
        
        print(f"{indent}{marker} [{node['t'].upper()}] {display_text}")

        # Process children
        if node_id in adj:
            # Sort children to keep logical order if possible (e.g. by bbox top)
            children_edges = adj[node_id]
            # Custom sort: if children have bbox, sort by top
            def get_top(edge):
                meta = nodes.get(edge["t"], {}).get("m", {})
                bbox = meta.get("bbox")
                return bbox[1] if bbox and len(bbox) > 1 else 0
            
            children_edges.sort(key=get_top)

            for e in children_edges:
                # Avoid printing metadata-only relationships in the main structural tree
                # for clarity (e.g. don't follow 'visual_evidence' back into paragraphs here)
                if e["r"] in ["contains", "body_text", "sub_content"]:
                    self._print_tree(e["t"], nodes, adj, level + 1, visited)

if __name__ == "__main__":
    # Internal CLI for quick debugging
    import sys
    if len(sys.argv) > 1:
        viewer = LoomViewer(sys.argv[1])
        viewer.audit()
