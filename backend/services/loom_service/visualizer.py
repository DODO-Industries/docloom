import os

class LoomVisualizer:
    @staticmethod
    def generate_html(loom_data, output_path):
        """Generates a premium, responsive side-by-side column visualization of the Loom DAG."""
        nodes = loom_data["g"]["n"]
        edges = loom_data["g"]["e"]

        # Build tree structure
        adj = {}
        for e in edges:
            f, t = e["f"], e["t"]
            if f not in adj: adj[f] = []
            adj[f].append(e)

        root_ids = [nid for nid, node in nodes.items() if node["t"] == "root"]
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocLoom Intelligence Map v1.1</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #05070a;
            --panel: rgba(17, 24, 39, 0.7);
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.3);
            --text: #f3f4f6;
            --muted: #9ca3af;
            --heading: #fbbf24;
            --page: #a5b4fc;
            --para: #d1d5db;
            --image: #34d399;
            --table: #f472b6;
            --border: rgba(255, 255, 255, 0.08);
            --glass: blur(12px) saturate(180%);
        }}

        body {{
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(244, 114, 182, 0.1) 0px, transparent 50%);
            color: var(--text);
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            margin: 0;
            padding: 60px 40px;
            min-height: 100vh;
            line-height: 1.5;
        }}

        .container {{ max-width: 1200px; margin: 0 auto; }}

        header {{
            margin-bottom: 80px;
            position: relative;
        }}

        h1 {{ 
            font-size: 3.5rem; 
            font-weight: 600; 
            margin: 0; 
            letter-spacing: -2px;
            background: linear-gradient(to right, #fff, #9ca3af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .subtitle {{ 
            color: var(--muted); 
            font-size: 1.1rem; 
            margin-top: 10px;
            font-weight: 300;
            letter-spacing: 1px;
            text-transform: uppercase;
        }}

        .page-container {{
            margin-bottom: 120px;
            position: relative;
        }}

        .page-header {{
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 4px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        
        .page-header::after {{
            content: '';
            flex: 1;
            height: 1px;
            background: linear-gradient(to right, var(--border), transparent);
        }}

        /* THE GRID SOLUTION */
        .layout-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            column-gap: 60px;
            row-gap: 30px;
        }}

        .node {{
            background: var(--panel);
            backdrop-filter: var(--glass);
            -webkit-backdrop-filter: var(--glass);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            position: relative;
            transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
            animation: fadeIn 0.8s ease-out backwards;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .node:hover {{
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 20px var(--accent-glow);
        }}

        /* Column Spanning Logic */
        .node[data-col="0"] {{ grid-column: 1 / span 2; }}
        .node[data-col="1"] {{ grid-column: 1; }}
        .node[data-col="2"] {{ grid-column: 2; }}

        .node-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.6rem;
            font-weight: 600;
            text-transform: uppercase;
            padding: 4px 12px;
            border-radius: 6px;
            margin-bottom: 16px;
            display: inline-block;
            letter-spacing: 1px;
        }}

        .tag-heading {{ background: rgba(251, 191, 36, 0.1); color: var(--heading); }}
        .tag-paragraph {{ background: rgba(255, 255, 255, 0.05); color: var(--para); }}
        .tag-image {{ background: rgba(52, 211, 153, 0.1); color: var(--image); }}
        .tag-table {{ background: rgba(244, 114, 182, 0.1); color: var(--table); }}

        .content {{ font-size: 1.05rem; line-height: 1.8; color: var(--para); }}
        .heading-content {{ 
            font-size: 1.8rem; 
            font-weight: 600; 
            color: var(--heading); 
            line-height: 1.3;
            letter-spacing: -0.5px;
        }}

        .meta-strip {{
            display: flex;
            gap: 20px;
            margin-top: 24px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            color: var(--muted);
            opacity: 0.5;
            border-top: 1px solid var(--border);
            padding-top: 16px;
        }}

        .image-box {{
            margin-top: 20px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        
        .image-box img {{ width: 100%; display: block; filter: grayscale(20%); transition: filter 0.3s; }}
        .image-box:hover img {{ filter: grayscale(0%); }}

        .table-box {{
            margin-top: 20px;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}

        table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
        td {{ padding: 14px; border: 1px solid var(--border); }}
        tr:nth-child(even) {{ background: rgba(255,255,255,0.02); }}

        .anchor-badge {{
            position: absolute;
            top: 24px;
            right: 24px;
            font-size: 0.6rem;
            font-weight: 600;
            background: var(--accent);
            color: #fff;
            padding: 4px 12px;
            border-radius: 100px;
            box-shadow: 0 4px 12px var(--accent-glow);
        }}

        @media (max-width: 900px) {{
            .layout-grid {{ grid-template-columns: 1fr; }}
            .node[data-col="0"], .node[data-col="1"], .node[data-col="2"] {{ grid-column: 1; }}
            h1 {{ font-size: 2.5rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>DocLoom Intelligence</h1>
            <div class="subtitle">Multi-Column Neural Graph Visualization</div>
        </header>

        {LoomVisualizer._render_root(root_ids[0] if root_ids else None, nodes, adj)}
    </div>
</body>
</html>
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    @staticmethod
    def _render_root(root_id, nodes, adj):
        if not root_id or root_id not in adj: return ""
        
        html = ""
        # Get all page nodes
        page_edges = [e for e in adj[root_id] if nodes[e["t"]]["t"] == "page"]
        # Sort by page number
        page_edges.sort(key=lambda e: nodes[e["t"]]["m"].get("num", 0))

        for pe in page_edges:
            page_id = pe["t"]
            page_node = nodes[page_id]
            
            html += f'<div class="page-container">'
            html += f'<div class="page-header"><span>//</span> Intelligence Layer: Page {page_node["m"].get("num")}</div>'
            
            # FLAT GRID for all children of page
            html += '<div class="layout-grid">'
            
            # Render all children in order (they are already sorted by Weaver)
            page_children_edges = adj.get(page_id, [])
            for ce in page_children_edges:
                html += LoomVisualizer._render_node_html(ce["t"], nodes, adj)
            
            html += '</div>' # End layout-grid
            html += '</div>' # End page-container
            
        return html

    @staticmethod
    def _render_node_html(node_id, nodes, adj):
        node = nodes[node_id]
        meta = node.get("m", {})
        node_type = node["t"]
        col_id = meta.get("col", 0)
        
        anchor_info = ""
        if node_id in adj:
            for e in adj[node_id]:
                if e["r"] == "visual_evidence":
                    anchor_info = '<div class="anchor-badge">Loom: Spatial Link</div>'
                elif e["r"] == "keyword_anchor":
                    anchor_info = '<div class="anchor-badge">Loom: Semantic Link</div>'

        extra = ""
        if node_type == "image" and meta.get("binary"):
            extra = f'<div class="image-box"><img src="data:image/png;base64,{meta["binary"]}"></div>'
        elif node_type == "table" and meta.get("table"):
            table_rows = "".join([f"<tr>{''.join([f'<td>{str(c)}</td>' for c in row])}</tr>" for row in meta["table"]])
            extra = f'<div class="table-box"><table>{table_rows}</table></div>'

        content_class = "heading-content" if node_type == "heading" else "paragraph-content"
        
        # Render nested children in flow
        nested_html = ""
        if node_id in adj:
            for e in adj[node_id]:
                if e["r"] in ["contains", "body_text", "sub_content"]:
                    nested_html += LoomVisualizer._render_node_html(e["t"], nodes, adj)

        return f"""
        <div class="node" data-col="{col_id}">
            {anchor_info}
            <span class="node-tag tag-{node_type}">{node_type}</span>
            <div class="content {content_class}">{node["c"]}</div>
            {extra}
            <div class="meta-strip">
                <span>UID: {node_id.split('_')[1] if '_' in node_id else node_id}</span>
                <span>COORD: {meta.get("bbox")}</span>
            </div>
            <div class="nested-flow">
                {nested_html}
            </div>
        </div>
        """
