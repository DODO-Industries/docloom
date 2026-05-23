import os

class LoomVisualizer:
    @staticmethod
    def generate_html(loom_data, output_path):
        """Generates a premium, cognitive-aware visualization of the Loom Knowledge Graph."""
        nodes = loom_data.get("g", {}).get("n", {})
        edges = loom_data.get("g", {}).get("e", [])
        v_version = loom_data.get("v", "1.2")
        shard_info = f"Shard: {loom_data.get('shard', 0)}" if "shard" in loom_data else ""

        # Build adjacency
        adj = {}
        cognitive_adj = {}
        for e in edges:
            f, t, r = e["f"], e["t"], e["r"]
            if r in ["contains", "primary_ingestion"]:
                if f not in adj: adj[f] = []
                adj[f].append(e)
            else:
                if f not in cognitive_adj: cognitive_adj[f] = []
                cognitive_adj[f].append(e)

        root_ids = [nid for nid, node in nodes.items() if node["t"] == "root"]
        constellation_ids = [nid for nid, node in nodes.items() if node["t"] == "constellation"]
        atlas_ids = [nid for nid, node in nodes.items() if node["t"] == "atlas"]
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocLoom | Cognitive Neural Graph v{v_version}</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #05070a;
            --panel: rgba(11, 15, 25, 0.8);
            --accent: #818cf8;
            --accent-glow: rgba(129, 140, 248, 0.3);
            --text: #f1f5f9;
            --muted: #475569;
            --shard-h: #fbbf24;
            --shard: #34d399;
            --image: #fb7185;
            --table: #c084fc;
            --atlas: #38bdf8;
            --constellation: #a78bfa;
            --border: rgba(255, 255, 255, 0.08);
            --glass: blur(20px) saturate(180%);
        }}

        body {{
            background-color: var(--bg);
            background-image: 
                radial-gradient(circle at 0% 0%, rgba(129, 140, 248, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 100% 100%, rgba(56, 189, 248, 0.08) 0%, transparent 50%);
            color: var(--text);
            font-family: 'Outfit', sans-serif;
            margin: 0; padding: 40px; min-height: 100vh;
        }}

        .container {{ max-width: 1400px; margin: 0 auto; }}

        header {{
            display: flex; justify-content: space-between; align-items: flex-end;
            margin-bottom: 50px; padding-bottom: 20px; border-bottom: 1px solid var(--border);
        }}

        h1 {{ 
            font-size: 2.4rem; font-weight: 600; margin: 0; letter-spacing: -1px;
            background: linear-gradient(135deg, #fff 20%, #818cf8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}

        .tag-badge {{
            font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--muted);
            letter-spacing: 2px; text-transform: uppercase;
        }}

        .intelligence-grid {{
            display: grid; grid-template-columns: 320px 1fr; gap: 40px;
        }}

        /* SIDEBAR: Cognitive Atlases */
        .sidebar {{ position: sticky; top: 40px; height: calc(100vh - 80px); overflow-y: auto; }}
        .sidebar::-webkit-scrollbar {{ width: 3px; }}
        .sidebar::-webkit-scrollbar-thumb {{ background: var(--border); }}

        .section-lbl {{
            font-size: 0.75rem; font-weight: 600; color: var(--accent);
            text-transform: uppercase; letter-spacing: 3px; margin-bottom: 20px;
        }}

        .atlas-card {{
            background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
            padding: 18px; margin-bottom: 15px; transition: 0.3s;
        }}
        .atlas-card:hover {{ border-color: var(--atlas); box-shadow: 0 0 20px rgba(56, 189, 248, 0.15); }}
        .atlas-title {{ font-weight: 600; color: var(--atlas); font-size: 0.85rem; margin-bottom: 8px; }}
        .atlas-meta {{ font-size: 0.65rem; color: var(--muted); display: flex; gap: 12px; }}

        .const-card {{ border-left: 3px solid var(--constellation); margin-bottom: 30px; }}
        .const-lbl {{ font-size: 0.6rem; color: var(--constellation); letter-spacing: 2px; margin-bottom: 5px; }}

        /* MAIN FLOW: Neurons (Shards) */
        .neuron-stream {{ display: flex; flex-direction: column; gap: 40px; }}
        .atlas-block {{ margin-bottom: 40px; }}
        .atlas-header {{
            font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--atlas);
            margin-bottom: 20px; display: flex; align-items: center; gap: 15px;
        }}
        .atlas-header::after {{ content: ''; flex: 1; height: 1px; background: var(--border); }}

        .const-header {{
            font-size: 1.2rem; font-weight: 600; color: var(--constellation);
            margin: 40px 0 20px 0; border-bottom: 2px solid var(--constellation);
            padding-bottom: 10px;
        }}

        .shard-container {{
            display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 18px;
        }}

        .neuron {{
            background: var(--panel); backdrop-filter: var(--glass);
            border: 1px solid var(--border); border-radius: 12px;
            padding: 20px; position: relative; transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        .neuron:hover {{
            transform: translateY(-4px); border-color: var(--accent);
            box-shadow: 0 15px 35px rgba(0,0,0,0.4), 0 0 20px var(--accent-glow);
        }}

        .n-meta {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .pill {{
            font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; padding: 2px 8px;
            border-radius: 4px; text-transform: uppercase;
        }}
        .pill-shard_heading {{ background: rgba(251, 191, 36, 0.1); color: var(--shard-h); }}
        .pill-shard {{ background: rgba(52, 211, 153, 0.05); color: var(--shard); }}
        .pill-image {{ background: rgba(251, 113, 133, 0.1); color: var(--image); }}
        .pill-table {{ background: rgba(192, 132, 252, 0.1); color: var(--table); }}

        .truth-score {{ font-family: 'JetBrains Mono'; font-size: 0.6rem; color: var(--accent); opacity: 0.8; }}
        .n-content {{ font-size: 0.92rem; line-height: 1.6; color: #cbd5e1; }}
        .h-text {{ font-size: 1.25rem; font-weight: 600; color: var(--shard-h); }}

        .concepts {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 15px; }}
        .concept-tag {{
            font-size: 0.6rem; background: rgba(255,255,255,0.03); color: var(--muted);
            padding: 1px 8px; border-radius: 100px; border: 1px solid var(--border);
        }}

        .synapse {{
            margin-top: 15px; padding-top: 12px; border-top: 1px solid var(--border);
            font-size: 0.65rem; color: var(--accent);
        }}
        .synapse-icon {{ opacity: 0.6; margin-right: 4px; }}

        .media-asset {{ margin-top: 15px; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }}
        .media-asset img {{ width: 100%; display: block; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.75rem; }}
        td {{ padding: 8px; border: 1px solid var(--border); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>DocLoom Cognitive Loom</h1>
                <div class="tag-badge">{shard_info} // COGNITIVE_LAYER_V{v_version}</div>
            </div>
            <div class="section-lbl">Neural Knowledge Architecture</div>
        </header>

        <div class="intelligence-grid">
            <aside class="sidebar">
                <div class="section-lbl">Emergent Hierarchy</div>
                {LoomVisualizer._render_sidebar(root_ids[0] if root_ids else None, nodes, adj)}
            </aside>

            <main class="neuron-stream">
                {LoomVisualizer._render_cognitive_flow(root_ids[0] if root_ids else None, nodes, adj, cognitive_adj)}
            </main>
        </div>
    </div>
</body>
</html>
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    @staticmethod
    def _render_sidebar(root_id, nodes, adj):
        if not root_id or root_id not in adj: return ""
        html = ""
        
        # 1. Constellations
        const_edges = [e for e in adj[root_id] if nodes[e["t"]]["t"] == "constellation"]
        if not const_edges:
            # Fallback to direct atlases if no constellations
            atlas_edges = [e for e in adj[root_id] if nodes[e["t"]]["t"] == "atlas"]
            for ae in atlas_edges:
                html += LoomVisualizer._render_atlas_card(ae["t"], nodes, adj)
            return html

        for ce in const_edges:
            cid = ce["t"]
            node = nodes[cid]
            html += f'<div class="const-card"><div class="const-lbl">CONSTELLATION</div>'
            html += f'<div class="atlas-title">{node["c"]}</div>'
            
            for ae in adj.get(cid, []):
                html += LoomVisualizer._render_atlas_card(ae["t"], nodes, adj)
            html += '</div>'
        return html

    @staticmethod
    def _render_atlas_card(aid, nodes, adj):
        node = nodes[aid]
        children_count = len(adj.get(aid, []))
        return f"""
        <div class="atlas-card">
            <div class="atlas-title"># {node['c']}</div>
            <div class="atlas-meta">
                <span>Shards: {children_count}</span>
                <span>Density: {'Critical' if children_count > 15 else 'Balanced'}</span>
            </div>
        </div>
        """

    @staticmethod
    def _render_cognitive_flow(root_id, nodes, adj, cognitive_adj):
        if not root_id or root_id not in adj: return ""
        
        html = ""
        # SECTION 7: Multi-Level Hierarchy
        # Start from Constellations or directly from Atlases
        child_edges = adj[root_id]
        
        if any(nodes[e["t"]]["t"] == "constellation" for e in child_edges):
            for ce in [e for e in child_edges if nodes[e["t"]]["t"] == "constellation"]:
                cid = ce["t"]
                const_node = nodes[cid]
                html += f'<div class="const-header">CONSTELLATION: {const_node["c"].upper()}</div>'
                for ae in adj.get(cid, []):
                    html += LoomVisualizer._render_atlas_block(ae["t"], nodes, adj, cognitive_adj)
        else:
            for ae in [e for e in child_edges if nodes[e["t"]]["t"] == "atlas"]:
                html += LoomVisualizer._render_atlas_block(ae["t"], nodes, adj, cognitive_adj)
                
        return html

    @staticmethod
    def _render_atlas_block(aid, nodes, adj, cognitive_adj):
        atlas_node = nodes[aid]
        html = f'<div class="atlas-block">'
        html += f'<div class="atlas-header">NEURAL_CLUSTER: {atlas_node["c"].upper()}</div>'
        html += '<div class="shard-container">'
        
        for ce in adj.get(aid, []):
            html += LoomVisualizer._render_neuron(ce["t"], nodes, cognitive_adj)
        
        html += '</div></div>'
        return html

    @staticmethod
    def _render_neuron(nid, nodes, cognitive_adj):
        node = nodes[nid]
        meta = node.get("m", {})
        ntype = node["t"]
        
        if ntype not in ["shard", "shard_heading", "image", "table"]: return ""

        concepts = meta.get("concepts", [])
        concepts_html = "".join([f'<span class="concept-tag">{c}</span>' for c in concepts])
        
        truth_score = meta.get("truth_score", 0.0)
        truth_html = f'<span class="truth-score">ACT: {truth_score:.2f}</span>' if truth_score > 0 else ""

        extra = ""
        if ntype == "image" and meta.get("binary"):
            extra = f'<div class="media-asset"><img src="data:image/png;base64,{meta["binary"]}"></div>'
        elif ntype == "table" and meta.get("table"):
            rows = "".join([f"<tr>{''.join([f'<td>{str(c)}</td>' for c in r])}</tr>" for r in meta["table"]])
            extra = f'<div class="media-asset"><table>{rows}</table></div>'

        # SECTION 4: Cognitive Synapses
        synapses_html = ""
        if nid in cognitive_adj:
            for e in cognitive_adj[nid]:
                target_node = nodes.get(e["t"], {})
                target_type = target_node.get("t", "neuron")
                synapses_html += f'<div class="synapse"><span class="synapse-icon">⚡</span> {e["r"].upper()}: {target_type} ({e["t"][-6:]})</div>'

        content_class = "h-text" if ntype == "shard_heading" else "n-content"

        return f"""
        <div class="neuron" id="{nid}">
            <div class="n-meta">
                <span class="pill pill-{ntype}">{ntype}</span>
                {truth_html}
            </div>
            <div class="{content_class}">{node['c']}</div>
            {extra}
            <div class="concepts">{concepts_html}</div>
            {synapses_html}
        </div>
        """
