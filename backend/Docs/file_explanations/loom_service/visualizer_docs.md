# Visualizer Docs (`visualizer.py`)

The `LoomVisualizer` generates a high-end, interactive HTML representation of the knowledge graph.

## 1. Design Aesthetics
The visualizer is built with modern web aesthetics in mind:
- **Glassmorphism:** Uses `backdrop-filter: blur` and translucent panels.
- **Neural Color Palette:**
    - **Atlases:** Sky Blue
    - **Constellations:** Deep Purple
    - **Shards:** Emerald Green
    - **Images/Tables:** Rose/Lavender
- **Interactive Hover Effects:** Nodes elevate and glow when hovered.

## 2. Structural Rendering
- **Sidebar (Emergent Hierarchy):** Shows the tree structure from Root down to Atlases.
- **Main Stream (Cognitive Flow):** Groups neurons by their Atlas clusters.
- **Synapses:** Visualizes relationships (Edges) between nodes, showing connection types like "Causality" or "Similarity."

## 3. Key Method: `generate_html`
This static method takes the graph data and writes a self-contained HTML file. It handles:
- **Image Injection:** Injects base64 image data directly into the HTML.
- **Table Formatting:** Renders extracted table data into clean, styled HTML tables.
- **Cognitive Metadata:** Displays `truth_score` (Activation) and extracted concepts for every node.

## 4. Interaction with other files
- Called by `LoomViewer.audit()` to produce a visual report of the document.
- Consumes the standard Loom JSON/Dict format produced by the Weaver.
