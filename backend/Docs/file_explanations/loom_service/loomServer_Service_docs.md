# LoomServerService Docs (`loomServer_Service.py`)

The `LoomServerService` is the high-level coordinator (the "Controller") that bridges the gap between raw document processing and AI-driven retrieval.

## 1. Core Responsibilities
- **Text Pre-processing:** Decomposing arbitrary text into Loom-ready shards.
- **Hybrid Retrieval:** Orchestrating the search across the knowledge graph.
- **Context Synthesis:** Assembling search results into a format suitable for an LLM.

## 2. Hybrid Cognitive Retrieval Workflow
This is the core of the **Neural Activation** process:

1. **Query Embedding:** Converts the user's question into a 384-dimensional vector.
2. **Entry Point Detection:**
    - **Concept Bridge:** First, it checks if any words in the query match hashed concepts in the graph.
    - **Heuristic Jump:** If no concepts match, it calculates the most similar Atlas/Constellation hub using centroid similarity.
3. **Local Activation:**
    - Starting from the entry point, it triggers a **Beam Search** via the `Navigator`.
    - This follows the "strongest" semantic edges to find related context.
4. **Context Assembly:**
    - Gathers the content, scores, and metadata from all nodes in the activation path.
    - Returns a "Synthesis" object containing the most relevant knowledge units.

## 3. Key Methods
- `process_text(text)`: Used to test how raw text is broken down into semantic units.
- `activate_loom(viewer, query)`: The main search function. It takes a `LoomViewer` instance and a query string, returning the retrieved context.

## 4. Interaction with other files
- Orchestrates `LoomNavigator` and `LoomTransformer`.
- Acts as the primary interface for external API routes or the main backend application.
