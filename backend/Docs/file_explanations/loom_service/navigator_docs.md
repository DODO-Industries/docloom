# Navigator Docs (`navigator.py`)

The `LoomNavigator` is the "search brain" of the service. It handles the traversal of the graph to find the most relevant context for a given query.

## 1. Core Responsibilities
- **Entry Point Detection:** Finding the best starting node in the graph.
- **Graph Traversal:** Implementing BFS and Beam Search.
- **Cognitive Jumps:** Jumping across the graph using concepts or centroids.
- **Lazy Load Coordination:** Requesting shards from the `Viewer` as it encounters new nodes.

## 2. Navigation Strategies

### A. Beam Search (Focused Reasoning)
Unlike a simple search, Beam Search follows the most "truthful" paths.
- **Width:** Limits the number of parallel paths explored.
- **Heuristic:** Uses `node.truth_score` and `edge.score` to calculate the next best hop.
- **Purpose:** Follows a chain of thought (e.g., Heading $\rightarrow$ Sub-Heading $\rightarrow$ Key Paragraph).

### B. Concept Jump
Uses the `Concept Bridge` (a hash map of keywords to node IDs).
1. Query is split into words.
2. Words are hashed (MD5).
3. Navigator lookups hashes in the bridge to find "jump points."

### C. Heuristic Centroid Jump
When no direct concepts are found:
1. Navigator calculates cosine similarity between the query embedding and all **Atlas Centroids**.
2. It "jumps" to the center of the most similar Atlas.

## 3. Key Methods
- `beam_search(start_id, width, depth)`: Performs the focused reasoning traversal.
- `bfs_traversal(start_id, depth)`: Performs a broad neighborhood search.
- `heuristic_jump(query_emb, centroids)`: Finds the best high-level entry point.

## 4. Interaction with other files
- Operates on data loaded by `LoomViewer`.
- Calls `node_resolver` (callback to Viewer) to trigger lazy-loading of shards.
