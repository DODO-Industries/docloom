# Viewer Docs (`viewer.py`)

The `LoomViewer` is the data management layer. It handles the loading, lazy-loading, and integrity auditing of Loom files.

## 1. Core Responsibilities
- **Data Loading:** Unpacking msgpack `.loom` files.
- **Lazy Loading (Sharding):** Loading only the Master Atlas initially and fetching data shards on-demand.
- **Activation Cache:** Keeping recently accessed shards in memory to prevent repeated disk reads.
- **Neural Audit:** Validating the graph for "orphans" and visualizing the hierarchy in the console.

## 2. Lazy Loading Logic
DocLoom uses a "Master/Shard" architecture to handle massive documents:
1. **Master Atlas** contains the hierarchy (Root $\rightarrow$ Constellation $\rightarrow$ Atlas) and the `shard_map`.
2. When `get_node(node_id)` is called:
    - If the node is structural (in Master), it's returned immediately.
    - If it's a data node, Viewer checks the `shard_map` to find which file contains it.
    - The specific `.loom_shard_N` file is loaded into the `shard_cache`.

## 3. Key Methods
- `load(path)`: Detects if the file is a Master or Standalone shard and initializes.
- `get_node(id)`: The primary data access method with built-in lazy-loading logic.
- `audit()`: Prints a tree view of the graph and checks for "Neural Gaps" (nodes with no incoming edges).
- `reasoning_audit()`: Runs a test Beam Search to demonstrate how the graph "thinks."

## 4. Interaction with other files
- Provides the data source for `LoomNavigator`.
- Uses `LoomVisualizer` to generate HTML reports.
- Used by `LoomServerService` to provide context for AI responses.
