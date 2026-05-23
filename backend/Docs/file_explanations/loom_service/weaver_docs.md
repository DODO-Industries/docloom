# Weaver Docs (`weaver.py`)

The `LoomWeaver` is the architect of the DocLoom system. It is responsible for transforming flat document extractions (paragraphs, images, tables) into a multi-layered, semantically linked graph.

## 1. Core Responsibilities
- **Structural Decomposition:** Splitting long text into atomic "shards."
- **Content-Addressable Hashing:** Ensuring every piece of knowledge has a unique, deterministic ID.
- **Hierarchical Clustering:** Organizing shards into Atlases and Constellations.
- **Semantic Linking:** Establishing top-K edges between related shards.
- **Graph Partitioning:** Serializing data into manageable msgpack shards.

## 2. Key Algorithms & Math

### A. Stable ID Generation (Blake2b)
To avoid duplicates and ensure consistency across document updates, IDs are generated using a 12-byte Blake2b hash of:
- Content string
- Node type
- Metadata (BBox, font size, page number)

### B. The Edge Scoring Function
Edges are created between nodes $A$ and $B$ if the score $S$ exceeds a threshold $\tau$ (default 0.6):
$$S = (0.2 \cdot J_{concepts}) + (0.4 \cdot \cos(\theta_{emb})) + (0.2 \cdot Struct) + (0.2 \cdot J_{entities})$$

### C. Knowledge Clustering (KMeans)
- **Atlases:** Content shards are clustered based on their semantic embeddings.
- **Constellations:** Atlas centroids are clustered to form high-level meta-structures.

## 3. Data Flow
1. **`weave()`**: Entry point. Processes all pages.
2. **`_add_node()`**: Creates nodes and builds the `Concept Bridge`.
3. **`_score_and_link_nodes()`**: Uses **ANN (Nearest Neighbors)** to find semantic neighbors without $O(N^2)$ complexity.
4. **`_save_shards()`**: Splits the final graph into a Master Atlas and multiple binary Shard files.

## 4. Interaction with other files
- Uses `LoomTransformer` for all NLP tasks (embeddings, concepts).
- Outputs `.loom` files consumed by `LoomViewer`.
