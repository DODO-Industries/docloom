# Loom Service: Technical Deep Dive (DocLoom Architecture v1.3)

This document provides an exhaustive explanation of the `loom_service`, its mathematical foundations, data structures, and component connections.

## 1. System Philosophy: Beyond RAG
The `loom_service` implements a **Cognitive Neural Graph**. Unlike standard RAG which retrieves isolated chunks of text, DocLoom preserves the **structural inheritance** and **semantic relationships** of a document, allowing an AI to "walk" the knowledge graph like a human mind.

---

## 2. Component Analysis

### A. `weaver.py` (The Architect)
**Role:** Transforms raw document extractions into the hierarchical Loom structure.
- **Content-Addressable Hashing (Blake2b):** Generates deterministic IDs based on content, bounding box, and font size. This ensures that identical content across revisions is deduplicated.
- **Structural Weaving:** Decomposes paragraphs into "Shards" (atomic units) while maintaining links to their parent headings.
- **Semantic Enrichment:** Calls the `Transformer` to attach embeddings, concepts, and entities to every node.
- **Clustering (KMeans):** 
    - **Atlases:** Groups ~20 shards into semantic clusters.
    - **Constellations:** Groups ~10 atlases into higher-level abstractions.
- **Serialization (Msgpack):** Partitions the graph into a `master_atlas` (hierarchy + bridge) and multiple `shards` (content + local edges) for efficient memory management.

### B. `transformer.py` (The Neural Engine)
**Role:** The NLP powerhouse.
- **Embeddings:** Uses `SentenceTransformer ('all-MiniLM-L6-v2')` for high-speed, high-accuracy vector representations.
- **Concept Extraction:** Uses **RAKE** (Rapid Automatic Keyword Extraction) to identify the "DNA" of a text block.
- **Entity Recognition:** Uses **SpaCy** (`en_core_web_sm`) to detect organizations, people, and locations.
- **Similarity Logic:** Implements both Jaccard (conceptual overlap) and Cosine (semantic similarity).

### C. `navigator.py` (The Search Engine)
**Role:** Navigates the graph to find relevant context.
- **Concept Jump:** Uses the `Concept Bridge` to immediately find nodes matching query keywords ($O(1)$ lookups).
- **Heuristic Jump:** Uses **Cosine Similarity** between query embeddings and **Atlas/Constellation Centroids** to find the right "neighborhood" in the graph.
- **Beam Search:** A focused traversal algorithm that follows high-scoring edges (`truth_score`) to assemble a coherent "reasoning path."
- **BFS Traversal:** Provides broad coverage when a query is general.

### D. `viewer.py` & `visualizer.py` (The Interface)
**Role:** Data access, auditing, and visualization.
- **Lazy Loading:** Implements an "Activation Cache." Only the master atlas is loaded initially; shards are pulled from disk only when a search "activates" them.
- **Neural Audit:** A debugging tool that prints the graph hierarchy and detects "Neural Gaps" (orphaned content).
- **Premium Visualization:** Generates a "Cognitive Loom" HTML interface with glassmorphism and interactive nodes.

### E. `loomServer_Service.py` (The Coordinator)
**Role:** The primary API entry point.
- **Hybrid Cognitive Retrieval:** Orchestrates the 3-step retrieval process:
    1. **Entry Point Detection:** Concept Bridge + Hub Jumper.
    2. **Local Activation:** Beam search from the entry point.
    3. **Context Assembly:** Synthesizing the path into a prompt-ready context.

---

## 3. Mathematical Foundations

### I. Edge Scoring Function ($S$)
When the Weaver connects two nodes ($A$ and $B$), it calculates a weighted score:
$$S(A, B) = (\alpha \cdot J(C_A, C_B)) + (\beta \cdot \cos(\theta_{E_A, E_B})) + (\gamma \cdot Struct) + (\delta \cdot J(Ent_A, Ent_B))$$
Where:
- $J$ = Jaccard Similarity
- $C$ = Concepts
- $E$ = Embeddings
- $Struct$ = 1.0 if on the same page, 0.0 otherwise
- $Ent$ = Named Entities

### II. Truth Resolution
Nodes aren't just present; they have a **Truth Score**. 
$$T_{new} = T_{current} + (Score \cdot 0.5)$$
This means nodes that are heavily referenced or semantically central to a topic become "brightest" in the graph.

---

## 4. Data Structure: The Loom Node
Every unit of knowledge is stored as a dictionary:
```json
{
  "t": "node_type (shard, atlas, constellation, image, table)",
  "c": "content_string",
  "m": {
    "emb": [1, 384],     // Vector Embedding
    "concepts": [],      // Keywords
    "entities": [],      // NER Tags
    "truth_score": 0.0,  // "Strength" of node
    "bbox": [x,y,w,h],   // Spatial position
    "binary": "base64"   // For images
  }
}
```

---

## 5. Connection Workflow
1. **Ingest:** PDF/Image extracted into flat JSON.
2. **Weave:** `LoomWeaver` hashes content $\rightarrow$ clusters nodes $\rightarrow$ scores edges.
3. **Partition:** Graph split into `atlas_name.loom` and `name_shard_N.loom`.
4. **Query:** `LoomServerService` embeds query $\rightarrow$ `Navigator` jumps to Atlas $\rightarrow$ `Navigator` beam searches Shards $\rightarrow$ `Viewer` lazy-loads content.

---

## 6. Current Stage: v1.3 (Advanced Expansion)
- [x] **Content-Addressable IDs:** No more duplicate nodes.
- [x] **Partitioned Graph:** Can handle 10,000+ page documents.
- [x] **Concept Bridge:** Sub-millisecond entry point detection.
- [x] **Hybrid Scoring:** Combines keyword, semantic, and structural logic.
- [x] **Lazy Loading:** Memory efficient activation.
