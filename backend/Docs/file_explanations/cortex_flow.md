# Cortex Flow Documentation

This document explains the core files and the data flow within the `cortex` directory (`backend/services/loom_service/cortex`). The Cortex is responsible for transforming raw data shards into a mathematically robust, physically-simulated knowledge graph.

## Main Files and Responsibilities

1. **`SubstrateWeaver.py` (The Orchestrator)**
   - Acts as the primary entry point for the weaving process.
   - Takes raw text data (shards) and calculates initial physical properties (mass, energy, phase) based on word count and embedding significance.
   - Uses `HyperVectorCreation.py` to convert standard dense embeddings into Hyperdimensional Computing (HDC) bipolar vectors.
   - Delegates the heavy lifting of graph creation to the `DynamicFieldSubstrateEngine`.
   - Packages the resulting nodes and edges into chunked `.loom` shards and a master substrate hub for persistence.

2. **`HyperVectorCreation.py`**
   - Implements the `HyperVectorEngine`.
   - Converts standard continuous float embeddings into very high-dimensional (e.g., 8000-D) bipolar {-1, 1} semantic basis vectors. This allows for noise-resistant similarity comparisons using Hamming distance.

3. **`latent_field_cognition/dynamic_field_substrate`**
   - The physics engine. `SubstrateWeaver` ingests shards into this engine.
   - Runs a real-time physics loop where shards interact gravitationally based on their embeddings and mass.
   - Dense regions of shards "collapse" to spawn "macro_shards" (emergent concepts).

4. **`neural_viewer.py`**
   - Provides the `NeuralViewer` and `LoomNavigator` classes.
   - Reads the generated `.loom` binary files (using msgpack) to navigate the constructed knowledge graph.

5. **`TemporalCognitiveStream.py` & `ResonanceField/`**
   - Advanced cognitive models handling time-based processing and wave interference (resonance) across the semantic field.

## The Full Detailed Flow

1. **Initialization (`weave` method in `SubstrateWeaver.py`)**
   - Data shards are ingested. Each shard gets physical properties: `mass` (from word count/embedding strength), `energy`, and a random `phase`.

2. **Semantic Enrichment**
   - Continuous vector embeddings are generated for each shard via an external LLM service.
   - These embeddings are transformed into HDC bipolar vectors by the `HyperVectorEngine`.

3. **Physics Simulation (The Dynamic Field)**
   - Shards and their continuous embeddings are pushed into the `DynamicFieldSubstrateEngine`.
   - Chronological transitions are recorded.
   - The physics loop is started for a set duration, causing shards to attract one another based on semantic similarity and mass (gravitational collapse).

4. **Emergence**
   - The engine identifies clusters and creates new "macro_shards" to represent these high-density regions.
   - Causal entanglements and containment edges are established between regular shards and macro shards.

5. **Crystallization (Serialization)**
   - The `SubstrateWeaver` halts the physics engine.
   - It partitions the nodes (data vs. structural) into manageable limits (e.g., 1000 nodes per shard).
   - Writes the data to disk as multiple child `.loom` shards and one `substrate_master.loom` file using `msgpack` for efficient binary storage.
