# Weaver Flow Documentation

This document explains the core files and the data flow within the `weaver` directory (`backend/services/loom_service/weaver`). The Weaver acts as the runtime memory controller and routing layer, managing how conceptual "shards" are stored in physical files ("crystals") and recalled into active memory ("RAM ledger").

## Main Files and Responsibilities

1. **`weaver_coordinator.py` (`WeaveBrainCoordinator`)**
   - The master orchestrator. Coordinates the ingestion of new shards and the recall of existing ones.
   - Manages an "Active Latent Cortex Ledger" (a RAM ledger) of currently active/awake thoughts.
   - Handles sub-second boot sequences (reconstructing state from snapshots and journals) and crystallizing checkpoints.

2. **`seed_core.py` (`UniverseSeedCore`)**
   - Uses a deterministic seed (from `universe.metric` or `atlas.capnp`) to generate micro-scaffolds and macro-offsets. This ensures deterministic routing and structural growth.

3. **`atlas_router.py` (`GlobalAtlasRouter`)**
   - Maintains the spatial topology of where data is stored.
   - Routes incoming momentum vectors to the most appropriate target `.loom` file (crystal) based on centroid proximities.
   - Handles "Cellular Division": if a crystal exceeds its maximum shard limit, the atlas splits it into a new sub-sector.

4. **`storage_physics.py` (`LatentFieldPhysicsEngine`)**
   - Calculates the physics of memory storage and recall.
   - Evaluates "Momentum Vectors" during ingestion.
   - Calculates "Wave Resonance" (excitation gain), "Kuramoto Coupling" (phase locking), and "Gravitational Attraction" between queries and stored shards during recall.

5. **`memory_fluidity.py` (`DynamicMemoryFluidity`)**
   - Manages spatiotemporal decay. Calculates how fast an active thought fades from the RAM ledger based on time and a decay rate (`lambda_base`).
   - Executes the "Deep Sleep Protocol" to evict dormant nodes.

6. **`substrate_layout.py` (`LoomSubstrate`)**
   - The lowest level I/O layer. Interacts directly with the physical binary files (`.loom`).
   - Handles zero-copy memory mapping to fetch vectors without loading everything into memory.

## The Full Detailed Flow

### Ingestion Flow (Writing)
1. **Seed Evaluation**: `WeaveBrainCoordinator` asks the `UniverseSeedCore` for a deterministic structural scaffold based on the shard ID.
2. **Field Interference**: The `LatentFieldPhysicsEngine` calculates a "Momentum Vector" combining the shard's true embedding, the seed scaffold, and its mass.
3. **Atlas Capture**: The `GlobalAtlasRouter` routes the momentum vector to the closest `.loom` crystal. If the crystal is too full, it fractures (cellular division) and registers a new sub-centroid.
4. **Substrate Execution**: `LoomSubstrate` appends the shard, metadata, and vectors to the target physical `.loom` file.
5. **Ledger Update**: The shard is added to the active RAM ledger and logged to the `cortex_journal.bin`.

### Recall Flow (Querying)
1. **Decay Check**: `WeaveBrainCoordinator` enforces continuous decay. Any node in the RAM ledger with activation $\leq 0.001$ is evicted (Deep Sleep).
2. **Topology Sweep**: The `GlobalAtlasRouter` identifies the closest target crystal for the incoming query vector.
3. **Zero-Copy Fetch**: `LoomSubstrate` memory maps the target crystal, scanning shards.
4. **Wave Resonance & Gravity**: The `LatentFieldPhysicsEngine` calculates similarity (HDC overlap) and calculates an excitation gain to wake up sleeping nodes, as well as a final gravitational score combining activation and mass.
5. **Kuramoto Coupling**: The active phase angles of all awake nodes in the RAM ledger are synchronized (phase locked) to mimic coherent thought.
6. **Reinforcement**: Top results have their activations boosted and are kept/added to the active RAM ledger.
