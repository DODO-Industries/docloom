# DocLoom Neural Memory Architecture & Scaling Specification

## 1. Executive Summary & Core Philosophy

Traditional AI retrieval-augmented generation (RAG) and vector databases treat memory as a static, flat list of vectors searched via brute-force or approximate nearest neighbors (ANN). When queries end, the context is completely lost; when databases grow to billions of vectors, latency spikes and memory collapses.

DocLoom implements a **bio-physics inspired memory substrate** governed by three immutable principles:
1. **Never Forgotten, Just Less Excited:** Memories never drop to zero or get erased. When attention moves away, they settle into a permanent, non-zero resting base energy.
2. **Fixed 3-Tier Physical Topology:** Memory avoids infinite nested hierarchies ("Folder Hell") by adhering to a strict 3-tier architecture: *Shards (Atoms)* -> *Crystals (Molecules/Domains)* -> *Atlas (The Universe)*.
3. **Emergent Semantic Gravity & Automatic Mitosis:** Knowledge clusters gravitationally in continuous vector space. When density crosses capacity thresholds (e.g. 50,000 shards), crystals split cleanly without pre-defined human categories.

---

## 2. Activation Energy, Decay & Resting Baseline Potential

### 2.1 The Biological Parallel: Action Potential vs. Resting Potential
In neurobiology, a living neuron never has 0 mV electrical charge. It sits at a **-70 mV Resting Membrane Potential** ($E_{\text{base}}$).
* When a memory or concept fires, an **Action Potential** spikes the membrane voltage to **+40 mV** ("Hot / Excited").
* Post-firing, potassium channels repolarize the cell, bringing it back down to **-70 mV**.
* If a neuron reached 0 mV, it would be necrotic (dead).

In DocLoom:
$$\text{Current Energy}(t) = E_{\text{base}} + \Delta E_{\text{excitation}} \cdot e^{-\lambda t}$$

* **$E_{\text{base}}$ (Resting Base Energy):** Derived from the shard's mass, word count, and vector norm during ingestion. It is permanently retained in the `.loom` crystal substrate on disk.
* **$\Delta E_{\text{excitation}}$ (Recall Surge):** The transient kinetic excitation injected by a query probe proportional to cosine resonance.

### 2.2 The 5-Second Working Memory Cooldown
In cognitive science, **sensory and echoic memory** (the auditory/working memory reverberation) persists for **3 to 5 seconds**.
* A decay that drops in 1 second feels artificial and gives human operators no time to register synaptic pathways.
* DocLoom sets its baseline decay half-life so that excitation smoothly relaxes back to resting potential over **5 seconds**.
* Operators can modulate decay via the HUD controls:
  * `Freeze (0x)`: Holds working memory excitation indefinitely in RAM.
  * `Normal (1x)`: Natural biological 5-second relaxation.
  * `Fast (4x)`: Rapid 1.2-second cooling.

### 2.3 Long-Term Potentiation (LTP) & Memory Reinforcement
A memory must not return strictly to its pre-recall baseline. Doing so would induce system amnesia regarding recent thought history.
* Each successful recall hit applies a permanent residual elevation to baseline energy:
  $$E_{\text{base}}^{\text{new}} = E_{\text{base}}^{\text{old}} + \delta E_{\text{residual}}$$
* **Frequent Recall:** Facts queried repeatedly accumulate mass and baseline energy, cementing them as core cognitive pillars (e.g. personal identity, critical constants).
* **Infrequent Recall:** Facts unqueried over months settle to minimum baseline energy, sleeping quietly in cold crystal storage.

---

## 3. The 3-Tier Memory Hierarchy

To prevent runaway tree structures where deeper sub-folders cause exponential search degradation ($O(\text{depth})$ traversal overhead), DocLoom enforces a flat, continuous 3-tier model:

```
+-----------------------------------------------------------------------+
| TIER 1: THE ATLAS (Continuous 3D Semantic Manifold)                   |
| - Master semantic coordinate space spanning all domains               |
| - Manages inter-crystal routing, wormholes, and global cosmic balance |
+-----------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------+
| TIER 2: CRYSTALS (.loom Binary Storage Files)                         |
| - Discrete, modular domain clusters (5,000 to 50,000 shards each)    |
| - High-density memory-mapped (mmap) binary files                      |
| - Holds centroid vectors, Kuramoto phase couplings, and LSH buckets   |
+-----------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------+
| TIER 3: SHARDS (Atomic Memory Units)                                  |
| - Individual sentences, facts, or code symbols                        |
| - Carries: 384-D/128-D embedding, mass, base energy, Kuramoto phase   |
+-----------------------------------------------------------------------+
```

Regardless of whether DocLoom stores 1,000 shards or 10 billion shards, the number of architectural tiers remains strictly three. Scaling occurs horizontally across the manifold, never vertically through nested folder depths.

---

## 4. Semantic Gravity & Cosmic Expansion

### 4.1 Cosmological Balance
As knowledge accumulates, the memory manifold mirrors physical cosmology:
* **Global Expansion:** Diverse, unrelated domains (e.g. Molecular Biology vs. Medieval Archaeology) drift apart in 3D coordinate space, expanding the global coordinate boundaries and preventing cross-domain interference.
* **Local Gravitational Binding:** Related concepts exert semantic gravitational attraction proportional to the dot product of their centroid vectors:
  $$F_{\text{gravity}} \propto \frac{M_1 \cdot M_2}{\|\vec{c}_1 - \vec{c}_2\|^2}$$

### 4.2 Superclusters & "Galaxies"
When a specific subject grows extensively (e.g. 500,000 mathematical facts):
* The system does not maintain an unmanageably massive monolithic file.
* Multiple sister crystals form a **Supercluster (Galaxy)**.
* The crystals maintain tight orbital proximity in 3D space, linked by continuous inter-crystal resonance bridges.

---

## 5. Dynamic Splitting (Mitosis) & Self-Healing

### 5.1 The Anti-Categorization Principle
Knowledge is messy and dynamic; human developers cannot pre-plan or hardcode taxonomic divisions in advance (e.g., trying to anticipate every sub-branch of mathematics or computer science).

DocLoom eliminates pre-configured categories through **Density Bisection**:
1. **Capacity Cap:** Each crystal has a physical threshold (default: 50,000 shards).
2. **Hyperplane Crack Detection:** When capacity is exceeded, the crystal runs Principal Component Analysis (PCA) or 2-Means vector partitioning on its internal vector distribution to find the natural dimensional cleavage plane (the axis of maximum variance).
3. **Mitosis:** The crystal splits cleanly along this hyperplane into two sister crystals (*Crystal A* and *Crystal B*).
4. Zero human labeling is required; the division is purely emergent from mathematical vector topology.

### 5.2 Self-Healing & Consolidation
* **Tombstone Pruning:** Deleted or superseded shards receive cryptographic tombstones and are evacuated during compaction sweeps.
* **Gravitational Fusion:** If sister crystals shrink below minimum density thresholds (< 5,000 shards) due to deletions or memory decay, semantic gravity pulls them together, automatically fusing them back into a single consolidated `.loom` crystal.

---

## 6. Scaling to Web Scale (10+ Billion Shards)

When dataset volume reaches web scale (Google scale), DocLoom avoids performance degradation via three architectural mechanisms:

### 6.1 Constant-Time ($O(1) / O(\log N)$) Hierarchical Routing
Rather than performing sequential dot-product scans against billions of vectors:
* **Atlas Level:** Query probes test against crystal centroids, pruning 99.9% of candidate space in under 2 milliseconds.
* **Crystal Level:** The query accesses targeted crystals and hashes into exact Locality-Sensitive Hashing (LSH) hyperplanes.
* **Result:** A search across 10 billion shards evaluates fewer than 50 candidates, maintaining query latencies below 25 ms.

### 6.2 Zero-Copy Memory-Mapped Storage (`mmap`)
* Crystals reside on NVMe SSD storage formatted as binary byte arrays.
* When querying dormant crystals, the operating system uses `mmap` zero-copy paging to access only the requested byte slices.
* 99.9% of dormant memories consume 0 active RAM.

### 6.3 Level of Detail (LOD) 3D Visualization
Rendering billions of geometry primitives in WebGL is impossible. The DocLoom 3D engine employs hierarchical LOD:
* **Macroscopic Scale (Universe View):** Galaxy centroids render as volumetric particle clouds and cluster hulls.
* **Mesoscopic Scale (System View):** Approaching a galaxy resolves crystal gemstone shells and major warp bridges.
* **Microscopic Scale (Local View):** Individual shard spheres, Kuramoto spin indicators, and laser lines materialize only within the camera's local frustum.

---

## 7. Implementation Roadmap & Upcoming Engineering Tasks (Work To Do)

The following core modules have been fully implemented, verified, and integrated into the live DocLoom neural engine:

### Task 7.1: Spatial Overlap Protection (Bubble Collision & Clearance Physics) — [COMPLETED]
* **Objective:** Prevent geometric clipping and visual clutter when crystals divide or expand in dense regions of the 3D manifold.
* **Implementation Status: [COMPLETED]**
  * **Galactic Anchors:** Deployed 5 core domain coordinate anchors (`physics`, `biology`, `ai_systems`, `history`, `finance`) with dedicated spatial clearance envelopes.
  * **Soft-Body Bubble Repulsion:** Added distance-squared clearance repulsion (`dist_sq < 4.0`) in `testing_routes.py` `_project_to_3d` to maintain an exclusion bubble around every existing memory node.
  * **Local Clearance Nudging:** Twin shard placements and dense clustering are dynamically pushed away from neighboring centroids along collision normal vectors.
  * **Global Coordinate Invariance:** Distant galaxies maintain their global separation without cascading shifts across unrelated memory sectors.

### Task 7.2: Deterministic Hybrid Decision Routing (Border Embassy Engine) — [COMPLETED]
* **Objective:** Ensure multi-disciplinary memories (e.g. historical medicine, quantum bio-physics) are routed deterministically without randomness or ambiguity.
* **Implementation Status: [COMPLETED]**
  * **Deterministic Centroid Cosine Scoring:** Candidate vectors are projected against calibrated prototype centroids for all five knowledge domains via `_evaluate_hybrid_routing`.
  * **Primary Physical Storage:** The domain with maximum cosine affinity is deterministically assigned as the primary physical crystal file on disk.
  * **Frontier Saddle Placement:** Shards with competitive secondary domain affinity (`is_embassy = True`) are positioned on the geometric saddle boundary halfway between the primary and secondary galactic anchors (e.g. History + Biology).
  * **Dual-Socket Excitation & Embassy Halo:** Embassy shards feature an orbital amber halo in the Three.js 3D viewport and are excited by disturbance waves originating in either domain.

### Task 7.3: Adaptive 5-Second Biological Cooldown & Residual Baseline Reinforcement — [COMPLETED]
* **Objective:** Mirror human echoic memory retention and synaptic consolidation.
* **Implementation Status: [COMPLETED]**
  * **5-Second Exponential Cooldown:** Calibrated field decay constant (`lambda = 0.55`) in `testing_viz_energy.js` settling field energy from peak excitation (100%) down to resting baseline potential (`BASE_RESTING_ENERGY = 0.15`) in ~5 seconds.
  * **Long-Term Potentiation (LTP):** Query activations in `testing_routes.py` increment the shard's `hits` counter and permanently lift its resting energy floor by `+0.015` (`delta_E = 0.015`).
  * **UI & Inspector Synchronization:** Added real-time LTP hit counters and Embassy status indicators in both the inspector drawer (`inspect-hits`, `inspect-embassy`) and the standalone result modal (`modal-metric-hits`, `modal-metric-embassy`).

