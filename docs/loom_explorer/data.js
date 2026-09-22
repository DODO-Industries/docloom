/**
 * DOCLOOM ARCHITECTURAL DRILL-DOWN DATA REGISTRY
 * Exhaustive catalog of every module, file, function, formula, and storage schema.
 */

const LOOM_DATA = {
  systemOverview: {
    title: "DocLoom Neuromorphic Cognitive Substrate",
    tagline: "A Physics-Driven, Procedural Memory Universe for Low-Power Edge Intelligence",
    coreParadigm: [
      {
        title: "Two Vector Worlds",
        desc: "Operational Substrate lives in continuous ℝ³⁸⁴ (MiniLM float32 vectors for routing, gravity, and recall). Symbolic Substrate lives in {-1, +1}⁸⁰⁰⁰ (HDC bipolar vectors for bit-packed algebra, permutation binding, and concept bundling)."
      },
      {
        title: "Procedural Universe",
        desc: "Every memory's resting socket is deterministically computed from a 64-bit seed without reading disk. Semantic mass physically warps shards off their sockets toward their meaning."
      },
      {
        title: "Append-Only Immutable Logs",
        desc: "Memory crystals (.loom) are pure append-only virtual file systems. Nothing is ever overwritten or deleted. Faster reads via zero-copy mmap (.idx), and consolidated memories are marked superseded, not destroyed."
      },
      {
        title: "Living Physics Dynamics",
        desc: "Memory decay follows ACT-R logarithmic friction. Memories synchronize phases through closed-form Kuramoto coupling, while a 2nd-order mass-spring-damper field drives thought assemblies and NREM/REM sleep."
      }
    ]
  },

  modules: [
    {
      id: "weaver",
      name: "Weaver Core & Storage Substrate",
      color: "#0e7f97",
      description: "Coordinates memory ingestion, atlas routing, physical binary layout, physics kernels, and living recall dynamics.",
      files: [
        {
          path: "module_loom/services/weaver/weaver_coordinator.py",
          name: "weaver_coordinator.py",
          lines: 2585,
          role: "Master Runtime Orchestrator",
          summary: "The central nervous system of DocLoom. Manages ingestion pipelines, serving/learning recall modes, living RAM ledger, Kuramoto phase synchrony, and sleep cycles.",
          classes: [
            {
              name: "WeaveBrainCoordinator",
              methods: [
                {
                  name: "ingest_shard(shard_id, true_vector, text, mass, metadata)",
                  desc: "Writes a single memory. Steps: 1) Evaluate seed socket; 2) Warp momentum vector; 3) Route to crystal via Atlas; 4) LSH bucket resolution; 5) Substrate append; 6) RAM ledger entry; 7) Cognitive tick; 8) Causal edge.",
                  inputs: "shard_id (str), true_vector (ndarray 384-D), text (str), mass (float)",
                  outputs: "target_crystal_path (str)",
                  storage: "Appends to crystal_*.loom and journals to universe.loom"
                },
                {
                  name: "recall(query_vector, top_k, query_phase, learn)",
                  desc: "Dual-mode memory retrieval. Serving Mode (learn=False): pure sub-2ms read-only top-k ranking. Learning Mode (learn=True): continuous decay, atlas routing, LSH Hamming jump, causal hop, gravitational scoring, Kuramoto phase locking, and wake/sleep plasticity.",
                  inputs: "query_vector (ndarray), top_k (int), learn (bool)",
                  outputs: "List of top_k dicts with shard_id, text, score, hits, activation",
                  storage: "Reads .idx mmap zero-copy; updates RAM ledger and journal if learn=True"
                },
                {
                  name: "process_cognitive_tick(resonance_input)",
                  desc: "Drives a 2nd-order mass-spring-damper field. Calculates field coherence order parameter, field turbulence entropy, and dynamic energy budget. Clusters resonating nodes into cognitive assemblies.",
                  inputs: "resonance_input: Dict[shard_id, intensity]",
                  outputs: "Optional[CognitiveAssembly]"
                },
                {
                  name: "maybe_sleep_cycle(force)",
                  desc: "Triggers consolidation when idle (≥120s) or surprise accumulates. NREM: reinforces member causal edges. REM: synthesizes dense semantic crystal via LLM/template, checks faithfulness, and marks sources superseded.",
                  inputs: "force (bool)",
                  outputs: "Optional[Dict] summarizing consolidated crystals"
                },
                {
                  name: "enforce_spatiotemporal_decay(current_time)",
                  desc: "Vectorized ACT-R exponential decay across all active RAM ledger states. Evicts dead memories (activation ≤ 0.001) into disk sleep with tombstones.",
                  inputs: "current_time (float)",
                  outputs: "None (mutates ram_ledger in place)"
                }
              ]
            }
          ]
        },
        {
          path: "module_loom/services/weaver/substrate_layout.py",
          name: "substrate_layout.py",
          lines: 1013,
          role: "Physical Binary Storage Layout (LOM2 / LIX2)",
          summary: "Implements the append-only binary virtual file system. No database dependency; pure C-struct packed files with mmap acceleration.",
          classes: [
            {
              name: "LoomStore",
              methods: [
                {
                  name: "append_shard(shard_id, vector, mass, meta)",
                  desc: "Appends [rec_len][type=1][id_len][id_bytes][vec_f32][mass_f32][meta_len][msgpack_meta] to .loom log. Zero rewrite.",
                  inputs: "shard_id (str), vector (ndarray), mass (float), meta (dict)",
                  outputs: "crystal_idx (int)"
                },
                {
                  name: "checkpoint()",
                  desc: "Flushes the in-memory delta into a disposable .idx snapshot ('LIX2') containing contiguous float32 matrix, norms, byte offsets, and bucket_ids.",
                  inputs: "None",
                  outputs: "None"
                },
                {
                  name: "gather_vectors(indices)",
                  desc: "Zero-copy gathers float32 candidate vectors directly from mmap buffer for sub-millisecond batch matrix multiplication.",
                  inputs: "indices (ndarray)",
                  outputs: "cand_vectors (ndarray shape [len, dim])"
                }
              ]
            }
          ]
        },
        {
          path: "module_loom/services/weaver/universe_container.py",
          name: "universe_container.py",
          lines: 231,
          role: "Single-File Universe Container (LUNI)",
          summary: "Consolidates multi-file brain sprawl into a single universe.loom file with a 1024-byte header, named binary segments, Table of Contents (TOC), and append-only journal tail.",
          classes: [
            {
              name: "UniverseContainer",
              methods: [
                {
                  name: "append_journal(blob)",
                  desc: "O(1) append of typed transaction records (JREC_TXN, JREC_CRYSTAL) at EOF without rewriting container header.",
                  inputs: "blob (bytes)",
                  outputs: "None"
                },
                {
                  name: "rebuild(segments, journal_tail)",
                  desc: "Atomic checkpoint rewrite using temporary file + os.replace. Packing named segments contiguously with msgpack TOC.",
                  inputs: "segments (dict), journal_tail (bytes)",
                  outputs: "None"
                }
              ]
            }
          ]
        },
        {
          path: "module_loom/services/weaver/storage_physics.py",
          name: "storage_physics.py",
          lines: 215,
          role: "Physics & Mechanics Math Kernels",
          summary: "Pure vectorized NumPy mathematical kernels for momentum warping, warped Newtonian gravity, wave excitation gain, and Kuramoto oscillator coupling.",
          classes: [
            {
              name: "LatentFieldPhysicsEngine",
              methods: [
                {
                  name: "calculate_momentum_vector(true_vector, seed_scaffold, mass)",
                  desc: "Warps seed socket toward semantic vector based on mass factor w = m / (SCAFFOLD_WARP + m).",
                  inputs: "true_vector (ndarray), seed_scaffold (ndarray), mass (float)",
                  outputs: "momentum (normalized ndarray)"
                },
                {
                  name: "calculate_gravitational_attraction_batch(cos_warped, mass_b, mass_a)",
                  desc: "Vectorized Newtonian gravity: G = (m_a * m_b) / max((1 - cos)², ε).",
                  inputs: "cos_warped (ndarray), mass_b (ndarray), mass_a (float)",
                  outputs: "grav (ndarray)"
                },
                {
                  name: "calculate_excitation_gain_batch(cos, query_phase, cell_phase)",
                  desc: "Phase-resonance gain: gain = η * cos * cos(θ_q - θ_cell).",
                  inputs: "cos (ndarray), query_phase (float), cell_phase (ndarray)",
                  outputs: "gain (ndarray)"
                }
              ]
            }
          ]
        },
        {
          path: "module_loom/services/weaver/atlas_router.py",
          name: "atlas_router.py",
          lines: 286,
          role: "Global Celestial Atlas & Centroid Router",
          summary: "Maintains centroids of all crystal galaxies. Routes vectors via maximum cosine similarity and triggers Cellular Division when a crystal is full.",
          classes: [
            {
              name: "GlobalAtlasRouter",
              methods: [
                {
                  name: "route_vector(vector)",
                  desc: "Finds the crystal whose centroid maximizes cosine similarity to the vector.",
                  inputs: "vector (ndarray)",
                  outputs: "target_crystal_path (str)"
                },
                {
                  name: "update_centroid(crystal_path, vector)",
                  desc: "Updates crystal center of mass via running mean: C_new = (C_old * n + v) / (n + 1).",
                  inputs: "crystal_path (str), vector (ndarray)",
                  outputs: "None"
                }
              ]
            }
          ]
        },
        {
          path: "module_loom/services/weaver/seed_core.py",
          name: "seed_core.py",
          lines: 74,
          role: "Procedural Universe Seed Engine",
          summary: "Derives 3D/high-dimensional coordinate sockets across macro, meso, and micro scales from a single 64-bit integer seed using deterministic SHA-256 RNG hashing.",
          classes: [
            {
              name: "UniverseSeedCore",
              methods: [
                {
                  name: "get_micro_socket(shard_id)",
                  desc: "RNG(SHA256(seed || 'micro_' + shard_id)).Normal(0, 0.1)^d.",
                  inputs: "shard_id (str)",
                  outputs: "socket (ndarray shape [dim])"
                },
                {
                  name: "get_macro_offset(crystal_name)",
                  desc: "Deterministic perturbation vector used for cellular division splits.",
                  inputs: "crystal_name (str)",
                  outputs: "offset (ndarray shape [dim])"
                }
              ]
            }
          ]
        },
        {
          path: "module_loom/services/weaver/memory_fluidity.py",
          name: "memory_fluidity.py",
          lines: 97,
          role: "ACT-R Cognitive Forgetting Curves",
          summary: "Simulates biological memory retention and forgetting. Frequent recall increases inertia, reducing future decay rates.",
          classes: [
            {
              name: "DynamicMemoryFluidity",
              methods: [
                {
                  name: "calculate_decay_batch(init_act, last_recalled, now, hits)",
                  desc: "λ_eff = λ / (1 + ln(1 + hits)). A(t) = A_0 * exp(-λ_eff * Δt).",
                  inputs: "init_act, last_recalled, now, hits (ndarrays)",
                  outputs: "decayed_activation (ndarray)"
                },
                {
                  name: "reinforce(current_time, hits)",
                  desc: "Increments recall hit count and updates timestamp upon query hit.",
                  inputs: "current_time (float), hits (int)",
                  outputs: "Tuple[current_time, hits + 1]"
                }
              ]
            }
          ]
        }
      ]
    },

    {
      id: "cortex",
      name: "Cortex & Hyperdimensional Computing",
      color: "#7a45c0",
      description: "Transforms text into symbolic 8000-D bipolar HDC vectors, handles predictive causal inference, and models emergent cognitive assemblies.",
      files: [
        {
          path: "module_loom/services/cortex/HyperVectorCreation.py",
          name: "HyperVectorCreation.py",
          lines: 395,
          role: "Hyperdimensional Computing (HDC) Algebra Engine",
          summary: "Implements D=8000 bipolar {-1, +1} vector symbolic architecture: Johnson-Lindenstrauss random projection, cyclic permutation binding, superposition bundling, and bit-packing.",
          classes: [
            {
              name: "HyperVectorEngine",
              methods: [
                {
                  name: "get_semantic_basis_vector(text, embedding)",
                  desc: "Bridges continuous ℝ³⁸⁴ to HDC {-1, +1}⁸⁰⁰⁰ via sign(P * t). Preserves semantic cosine similarity in symbolic space.",
                  inputs: "text (str), embedding (ndarray)",
                  outputs: "bipolar_vector (ndarray int8)"
                },
                {
                  name: "permute(vector, shift)",
                  desc: "Cyclic roll ρ^p(v) to encode sequential order or structural roles.",
                  inputs: "vector (ndarray), shift (int)",
                  outputs: "permuted_vector (ndarray)"
                },
                {
                  name: "bundle(vectors, weights)",
                  desc: "Superposition of multiple vectors: B = sign(Σ w_i v_i / ||w||). Remembers all member concepts simultaneously.",
                  inputs: "vectors (List[ndarray]), weights (List[float])",
                  outputs: "bundled_vector (ndarray)"
                },
                {
                  name: "pack_bipolar(vector)",
                  desc: "Compresses 8000 bipolar elements into 1000 bytes (1 bit per dimension) + base64 string.",
                  inputs: "vector (ndarray)",
                  outputs: "packed_str (str)"
                }
              ]
            }
          ]
        },
        {
          path: "module_loom/services/cortex/SubstrateWeaver.py",
          name: "SubstrateWeaver.py",
          lines: 231,
          role: "Bulk Document Ingestion & HDC Graph Builder",
          summary: "Extracts entities and concepts, converts text to HDC vectors, calculates physics scalars (mass, energy, phase), and constructs a spring-physics graph before writing to LoomStore.",
          classes: [
            {
              name: "SubstrateWeaver",
              methods: [
                {
                  name: "weave(data, output_path, incremental)",
                  desc: "Batch-weaves documents: semantic enrichment, HDC projection, mass/energy/phase derivation, concept bridge construction, and chronological transition stitching.",
                  inputs: "data (List[dict]), output_path (str)",
                  outputs: "dict with graph metrics"
                }
              ]
            }
          ]
        },
        {
          path: "module_loom/services/cortex/latent_field_cognition/predictive_processing.py",
          name: "predictive_processing.py",
          lines: 275,
          role: "Active Inference & Causal Graph",
          summary: "Directed graph engine linking memories by causality. Discards temporal noise using prediction-error discounting; prunes stale unreinforced edges.",
          classes: [
            {
              name: "CausalGraphs",
              methods: [
                {
                  name: "record_transition(from_sid, to_sid, weight, prediction_error)",
                  desc: "Updates directed edge with surprise discounting: strength = w * (1 - error). Rejects edges below CAUSAL_THRESHOLD.",
                  inputs: "from_sid (str), to_sid (str), weight (float), prediction_error (float)",
                  outputs: "None"
                },
                {
                  name: "decay_and_prune(stale_age, prune_threshold)",
                  desc: "Ages out unreinforced causal edges; removes disconnected orphan nodes.",
                  inputs: "stale_age (float), prune_threshold (float)",
                  outputs: "int (number of edges pruned)"
                }
              ]
            }
          ]
        },
        {
          path: "module_loom/services/cortex/latent_field_cognition/cognitive_field_substrate.py",
          name: "cognitive_field_substrate.py",
          lines: 390,
          role: "Thermodynamic Latent Field Substrate",
          summary: "Computes Kuramoto coherence order parameter r ∈ [0, 1], field entropy turbulence S, and updates thermodynamic energy budget.",
          classes: [
            {
              name: "CognitiveMetrics",
              methods: [
                {
                  name: "compute_order_parameter(vectors)",
                  desc: "Kuramoto order parameter r = (1/N) Σ (x_i / ||x_i||) • x̄. Quantifies alignment of active thoughts.",
                  inputs: "vectors (ndarray [N, dim])",
                  outputs: "r (float in [0, 1])"
                },
                {
                  name: "compute_field_entropy(current_field, prev_field)",
                  desc: "Turbulence S = clip(||L - L_prev||, 0, 1). Measures destabilization rate.",
                  inputs: "current_field (ndarray), prev_field (ndarray)",
                  outputs: "S (float)"
                }
              ]
            }
          ]
        },
        {
          path: "module_loom/services/cortex/latent_field_cognition/attractor_basin_compilation.py",
          name: "attractor_basin_compilation.py",
          lines: 236,
          role: "Attractor Basin & Assembly Compiler",
          summary: "Clusters co-resonating shards into single-concept cognitive assemblies guarded by a strict coherence gate (ASSEMBLY_COHERENCE_MIN). Tracks recurring attractor basins for sleep consolidation.",
          classes: [
            {
              name: "AssemblyCompilation",
              methods: [
                {
                  name: "compile_assembly(resonance_nodes, dominant_concept)",
                  desc: "Filters co-resonating nodes against group anchor. If coherent, builds a CognitiveAssembly and increments concept frequency in attractor memory.",
                  inputs: "resonance_nodes (dict), dominant_concept (str)",
                  outputs: "Optional[CognitiveAssembly]"
                }
              ]
            }
          ]
        }
      ]
    },

    {
      id: "services_and_api",
      name: "API, Visualizer, Embeddings & LLM Services",
      color: "#2f8f5b",
      description: "FastAPI network boundaries, real-time WebGL streaming, circuit-broken embeddings, and LLM sleep consolidation.",
      files: [
        {
          path: "module_loom/app.py",
          name: "app.py",
          lines: 70,
          role: "FastAPI Application Entry Point",
          summary: "Mounts /embed and /loom/neuro endpoints with CORS and lifespan hooks to cleanly flush WeaveBrainCoordinator state on shutdown.",
          classes: [
            {
              name: "FastAPI App",
              methods: [
                {
                  name: "lifespan(app)",
                  desc: "Context manager that logs startup and invokes close_coordinator() on shutdown.",
                  inputs: "app (FastAPI)",
                  outputs: "AsyncGenerator"
                }
              ]
            }
          ]
        },
        {
          path: "module_loom/routes/visualizer_routes.py",
          name: "visualizer_routes.py",
          lines: 805,
          role: "WebSocket Neuromorphic Streamer",
          summary: "Streams live brain states to WebGL front end: 128-bit LSH Hamming sweeps, spatiotemporal wave bursts, Kuramoto phase coloring, and causal graph edge networks.",
          classes: [
            {
              name: "NeuroVisualizer Router",
              methods: [
                {
                  name: "websocket_neuro_feed(websocket)",
                  desc: "Broadcasts real-time frames with node coordinates, activations, oscillator phases, and crystal cluster groupings.",
                  inputs: "WebSocket connection",
                  outputs: "JSON event stream"
                }
              ]
            }
          ]
        },
        {
          path: "module_loom/services/embedding/embedding_service.py",
          name: "embedding_service.py",
          lines: 240,
          role: "4-Tier Circuit-Broken Embedding Provider",
          summary: "Converts text to 384-D vectors. Resilient 4-tier fallback (in-process -> HTTP /embed -> local SentenceTransformer -> deterministic dummy) with 120s circuit breaker.",
          classes: [
            {
              name: "EmbeddingService",
              methods: [
                {
                  name: "embed_text(text)",
                  desc: "Produces normalized 384-D float32 vector safely without hanging batch writes.",
                  inputs: "text (str)",
                  outputs: "ndarray float32"
                }
              ]
            }
          ]
        },
        {
          path: "module_loom/services/completion/completion_service.py",
          name: "completion_service.py",
          lines: 195,
          role: "Sleep Consolidation LLM Completion Service",
          summary: "Provides structured prompting to synthesize recurring memory shards into dense semantic crystals during the sleep cycle.",
          classes: [
            {
              name: "CompletionService",
              methods: [
                {
                  name: "complete(messages, max_tokens, temperature)",
                  desc: "Queries LLM provider (OpenAI/Ollama/Anthropic) with strict anti-hallucination system prompts.",
                  inputs: "messages (List[dict]), max_tokens (int)",
                  outputs: "str (synthesized text)"
                }
              ]
            }
          ]
        },
        {
          path: "module_loom/services/eval/faithfulness_judge.py",
          name: "faithfulness_judge.py",
          lines: 185,
          role: "Sleep-Cycle Faithfulness & Entailment Auditor",
          summary: "Validates LLM-synthesized crystals against source shards before committing. If hallucination is detected, falls back to deterministic template.",
          classes: [
            {
              name: "FaithfulnessJudge",
              methods: [
                {
                  name: "judge(synthesis, sources)",
                  desc: "Scores semantic claim entailment. Returns faithfulness score in [0, 1].",
                  inputs: "synthesis (str), sources (List[str])",
                  outputs: "FaithfulnessResult(score, status)"
                }
              ]
            }
          ]
        }
      ]
    }
  ],

  formulas: [
    {
      id: "scaffold_warp",
      name: "Procedural Seed Scaffold & Mass Momentum Warp",
      file: "storage_physics.py & seed_core.py",
      latex: "\\mathbf{s} = \\text{RNG}\\big(\\text{SHA256}(\\text{seed} \\parallel \\text{'micro\\_'} + \\text{id})\\big).\\mathcal{N}(0, 0.1)^d",
      latex2: "w = \\frac{m}{\\text{SCAFFOLD\\_WARP} + \\max(0, m)}, \\qquad \\mathbf{p} = \\frac{\\mathbf{s} + w(\\mathbf{t} - \\mathbf{s})}{\\|\\mathbf{s} + w(\\mathbf{t} - \\mathbf{s})\\|}",
      explanation: "Resting sockets are derived from a 64-bit seed without reading disk. Semantic mass warps the shard toward its true vector t. Light memories stay near their procedural socket; heavy memories snap to their true semantic meaning."
    },
    {
      id: "gravity",
      name: "Warped Newtonian Gravitational Attraction",
      file: "storage_physics.py",
      latex: "G_i = \\frac{m_q \\cdot m_i}{\\max\\big((1 - \\cos_i)^2,\\ \\varepsilon\\big)}, \\qquad \\varepsilon = 10^{-4}",
      latex2: "\\text{Score}_i = G_i \\cdot \\big(1 + \\text{RANK\\_ACTIVATION\\_WEIGHT} \\cdot A_i\\big)",
      explanation: "Newton's law applied to memory distance (1 - cos). As similarity approaches 1, distance approaches 0 and gravity skyrockets. Mass allows heavy core memories to outrank superficial close matches."
    },
    {
      id: "kuramoto",
      name: "Closed-Form Kuramoto Phase Coupling",
      file: "weaver_coordinator.py",
      latex: "S_1 = \\sum_j a_j \\sin\\theta_j, \\qquad S_2 = \\sum_j a_j \\cos\\theta_j",
      latex2: "\\Delta\\theta_i = \\frac{K}{N}\\big(\\cos\\theta_i \\cdot S_1 - \\sin\\theta_i \\cdot S_2\\big), \\quad \\theta_i \\leftarrow ((\\theta_i + \\Delta\\theta_i + \\pi) \\bmod 2\\pi) - \\pi",
      explanation: "Calculates Kuramoto coupled oscillator phase-locking in linear O(N) time instead of O(N²) by expanding the sine subtraction identity into two global scalar sums. Synchronizes co-active memories."
    },
    {
      id: "hdc_proj",
      name: "Johnson-Lindenstrauss HDC Projection & Binarization",
      file: "HyperVectorCreation.py",
      latex: "P_{ij} \\sim \\mathcal{N}\\left(0,\\ \\frac{1}{d}\\right), \\qquad \\mathbf{h}^{\\text{sem}} = \\text{sign}(P \\cdot \\mathbf{t}) \\in \\{-1, +1\\}^{8000}",
      latex2: "\\text{sim}_{HDC}(\\mathbf{a}, \\mathbf{b}) = \\frac{\\mathbf{a} \\cdot \\mathbf{b}}{D} = 1 - \\frac{2 \\cdot H(\\mathbf{a}, \\mathbf{b})}{D}",
      explanation: "Bridges continuous ℝ³⁸⁴ embedding space into 8000-D bipolar space. Preserves semantic cosine similarity while unlocking bitwise XOR binding, majority-gate bundling, and 1-bit-per-dimension packing."
    },
    {
      id: "actr_decay",
      name: "ACT-R Cognitive Forgetting Curve",
      file: "memory_fluidity.py",
      latex: "\\lambda_{\\text{eff}} = \\frac{\\lambda}{1 + \\ln(1 + \\text{hits})}, \\qquad A(t) = A_0 \\cdot e^{-\\lambda_{\\text{eff}} \\cdot \\Delta t}",
      latex2: "A \\le 0.001 \\implies \\text{Tombstone Eviction to Disk Sleep}",
      explanation: "Active memories continuously fade from RAM into disk sleep. Memories recalled frequently build friction, decaying progressively slower over time."
    },
    {
      id: "latent_field",
      name: "Second-Order Mass-Spring-Damper Latent Field",
      file: "cognitive_field_substrate.py",
      latex: "\\mathbf{v}_{\\text{field}} \\leftarrow \\beta \\mathbf{v}_{\\text{field}} + g_{\\text{in}} \\mathbf{u} + g_{\\text{sync}}(\\mathbf{w} - \\mathbf{L}_{\\text{prev}}) - \\gamma \\mathbf{w}",
      latex2: "\\mathbf{L} \\leftarrow \\frac{\\mathbf{L}_{\\text{prev}} + \\mathbf{v}_{\\text{field}}}{\\|\\cdot\\|}, \\qquad \\mathbf{w} \\leftarrow \\frac{(1-\\alpha)\\mathbf{w} + \\alpha \\mathbf{L}}{\\|\\cdot\\|}",
      explanation: "The shared latent field acts as a damped harmonic oscillator driven by recent inputs u, pulled toward working memory w, with competitive lateral inhibition γ."
    }
  ],

  binaryStorageLayouts: [
    {
      name: "universe.loom Container File ('LUNI')",
      file: "universe_container.py",
      magic: "LUNI (0x4C554E49)",
      headerSize: "1024 Bytes",
      structFormat: "<4sHQIIIQQ (magic[4], version[u16], seed[u64], dim[u32], flags[u32], gen[u32], toc_offset[u64], toc_len[u64])",
      body: "Contiguous named binary segments (atlas matrix, cortex tensors, RAM ledger snapshot). Zero-copy mmap.",
      tail: "Append-only journal tail. Records: JREC_TXN (u8=1, u16 len, u64 shard_hash, f32 act, f64 ts) & JREC_CRYSTAL (u8=2, u16 len, u8 state, u32 shards, u16 path_len, path, centroid[dim])."
    },
    {
      name: "Crystal Log File ('LOM2')",
      file: "substrate_layout.py",
      magic: "LOM2 (0x4C4F4D32)",
      headerSize: "64 Bytes",
      structFormat: "<4sHQIIQQ (magic[4], version[u16], seed[u64], vector_dim[u32], flags[u32], record_count[u64], data_end[u64])",
      body: "Append-only record stream. Never rewritten. Record format: [rec_len u32][type u8]. REC_SHARD: [id_len u16][id_utf8][vec float32 x dim][mass float32][meta_len u32][meta msgpack]. REC_LEADER: [16 bytes packed LSH bits].",
      tail: "Grows monotonically on every ingest. Crash-safe by construction."
    },
    {
      name: "Index Acceleration Snapshot ('LIX2')",
      file: "substrate_layout.py",
      magic: "LIX2 (0x4C495832)",
      headerSize: "32 Bytes",
      structFormat: "<4sHQQII (magic[4], version[u16], covered_data_end[u64], n_records[u64], dim[u32], n_leaders[u32])",
      body: "Zero-copy mmap layout: offsets[u64 x n], mass[f32 x n], norms[f32 x n], vectors[f32 x n x dim], leaders[16 bytes x L], ids[(u16 len + utf8) x n], bucket_ids[u16 x n].",
      tail: "Disposable derived cache. If missing or torn, rebuilt automatically from the .loom log."
    }
  ],

  futureVision: [
    {
      id: "continuous_injection",
      phase: "Phase 1 (Near-Term)",
      tag: "Soft Prompting",
      color: "#00d2d3",
      title: "Direct Continuous Vector Injection (Zero-Token Interface)",
      status: "Architected — Ready for Implementation",
      summary: "Bypass the string conversion and tokenization bottleneck entirely. Inject DocLoom memory vectors directly into small LLM Transformer layers as virtual continuous tokens.",
      breakthrough: "Eliminates string decoding and prompt formatting. Reduces context window overhead by ~90% and shrinks Time-To-First-Token (TTFT) to < 10 milliseconds on pure CPU.",
      deliverables: [
        "1-Layer Linear Projection Adapter (W_proj: 384 -> D_model)",
        "Direct input into inputs_embeds without tokenizer lookup",
        "Dense memory compression: 100 words represented in 1 virtual vector"
      ]
    },
    {
      id: "native_multimodal",
      phase: "Phase 2 (Mid-Term)",
      tag: "Multi-Sensor Fusion",
      color: "#a55eea",
      title: "Native Unified Multimodal Sensor Ingestion",
      status: "Concept Designed",
      summary: "Directly embed video bounding boxes (ByteTrack), audio waveforms, LiDAR points, and text into a shared joint continuous ℝ³⁸⁴ and 8000-D HDC space.",
      breakthrough: "Stops parsing images or audio into text strings first. A car screech sound, a video collision bbox, and the word 'crash' will land in the exact same crystal galaxy and attractor basin natively.",
      deliverables: [
        "MobileCLIP / SigLIP visual patch encoder integration",
        "Whisper / CLAP audio waveform projection adapter",
        "ByteTrack trajectory velocity vector encoder for autonomous edge tracking"
      ]
    },
    {
      id: "rust_silicon",
      phase: "Phase 3 (Hardware Milestone)",
      tag: "NPU & Bare Silicon",
      color: "#2ecc71",
      title: "Pure C++20 / Rust Micro-Engine (libdocloom)",
      status: "Roadmap Item",
      summary: "Port the core .loom storage layout, Kuramoto ODE solver, and bitwise LSH/HDC kernels into a zero-allocation, header-only C++20 / Rust library.",
      breakthrough: "Completely cuts out the Python runtime. Runs bare-metal on low-power edge chips (Intel NPU, Hailo-8, Rockchip RK3588, Raspberry Pi, Apple Neural Engine) consuming < 0.5 Watts.",
      deliverables: [
        "Header-only libdocloom with static memory footprint",
        "Hardware-accelerated bitwise AVX-512 / ARM NEON PopCount LUT",
        "Zero-copy direct DMA mmap for flash storage"
      ]
    },
    {
      id: "online_plasticity",
      phase: "Phase 4 (Advanced Cognition)",
      tag: "Online Learning",
      color: "#f39c12",
      title: "Bi-Directional Online Plasticity & LoRA Adaptation",
      status: "Research Frontier",
      summary: "Enable bidirectional learning where stabilized attractor basins in DocLoom trigger rank-1 parameter updates in the AI model's Low-Rank Adapters (LoRA).",
      breakthrough: "The AI genuinely learns new concepts permanently from daily conversation without requiring massive retraining runs or catastrophic forgetting.",
      deliverables: [
        "Attractor-driven LoRA weight updates on idle sleep cycles",
        "Hebbian weight modification in downstream attention layers",
        "Anti-catastrophic forgetting barrier guarded by FaithfulnessJudge"
      ]
    },
    {
      id: "swarm_loom",
      phase: "Phase 5 (Decentralized Scale)",
      tag: "Swarm Intelligence",
      color: "#3498db",
      title: "Decentralized Swarm Loom (Federated Peer-to-Peer Sync)",
      status: "Long-Range Vision",
      summary: "Multiple edge cameras or IoT nodes running local DocLoom crystals synchronized via lightweight gossip protocols.",
      breakthrough: "Instead of uploading gigabytes of raw video to central servers, edge devices sync only 8000-bit HDC signatures and causal edges across a shared 64-bit genesis universe seed.",
      deliverables: [
        "Peer-to-peer crystal gossip protocol",
        "Differential HDC bundle replication",
        "Decentralized causal graph consensus for smart cities and vehicle fleets"
      ]
    }
  ]
};

