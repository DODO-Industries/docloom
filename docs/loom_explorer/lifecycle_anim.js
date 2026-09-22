/**
 * DOCLOOM DATA LIFECYCLE INTERACTIVE ANIMATION & STEPPER
 * Animates a single piece of data through its entire lifetime:
 * 1. Raw Text Arrival
 * 2. Embedding & Physics Scalars
 * 3. Seed Socket & Momentum Warp
 * 4. Atlas Galaxy Routing
 * 5. 128-bit LSH Resonance-Jump Bucketing
 * 6. Disk Binary Append (.loom / .idx)
 * 7. Active RAM Ledger & Cognitive Tick
 * 8. Spatiotemporal Decay (Forgetting)
 * 9. Query Recall, Gravity & Kuramoto Sync
 * 10. Sleep Consolidation (NREM / REM)
 */

const LIFECYCLE_STAGES = [
  {
    id: 1,
    title: "1. Raw Intake (Perception)",
    short: "Raw Intake",
    tag: "Input Boundary",
    color: "#00d2d3",
    desc: "A piece of information arrives into DocLoom (e.g., 'Car A braked suddenly at the intersection'). At this stage, it is just an unformatted text string.",
    file: "module_loom/routes/embedding_routes.py",
    func: "embed_endpoint() -> embedding_service.embed_text()",
    dataState: {
      format: "Raw Text String",
      content: '"Car A braked suddenly at the intersection"',
      length: "42 characters (7 words)"
    },
    mechanicalAction: "System receives payload, assigns unique Blake2b hash ID: shard_8f4a1c09"
  },
  {
    id: 2,
    title: "2. Embedding & Physics Scalars",
    short: "Physics Scalars",
    tag: "Vector & Energy",
    color: "#a55eea",
    desc: "The text is passed through MiniLM to get a continuous 384-D float vector t. Then, DocLoom computes its physical personality: Mass m (information heaviness), Energy E, and Phase angle θ.",
    file: "module_loom/services/cortex/SubstrateWeaver.py",
    func: "weave() -> mass, energy, phase calculation",
    dataState: {
      vector: "Float32 array [384 dims] (||t|| = 1.0)",
      mass: "m = ln(7+1) * Σ|t_i| = 2.079 * 21.4 ≈ 44.5 (Heavy memory)",
      energy: "E = (1/384) * Σ|t_i| = 0.055",
      phase: "θ = atan2(sum_first_half, sum_second_half) = 2.41 rad"
    },
    mechanicalAction: "Non-zero phase seeds the Kuramoto oscillator so coupling works. Mass determines gravitational attraction."
  },
  {
    id: 3,
    title: "3. Procedural Seed Socket & Momentum Warp",
    short: "Seed & Warp",
    tag: "Procedural Physics",
    color: "#00d2d3",
    desc: "Instead of indexing immediately, DocLoom derives a procedural resting socket s from the 64-bit universe seed and shard_id (costing 0 disk I/O). The semantic mass m then physically warps the memory off its socket toward its true meaning t.",
    file: "module_loom/services/weaver/storage_physics.py",
    func: "LatentFieldPhysicsEngine.calculate_momentum_vector()",
    dataState: {
      socket: "s = RNG(SHA256(seed || 'micro_shard_8f4a1c09')) ~ Normal(0, 0.1)³⁸⁴",
      warpFactor: "w = 44.5 / (1.0 + 44.5) = 0.978 (97.8% warped toward meaning)",
      momentum: "p = norm(s + 0.978 * (t - s))"
    },
    mechanicalAction: "Heavy memory snaps almost completely to its true meaning; light memories stay anchored near seed sockets."
  },
  {
    id: 4,
    title: "4. Celestial Atlas Routing",
    short: "Atlas Routing",
    tag: "Galaxy Navigation",
    color: "#f39c12",
    desc: "The Global Atlas maintains centroids for all crystal galaxies (.loom files). It computes the cosine similarity between the shard's vector and each crystal's centroid, routing it to the nearest galaxy.",
    file: "module_loom/services/weaver/atlas_router.py",
    func: "GlobalAtlasRouter.route_vector(vector)",
    dataState: {
      availableCrystals: "['traffic_accidents.loom', 'weather_conditions.loom']",
      selected: "traffic_accidents.loom (Centroid Cosine = 0.842)",
      action: "Headroom checked (<500,000 shards). If full, Cellular Division splits crystal."
    },
    mechanicalAction: "Crystal centroid updates via running mean: C_new = (C_old * N + t) / (N + 1)."
  },
  {
    id: 5,
    title: "5. 128-bit LSH Resonance-Jump Bucketing",
    short: "LSH Bucketing",
    tag: "Microsecond Partition",
    color: "#2ecc71",
    desc: "Within the crystal, DocLoom maps the vector into a 128-bit sign signature (16 packed bytes). It computes Hamming distance against neighborhood leaders using a byte-wise popcount LUT.",
    file: "module_loom/services/weaver/weaver_coordinator.py",
    func: "WeaveBrainCoordinator._resolve_bucket()",
    dataState: {
      signature: "128-bit packed array: 10110010... (16 bytes)",
      minHammingDist: "18 bits difference (<= 32 threshold)",
      outcome: "⚡ ASSIMILATION into Leader Bucket #7"
    },
    mechanicalAction: "Assigns bucket_id = 7. Future recall queries will jump directly to bucket #7 in microseconds!"
  },
  {
    id: 6,
    title: "6. Physical Disk Append Execution",
    short: "Binary Append",
    tag: "Immutable Storage",
    color: "#3498db",
    desc: "The memory is appended to the binary .loom log (LOM2 format) in one seek and write. Existing records are never touched. The universe container journals the write for crash safety.",
    file: "module_loom/services/weaver/substrate_layout.py",
    func: "LoomStore.append_shard() & append_journal()",
    dataState: {
      binaryFile: "assets/.brain_data/traffic_accidents.loom",
      recordBytes: "[u32 rec_len][u8 type=1][u16 id_len][id_bytes][float32 x 384][float32 mass][u32 meta_len][msgpack_meta]",
      indexSnapshot: ".idx mmap updated with offset, mass, precomputed norm, and bucket_id"
    },
    mechanicalAction: "Permanent storage complete. Shard can survive crashes, restarts, and power loss."
  },
  {
    id: 7,
    title: "7. Active Thought & Cognitive Tick",
    short: "Cognitive Tick",
    tag: "Field Dynamics",
    color: "#e84393",
    desc: "The new memory enters the active RAM ledger with activation A = 1.0. It delivers a physical impulse to the shared latent field L (a 2nd-order mass-spring-damper) and links to the preceding memory in the causal graph.",
    file: "module_loom/services/weaver/weaver_coordinator.py",
    func: "process_cognitive_tick({shard_id: 0.8})",
    dataState: {
      ramLedger: "{activation: 1.0, phase: 2.41, hits: 0, energy: 1.0, resonance: 0.82}",
      latentVelocity: "v_field updated with input pulse g_in * u - gamma * w",
      causalEdge: "Edge created: shard_previous -> shard_8f4a1c09 (weight=1.0)"
    },
    mechanicalAction: "The memory is now awake and interacting with current working memory."
  },
  {
    id: 8,
    title: "8. Spatiotemporal Decay (Forgetting)",
    short: "Forgetting Curve",
    tag: "ACT-R Decay",
    color: "#e67e22",
    desc: "As time passes without this memory being queried, its activation decays continuously following the ACT-R logarithmic forgetting curve: A(t) = A_0 * exp(-lambda_eff * dt).",
    file: "module_loom/services/weaver/memory_fluidity.py",
    func: "DynamicMemoryFluidity.calculate_decay_batch()",
    dataState: {
      elapsedTime: "Δt = 360 seconds (6 minutes of inactivity)",
      decayRate: "λ_eff = 0.05 / (1 + ln(1 + 0 hits)) = 0.05",
      activationDrop: "A(t): 1.000 -> 0.740 -> 0.320 -> 0.080"
    },
    mechanicalAction: "If activation drops below 0.001, it is evicted from RAM into disk sleep (with a tombstone log)."
  },
  {
    id: 9,
    title: "9. Query Recall, Gravity & Kuramoto Sync",
    short: "Query & Wake",
    tag: "Plasticity & Recall",
    color: "#00cec9",
    desc: "A user asks: 'Did a vehicle stop suddenly?' The query routes to traffic_accidents.loom, jumps to bucket #7, and calculates warped Newtonian gravity. The memory wins top-k, hits increment (+1), activation spikes to 1.0, and its oscillator locks phase with other active thoughts.",
    file: "module_loom/services/weaver/weaver_coordinator.py",
    func: "WeaveBrainCoordinator.recall(learn=True)",
    dataState: {
      cosineSim: "cos = 0.91",
      gravityScore: "G = (1.0 * 44.5) / max((1 - 0.91)², 10^-4) = 44.5 / 0.0081 ≈ 5493.8",
      rankScore: "Score = G * (1 + 0.25 * A) = 5493.8 * 1.02 ≈ 5603.7 (#1 Winner!)",
      kuramotoSync: "Phase pulled toward group phase: θ = 2.41 -> 2.18 rad (Synchronized!)"
    },
    mechanicalAction: "Hits incremented to 1; effective decay rate drops (memory will resist forgetting longer next time!)."
  },
  {
    id: 10,
    title: "10. Sleep Consolidation (NREM / REM)",
    short: "Sleep Synthesis",
    tag: "Permanent Crystal",
    color: "#6c5ce7",
    desc: "When the system goes idle (>= 120s), a sleep cycle runs. Memories in recurring attractor basins are bundled. An LLM synthesizes a dense summary crystal, verified by FaithfulnessJudge. The original shard is marked 'superseded' (never deleted).",
    file: "module_loom/services/weaver/weaver_coordinator.py",
    func: "maybe_sleep_cycle() -> _synthesize_semantic_text()",
    dataState: {
      recurringConcept: "'Sudden braking incident at junction' (Frequency = 4)",
      synthesizedCrystal: "crystal_9a8b7c: 'Vehicle collision preceded by sudden braking at intersection...'",
      faithfulness: "Score = 0.95 (Faithfulness verified, 0 hallucinations)",
      provenance: "shard_8f4a1c09 marked superseded_by = crystal_9a8b7c"
    },
    mechanicalAction: "Fast recall now routes to the clean summary crystal, while the original raw shard is safely preserved on disk forever!"
  }
];

class LifecycleAnimationController {
  constructor() {
    this.currentStep = 0;
    this.isPlaying = false;
    this.timer = null;
    this.canvas = document.getElementById("lifecycle-canvas");
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.init();
  }

  init() {
    this.renderButtons();
    this.showStep(0);
    this.drawCanvas();
    this.setupListeners();
  }

  renderButtons() {
    const container = document.getElementById("lifecycle-steps-bar");
    if (!container) return;

    container.innerHTML = LIFECYCLE_STAGES.map((st, idx) => `
      <button class="step-btn ${idx === 0 ? 'active' : ''}" data-idx="${idx}" onclick="window.lifecycleController.goToStep(${idx})">
        <span class="step-num">${idx + 1}</span>
        <span class="step-label">${st.short}</span>
      </button>
    `).join("");
  }

  goToStep(idx) {
    this.currentStep = Math.max(0, Math.min(LIFECYCLE_STAGES.length - 1, idx));
    this.showStep(this.currentStep);
    this.drawCanvas();
  }

  next() {
    if (this.currentStep < LIFECYCLE_STAGES.length - 1) {
      this.goToStep(this.currentStep + 1);
    } else {
      this.goToStep(0);
    }
  }

  prev() {
    if (this.currentStep > 0) {
      this.goToStep(this.currentStep - 1);
    }
  }

  togglePlay() {
    this.isPlaying = !this.isPlaying;
    const playBtn = document.getElementById("lifecycle-play-btn");
    if (playBtn) playBtn.textContent = this.isPlaying ? "⏸ Pause Tour" : "▶ Play Full Lifecycle Tour";

    if (this.isPlaying) {
      this.timer = setInterval(() => {
        if (this.currentStep >= LIFECYCLE_STAGES.length - 1) {
          this.goToStep(0);
        } else {
          this.next();
        }
      }, 4200);
    } else {
      clearInterval(this.timer);
    }
  }

  showStep(idx) {
    const stage = LIFECYCLE_STAGES[idx];
    if (!stage) return;

    // Update active button
    document.querySelectorAll(".step-btn").forEach((btn, i) => {
      btn.classList.toggle("active", i === idx);
    });

    // Populate card details
    document.getElementById("stage-title").textContent = stage.title;
    document.getElementById("stage-tag").textContent = stage.tag;
    document.getElementById("stage-tag").style.color = stage.color;
    document.getElementById("stage-desc").textContent = stage.desc;
    document.getElementById("stage-file").textContent = stage.file;
    document.getElementById("stage-func").textContent = stage.func;
    document.getElementById("stage-action").textContent = stage.mechanicalAction;

    // Format Data State
    const dataContainer = document.getElementById("stage-data-box");
    if (dataContainer) {
      dataContainer.innerHTML = Object.entries(stage.dataState).map(([k, v]) => `
        <div style="margin-bottom: 6px;">
          <strong style="color: var(--text-dim); text-transform: uppercase; font-size: 11px;">${k}:</strong>
          <span style="color: ${stage.color}; font-family: var(--font-mono); font-size: 12px; margin-left: 6px;">${v}</span>
        </div>
      `).join("");
    }
  }

  drawCanvas() {
    if (!this.canvas || !this.ctx) return;
    const { width, height } = this.canvas;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, width, height);

    const total = LIFECYCLE_STAGES.length;
    const pad = 40;
    const stepW = (width - pad * 2) / (total - 1);
    const cy = height / 2;

    // Draw connecting railway line
    ctx.beginPath();
    ctx.moveTo(pad, cy);
    ctx.lineTo(width - pad, cy);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 4;
    ctx.stroke();

    // Draw completed progress line
    const curX = pad + this.currentStep * stepW;
    ctx.beginPath();
    ctx.moveTo(pad, cy);
    ctx.lineTo(curX, cy);
    ctx.strokeStyle = LIFECYCLE_STAGES[this.currentStep].color;
    ctx.lineWidth = 4;
    ctx.stroke();

    // Draw station dots
    for (let i = 0; i < total; i++) {
      const x = pad + i * stepW;
      const isPast = i < this.currentStep;
      const isCurrent = i === this.currentStep;

      ctx.beginPath();
      ctx.arc(x, cy, isCurrent ? 12 : 7, 0, Math.PI * 2);

      if (isCurrent) {
        ctx.fillStyle = LIFECYCLE_STAGES[i].color;
        ctx.shadowColor = LIFECYCLE_STAGES[i].color;
        ctx.shadowBlur = 18;
      } else if (isPast) {
        ctx.fillStyle = LIFECYCLE_STAGES[i].color;
        ctx.shadowBlur = 0;
      } else {
        ctx.fillStyle = "#21262d";
        ctx.shadowBlur = 0;
      }

      ctx.fill();
      ctx.shadowBlur = 0;

      // Label number
      ctx.font = isCurrent ? "bold 11px sans-serif" : "10px sans-serif";
      ctx.fillStyle = isCurrent ? "#ffffff" : "#8b949e";
      ctx.textAlign = "center";
      ctx.fillText((i + 1).toString(), x, cy - 18);
    }
  }

  setupListeners() {
    const playBtn = document.getElementById("lifecycle-play-btn");
    const nextBtn = document.getElementById("lifecycle-next-btn");
    const prevBtn = document.getElementById("lifecycle-prev-btn");

    if (playBtn) playBtn.addEventListener("click", () => this.togglePlay());
    if (nextBtn) nextBtn.addEventListener("click", () => this.next());
    if (prevBtn) prevBtn.addEventListener("click", () => this.prev());
  }
}

window.initLifecycleAnimation = function() {
  window.lifecycleController = new LifecycleAnimationController();
};
