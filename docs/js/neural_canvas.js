/**
 * DocLoom 60M Neural Reasoning Cortex — Ultra-Smooth 60FPS Batched Canvas Visualizer
 * Fully optimized: batched vector paths, zero shadowBlur stalls, lifecycle auto-pause.
 */
(function() {
  let canvas, ctx, animId = null;
  let animSpeed = 1.0;
  let pulseWave = -1.0;
  let time = 0;
  let isRunning = false;
  let eventsBound = false;

  const layers = [
    {
      id: "input",
      name: "Token Input",
      sub: "Text Stream",
      xRatio: 0.08,
      nodes: 5,
      color: "#94a3b8",
      halo: "rgba(148, 163, 184, 0.2)",
      file: "module_AI/model/tokenizer.py",
      tensor: "[Batch=1, SeqLen=16]",
      params: "Vocabulary V=32,000",
      desc: "Input tokens tokenized via Byte-Pair Encoding (BPE). Discrete token IDs fed into continuous space."
    },
    {
      id: "embed",
      name: "Token Embedding",
      sub: "32k x 512d",
      xRatio: 0.22,
      nodes: 7,
      color: "#06b6d4",
      halo: "rgba(6, 182, 212, 0.25)",
      file: "module_AI/model/backbone.py:28",
      tensor: "[Batch=1, SeqLen=16, Dim=512]",
      params: "16,384,000 Params",
      desc: "Projects token IDs to 512d continuous space with Rotary Positional Embeddings (RoPE)."
    },
    {
      id: "memory",
      name: "Memory Projection",
      sub: "12 Latent Slots",
      xRatio: 0.38,
      nodes: 8,
      isMemory: true,
      color: "#10b981",
      halo: "rgba(16, 185, 129, 0.3)",
      file: "module_AI/model/memory_projection.py",
      tensor: "[Batch=1, Slots=12, Dim=512]",
      params: "3,145,728 Params",
      desc: "Transforms 384d semantic vectors + 9d living physics ledger (393d) into 12 continuous latent tokens."
    },
    {
      id: "cross_attn",
      name: "Cross-Attention",
      sub: "8 Blocks x 8 Heads",
      xRatio: 0.55,
      nodes: 8,
      color: "#8b5cf6",
      halo: "rgba(139, 92, 246, 0.3)",
      file: "module_AI/model/docloom_model.py:44",
      tensor: "Q:[1,16,512], K,V:[1,12,512]",
      params: "8 Layers x 8 Heads (dk=64)",
      desc: "Text queries cross-attend to 12 memory slot tokens. Zero prompt bloat."
    },
    {
      id: "ffn",
      name: "SwiGLU FFN",
      sub: "512 -> 2048 -> 512",
      xRatio: 0.70,
      nodes: 7,
      color: "#38bdf8",
      halo: "rgba(56, 189, 248, 0.25)",
      file: "module_AI/model/backbone.py:85",
      tensor: "[Batch=1, SeqLen=16, 2048]",
      params: "3,145,728 Params / Block",
      desc: "Gated feed-forward network with Swish activation: (swish(x W_gate) * x W_up) W_down."
    },
    {
      id: "lora",
      name: "LoRA QA Skill",
      sub: "Rank r=16, α=32",
      xRatio: 0.84,
      nodes: 6,
      color: "#f59e0b",
      halo: "rgba(245, 158, 11, 0.3)",
      file: "module_AI/model/lora.py:18",
      tensor: "ΔW = (α/r) · (B · A)",
      params: "3,014,656 Params (Swappable)",
      desc: "Low-rank adapters enforcing crisp evidence citations in <|think|> before answering."
    },
    {
      id: "output",
      name: "Logits / Head",
      sub: "32k Softmax",
      xRatio: 0.95,
      nodes: 5,
      color: "#f43f5e",
      halo: "rgba(244, 63, 94, 0.25)",
      file: "module_AI/model/docloom_model.py:120",
      tensor: "[Batch=1, SeqLen=16, 32000]",
      params: "Tied to Embedding Weights",
      desc: "Computes vocabulary token probabilities. Top-p (0.9), temperature (0.7), repetition penalty (1.15)."
    }
  ];

  // Traveling photon particles
  const particles = [];
  for (let i = 0; i < 35; i++) {
    particles.push({
      layerIdx: Math.floor(Math.random() * (layers.length - 1)),
      progress: Math.random(),
      speed: 0.010 + Math.random() * 0.012,
      startNode: Math.floor(Math.random() * 5),
      endNode: Math.floor(Math.random() * 5)
    });
  }

  function start() {
    canvas = document.getElementById('neuralCanvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    resize();

    if (!eventsBound) {
      canvas.addEventListener('click', handleCanvasClick);
      canvas.addEventListener('mousemove', handleCanvasHover);
      window.addEventListener('resize', resize);
      eventsBound = true;
    }

    if (!isRunning) {
      isRunning = true;
      loop();
    }
  }

  function stop() {
    isRunning = false;
    if (animId) {
      cancelAnimationFrame(animId);
      animId = null;
    }
  }

  function resize() {
    if (!canvas || !canvas.parentElement) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    if (rect.width > 0) {
      canvas.width = rect.width;
      canvas.height = 360;
    }
  }

  function loop() {
    if (!isRunning) return;
    render();
    animId = requestAnimationFrame(loop);
  }

  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    time += 0.025 * animSpeed;

    const h = canvas.height;
    const w = canvas.width;

    if (pulseWave >= 0) {
      pulseWave += 0.018 * animSpeed;
      if (pulseWave > 1.25) pulseWave = -1.0;
    }

    // --- 1. BATCHED SYNAPTIC CONNECTIONS ---
    // Standard synapses batched into a single stroke call
    ctx.beginPath();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 0.8;

    for (let i = 0; i < layers.length - 1; i++) {
      const l1 = layers[i];
      const l2 = layers[i + 1];
      if (l1.isMemory || l2.isMemory) continue;

      const x1 = w * l1.xRatio;
      const x2 = w * l2.xRatio;
      const step1 = (h - 120) / (l1.nodes - 1);
      const step2 = (h - 120) / (l2.nodes - 1);

      for (let j = 0; j < l1.nodes; j++) {
        const y1 = 60 + j * step1;
        for (let k = 0; k < l2.nodes; k++) {
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, 60 + k * step2);
        }
      }
    }
    ctx.stroke();

    // Memory cross-attention synapses batched into green stroke
    ctx.beginPath();
    ctx.strokeStyle = "rgba(16, 185, 129, 0.12)";
    ctx.lineWidth = 1.2;

    for (let i = 0; i < layers.length - 1; i++) {
      const l1 = layers[i];
      const l2 = layers[i + 1];
      if (!l1.isMemory && !l2.isMemory) continue;

      const x1 = w * l1.xRatio;
      const x2 = w * l2.xRatio;
      const step1 = (h - 120) / (l1.nodes - 1);
      const step2 = (h - 120) / (l2.nodes - 1);

      for (let j = 0; j < l1.nodes; j++) {
        const y1 = 60 + j * step1;
        for (let k = 0; k < l2.nodes; k++) {
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, 60 + k * step2);
        }
      }
    }
    ctx.stroke();

    // Dynamic shockwave pulse overlay (batched)
    if (pulseWave >= 0) {
      ctx.beginPath();
      ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
      ctx.lineWidth = 1.6;

      for (let i = 0; i < layers.length - 1; i++) {
        const l1 = layers[i];
        const dist = Math.abs(l1.xRatio - pulseWave);
        if (dist > 0.12) continue;

        const l2 = layers[i + 1];
        const x1 = w * l1.xRatio;
        const x2 = w * l2.xRatio;
        const step1 = (h - 120) / (l1.nodes - 1);
        const step2 = (h - 120) / (l2.nodes - 1);

        for (let j = 0; j < l1.nodes; j++) {
          const y1 = 60 + j * step1;
          for (let k = 0; k < l2.nodes; k++) {
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, 60 + k * step2);
          }
        }
      }
      ctx.stroke();
    }

    // --- 2. BATCHED TRAVELING PHOTON PARTICLES ---
    ctx.beginPath();
    ctx.fillStyle = "#67e8f9";

    particles.forEach(p => {
      p.progress += p.speed * animSpeed;
      if (p.progress >= 1.0) {
        p.progress = 0;
        p.layerIdx = (p.layerIdx + 1) % (layers.length - 1);
        p.startNode = Math.floor(Math.random() * layers[p.layerIdx].nodes);
        p.endNode = Math.floor(Math.random() * layers[p.layerIdx + 1].nodes);
      }

      const l1 = layers[p.layerIdx];
      const l2 = layers[p.layerIdx + 1];
      const x1 = w * l1.xRatio;
      const x2 = w * l2.xRatio;
      const y1 = 60 + p.startNode * ((h - 120) / (l1.nodes - 1));
      const y2 = 60 + p.endNode * ((h - 120) / (l2.nodes - 1));

      const curX = x1 + (x2 - x1) * p.progress;
      const curY = y1 + (y2 - y1) * p.progress;

      ctx.moveTo(curX + 2.5, curY);
      ctx.arc(curX, curY, 2.5, 0, Math.PI * 2);
    });
    ctx.fill();

    // --- 3. LAYER NODES & LABELS ---
    layers.forEach((l, i) => {
      const cx = w * l.xRatio;
      const step = (h - 120) / (l.nodes - 1);

      // Label text
      ctx.fillStyle = l.color;
      ctx.font = "bold 12px 'Outfit', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(l.name, cx, 24);

      ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.fillText(l.sub, cx, 38);

      // Halos (Fast double-ring, zero shadowBlur)
      ctx.beginPath();
      ctx.fillStyle = l.halo;
      for (let j = 0; j < l.nodes; j++) {
        const cy = 60 + j * step;
        ctx.moveTo(cx + 9, cy);
        ctx.arc(cx, cy, 9, 0, Math.PI * 2);
      }
      ctx.fill();

      // Core circles
      ctx.beginPath();
      ctx.fillStyle = l.color;
      for (let j = 0; j < l.nodes; j++) {
        const cy = 60 + j * step;
        ctx.moveTo(cx + 5, cy);
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      }
      ctx.fill();

      // Memory slot ring indicators
      if (l.isMemory) {
        ctx.beginPath();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.2;
        for (let j = 0; j < l.nodes; j++) {
          const cy = 60 + j * step;
          ctx.moveTo(cx + 7, cy);
          ctx.arc(cx, cy, 7, 0, Math.PI * 2);
        }
        ctx.stroke();

        ctx.font = "bold 11px 'Outfit', sans-serif";
        ctx.fillStyle = "#10b981";
        ctx.fillText("▲ Living Physics [393d] Injection ▲", cx, h - 14);
      }
    });
  }

  function handleCanvasClick(e) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const w = canvas.width;

    let clickedLayer = null;
    layers.forEach(l => {
      const cx = w * l.xRatio;
      if (Math.abs(mouseX - cx) < 35) clickedLayer = l;
    });

    if (clickedLayer) showLayerHUD(clickedLayer);
  }

  function handleCanvasHover(e) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const w = canvas.width;

    let hovered = false;
    layers.forEach(l => {
      const cx = w * l.xRatio;
      if (Math.abs(mouseX - cx) < 35) hovered = true;
    });
    canvas.style.cursor = hovered ? "pointer" : "default";
  }

  function showLayerHUD(layer) {
    const modal = document.getElementById('neuralHUD');
    if (!modal) return;

    modal.innerHTML = `
      <button class="hud-close" onclick="document.getElementById('neuralHUD').style.display='none'">&times;</button>
      <div style="font-family:'Outfit'; font-size:16px; font-weight:800; color:${layer.color}; margin-bottom:4px;">
        ${layer.name} (${layer.sub})
      </div>
      <div style="font-size:11px; color:#94a3b8; font-family:'JetBrains Mono'; margin-bottom:12px;">
        ${layer.file}
      </div>
      <div style="background:#03060a; border:1px solid rgba(255,255,255,0.06); padding:10px; border-radius:6px; margin-bottom:10px; font-family:'JetBrains Mono'; font-size:12px;">
        <div style="color:#38bdf8;"><strong>Tensor Shape:</strong> ${layer.tensor}</div>
        <div style="color:#f59e0b; margin-top:4px;"><strong>Parameters:</strong> ${layer.params}</div>
      </div>
      <p style="font-size:12.5px; color:#cbd5e1; line-height:1.6;">
        ${layer.desc}
      </p>
    `;
    modal.style.display = 'block';
  }

  window.triggerNeuralPulse = function() {
    pulseWave = 0.0;
  };

  window.setNeuralSpeed = function(val) {
    animSpeed = parseFloat(val);
    const lbl = document.getElementById('neuralSpeedVal');
    if (lbl) lbl.innerText = animSpeed.toFixed(1) + "x";
  };

  window.startNeuralCanvas = start;
  window.stopNeuralCanvas = stop;
  window.initNeuralCanvas = start;
})();
