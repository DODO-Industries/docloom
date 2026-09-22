/**
 * DOCLOOM INTERACTIVE SIMULATIONS & VISUAL CHARTS
 * Provides live physical simulations:
 * 1. Kuramoto Coupled Oscillator Phase Synchronization
 * 2. Procedural Seed Socket vs Semantic Mass Momentum Warp
 * 3. Warped Newtonian Gravity vs Cosine Distance Curve
 * 4. 128-bit LSH Hamming Jump Assimilate vs Accommodate
 * 5. Causal Spreading Activation Interactive Network
 */

// ============================================================================
// 1. KURAMOTO PHASE OSCILLATOR SIMULATION
// ============================================================================
class KuramotoSimulation {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    this.N = 32;
    this.phases = new Float32Array(this.N);
    this.omegas = new Float32Array(this.N);
    this.K = 0.15; // Coupling strength
    this.running = true;
    this.init();
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  init() {
    for (let i = 0; i < this.N; i++) {
      this.phases[i] = Math.random() * Math.PI * 2;
      this.omegas[i] = (Math.random() - 0.5) * 0.04; // small natural frequency variation
    }
  }

  step() {
    // Closed-form O(N) Kuramoto update (from DocLoom's weaver_coordinator.py)
    let s1 = 0, s2 = 0;
    for (let j = 0; j < this.N; j++) {
      s1 += Math.sin(this.phases[j]);
      s2 += Math.cos(this.phases[j]);
    }
    const dt = 0.8;
    for (let i = 0; i < this.N; i++) {
      const d_theta = (this.K / this.N) * (Math.cos(this.phases[i]) * s1 - Math.sin(this.phases[i]) * s2);
      this.phases[i] = (this.phases[i] + this.omegas[i] + d_theta * dt + Math.PI * 2) % (Math.PI * 2);
    }
  }

  getOrderParameter() {
    let sumCos = 0, sumSin = 0;
    for (let i = 0; i < this.N; i++) {
      sumCos += Math.cos(this.phases[i]);
      sumSin += Math.sin(this.phases[i]);
    }
    return Math.sqrt(sumCos * sumCos + sumSin * sumSin) / this.N;
  }

  animate() {
    if (this.running) this.step();
    this.draw();
    requestAnimationFrame(this.animate);
  }

  draw() {
    const { width, height } = this.canvas;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, width, height);

    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.38;

    // Draw reference circle
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw center coherence hub
    const r = this.getOrderParameter();
    ctx.beginPath();
    ctx.arc(cx, cy, 6 + r * 16, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(165, 94, 234, ${0.3 + r * 0.6})`;
    ctx.fill();

    // Draw oscillators on the ring
    for (let i = 0; i < this.N; i++) {
      const th = this.phases[i];
      const px = cx + Math.cos(th) * radius;
      const py = cy + Math.sin(th) * radius;

      // Color based on phase angle
      const hue = Math.floor((th / (Math.PI * 2)) * 360);
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fillStyle = `hsl(${hue}, 85%, 60%)`;
      ctx.shadowColor = `hsl(${hue}, 85%, 60%)`;
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Display order parameter r badge
    const badge = document.getElementById("kuramoto-order-badge");
    if (badge) {
      badge.textContent = `Coherence r = ${r.toFixed(3)} ${r > 0.85 ? '⚡ Synchronized Phase Lock' : '〰️ Dispersed'}`;
    }
  }
}

// ============================================================================
// 2. MASS MOMENTUM WARP INTERACTIVE VISUALIZER
// ============================================================================
class MomentumWarpVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    this.mass = 1.0;
    this.scaffoldWarp = 1.0;
    this.draw();
  }

  setMass(m) {
    this.mass = parseFloat(m);
    this.draw();
  }

  draw() {
    const { width, height } = this.canvas;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, width, height);

    const pad = 60;
    const socketX = pad;
    const socketY = height / 2;
    const meaningX = width - pad;
    const meaningY = height / 2;

    // w = m / (SCAFFOLD_WARP + m)
    const w = this.mass / (this.scaffoldWarp + Math.max(0, this.mass));
    const currentX = socketX + w * (meaningX - socketX);
    const currentY = socketY;

    // Connecting line
    ctx.beginPath();
    ctx.moveTo(socketX, socketY);
    ctx.lineTo(meaningX, meaningY);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 4;
    ctx.stroke();

    // Traversed path line
    ctx.beginPath();
    ctx.moveTo(socketX, socketY);
    ctx.lineTo(currentX, currentY);
    ctx.strokeStyle = "var(--accent-teal, #00d2d3)";
    ctx.lineWidth = 4;
    ctx.stroke();

    // Resting Socket s
    ctx.beginPath();
    ctx.arc(socketX, socketY, 10, 0, Math.PI * 2);
    ctx.fillStyle = "#8b949e";
    ctx.fill();
    ctx.font = "12px monospace";
    ctx.fillStyle = "#8b949e";
    ctx.fillText("Seed Socket s (m≈0)", socketX - 25, socketY - 18);

    // True Meaning t
    ctx.beginPath();
    ctx.arc(meaningX, meaningY, 10, 0, Math.PI * 2);
    ctx.fillStyle = "#a55eea";
    ctx.fill();
    ctx.fillText("True Vector t (m→∞)", meaningX - 70, meaningY - 18);

    // Warped Momentum Position p
    ctx.beginPath();
    ctx.arc(currentX, currentY, 12, 0, Math.PI * 2);
    ctx.fillStyle = "#00d2d3";
    ctx.shadowColor = "#00d2d3";
    ctx.shadowBlur = 15;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Update text readouts
    const statsEl = document.getElementById("warp-stats");
    if (statsEl) {
      statsEl.innerHTML = `
        <strong>Mass (m):</strong> ${this.mass.toFixed(1)} &nbsp;|&nbsp; 
        <strong>Warp Factor (w):</strong> ${w.toFixed(3)} (${(w * 100).toFixed(1)}% toward semantic position) &nbsp;|&nbsp;
        <strong>Interpretation:</strong> ${w > 0.8 ? "Heavy shard snapped to meaning" : w < 0.2 ? "Light shard anchored near seed socket" : "Hybrid momentum state"}
      `;
    }
  }
}

// ============================================================================
// 3. WARPED NEWTONIAN GRAVITY VS COSINE CHART
// ============================================================================
class GravityCurveVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    this.massA = 1.0;
    this.massB = 1.0;
    this.draw();
  }

  setMassB(m) {
    this.massB = parseFloat(m);
    this.draw();
  }

  draw() {
    const { width, height } = this.canvas;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, width, height);

    const padLeft = 40;
    const padBottom = 30;
    const padTop = 20;
    const padRight = 20;
    const plotW = width - padLeft - padRight;
    const plotH = height - padBottom - padTop;

    // Axes
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, height - padBottom);
    ctx.lineTo(width - padRight, height - padBottom);
    ctx.stroke();

    // Axis Labels
    ctx.font = "10px monospace";
    ctx.fillStyle = "#8b949e";
    ctx.fillText("cos = 0.0", padLeft, height - 10);
    ctx.fillText("cos = 1.0", width - padRight - 50, height - 10);
    ctx.fillText("Gravity G", 5, padTop + 10);

    // Plot G = (m_a * m_b) / max((1 - cos)^2, 0.0001) clamped for plot
    const maxPlotG = 250;
    ctx.beginPath();
    for (let px = 0; px <= plotW; px++) {
      const cos = px / plotW;
      const dist = 1 - cos;
      const denom = Math.max(dist * dist, 0.0001);
      const G = (this.massA * this.massB) / denom;
      const clampedG = Math.min(G, maxPlotG);

      const py = height - padBottom - (clampedG / maxPlotG) * plotH;
      if (px === 0) ctx.moveTo(padLeft + px, py);
      else ctx.lineTo(padLeft + px, py);
    }
    ctx.strokeStyle = "#00d2d3";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = "#00d2d3";
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}

// ============================================================================
// 4. LSH 128-BIT HAMMING ASSIMILATE VS ACCOMMODATE PLAYGROUND
// ============================================================================
function runLshHammingDemo(bitsDiff = 20) {
  const container = document.getElementById("lsh-demo-container");
  if (!container) return;

  const threshold = 32;
  const isAssimilate = bitsDiff <= threshold;

  container.innerHTML = `
    <div style="display: flex; gap: 20px; align-items: center; flex-wrap: wrap;">
      <div style="flex: 1; min-width: 260px;">
        <label style="font-size: 13px; font-family: monospace; display: block; margin-bottom: 8px;">
          Simulated Hamming Distance: <strong>${bitsDiff} / 128 bits</strong>
        </label>
        <input type="range" min="0" max="80" value="${bitsDiff}" style="width: 100%;" 
               oninput="runLshHammingDemo(parseInt(this.value))">
      </div>
      <div style="padding: 12px 18px; border-radius: 8px; font-family: monospace; font-size: 13px; 
                  background: ${isAssimilate ? 'rgba(46, 204, 113, 0.15)' : 'rgba(235, 94, 234, 0.15)'}; 
                  border: 1px solid ${isAssimilate ? '#2ecc71' : '#a55eea'}; 
                  color: ${isAssimilate ? '#2ecc71' : '#a55eea'};">
        ${isAssimilate ? '⚡ ASSIMILATE: Joins existing Leader Bucket' : '🌟 ACCOMMODATE: Spawns New Leader Neighborhood'}
      </div>
    </div>
  `;
}

// Global visualizer instances
let kuramotoSim = null;
let momentumWarpViz = null;
let gravityViz = null;

window.initLoomSimulations = function() {
  kuramotoSim = new KuramotoSimulation("kuramoto-canvas");
  momentumWarpViz = new MomentumWarpVisualizer("warp-canvas");
  gravityViz = new GravityCurveVisualizer("gravity-canvas");
  runLshHammingDemo(22);

  // Bind sliders
  const kSlider = document.getElementById("kuramoto-k-slider");
  if (kSlider) {
    kSlider.addEventListener("input", (e) => {
      if (kuramotoSim) kuramotoSim.K = parseFloat(e.target.value);
      const valEl = document.getElementById("kuramoto-k-val");
      if (valEl) valEl.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  const warpSlider = document.getElementById("warp-mass-slider");
  if (warpSlider) {
    warpSlider.addEventListener("input", (e) => {
      if (momentumWarpViz) momentumWarpViz.setMass(e.target.value);
    });
  }

  const gravSlider = document.getElementById("gravity-mass-slider");
  if (gravSlider) {
    gravSlider.addEventListener("input", (e) => {
      if (gravityViz) gravityViz.setMassB(e.target.value);
      const valEl = document.getElementById("gravity-mass-val");
      if (valEl) valEl.textContent = parseFloat(e.target.value).toFixed(1);
    });
  }
};
