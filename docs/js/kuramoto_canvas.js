/**
 * DocLoom Kuramoto Non-Linear Coupled Oscillator & Wave Resonance Canvas
 * Fully optimized: batched vector paths, zero shadowBlur stalls, lifecycle auto-pause.
 */
(function() {
  let canvas, ctx, animId = null;
  let particles = [];
  let numParticles = 50;
  let K_coupling = 0.55;
  let omegaSpread = 0.05;
  let time = 0;
  let isRunning = false;
  let eventsBound = false;

  const clusters = [
    { name: "Medical Crystal", color: "#06b6d4", halo: "rgba(6, 182, 212, 0.25)", baseFreq: 0.02 },
    { name: "Code Crystal", color: "#8b5cf6", halo: "rgba(139, 92, 246, 0.25)", baseFreq: -0.015 },
    { name: "Legal Crystal", color: "#10b981", halo: "rgba(16, 185, 129, 0.25)", baseFreq: 0.035 }
  ];

  function start() {
    canvas = document.getElementById('kuramotoCanvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    resize();

    if (!eventsBound) {
      window.addEventListener('resize', resize);
      eventsBound = true;
    }

    if (particles.length === 0) resetParticles();

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

  function resetParticles() {
    particles = [];
    for (let i = 0; i < numParticles; i++) {
      const cIdx = i % clusters.length;
      const c = clusters[cIdx];
      particles.push({
        id: i,
        cluster: cIdx,
        theta: Math.random() * Math.PI * 2,
        omega: c.baseFreq + (Math.random() - 0.5) * omegaSpread,
        color: c.color,
        halo: c.halo
      });
    }
  }

  function loop() {
    if (!isRunning) return;
    updatePhysics();
    render();
    animId = requestAnimationFrame(loop);
  }

  // Numerical Integration: dθ_i / dt = ω_i + (K / N) * ∑ sin(θ_j - θ_i)
  function updatePhysics() {
    const N = particles.length;
    const dTheta = new Float32Array(N);

    for (let i = 0; i < N; i++) {
      let couplingSum = 0;
      const p_i = particles[i];
      for (let j = 0; j < N; j++) {
        couplingSum += Math.sin(particles[j].theta - p_i.theta);
      }
      dTheta[i] = p_i.omega + (K_coupling / N) * couplingSum;
    }

    for (let i = 0; i < N; i++) {
      particles[i].theta = (particles[i].theta + dTheta[i]) % (Math.PI * 2);
      if (particles[i].theta < 0) particles[i].theta += Math.PI * 2;
    }

    time += 0.035;
  }

  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const w = canvas.width;
    const h = canvas.height;

    const cx1 = w * 0.28;
    const cy1 = h * 0.5;
    const radius = Math.min(cx1 - 40, h * 0.38);
    const N = particles.length;

    // --- 1. POLAR PHASE SPACE ---
    // Reference circle & axes
    ctx.beginPath();
    ctx.arc(cx1, cy1, radius, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1.2;
    ctx.moveTo(cx1 - radius - 10, cy1);
    ctx.lineTo(cx1 + radius + 10, cy1);
    ctx.moveTo(cx1, cy1 - radius - 10);
    ctx.lineTo(cx1, cy1 + radius + 10);
    ctx.stroke();

    // Calculate Complex Order Parameter: r * exp(i * psi)
    let sumCos = 0;
    let sumSin = 0;
    for (let i = 0; i < N; i++) {
      sumCos += Math.cos(particles[i].theta);
      sumSin += Math.sin(particles[i].theta);
    }
    const orderR = Math.sqrt(sumCos * sumCos + sumSin * sumSin) / N;
    const orderPsi = Math.atan2(sumSin, sumCos);

    // Batched Harmonic Resonance Springs
    ctx.beginPath();
    ctx.strokeStyle = "rgba(6, 182, 212, 0.2)";
    ctx.lineWidth = 0.8;

    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        let diff = Math.abs(particles[i].theta - particles[j].theta);
        if (diff > Math.PI) diff = Math.PI * 2 - diff;
        if (diff < 0.22) {
          ctx.moveTo(cx1 + Math.cos(particles[i].theta) * radius, cy1 + Math.sin(particles[i].theta) * radius);
          ctx.lineTo(cx1 + Math.cos(particles[j].theta) * radius, cy1 + Math.sin(particles[j].theta) * radius);
        }
      }
    }
    ctx.stroke();

    // Batched Particle Halos
    for (let c = 0; c < clusters.length; c++) {
      ctx.beginPath();
      ctx.fillStyle = clusters[c].halo;
      for (let i = 0; i < N; i++) {
        if (particles[i].cluster === c) {
          const px = cx1 + Math.cos(particles[i].theta) * radius;
          const py = cy1 + Math.sin(particles[i].theta) * radius;
          ctx.moveTo(px + 7, py);
          ctx.arc(px, py, 7, 0, Math.PI * 2);
        }
      }
      ctx.fill();
    }

    // Batched Particle Cores
    for (let c = 0; c < clusters.length; c++) {
      ctx.beginPath();
      ctx.fillStyle = clusters[c].color;
      for (let i = 0; i < N; i++) {
        if (particles[i].cluster === c) {
          const px = cx1 + Math.cos(particles[i].theta) * radius;
          const py = cy1 + Math.sin(particles[i].theta) * radius;
          ctx.moveTo(px + 4, py);
          ctx.arc(px, py, 4, 0, Math.PI * 2);
        }
      }
      ctx.fill();
    }

    // Complex Order Parameter Vector: r(t) * e^(i * psi)
    const vecX = cx1 + Math.cos(orderPsi) * (radius * orderR);
    const vecY = cy1 + Math.sin(orderPsi) * (radius * orderR);

    ctx.beginPath();
    ctx.moveTo(cx1, cy1);
    ctx.lineTo(vecX, vecY);
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(vecX, vecY, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#f59e0b";
    ctx.fill();

    // Polar Telemetry
    ctx.font = "bold 13px 'Outfit', sans-serif";
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.fillText("Phase Space: θ_i ∈ [-π, +π)", cx1, 24);

    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = orderR > 0.6 ? "#10b981" : "#94a3b8";
    ctx.fillText(`Coherence r = ${orderR.toFixed(3)}  |  Ψ = ${(orderPsi * 180 / Math.PI).toFixed(0)}°`, cx1, h - 14);

    // --- 2. RIGHT PANEL: TRAVELING WAVE OSCILLOSCOPE ---
    const cx2Start = w * 0.58;
    const cx2Width = w * 0.38;
    const cy2 = h * 0.5;

    // Grid lines
    ctx.beginPath();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    ctx.moveTo(cx2Start, cy2);
    ctx.lineTo(cx2Start + cx2Width, cy2);
    ctx.moveTo(cx2Start, cy2 - 30);
    ctx.lineTo(cx2Start + cx2Width, cy2 - 30);
    ctx.moveTo(cx2Start, cy2 + 30);
    ctx.lineTo(cx2Start + cx2Width, cy2 + 30);
    ctx.stroke();

    // Waveform: optimized step size x += 4 (fast linear interpolation)
    ctx.beginPath();
    ctx.strokeStyle = orderR > 0.6 ? "#06b6d4" : "rgba(148, 163, 184, 0.8)";
    ctx.lineWidth = 2.2;

    const sampleCount = Math.min(8, N);
    const ampScale = 50 / sampleCount;

    for (let x = 0; x <= cx2Width; x += 4) {
      const spatialPhase = (x / cx2Width) * Math.PI * 4;
      let totalY = 0;
      for (let s = 0; s < sampleCount; s++) {
        totalY += Math.sin(spatialPhase - time * 2 + particles[s].theta) * ampScale;
      }
      const plotY = cy2 + totalY;
      if (x === 0) ctx.moveTo(cx2Start, plotY);
      else ctx.lineTo(cx2Start + x, plotY);
    }
    ctx.stroke();

    // Oscilloscope Telemetry
    ctx.font = "bold 13px 'Outfit', sans-serif";
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.fillText("Traveling Wave Constructive Interference", cx2Start + cx2Width / 2, 24);

    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#38bdf8";
    const gainDb = (20 * Math.log10(1.0 + orderR * 3.5)).toFixed(1);
    ctx.fillText(`Resonance Gain: +${gainDb} dB  |  Constructive Peak: ${(orderR * 100).toFixed(0)}%`, cx2Start + cx2Width / 2, h - 14);
  }

  window.updateKuramotoK = function(val) {
    K_coupling = parseFloat(val) / 100.0;
    const lbl = document.getElementById('kVal');
    if (lbl) lbl.innerText = K_coupling.toFixed(2);
  };

  window.injectKuramotoShockwave = function() {
    const targetTheta = Math.random() * Math.PI * 2;
    for (let i = 0; i < Math.floor(particles.length * 0.5); i++) {
      particles[i].theta = targetTheta + (Math.random() - 0.5) * 0.08;
    }
  };

  window.startKuramotoCanvas = start;
  window.stopKuramotoCanvas = stop;
  window.initKuramotoCanvas = start;
})();
