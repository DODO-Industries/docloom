/**
 * DocLoom Spatiotemporal Biological Decay & Deep-Sleep Eviction Sandbox
 * Formulates the ACT-R / Ebbinghaus hybrid forgetting curve and memory pool RAM clamping.
 */
(function() {
  let decayCanvas, dCtx;

  function initDecayPlot() {
    decayCanvas = document.getElementById('decayCurveCanvas');
    if (!decayCanvas) return;
    dCtx = decayCanvas.getContext('2d');
    decayCanvas.width = decayCanvas.parentElement.clientWidth;
    decayCanvas.height = 240;
    updateDecaySim();
  }

  function renderCurve(curHours, curHits, curAct, lambda) {
    if (!dCtx) return;
    dCtx.clearRect(0, 0, decayCanvas.width, decayCanvas.height);
    const w = decayCanvas.width;
    const h = decayCanvas.height;

    const padLeft = 50;
    const padBottom = 35;
    const padTop = 20;
    const padRight = 30;

    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    // Grid lines
    dCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    dCtx.lineWidth = 1;

    for (let yStep = 0; yStep <= 1.0; yStep += 0.25) {
      const y = padTop + plotH * (1.0 - yStep);
      dCtx.beginPath();
      dCtx.moveTo(padLeft, y);
      dCtx.lineTo(padLeft + plotW, y);
      dCtx.stroke();

      dCtx.fillStyle = "#64748b";
      dCtx.font = "10px 'JetBrains Mono'";
      dCtx.textAlign = "right";
      dCtx.fillText(yStep.toFixed(2), padLeft - 8, y + 3);
    }

    for (let xStep = 0; xStep <= 120; xStep += 24) {
      const x = padLeft + (xStep / 120) * plotW;
      dCtx.beginPath();
      dCtx.moveTo(x, padTop);
      dCtx.lineTo(x, padTop + plotH);
      dCtx.stroke();

      dCtx.fillStyle = "#64748b";
      dCtx.font = "10px 'JetBrains Mono'";
      dCtx.textAlign = "center";
      dCtx.fillText(xStep + "h", x, padTop + plotH + 18);
    }

    // Eviction Threshold Line at A = 0.001
    const evictY = padTop + plotH * (1.0 - 0.001);
    dCtx.beginPath();
    dCtx.setLineDash([4, 4]);
    dCtx.strokeStyle = "rgba(244, 63, 94, 0.6)";
    dCtx.lineWidth = 1.2;
    dCtx.moveTo(padLeft, evictY);
    dCtx.lineTo(padLeft + plotW, evictY);
    dCtx.stroke();
    dCtx.setLineDash([]);

    dCtx.fillStyle = "#f43f5e";
    dCtx.font = "bold 10px 'JetBrains Mono'";
    dCtx.textAlign = "right";
    dCtx.fillText("Deep-Sleep Eviction Horizon (A <= 0.001)", padLeft + plotW - 6, evictY - 6);

    // Plot Theoretical Decay Trajectory for current Hits
    const hitBonus = 1.0 + 0.15 * Math.log(curHits + 1);
    dCtx.beginPath();
    dCtx.strokeStyle = "#06b6d4";
    dCtx.lineWidth = 2.5;

    for (let t = 0; t <= 120; t += 1) {
      const decay = Math.exp(-lambda * (t / 12.0));
      const a = Math.max(0, Math.min(1.0, decay * hitBonus));
      const x = padLeft + (t / 120) * plotW;
      const y = padTop + plotH * (1.0 - a);

      if (t === 0) dCtx.moveTo(x, y);
      else dCtx.lineTo(x, y);
    }
    dCtx.stroke();

    // Draw Operating Point (Current Time, Current Act)
    const curX = padLeft + (Math.min(120, curHours) / 120) * plotW;
    const curY = padTop + plotH * (1.0 - curAct);

    dCtx.beginPath();
    dCtx.arc(curX, curY, 6, 0, Math.PI * 2);
    dCtx.fillStyle = curAct <= 0.001 ? "#f43f5e" : "#10b981";
    dCtx.shadowColor = curAct <= 0.001 ? "rgba(244, 63, 94, 0.9)" : "rgba(16, 185, 129, 0.9)";
    dCtx.shadowBlur = 14;
    dCtx.fill();
    dCtx.shadowBlur = 0;

    dCtx.beginPath();
    dCtx.arc(curX, curY, 10, 0, Math.PI * 2);
    dCtx.strokeStyle = "#fff";
    dCtx.lineWidth = 1.5;
    dCtx.stroke();
  }

  window.updateDecaySim = function() {
    const hours = parseFloat(document.getElementById('slider-time').value);
    const hits = parseFloat(document.getElementById('slider-hits').value);
    const lambda = 0.55;

    document.getElementById('val-time').innerText = hours + ' Hours (' + (hours / 24).toFixed(1) + ' Days)';
    document.getElementById('val-hits').innerText = hits + ' Hits';

    const decayFactor = Math.exp(-lambda * (hours / 12.0));
    const hitBonus = 1.0 + 0.15 * Math.log(hits + 1);
    const activation = Math.max(0, Math.min(1.0, 1.0 * decayFactor * hitBonus));

    document.getElementById('sim-act-val').innerText = activation.toFixed(4);

    const badge = document.getElementById('sim-badge');
    const desc = document.getElementById('sim-desc');
    const formula = document.getElementById('sim-formula');
    const ramMeter = document.getElementById('sim-ram-meter');

    if (activation <= 0.001 || (hours >= 48 && hits <= 1)) {
      badge.className = 'sim-status-badge status-evicted';
      badge.innerText = 'PURGED FROM RAM (DEEP SLEEP)';
      badge.style.background = 'rgba(244, 63, 94, 0.15)';
      badge.style.color = '#f43f5e';
      badge.style.border = '1px solid rgba(244, 63, 94, 0.3)';

      desc.innerHTML = `<strong>Memory State:</strong> Dormant in <code>crystal_*.loom</code> binary container on disk. Occupies <strong>0 bytes</strong> of volatile RAM. Awakens instantly upon vector query hit via zero-copy <code>mmap</code>.`;
      formula.innerHTML = `
        <span style="color:#f43f5e;">A(t) = ${activation.toFixed(5)} &le; 0.001</span> &mdash;&gt; 
        <strong>Executed in weaver_coordinator.py:</strong> <code>del self.ram_ledger[shard_id]</code>.
      `;
      if (ramMeter) ramMeter.style.width = '18%';
    } else {
      badge.className = 'sim-status-badge status-active';
      badge.innerText = 'ACTIVE IN WORKING RAM';
      badge.style.background = 'rgba(16, 185, 129, 0.15)';
      badge.style.color = '#10b981';
      badge.style.border = '1px solid rgba(16, 185, 129, 0.3)';

      desc.innerHTML = `<strong>Memory State:</strong> Resident in <code>ram_ledger</code>. Physics state actively oscillating in Kuramoto phase space. Instantaneous sub-millisecond retrieval.`;
      formula.innerHTML = `
        <span style="color:#38bdf8;">A(t) = 1.0 &times; e^(-0.55 &times; ${(hours/12.0).toFixed(2)}) &times; ${hitBonus.toFixed(2)} = <strong>${activation.toFixed(4)}</strong></span> &mdash;&gt; 
        Retained in active working set.
      `;
      if (ramMeter) ramMeter.style.width = (25 + activation * 50) + '%';
    }

    renderCurve(hours, hits, activation, lambda);
  };

  window.initDecaySimulator = function() {
    initDecayPlot();
    window.addEventListener('resize', initDecayPlot);
  };
})();
