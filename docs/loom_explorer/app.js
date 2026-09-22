/**
 * DOCLOOM ARCHITECTURE EXPLORER INTERACTIVE CONTROLLER
 * Renders data dynamically, manages live search, theme toggles, and card expansions.
 */

document.addEventListener("DOMContentLoaded", () => {
  renderSystemOverview();
  renderModules();
  renderFormulas();
  renderStorageLayouts();
  renderFutureVision();
  setupSearch();
  setupThemeToggle();
  setupScrollSpy();
  if (window.initLoomSimulations) window.initLoomSimulations();
  if (window.initLifecycleAnimation) window.initLifecycleAnimation();
});

// 1. Render Core Paradigms
function renderSystemOverview() {
  const container = document.getElementById("paradigm-grid");
  if (!container || !LOOM_DATA.systemOverview) return;

  container.innerHTML = LOOM_DATA.systemOverview.coreParadigm.map(item => `
    <div class="card">
      <div class="card-header">
        <h3 class="card-title">${item.title}</h3>
      </div>
      <p style="font-size: 14px; color: var(--text-muted);">${item.desc}</p>
    </div>
  `).join("");
}

// 2. Render File Drill-Down
function renderModules(filterText = "") {
  const container = document.getElementById("modules-container");
  if (!container) return;

  const query = filterText.toLowerCase().trim();
  let html = "";

  LOOM_DATA.modules.forEach(mod => {
    // Filter matching files
    const matchedFiles = mod.files.filter(f => {
      if (!query) return true;
      const inName = f.name.toLowerCase().includes(query);
      const inRole = f.role.toLowerCase().includes(query);
      const inSummary = f.summary.toLowerCase().includes(query);
      const inMethods = f.classes.some(c => 
        c.name.toLowerCase().includes(query) || 
        c.methods.some(m => m.name.toLowerCase().includes(query) || m.desc.toLowerCase().includes(query))
      );
      return inName || inRole || inSummary || inMethods;
    });

    if (matchedFiles.length === 0) return;

    html += `
      <div class="module-group" id="mod-${mod.id}">
        <div class="module-header">
          <div class="module-color-dot" style="background-color: ${mod.color};"></div>
          <div>
            <h2 style="font-size: 20px; font-weight: 700; font-family: var(--font-heading);">${mod.name}</h2>
            <p style="font-size: 13px; color: var(--text-muted);">${mod.description}</p>
          </div>
        </div>
    `;

    matchedFiles.forEach(file => {
      html += `
        <div class="file-card" data-filename="${file.name.toLowerCase()}">
          <div class="file-card-top" onclick="toggleFileCard(this)">
            <div>
              <span class="file-name">${file.name}</span>
              <span style="font-size: 13px; color: var(--text-muted); margin-left: 10px;">${file.role}</span>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
              <span class="file-meta-pill">${file.lines} lines</span>
              <span style="font-size: 14px; color: var(--text-dim);">▼</span>
            </div>
          </div>
          <div class="file-card-body">
            <div class="file-summary">${file.summary}</div>
            <div style="font-size: 12px; font-family: var(--font-mono); color: var(--text-dim); margin-bottom: 12px;">
              Path: ${file.path}
            </div>

            ${file.classes.map(c => `
              <div class="class-box">
                <div class="class-title">class ${c.name}</div>
                ${c.methods.map(m => `
                  <div class="method-row">
                    <div class="method-name">def ${m.name}</div>
                    <div class="method-desc">${m.desc}</div>
                    <div class="method-io">
                      <strong>Args:</strong> ${m.inputs}<br/>
                      ${m.outputs ? `<strong>Returns:</strong> ${m.outputs}<br/>` : ''}
                      ${m.storage ? `<strong>Persistence:</strong> ${m.storage}` : ''}
                    </div>
                  </div>
                `).join("")}
              </div>
            `).join("")}
          </div>
        </div>
      `;
    });

    html += `</div>`;
  });

  if (html === "") {
    container.innerHTML = `<div style="padding: 30px; text-align: center; color: var(--text-muted);">No matching files or functions found for "${query}"</div>`;
  } else {
    container.innerHTML = html;
  }
}

// 3. Render Formulas (with KaTeX formatting)
function renderFormulas() {
  const container = document.getElementById("formulas-container");
  if (!container || !LOOM_DATA.formulas) return;

  container.innerHTML = LOOM_DATA.formulas.map(form => {
    let renderedMath1 = escapeHtml(form.latex);
    let renderedMath2 = form.latex2 ? escapeHtml(form.latex2) : '';

    if (window.katex) {
      try {
        renderedMath1 = katex.renderToString(form.latex, { throwOnError: false, displayMode: true });
        if (form.latex2) {
          renderedMath2 = katex.renderToString(form.latex2, { throwOnError: false, displayMode: true });
        }
      } catch (err) {
        console.warn("KaTeX render error:", err);
      }
    }

    return `
      <div class="formula-card">
        <h3 style="font-size: 17px; font-weight: 700; margin-bottom: 6px; font-family: var(--font-heading); color: var(--text-main);">
          ${form.name}
        </h3>
        <div style="font-size: 12px; font-family: var(--font-mono); color: var(--text-dim); margin-bottom: 8px;">
          Implemented in: ${form.file}
        </div>
        <div class="math-display">
          ${renderedMath1}
          ${renderedMath2 ? `<div style="margin-top: 8px;">${renderedMath2}</div>` : ''}
        </div>
        <div class="math-explanation">${form.explanation}</div>
      </div>
    `;
  }).join("");
}

// 4. Render Binary Storage Layouts
function renderStorageLayouts() {
  const container = document.getElementById("storage-container");
  if (!container || !LOOM_DATA.binaryStorageLayouts) return;

  container.innerHTML = LOOM_DATA.binaryStorageLayouts.map(layout => `
    <div class="byte-card">
      <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px;">
        <h3 style="font-size: 16px; font-weight: 700; font-family: var(--font-mono); color: var(--accent-teal);">
          ${layout.name}
        </h3>
        <span class="file-meta-pill">Magic: ${layout.magic}</span>
      </div>
      <div class="byte-row">
        <div class="byte-label">Source File:</div>
        <div class="byte-value">${layout.file}</div>
      </div>
      <div class="byte-row">
        <div class="byte-label">Header Size:</div>
        <div class="byte-value">${layout.headerSize}</div>
      </div>
      <div class="byte-row">
        <div class="byte-label">C-Struct Header:</div>
        <div class="byte-value" style="color: var(--accent-purple);">${layout.structFormat}</div>
      </div>
      <div class="byte-row">
        <div class="byte-label">Body Layout:</div>
        <div class="byte-value">${layout.body}</div>
      </div>
      <div class="byte-row">
        <div class="byte-label">Append / Tail:</div>
        <div class="byte-value">${layout.tail}</div>
      </div>
    </div>
  `).join("");
}

// 4b. Render Future Vision & Next-Gen Roadmap
function renderFutureVision() {
  const container = document.getElementById("future-vision-container");
  if (!container || !LOOM_DATA.futureVision) return;

  container.innerHTML = LOOM_DATA.futureVision.map(item => `
    <div class="card" style="border-left: 4px solid ${item.color}; margin-bottom: 20px;">
      <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
        <div>
          <span style="font-size: 11px; font-weight: 700; color: ${item.color}; text-transform: uppercase; letter-spacing: 0.08em;">
            ${item.phase} &bull; ${item.tag}
          </span>
          <h3 style="font-family: var(--font-heading); font-size: 19px; font-weight: 700; color: var(--text-main); margin-top: 4px;">
            ${item.title}
          </h3>
        </div>
        <span class="file-meta-pill" style="border-color: ${item.color}; color: ${item.color}; font-size: 11px;">
          ${item.status}
        </span>
      </div>
      <p style="font-size: 14px; color: var(--text-muted); margin-bottom: 12px; line-height: 1.6;">
        ${item.summary}
      </p>
      <div style="background: var(--bg-primary); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 12px; margin-bottom: 12px;">
        <strong style="font-size: 12px; color: var(--accent-teal); text-transform: uppercase; letter-spacing: 0.05em;">Why it's a Breakthrough:</strong>
        <p style="font-size: 13px; color: var(--text-main); margin-top: 4px;">${item.breakthrough}</p>
      </div>
      <div>
        <strong style="font-size: 11px; color: var(--text-dim); text-transform: uppercase;">Key Deliverables:</strong>
        <ul style="margin-left: 20px; margin-top: 6px; font-size: 13px; color: var(--text-muted); line-height: 1.6;">
          ${item.deliverables.map(d => `<li>${d}</li>`).join("")}
        </ul>
      </div>
    </div>
  `).join("");
}

// 5. Card Toggle
function toggleFileCard(headerEl) {
  const body = headerEl.nextElementSibling;
  const arrow = headerEl.querySelector("span:last-child");
  if (body.style.display === "none") {
    body.style.display = "block";
    arrow.textContent = "▼";
  } else {
    body.style.display = "none";
    arrow.textContent = "▶";
  }
}

// 6. Search Filter Setup
function setupSearch() {
  const searchInput = document.getElementById("global-search");
  if (!searchInput) return;

  searchInput.addEventListener("input", (e) => {
    renderModules(e.target.value);
  });
}

// 7. Theme Toggle
function setupThemeToggle() {
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;

  btn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    btn.textContent = next === "dark" ? "🌙 Dark Mode" : "☀️ Light Mode";
  });
}

// 8. Navigation ScrollSpy
function setupScrollSpy() {
  const links = document.querySelectorAll(".nav-link[href^='#']");
  links.forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetId = link.getAttribute("href").substring(1);
      const targetEl = document.getElementById(targetId);
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: "smooth" });
        links.forEach(l => l.classList.remove("active"));
        link.classList.add("active");
      }
    });
  });
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
