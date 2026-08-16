(function () {
  "use strict";

  const STAGES = [
    "Preparing data",
    "Classical baselines (GBR / XGBoost)",
    "Self-supervised pretraining",
    "Multi-task fine-tuning (physics-informed loss)",
    "MC-Dropout + EVT threshold calibration",
    "Adaptive gating ensemble",
    "Holdout evaluation",
    "Statistical significance tests",
    "Architectural ablation study",
    "Explainability (Integrated Gradients)",
    "Rendering figures",
  ];

  const els = {
    runBtn: document.getElementById("run-btn"),
    runHint: document.getElementById("run-hint"),
    modeSelect: document.getElementById("mode-select"),
    seedInput: document.getElementById("seed-input"),
    checklist: document.getElementById("checklist"),
    statusPill: document.getElementById("global-status"),
    terminal: document.getElementById("terminal"),
    clock: document.getElementById("clock"),
    chartGrid: document.getElementById("chart-grid"),
    ablationTable: document.querySelector("#ablation-table tbody"),
    sensorBars: document.getElementById("sensor-bars"),
    overviewEmpty: document.getElementById("overview-empty"),
  };

  let seenLogs = 0;
  let pollTimer = null;
  let running = false;

  // ---------------- Checklist ----------------
  function buildChecklist() {
    els.checklist.innerHTML = "";
    STAGES.forEach((name, i) => {
      const li = document.createElement("li");
      li.id = `stage-${i}`;
      li.innerHTML = `<span class="idx">${String(i).padStart(2, "0")}</span>
                      <span class="lamp"></span>
                      <span class="label">${name}</span>`;
      els.checklist.appendChild(li);
    });
  }

  function setChecklistState(activeIndex) {
    STAGES.forEach((_, i) => {
      const li = document.getElementById(`stage-${i}`);
      li.classList.remove("active", "complete");
      if (i < activeIndex) li.classList.add("complete");
      else if (i === activeIndex) li.classList.add("active");
    });
  }

  // ---------------- Clock ----------------
  function tickClock() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    els.clock.textContent = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  }
  setInterval(tickClock, 1000);
  tickClock();

  // ---------------- Status pill ----------------
  function setStatus(state, text) {
    els.statusPill.dataset.state = state;
    els.statusPill.querySelector(".status-text").textContent = text;
  }

  // ---------------- Tabs ----------------
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });

  // ---------------- Terminal ----------------
  function appendLogLines(lines) {
    lines.forEach((text) => {
      const div = document.createElement("div");
      div.className = "term-line";
      if (text.startsWith("---")) div.classList.add("stage");
      else if (text.startsWith("ERROR") || text.startsWith("Traceback")) div.classList.add("err");
      div.textContent = text;
      els.terminal.appendChild(div);
    });
    els.terminal.scrollTop = els.terminal.scrollHeight;
  }

  // ---------------- Gauges (SVG semicircle dials) ----------------
  const GAUGE_NS = "http://www.w3.org/2000/svg";
  function polar(cx, cy, r, angleDeg) {
    const a = ((angleDeg - 180) * Math.PI) / 180;
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  }
  function arcPath(cx, cy, r, startDeg, endDeg) {
    const [sx, sy] = polar(cx, cy, r, startDeg);
    const [ex, ey] = polar(cx, cy, r, endDeg);
    const large = endDeg - startDeg > 180 ? 1 : 0;
    return `M ${sx} ${sy} A ${r} ${r} 0 ${large} 1 ${ex} ${ey}`;
  }

  function initGauge(svgId) {
    const svg = document.getElementById(svgId);
    svg.innerHTML = "";
    const cx = 80, cy = 88, r = 66;
    const track = document.createElementNS(GAUGE_NS, "path");
    track.setAttribute("d", arcPath(cx, cy, r, 0, 180));
    track.setAttribute("stroke", "#1c2740");
    track.setAttribute("stroke-width", "12");
    track.setAttribute("fill", "none");
    track.setAttribute("stroke-linecap", "round");
    svg.appendChild(track);

    const fill = document.createElementNS(GAUGE_NS, "path");
    fill.setAttribute("id", svgId + "-fill");
    fill.setAttribute("d", arcPath(cx, cy, r, 0, 0.001));
    fill.setAttribute("stroke", "#2c3c60");
    fill.setAttribute("stroke-width", "12");
    fill.setAttribute("fill", "none");
    fill.setAttribute("stroke-linecap", "round");
    svg.appendChild(fill);

    // tick marks
    for (let i = 0; i <= 10; i++) {
      const deg = (i / 10) * 180;
      const [x1, y1] = polar(cx, cy, r + 9, deg);
      const [x2, y2] = polar(cx, cy, r + 15, deg);
      const tick = document.createElementNS(GAUGE_NS, "line");
      tick.setAttribute("x1", x1); tick.setAttribute("y1", y1);
      tick.setAttribute("x2", x2); tick.setAttribute("y2", y2);
      tick.setAttribute("stroke", "#2c3c60"); tick.setAttribute("stroke-width", "1.3");
      svg.appendChild(tick);
    }

    const needle = document.createElementNS(GAUGE_NS, "line");
    needle.setAttribute("id", svgId + "-needle");
    needle.setAttribute("x1", cx); needle.setAttribute("y1", cy);
    const [nx, ny] = polar(cx, cy, r - 14, 0);
    needle.setAttribute("x2", nx); needle.setAttribute("y2", ny);
    needle.setAttribute("stroke", "#f5a623");
    needle.setAttribute("stroke-width", "2.4");
    needle.setAttribute("stroke-linecap", "round");
    svg.appendChild(needle);

    const hub = document.createElementNS(GAUGE_NS, "circle");
    hub.setAttribute("cx", cx); hub.setAttribute("cy", cy); hub.setAttribute("r", 4.5);
    hub.setAttribute("fill", "#f5a623");
    svg.appendChild(hub);
  }

  function setGauge(svgId, fraction, color) {
    fraction = Math.max(0, Math.min(1, fraction));
    const cx = 80, cy = 88, r = 66;
    const deg = fraction * 180;
    const fill = document.getElementById(svgId + "-fill");
    const needle = document.getElementById(svgId + "-needle");
    if (!fill || !needle) return;
    fill.setAttribute("d", arcPath(cx, cy, r, 0, Math.max(deg, 0.001)));
    fill.setAttribute("stroke", color);
    needle.setAttribute("stroke", color);
    const [nx, ny] = polar(cx, cy, r - 14, deg);
    needle.setAttribute("x2", nx);
    needle.setAttribute("y2", ny);
    const hub = document.querySelector(`#${svgId} circle`);
    if (hub) hub.setAttribute("fill", color);
  }

  ["gauge-auc", "gauge-f1", "gauge-rmse", "gauge-ece"].forEach(initGauge);

  // ---------------- Results rendering ----------------
  function fmt(v, digits = 3) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    return Number(v).toFixed(digits);
  }

  function renderResults(res) {
    els.overviewEmpty.style.display = "none";

    const m = res.metrics;
    document.getElementById("val-auc").textContent = fmt(m.anomaly_auc_roc);
    document.getElementById("val-f1").textContent = fmt(m.fault_macro_f1);
    document.getElementById("val-rmse").textContent = fmt(m.rul_rmse_gated_ensemble, 2);
    document.getElementById("val-ece").textContent = fmt(m.ece_failure_class);

    setGauge("gauge-auc", m.anomaly_auc_roc ?? 0, "#4fd1c5");
    setGauge("gauge-f1", m.fault_macro_f1 ?? 0, "#4fd1c5");
    setGauge("gauge-rmse", Math.max(0, 1 - (m.rul_rmse_gated_ensemble ?? 50) / 60), "#f5a623");
    setGauge("gauge-ece", Math.max(0, 1 - (m.ece_failure_class ?? 0.5) / 0.4), "#f5a623");

    document.getElementById("s-source").textContent = res.data_source;
    document.getElementById("s-device").textContent = res.device;
    document.getElementById("s-n").textContent = res.n_sequences.toLocaleString();
    document.getElementById("s-split").textContent =
      `${res.splits.train} / ${res.splits.val} / ${res.splits.holdout}`;
    document.getElementById("s-threshold").textContent =
      `${fmt(res.threshold.threshold, 4)}  (${res.threshold.method})`;
    document.getElementById("s-rmse-deep").textContent = fmt(m.rul_rmse_deep_head, 2);
    document.getElementById("s-rmse-gbr").textContent = fmt(m.rul_rmse_gbr, 2);

    document.getElementById("s-mcnemar").textContent =
      `stat=${fmt(res.significance.mcnemar_stat)}  p=${fmt(res.significance.mcnemar_p, 4)}`;
    document.getElementById("s-wilcoxon").textContent =
      `stat=${fmt(res.significance.wilcoxon_stat)}  p=${fmt(res.significance.wilcoxon_p, 4)}`;

    const uTable = document.getElementById("uncertainty-table");
    uTable.innerHTML = "";
    Object.entries(res.uncertainty).forEach(([name, val]) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${name}</td><td class="v">${fmt(val, 5)}</td>`;
      uTable.appendChild(tr);
    });

    // Charts
    const chartTitles = {
      training_curves: "Pretraining + Fine-Tuning Loss Curves",
      roc_curve: "ROC Curve — Anomaly Detection (Holdout)",
      rul_scatter: "Predicted vs. True RUL",
      fusion_weights: "Adaptive Attention-Fusion Weights (α)",
      ablation: "Ablation Study — RUL RMSE",
      sensor_importance: "Sensor Ranking — Integrated Gradients",
    };
    els.chartGrid.innerHTML = "";
    Object.entries(res.figures).forEach(([key, b64]) => {
      const card = document.createElement("div");
      card.className = "chart-card";
      card.innerHTML = `<img src="data:image/png;base64,${b64}" alt="${chartTitles[key] || key}">
                         <div class="chart-title">${chartTitles[key] || key}</div>`;
      els.chartGrid.appendChild(card);
    });

    // Ablation table
    els.ablationTable.innerHTML = "";
    res.ablation_table.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${row.variant}</td><td>${fmt(row.RUL_RMSE, 2)}</td><td>${fmt(row.Anomaly_AUC, 3)}</td>`;
      els.ablationTable.appendChild(tr);
    });

    // Sensor bars
    els.sensorBars.innerHTML = "";
    const maxImp = Math.max(...res.feature_importance.map((f) => f.importance), 1e-9);
    res.feature_importance.forEach((f) => {
      const row = document.createElement("div");
      row.className = "sensor-row";
      const pct = (f.importance / maxImp) * 100;
      row.innerHTML = `<div class="name">${f.sensor}</div>
                        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
                        <div class="score">${f.importance.toFixed(4)}</div>`;
      els.sensorBars.appendChild(row);
    });
  }

  // ---------------- Polling ----------------
  async function poll() {
    try {
      const res = await fetch(`/api/status?since=${seenLogs}`);
      const data = await res.json();

      if (data.new_logs && data.new_logs.length) {
        appendLogLines(data.new_logs);
        seenLogs = data.log_count;
      }

      if (data.stage_index >= 0) setChecklistState(data.stage_index);

      if (data.running) {
        setStatus("running", `RUNNING · ${data.stage_name || "…"}`);
      } else if (data.done && data.error) {
        setStatus("error", "ERROR");
        appendLogLines([`ERROR: ${data.error}`]);
        stopRun(true);
      } else if (data.done && data.results) {
        setStatus("done", "COMPLETE");
        setChecklistState(STAGES.length);
        renderResults(data.results);
        stopRun(true);
      }
    } catch (e) {
      // transient network hiccup while polling; keep trying
    }
  }

  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(poll, 700);
  }

  function stopRun(keepStatus) {
    running = false;
    els.runBtn.classList.remove("is-running");
    els.runBtn.querySelector(".run-switch-label").innerHTML =
      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M7 5v14l11-7z" fill="currentColor"/></svg> RUN PIPELINE`;
    els.runBtn.disabled = false;
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    // one last poll shortly after, in case results landed between ticks
    setTimeout(poll, 400);
  }

  // ---------------- Run trigger ----------------
  async function startRun() {
    if (running) return;
    running = true;
    seenLogs = 0;
    els.terminal.innerHTML = "";
    els.runBtn.disabled = true;
    els.runBtn.classList.add("is-running");
    els.runBtn.querySelector(".run-switch-label").innerHTML =
      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><rect x="6" y="6" width="12" height="12" fill="currentColor"/></svg> RUNNING…`;
    setStatus("running", "INITIALIZING");
    setChecklistState(-1);

    const mode = els.modeSelect.value;
    const seed = parseInt(els.seedInput.value || "42", 10);

    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, seed }),
      });
      if (!res.ok) {
        const body = await res.json();
        appendLogLines([`Could not start: ${body.message || res.statusText}`]);
        stopRun();
        return;
      }
    } catch (e) {
      appendLogLines([`Could not reach backend: ${e}`]);
      stopRun();
      return;
    }
    els.runBtn.disabled = false;
    startPolling();
  }

  els.runBtn.addEventListener("click", startRun);

  buildChecklist();
})();
