const SAMPLE_FILES = {
  normal: "/samples/normal.json",
  suspicious_network: "/samples/suspicious_network.json",
  suspicious_process: "/samples/suspicious_process.json",
  combined_attack: "/samples/combined_attack.json",
  hardware_tamper: "/samples/hardware_tamper.json",
};

const NETWORK_FEATURE_META = [
  { key: "packet_rate", label: "Packet rate", unit: "pps", softMax: 280 },
  { key: "bytes_per_sec", label: "Bytes/sec", unit: "Bps", softMax: 110000 },
  { key: "avg_packet_size", label: "Avg packet size", unit: "B", softMax: 520 },
  { key: "tcp_syn_rate", label: "TCP SYN rate", unit: "pps", softMax: 14 },
  { key: "failed_login_rate", label: "Failed login rate", unit: "ppm", softMax: 3 },
  { key: "new_connection_rate", label: "New connections", unit: "cps", softMax: 18 },
  { key: "external_ip_ratio", label: "External IP ratio", unit: "", softMax: 0.18 },
  { key: "dns_query_rate", label: "DNS query rate", unit: "qps", softMax: 14 },
];

const PROCESS_FEATURE_META = [
  { key: "reactor_temp_c", label: "Reactor temp", unit: "C", color: "#38bdf8" },
  { key: "reactor_pressure_bar", label: "Reactor pressure", unit: "bar", color: "#22c55e" },
  { key: "valve_position_pct", label: "Valve position", unit: "%", color: "#f59e0b" },
  { key: "motor_current_a", label: "Motor current", unit: "A", color: "#fb7185" },
];

const HARDWARE_FEATURE_META = [
  { key: "vcc_voltage_v", label: "VCC voltage", unit: "V", color: "#38bdf8" },
  { key: "cpu_current_ma", label: "CPU current", unit: "mA", color: "#818cf8" },
  { key: "clock_jitter_ns", label: "Clock jitter", unit: "ns", color: "#f59e0b" },
  { key: "board_temp_c", label: "Board temp", unit: "C", color: "#22c55e" },
  { key: "brownout_flag", label: "Brownout flag", unit: "", color: "#fb7185" },
  { key: "reset_count_delta", label: "Reset delta", unit: "", color: "#f97316" },
];

const HARDWARE_STATE_META = [
  { key: "chassis_open", label: "Chassis open" },
  { key: "tamper_switch", label: "Tamper switch" },
  { key: "jtag_active", label: "JTAG active" },
  { key: "uart_active", label: "UART active" },
  { key: "unexpected_usb", label: "Unexpected USB" },
  { key: "usb_hid_burst", label: "USB HID burst" },
];

let payloadPreviewTimer = null;

function asFiniteNumber(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function formatMetric(value, unit = "") {
  const num = asFiniteNumber(value);
  if (num == null) return "-";
  const formatted = Math.abs(num) >= 1000
    ? num.toLocaleString(undefined, { maximumFractionDigits: 1 })
    : num.toLocaleString(undefined, { maximumFractionDigits: 3 });
  return unit ? `${formatted} ${unit}` : formatted;
}

function escapeAttr(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function setPayloadEditorValue(payload) {
  document.getElementById("payloadInput").value = JSON.stringify(payload, null, 2);
  renderPayloadInsights(payload);
}

function severityBadgeClass(sev) {
  const s = String(sev || "").toLowerCase();
  if (s === "critical") return "badge badge-critical";
  if (s === "high") return "badge badge-high";
  if (s === "medium") return "badge badge-medium";
  return "badge badge-low";
}

async function loadSample(name) {
  const res = await fetch(`${SAMPLE_FILES[name]}?t=${Date.now()}`, { cache: "no-store" });
  const sample = await res.json();
  setPayloadEditorValue(sample);
}

/** Normalize CRLF / old Mac line endings. */
function normalizeNewlines(text) {
  return String(text).replace(/\r\n/g, "\n").replace(/\r/g, "\n");
}

/**
 * Parse one CSV line with RFC-style quoted fields (commas inside quotes, "" for literal quote).
 */
function parseCsvLine(line) {
  const result = [];
  let cur = "";
  let i = 0;
  let inQuotes = false;
  while (i < line.length) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i += 1;
        continue;
      }
      cur += c;
      i += 1;
    } else {
      if (c === '"') {
        inQuotes = true;
        i += 1;
        continue;
      }
      if (c === ",") {
        result.push(cur.trim());
        cur = "";
        i += 1;
        continue;
      }
      cur += c;
      i += 1;
    }
  }
  result.push(cur.trim());
  return result;
}

/** Split into non-empty lines; drop trailing blank lines (common after CRLF save). */
function splitCsvLines(text) {
  const lines = normalizeNewlines(text).split("\n");
  while (lines.length && lines[lines.length - 1].trim() === "") lines.pop();
  return lines.filter((line) => line.trim().length > 0);
}

function csvToPayload(text) {
  const lines = splitCsvLines(text);
  if (lines.length < 2) throw new Error("CSV must have a header row and at least one data row");
  const headers = parseCsvLine(lines[0]).map((h) => h.trim());
  const rows = [];
  for (let r = 1; r < lines.length; r += 1) {
    const values = parseCsvLine(lines[r]);
    const obj = {};
    headers.forEach((h, i) => {
      const raw = values[i] != null ? String(values[i]).trim() : "";
      const num = Number(raw);
      obj[h] = Number.isFinite(num) ? num : raw;
    });
    rows.push(obj);
  }
  if (rows.length === 0) throw new Error("CSV must contain at least one data row");
  const latest = rows[rows.length - 1];
  return {
    site_id: "demo-site",
    asset_id: "plc-01",
    timestamp: new Date().toISOString(),
    network: {
      packet_rate: latest.packet_rate,
      bytes_per_sec: latest.bytes_per_sec,
      avg_packet_size: latest.avg_packet_size,
      tcp_syn_rate: latest.tcp_syn_rate,
      failed_login_rate: latest.failed_login_rate,
      new_connection_rate: latest.new_connection_rate,
      external_ip_ratio: latest.external_ip_ratio,
      dns_query_rate: latest.dns_query_rate,
    },
    process_sequence: rows.map((r) => ({
      reactor_temp_c: r.reactor_temp_c,
      reactor_pressure_bar: r.reactor_pressure_bar,
      valve_position_pct: r.valve_position_pct,
      motor_current_a: r.motor_current_a,
    })),
  };
}

function renderEmptyState(targetId, message) {
  const el = document.getElementById(targetId);
  if (!el) return;
  el.innerHTML = `<div class="viz-empty">${escapeHtml(message)}</div>`;
}

function buildOverviewCard(label, value) {
  return `
    <article class="overview-card">
      <span class="overview-label">${escapeHtml(label)}</span>
      <div class="overview-value">${escapeHtml(value)}</div>
    </article>`;
}

function buildSparkline(values, color) {
  if (!Array.isArray(values) || values.length === 0) {
    return '<div class="viz-empty">No time-series values</div>';
  }

  const width = 320;
  const height = 88;
  const pad = 8;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const usableWidth = width - pad * 2;
  const usableHeight = height - pad * 2;
  const step = values.length > 1 ? usableWidth / (values.length - 1) : 0;

  const points = values.map((value, index) => {
    const x = pad + index * step;
    const y = height - pad - ((value - min) / range) * usableHeight;
    return [x, y];
  });

  const polyline = points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  const area = [
    `${pad},${height - pad}`,
    ...points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`),
    `${points[points.length - 1][0].toFixed(2)},${height - pad}`,
  ].join(" ");

  const dot = points[points.length - 1];

  return `
    <svg class="sparkline" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="Time-series preview">
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="rgba(255,255,255,0.10)" stroke-width="1" />
      <polygon points="${area}" fill="${escapeAttr(color)}22"></polygon>
      <polyline points="${polyline}" fill="none" stroke="${escapeAttr(color)}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"></polyline>
      <circle cx="${dot[0].toFixed(2)}" cy="${dot[1].toFixed(2)}" r="3.5" fill="${escapeAttr(color)}"></circle>
    </svg>`;
}

function renderSeriesCards(targetId, sequence, meta) {
  const target = document.getElementById(targetId);
  if (!target) return;
  if (!Array.isArray(sequence) || sequence.length === 0) {
    target.innerHTML = '<div class="viz-empty">No sequence data in this payload.</div>';
    return;
  }

  target.innerHTML = meta.map((feature) => {
    const values = sequence
      .map((row) => asFiniteNumber(row && row[feature.key]))
      .filter((value) => value != null);

    if (values.length === 0) {
      return `
        <article class="series-card">
          <div class="series-head">
            <span class="series-title">${escapeHtml(feature.label)}</span>
            <span class="series-latest">latest -</span>
          </div>
          <div class="viz-empty">No numeric values</div>
        </article>`;
    }

    const latest = values[values.length - 1];
    const min = Math.min(...values);
    const max = Math.max(...values);

    return `
      <article class="series-card">
        <div class="series-head">
          <span class="series-title">${escapeHtml(feature.label)}</span>
          <span class="series-latest">latest ${escapeHtml(formatMetric(latest, feature.unit))}</span>
        </div>
        ${buildSparkline(values, feature.color)}
        <div class="spark-stats">
          <span>min ${escapeHtml(formatMetric(min, feature.unit))}</span>
          <span>max ${escapeHtml(formatMetric(max, feature.unit))}</span>
          <span>${values.length} pts</span>
        </div>
      </article>`;
  }).join("");
}

function renderNetworkProfile(network) {
  const target = document.getElementById("networkProfileChart");
  if (!target) return;
  if (!network || typeof network !== "object") {
    target.innerHTML = '<div class="viz-empty">No network snapshot in this payload.</div>';
    return;
  }

  target.innerHTML = NETWORK_FEATURE_META.map((feature) => {
    const value = asFiniteNumber(network[feature.key]);
    if (value == null) {
      return `
        <div class="bar-row">
          <div class="bar-head">
            <span class="bar-label">${escapeHtml(feature.label)}</span>
            <span class="bar-value">missing</span>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width: 0%"></div></div>
          <div class="bar-meta"><span>current -</span><span>soft max ${escapeHtml(formatMetric(feature.softMax, feature.unit))}</span></div>
        </div>`;
    }

    const ratio = feature.softMax > 0 ? value / feature.softMax : 0;
    const fill = Math.max(0, Math.min(ratio / 2, 1)) * 100;
    const tone = ratio >= 1.5 ? "danger" : ratio >= 1 ? "warn" : "";
    const ratioText = `${ratio.toFixed(2)}x limit`;

    return `
      <div class="bar-row">
        <div class="bar-head">
          <span class="bar-label">${escapeHtml(feature.label)}</span>
          <span class="bar-value">${escapeHtml(formatMetric(value, feature.unit))}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill ${tone}" style="width: ${fill.toFixed(1)}%"></div>
        </div>
        <div class="bar-meta">
          <span>${escapeHtml(ratioText)}</span>
          <span>soft max ${escapeHtml(formatMetric(feature.softMax, feature.unit))}</span>
        </div>
      </div>`;
  }).join("");
}

function renderHardwareStates(hardwareState) {
  const target = document.getElementById("hardwareStatePills");
  if (!target) return;
  if (!hardwareState || typeof hardwareState !== "object") {
    target.innerHTML = '<div class="viz-empty">No hardware state flags in this payload.</div>';
    return;
  }

  target.innerHTML = HARDWARE_STATE_META.map((feature) => {
    const active = Number(hardwareState[feature.key]) === 1;
    return `<span class="state-pill ${active ? "active" : ""}">${escapeHtml(feature.label)}: ${active ? "ON" : "OFF"}</span>`;
  }).join("");
}

function renderPayloadInsights(payload) {
  const noteEl = document.getElementById("payloadInsightsNote");
  const overviewEl = document.getElementById("payloadOverview");
  if (!noteEl || !overviewEl) return;

  const processSequence = Array.isArray(payload && payload.process_sequence) ? payload.process_sequence : [];
  const hardwareSequence = Array.isArray(payload && payload.hardware_sequence) ? payload.hardware_sequence : [];
  const hardwareState = payload && typeof payload.hardware_state === "object" ? payload.hardware_state : {};
  const activeFlags = HARDWARE_STATE_META.filter((feature) => Number(hardwareState[feature.key]) === 1).length;
  const networkCount = payload && payload.network && typeof payload.network === "object"
    ? NETWORK_FEATURE_META.filter((feature) => payload.network[feature.key] != null).length
    : 0;

  noteEl.textContent = `Previewing ${processSequence.length} process points, ${hardwareSequence.length} hardware points, and ${activeFlags} active hardware flags.`;

  overviewEl.innerHTML = [
    buildOverviewCard("Site", payload && payload.site_id ? String(payload.site_id) : "-"),
    buildOverviewCard("Asset", payload && payload.asset_id ? String(payload.asset_id) : "-"),
    buildOverviewCard("Timestamp", payload && payload.timestamp ? String(payload.timestamp) : "-"),
    buildOverviewCard("Coverage", `${networkCount}/${NETWORK_FEATURE_META.length} network, ${processSequence.length} process, ${hardwareSequence.length} hardware`),
  ].join("");

  renderNetworkProfile(payload && payload.network);
  renderHardwareStates(hardwareState);
  renderSeriesCards("processSeriesCharts", processSequence, PROCESS_FEATURE_META);
  renderSeriesCards("hardwareSeriesCharts", hardwareSequence, HARDWARE_FEATURE_META);
}

function renderPayloadPreviewMessage(message) {
  const noteEl = document.getElementById("payloadInsightsNote");
  const overviewEl = document.getElementById("payloadOverview");
  if (noteEl) noteEl.textContent = message;
  if (overviewEl) {
    overviewEl.innerHTML = [
      buildOverviewCard("Site", "-"),
      buildOverviewCard("Asset", "-"),
      buildOverviewCard("Timestamp", "-"),
      buildOverviewCard("Coverage", "-"),
    ].join("");
  }
  renderEmptyState("networkProfileChart", message);
  renderEmptyState("hardwareStatePills", message);
  renderEmptyState("processSeriesCharts", message);
  renderEmptyState("hardwareSeriesCharts", message);
}

function refreshPayloadInsightsFromEditor() {
  const raw = document.getElementById("payloadInput").value.trim();
  if (!raw) {
    renderPayloadPreviewMessage("Paste JSON or load a sample to see the payload charts.");
    return;
  }
  try {
    renderPayloadInsights(JSON.parse(raw));
  } catch {
    renderPayloadPreviewMessage("Payload preview is waiting for valid JSON.");
  }
}

document.getElementById("payloadInput").addEventListener("input", () => {
  clearTimeout(payloadPreviewTimer);
  payloadPreviewTimer = setTimeout(refreshPayloadInsightsFromEditor, 120);
});

document.getElementById("fileInput").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const payload = file.name.endsWith(".csv") ? csvToPayload(text) : JSON.parse(text);
    setPayloadEditorValue(payload);
  } catch (err) {
    alert("Could not load file: " + err.message);
  }
});

async function runDetection() {
  try {
    const raw = document.getElementById("payloadInput").value;
    const payload = JSON.parse(raw);
    const res = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.error);
    renderDetection(data.risk);
    await refreshHistory();
    await refreshStatus();
  } catch (err) {
    alert("Detection failed: " + err.message);
  }
}

function renderDetection(risk) {
  const statusEl = document.getElementById("statusCard");
  const sys = risk.system_status || "-";
  const sev = risk.severity || "-";
  statusEl.innerHTML = `${escapeHtml(sys)} <span class="${severityBadgeClass(sev)}">${escapeHtml(sev)}</span>`;

  document.getElementById("riskCard").textContent =
    typeof risk.fused_risk_score === "number" ? risk.fused_risk_score.toFixed(3) : "-";
  document.getElementById("networkCard").textContent =
    typeof risk.network_score === "number" ? risk.network_score.toFixed(3) : "-";
  document.getElementById("processCard").textContent =
    typeof risk.process_score === "number" ? risk.process_score.toFixed(3) : "-";
  document.getElementById("hardwareCard").textContent =
    typeof risk.hardware_score === "number" ? risk.hardware_score.toFixed(3) : "-";

  renderExplanation(risk.explanation || {});
  renderAiAnalysis(risk.ai_analysis || null);
  document.getElementById("latestDetection").textContent = JSON.stringify(risk, null, 2);
}

function formatScore(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "-";
}

function renderFactorItems(items) {
  const filtered = Array.isArray(items) ? items.filter((item) => Number(item && item.score) > 1e-6) : [];
  if (filtered.length === 0) {
    return '<li class="explanation-empty">No strong factors detected.</li>';
  }
  return filtered
    .map((item) => {
      const label = item.label || item.signal || item.feature || "factor";
      const value = item.value != null ? item.value : item.latest_value;
      const score = item.score;
      return `
        <li>
          <span class="factor-label">${escapeHtml(String(label))}</span>
          <span class="factor-meta">score ${formatScore(score)} - value ${escapeHtml(String(value))}</span>
        </li>`;
    })
    .join("");
}

function renderExplanation(explanation) {
  const panel = document.getElementById("explanationPanel");
  const summaryEl = document.getElementById("explanationSummary");
  const networkEl = document.getElementById("networkExplanation");
  const processEl = document.getElementById("processExplanation");
  const hardwareEl = document.getElementById("hardwareExplanation");
  const rulesEl = document.getElementById("hardwareRuleHits");

  if (!panel || !summaryEl || !networkEl || !processEl || !hardwareEl || !rulesEl) return;

  const hasContent = explanation && Object.keys(explanation).length > 0;
  if (!hasContent) {
    panel.hidden = true;
    return;
  }

  panel.hidden = false;
  summaryEl.innerHTML = `<strong>Likely contributors:</strong> ${escapeHtml(String(explanation.summary || "No summary available"))}`;
  networkEl.innerHTML = renderFactorItems(explanation.network && explanation.network.top_factors);
  processEl.innerHTML = renderFactorItems(explanation.process && explanation.process.top_factors);
  hardwareEl.innerHTML = renderFactorItems(explanation.hardware && explanation.hardware.top_factors);

  const ruleHits = (explanation.hardware && explanation.hardware.rule_hits) || [];
  rulesEl.innerHTML = ruleHits.length
    ? ruleHits
        .map((hit) => {
          const label = hit.label || hit.signal || "hardware flag";
          return `<span class="explanation-rule">${escapeHtml(String(label))}</span>`;
        })
        .join("")
    : '<span class="explanation-empty">No hardware rule hits.</span>';
}

function renderSimpleList(items, emptyMessage) {
  const values = Array.isArray(items) ? items.filter((item) => String(item || "").trim()) : [];
  if (values.length === 0) {
    return `<li class="explanation-empty">${escapeHtml(emptyMessage)}</li>`;
  }
  return values.map((item) => `<li>${escapeHtml(String(item))}</li>`).join("");
}

function renderAiAnalysis(analysis) {
  const panel = document.getElementById("aiAnalystPanel");
  const modelEl = document.getElementById("aiAnalystModel");
  const summaryEl = document.getElementById("aiAnalystSummary");
  const statusEl = document.getElementById("aiAnalystStatus");
  const confidenceEl = document.getElementById("aiAnalystConfidence");
  const whatEl = document.getElementById("aiAnalystWhat");
  const whereEl = document.getElementById("aiAnalystWhere");
  const whyEl = document.getElementById("aiAnalystWhy");
  const evidenceEl = document.getElementById("aiAnalystEvidence");
  const actionsEl = document.getElementById("aiAnalystActions");
  const recoveryEl = document.getElementById("aiAnalystRecovery");

  if (!panel || !modelEl || !summaryEl || !statusEl || !confidenceEl || !whatEl || !whereEl || !whyEl || !evidenceEl || !actionsEl || !recoveryEl) {
    return;
  }

  if (!analysis || typeof analysis !== "object") {
    panel.hidden = true;
    return;
  }

  panel.hidden = false;
  const status = String(analysis.status || "unknown").toLowerCase();
  const confidence = String(analysis.confidence || "low").toLowerCase();
  modelEl.textContent = analysis.model ? `model ${analysis.model}` : "";
  summaryEl.textContent = String(analysis.summary || "AI analyst did not return a summary.");
  statusEl.textContent = status;
  statusEl.className = `analyst-pill ${status === "available" ? "low" : status === "error" ? "high" : "medium"}`;
  confidenceEl.textContent = `confidence ${confidence}`;
  confidenceEl.className = `analyst-pill ${confidence}`;
  whatEl.textContent = String(analysis.what_happened || "-");
  whereEl.textContent = String(analysis.where_it_happened || "-");
  whyEl.textContent = String(analysis.why_it_happened || "-");
  evidenceEl.innerHTML = renderSimpleList(analysis.evidence, "No analyst evidence provided.");
  actionsEl.innerHTML = renderSimpleList(analysis.immediate_actions, "No immediate actions provided.");
  recoveryEl.innerHTML = renderSimpleList(analysis.recovery_plan, "No recovery plan provided.");
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

async function clearTimeline() {
  if (!confirm("Clear all incidents from the timeline? This cannot be undone.")) return;
  try {
    const res = await fetch("/api/history", { method: "DELETE" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.error || "Clear failed");
    await refreshHistory();
  } catch (err) {
    alert("Could not clear timeline: " + err.message);
  }
}

async function refreshHistory() {
  const res = await fetch("/api/history");
  const data = await res.json();
  const tbody = document.getElementById("historyTable");
  const emptyHint = document.getElementById("historyEmpty");
  tbody.innerHTML = "";
  if (!data.history || data.history.length === 0) {
    emptyHint.hidden = false;
    return;
  }
  emptyHint.hidden = true;
  data.history.forEach((row) => {
    const tr = document.createElement("tr");
    const sev = row.severity || "";
    tr.innerHTML = `
      <td>${escapeHtml(String(row.timestamp))}</td>
      <td>${escapeHtml(String(row.site_id))}</td>
      <td>${escapeHtml(String(row.asset_id))}</td>
      <td><span class="${severityBadgeClass(sev)}">${escapeHtml(sev)}</span></td>
      <td>${Number.isFinite(Number(row.risk_score)) ? Number(row.risk_score).toFixed(3) : "-"}</td>
      <td class="reason-cell">${escapeHtml(String(row.reason))}</td>`;
    tbody.appendChild(tr);
  });
}

async function refreshStatus() {
  const badgeEl = document.getElementById("modelBadge");
  const pillEl = document.getElementById("envPill");
  badgeEl.textContent = "";
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const degraded = data.system_status === "degraded" || (data.model_errors && data.model_errors.length > 0);
    pillEl.textContent = degraded ? "Degraded" : "Operations";
    pillEl.classList.toggle("degraded", degraded);
    if (data.model_errors && data.model_errors.length > 0) {
      badgeEl.textContent = "Model: " + data.model_errors.length + " issue(s) - check /api/status";
    }
  } catch {
    pillEl.textContent = "Status unknown";
    pillEl.classList.add("degraded");
  }
}

renderPayloadPreviewMessage("Loading sample payload...");
loadSample("normal");
refreshHistory();
refreshStatus();
