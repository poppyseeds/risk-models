const SAMPLE_FILES = {
  normal: "/samples/normal.json",
  suspicious_network: "/samples/suspicious_network.json",
  suspicious_process: "/samples/suspicious_process.json",
  combined_attack: "/samples/combined_attack.json",
  hardware_tamper: "/samples/hardware_tamper.json",
};

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
  document.getElementById("payloadInput").value = JSON.stringify(sample, null, 2);
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

document.getElementById("fileInput").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const text = await file.text();
  const payload = file.name.endsWith(".csv") ? csvToPayload(text) : JSON.parse(text);
  document.getElementById("payloadInput").value = JSON.stringify(payload, null, 2);
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
  const sys = risk.system_status || "—";
  const sev = risk.severity || "—";
  statusEl.innerHTML = `${escapeHtml(sys)} <span class="${severityBadgeClass(sev)}">${escapeHtml(sev)}</span>`;

  document.getElementById("riskCard").textContent =
    typeof risk.fused_risk_score === "number" ? risk.fused_risk_score.toFixed(3) : "—";
  document.getElementById("networkCard").textContent =
    typeof risk.network_score === "number" ? risk.network_score.toFixed(3) : "—";
  document.getElementById("processCard").textContent =
    typeof risk.process_score === "number" ? risk.process_score.toFixed(3) : "—";
  document.getElementById("hardwareCard").textContent =
    typeof risk.hardware_score === "number" ? risk.hardware_score.toFixed(3) : "—";
  document.getElementById("latestDetection").textContent = JSON.stringify(risk, null, 2);
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
      <td>${Number.isFinite(Number(row.risk_score)) ? Number(row.risk_score).toFixed(3) : "—"}</td>
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
      badgeEl.textContent = "Model: " + data.model_errors.length + " issue(s) — check /api/status";
    }
  } catch {
    pillEl.textContent = "Status unknown";
    pillEl.classList.add("degraded");
  }
}

loadSample("normal");
refreshHistory();
refreshStatus();
