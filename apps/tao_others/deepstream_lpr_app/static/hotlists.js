const summaryMeta = document.getElementById("summaryMeta");
const uniquePlatesStat = document.getElementById("uniquePlatesStat");
const totalEntriesStat = document.getElementById("totalEntriesStat");
const localEntriesStat = document.getElementById("localEntriesStat");
const lastReloadStat = document.getElementById("lastReloadStat");
const sourceStatusList = document.getElementById("sourceStatusList");
const entriesTableBody = document.getElementById("entriesTableBody");
const sourceKindFilter = document.getElementById("sourceKindFilter");
const messageBanner = document.getElementById("messageBanner");
const reloadBtn = document.getElementById("reloadBtn");
const refreshBtn = document.getElementById("refreshBtn");
const agencyUploadForm = document.getElementById("agencyUploadForm");
const manualEntryForm = document.getElementById("manualEntryForm");
const csvUploadForm = document.getElementById("csvUploadForm");
const manualDate = document.getElementById("manualDate");

let hotlistStatus = null;

function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

function formatDateTime(value) {
  if (!value) {
    return "Never";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString([], { hour12: false });
}

function showMessage(text, tone = "success") {
  if (!text) {
    messageBanner.textContent = "";
    messageBanner.className = "message-banner hidden";
    return;
  }
  messageBanner.textContent = text;
  messageBanner.className = `message-banner is-${tone}`;
}

function detailToText(detail) {
  if (!detail) {
    return "Request failed.";
  }
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail.map((item) => detailToText(item)).join("; ");
  }
  if (detail.message && detail.errors && typeof detail.errors === "object") {
    const parts = [];
    for (const [key, value] of Object.entries(detail.errors)) {
      parts.push(`${key}: ${detailToText(value)}`);
    }
    return `${detail.message} ${parts.join("; ")}`.trim();
  }
  if (detail.message) {
    return String(detail.message);
  }
  return JSON.stringify(detail);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(detailToText(payload.detail || payload));
  }
  return payload;
}

function renderSummary(status) {
  hotlistStatus = status;
  uniquePlatesStat.textContent = String(status.unique_plates ?? 0);
  totalEntriesStat.textContent = String(status.total_entries ?? 0);
  localEntriesStat.textContent = String(status.local_entry_count ?? 0);
  lastReloadStat.textContent = formatDateTime(status.last_reload_utc);
  summaryMeta.textContent = `Hotlist root: ${status.hotlist_root}`;
}

function sourceBadge(detail) {
  const badgeClass = detail.source_kind === "local" ? "badge is-local" : "badge is-agency";
  return `<span class="${badgeClass}">${detail.source_kind === "local" ? "Local" : "Agency"}</span>`;
}

function renderSourceStatus(status) {
  const details = Array.isArray(status.source_details) ? status.source_details : [];
  if (!details.length) {
    sourceStatusList.innerHTML = '<div class="empty-state">No source details available.</div>';
    return;
  }

  sourceStatusList.innerHTML = details.map((detail) => `
    <div class="source-card">
      <div class="source-card-header">
        <div class="source-card-title">${detail.label}</div>
        ${sourceBadge(detail)}
      </div>
      <div class="source-card-meta">
        <div>File: ${detail.file_name || "-"}</div>
        <div>Records: ${detail.record_count ?? 0} | Unique plates: ${detail.unique_plates ?? 0}</div>
        <div>Updated: ${formatDateTime(detail.updated_utc)}</div>
        <div>Last import: ${formatDateTime(detail.last_import_utc)}</div>
      </div>
    </div>
  `).join("");
}

function renderEntries(payload) {
  const entries = Array.isArray(payload.entries) ? payload.entries : [];
  if (!entries.length) {
    entriesTableBody.innerHTML = '<tr><td colspan="7" class="empty-state">No entries found for this filter.</td></tr>';
    return;
  }

  entriesTableBody.innerHTML = entries.map((entry) => {
    const isLocal = entry.source_kind === "local";
    const action = isLocal
      ? `<button class="button button-danger delete-entry-btn" type="button" data-entry-id="${entry.entry_id}">Delete</button>`
      : '<span class="empty-state">Agency managed</span>';
    return `
      <tr>
        <td>${entry.plate}</td>
        <td>${entry.list_type} · ${entry.list_label}</td>
        <td>${sourceBadge(entry)}</td>
        <td>${entry.state}</td>
        <td>${entry.county_code}</td>
        <td>${entry.entry_date}</td>
        <td>${action}</td>
      </tr>
    `;
  }).join("");

  document.querySelectorAll(".delete-entry-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const entryId = button.dataset.entryId;
      if (!entryId) {
        return;
      }
      if (!window.confirm("Remove this local hotlist entry?")) {
        return;
      }
      try {
        await fetchJson(`/api/hotlist/local-entry/${encodeURIComponent(entryId)}`, { method: "DELETE" });
        showMessage("Local hotlist entry removed.");
        await loadPageData();
      } catch (error) {
        showMessage(error.message, "error");
      }
    });
  });
}

function summarizeUploadWarnings(warnings) {
  const parts = [];
  for (const [fileName, issues] of Object.entries(warnings || {})) {
    if (!Array.isArray(issues) || !issues.length) {
      continue;
    }

    const lines = issues
      .map((issue) => Number(issue?.line))
      .filter((line) => Number.isFinite(line) && line > 0)
      .slice(0, 3);
    const lineSuffix = lines.length ? ` (lines ${lines.join(", ")}${issues.length > lines.length ? ", ..." : ""})` : "";
    const rowLabel = issues.length === 1 ? "row" : "rows";
    parts.push(`${fileName} ignored ${issues.length} ${rowLabel}${lineSuffix}`);
  }
  return parts.join("; ");
}

async function loadEntries() {
  const sourceKind = sourceKindFilter.value || "all";
  const payload = await fetchJson(`/api/hotlist/entries?source_kind=${encodeURIComponent(sourceKind)}&limit=500`);
  renderEntries(payload);
}

async function loadPageData() {
  const status = await fetchJson("/api/hotlist/status", { cache: "no-store" });
  renderSummary(status);
  renderSourceStatus(status);
  await loadEntries();
}

agencyUploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(agencyUploadForm);
  const hasAnyFile = ["svs_file", "slr_file", "sfr_file"].some((name) => {
    const file = formData.get(name);
    return file instanceof File && file.size > 0;
  });
  if (!hasAnyFile) {
    showMessage("Choose at least one agency file to upload.", "error");
    return;
  }

  try {
    const result = await fetchJson("/api/hotlist/upload", { method: "POST", body: formData });
    agencyUploadForm.reset();
    const warningSummary = summarizeUploadWarnings(result.warnings);
    const warningSuffix = warningSummary ? ` Warnings: ${warningSummary}.` : "";
    showMessage(`Agency hotlists updated: ${result.updated_sources.join(", ")}.${warningSuffix}`);
    await loadPageData();
  } catch (error) {
    showMessage(error.message, "error");
  }
});

manualEntryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    plate: document.getElementById("manualPlate").value,
    list_type: document.getElementById("manualListType").value,
    state: document.getElementById("manualState").value,
    county_code: document.getElementById("manualCounty").value,
    entry_date: document.getElementById("manualDate").value || null,
  };

  try {
    const result = await fetchJson("/api/hotlist/local-entry", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    manualEntryForm.reset();
    manualDate.value = todayIso();
    showMessage(result.created ? "Local hotlist entry added." : "Matching local entry already exists.");
    await loadPageData();
  } catch (error) {
    showMessage(error.message, "error");
  }
});

csvUploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(csvUploadForm);
  const file = formData.get("csv_file");
  if (!(file instanceof File) || file.size <= 0) {
    showMessage("Choose a CSV file to import.", "error");
    return;
  }

  try {
    const result = await fetchJson("/api/hotlist/local-csv", { method: "POST", body: formData });
    csvUploadForm.reset();
    const issues = Array.isArray(result.issues) && result.issues.length
      ? ` ${result.issues.length} rows were rejected.`
      : "";
    showMessage(`Imported ${result.imported_count} local entries.${issues}`);
    await loadPageData();
  } catch (error) {
    showMessage(error.message, "error");
  }
});

reloadBtn.addEventListener("click", async () => {
  try {
    await fetchJson("/api/hotlist/reload", { method: "POST" });
    showMessage("Hotlists reloaded from disk.");
    await loadPageData();
  } catch (error) {
    showMessage(error.message, "error");
  }
});

refreshBtn.addEventListener("click", async () => {
  try {
    await loadPageData();
    showMessage("Hotlist status refreshed.");
  } catch (error) {
    showMessage(error.message, "error");
  }
});

sourceKindFilter.addEventListener("change", async () => {
  try {
    await loadEntries();
  } catch (error) {
    showMessage(error.message, "error");
  }
});

manualDate.value = todayIso();
loadPageData().catch((error) => {
  showMessage(error.message, "error");
});