const previewImage = document.getElementById("previewImage");
const zoomImage = document.getElementById("zoomImage");
const previewEmpty = document.getElementById("previewEmpty");
const zoomEmpty = document.getElementById("zoomEmpty");
const selectedSourceLabel = document.getElementById("selectedSourceLabel");
const selectedSourceMeta = document.getElementById("selectedSourceMeta");
const previewTimestamp = document.getElementById("previewTimestamp");
const cameraWarning = document.getElementById("cameraWarning");
const runtimeSummary = document.getElementById("runtimeSummary");
const sourceList = document.getElementById("sourceList");
const overlayToolbar = document.getElementById("overlayToolbar");
const previewShell = document.getElementById("previewShell");
const previewDetections = document.getElementById("previewDetections");
const diagnosticsPanel = document.getElementById("diagnosticsPanel");
const controlsPanel = document.getElementById("controlsPanel");
const networkAccessPanel = document.getElementById("networkAccessPanel");

let selectedSource = null;
let lastControlsFingerprint = "";
let latestPresets = [];
let currentSources = [];
let currentPreviewUrl = "";
let displayedPreviewUrl = "";
let previewLoadToken = 0;
let previewWarningMessage = "";
let inlineRenameSourceKey = null;

function absolutePreviewUrl(url) {
  try {
    return new URL(url, window.location.href).href;
  } catch (_error) {
    return url;
  }
}

const DEFAULT_CAMERA_PRESETS = [
  { id: "alpr_day", label: "ALPR Day", description: "Manual exposure and locked white balance for daytime traffic." },
  { id: "alpr_night", label: "ALPR Night", description: "Longer manual exposure, more gain, and locked white balance for low light." },
  { id: "factory_auto", label: "Factory Auto", description: "Restore camera auto exposure defaults." },
];

const overlayState = {
  grid: true,
  crosshair: true,
  safeZone: true,
  zoom: true,
};

const overlayOptions = [
  { key: "grid", label: "Thirds Grid", className: "show-grid" },
  { key: "crosshair", label: "Crosshair", className: "show-crosshair" },
  { key: "safeZone", label: "Safe Zone", className: "show-safe-zone" },
  { key: "zoom", label: "Center Zoom", className: null },
];

function formatTime(value) {
  if (!value) {
    return "never";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleTimeString([], { hour12: false });
}

function formatDateTime(value) {
  if (!value) {
    return "never";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString([], { hour12: false });
}

function metricChip(text) {
  const chip = document.createElement("span");
  chip.className = "metric-chip";
  chip.textContent = text;
  return chip;
}

function setStatusText(element, text, tone = "") {
  element.textContent = text;
  element.className = `control-status${tone ? ` ${tone}` : ""}`;
}

function selectedSourceData() {
  return Array.isArray(currentSources)
    ? currentSources.find((source) => source.source === selectedSource) || null
    : null;
}

function renderCameraWarning(source = selectedSourceData()) {
  const message = previewWarningMessage || source?.preview_warning || "";
  if (!message) {
    cameraWarning.textContent = "";
    cameraWarning.classList.add("hidden");
    return;
  }

  cameraWarning.textContent = message;
  cameraWarning.classList.remove("hidden");
}

function focusStateLabel(focusState) {
  switch (focusState) {
    case "in_focus":
      return "In focus";
    case "approaching_focus":
      return "Near focus";
    default:
      return "Unreadable";
  }
}

function focusStateTone(focusState) {
  switch (focusState) {
    case "in_focus":
      return "green";
    case "approaching_focus":
      return "yellow";
    default:
      return "red";
  }
}

function renderPreviewDetections(source = selectedSourceData()) {
  previewDetections.innerHTML = "";

  const detections = Array.isArray(source?.preview_detections) ? source.preview_detections : [];
  const overlayWidth = Number(source?.preview_overlay_width || 0);
  const overlayHeight = Number(source?.preview_overlay_height || 0);

  if (
    !source
    || source.preview_mode !== "runtime"
    || detections.length === 0
    || overlayWidth <= 0
    || overlayHeight <= 0
  ) {
    previewDetections.classList.add("hidden");
    return;
  }

  for (const detection of detections) {
    const left = Number(detection.left || 0);
    const top = Number(detection.top || 0);
    const width = Number(detection.width || 0);
    const height = Number(detection.height || 0);
    if (width <= 0 || height <= 0) {
      continue;
    }

    const focusState = typeof detection.focus_state === "string" ? detection.focus_state : "out_of_focus";
    const tone = focusStateTone(focusState);
    const plateText = typeof detection.plate === "string" ? detection.plate.trim() : "";
    const confidence = Number(detection.confidence || 0);

    const box = document.createElement("div");
    box.className = `preview-detection is-${tone}`;
    box.style.left = `${(left / overlayWidth) * 100}%`;
    box.style.top = `${(top / overlayHeight) * 100}%`;
    box.style.width = `${(width / overlayWidth) * 100}%`;
    box.style.height = `${(height / overlayHeight) * 100}%`;

    const tag = document.createElement("div");
    tag.className = "preview-detection-tag";
    const confidenceSuffix = confidence > 0 ? ` (${confidence}%)` : "";
    tag.textContent = plateText || `${focusStateLabel(focusState)}${confidenceSuffix}`;
    if (plateText && confidence > 0) {
      tag.textContent = `${plateText}${confidenceSuffix}`;
    }
    tag.title = `${focusStateLabel(focusState)}${confidenceSuffix}`;

    box.appendChild(tag);
    previewDetections.appendChild(box);
  }

  previewDetections.classList.toggle("hidden", previewDetections.childElementCount === 0);
}

function resetPreviewState(message) {
  currentPreviewUrl = "";
  displayedPreviewUrl = "";
  previewImage.removeAttribute("src");
  zoomImage.removeAttribute("src");
  previewImage.classList.add("hidden");
  zoomImage.classList.add("hidden");
  previewDetections.innerHTML = "";
  previewDetections.classList.add("hidden");
  previewEmpty.classList.remove("hidden");
  zoomEmpty.classList.remove("hidden");
  previewEmpty.textContent = message;
  zoomEmpty.textContent = message;
}

function setPreviewLoading(message) {
  if (displayedPreviewUrl) {
    previewEmpty.classList.add("hidden");
    if (overlayState.zoom) {
      zoomEmpty.classList.add("hidden");
    }
    return;
  }

  previewImage.classList.add("hidden");
  zoomImage.classList.add("hidden");
  previewEmpty.classList.remove("hidden");
  zoomEmpty.classList.remove("hidden");
  previewEmpty.textContent = message;
  zoomEmpty.textContent = overlayState.zoom ? message : "Center zoom disabled.";
}

function commitPreviewImage(imagePath) {
  previewImage.src = imagePath;
  previewImage.classList.remove("hidden");
  previewEmpty.classList.add("hidden");

  displayedPreviewUrl = imagePath;
  previewWarningMessage = "";
  renderCameraWarning();

  if (overlayState.zoom) {
    zoomImage.src = imagePath;
    zoomImage.classList.remove("hidden");
    zoomEmpty.classList.add("hidden");
  } else {
    zoomImage.classList.add("hidden");
    zoomEmpty.classList.remove("hidden");
    zoomEmpty.textContent = "Center zoom disabled.";
  }
}

function queuePreviewImage(imagePath) {
  if (!imagePath || imagePath === currentPreviewUrl) {
    return;
  }

  currentPreviewUrl = imagePath;
  const loadToken = ++previewLoadToken;
  setPreviewLoading("Loading preview frame...");

  const loader = new Image();
  loader.onload = () => {
    if (loadToken !== previewLoadToken || currentPreviewUrl !== imagePath) {
      return;
    }
    commitPreviewImage(imagePath);
  };
  loader.onerror = () => {
    if (loadToken !== previewLoadToken || currentPreviewUrl !== imagePath) {
      return;
    }
    previewWarningMessage = "Camera link unstable or disconnected. Check the USB cable, port, or power.";
    renderCameraWarning();
    if (!displayedPreviewUrl) {
      resetPreviewState("Preview image could not be loaded. Hard refresh the page and try again.");
    }
  };
  loader.src = imagePath;
}

function renderOverlayToolbar() {
  overlayToolbar.innerHTML = "";
  for (const option of overlayOptions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `overlay-button${overlayState[option.key] ? " active" : ""}`;
    button.textContent = option.label;
    button.onclick = () => {
      overlayState[option.key] = !overlayState[option.key];
      renderOverlayToolbar();
      applyOverlayState();
    };
    overlayToolbar.appendChild(button);
  }
}

function applyOverlayState() {
  for (const option of overlayOptions) {
    if (option.className) {
      previewShell.classList.toggle(option.className, Boolean(overlayState[option.key]));
    }
  }

  if (!overlayState.zoom) {
    zoomImage.classList.add("hidden");
    zoomEmpty.classList.remove("hidden");
    zoomEmpty.textContent = "Center zoom disabled.";
  } else if (displayedPreviewUrl) {
    zoomImage.classList.remove("hidden");
    zoomEmpty.classList.add("hidden");
  }
}

function diagnosticsRow(label, value) {
  const row = document.createElement("div");
  row.className = "diagnostic-row";

  const labelNode = document.createElement("div");
  labelNode.className = "diagnostic-label";
  labelNode.textContent = label;

  const valueNode = document.createElement("div");
  valueNode.className = "diagnostic-value";
  valueNode.textContent = value;

  row.appendChild(labelNode);
  row.appendChild(valueNode);
  return row;
}

function renderNetworkAccess(items) {
  networkAccessPanel.innerHTML = "";

  const accessItems = Array.isArray(items) ? items : [];
  if (accessItems.length === 0) {
    networkAccessPanel.textContent = "No active network access URLs reported yet.";
    return;
  }

  for (const item of accessItems) {
    const link = document.createElement("a");
    const medium = String(item.medium || "network").toLowerCase();
    link.className = `network-access-link network-${medium}`;
    link.href = item.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = `${String(item.medium || "network").toUpperCase()}${item.interface ? ` ${item.interface}` : ""} - ${item.address}`;

    const note = document.createElement("div");
    note.className = "network-access-note";
    note.textContent = item.note ? `${item.note} | ${item.url}` : item.url;

    const row = document.createElement("div");
    row.className = "network-access-item";
    row.appendChild(link);
    row.appendChild(note);
    networkAccessPanel.appendChild(row);
  }
}

function controlFingerprint(source) {
  return JSON.stringify({
    source: source?.source || null,
    device: source?.device_path || null,
    controls: source?.camera_controls || [],
  });
}

function createControlValueInput(control) {
  const controlType = String(control.type || "").toLowerCase();

  if (controlType === "menu" || controlType === "bool") {
    const select = document.createElement("select");
    const options = Array.isArray(control.options) && control.options.length > 0
      ? control.options
      : [
          { value: 0, label: "Off" },
          { value: 1, label: "On" },
        ];

    for (const option of options) {
      const item = document.createElement("option");
      item.value = String(option.value);
      item.textContent = option.label;
      if (String(option.value) === String(control.value)) {
        item.selected = true;
      }
      select.appendChild(item);
    }
    return select;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "control-input-stack";

  const range = document.createElement("input");
  range.type = "range";
  range.min = String(control.min ?? 0);
  range.max = String(control.max ?? 100);
  range.step = String(control.step ?? 1);
  range.value = String(control.value ?? control.default ?? 0);

  const number = document.createElement("input");
  number.type = "number";
  number.min = range.min;
  number.max = range.max;
  number.step = range.step;
  number.value = range.value;

  range.oninput = () => {
    number.value = range.value;
  };
  number.oninput = () => {
    range.value = number.value;
  };

  wrapper.appendChild(range);
  wrapper.appendChild(number);
  wrapper.valueInput = number;
  return wrapper;
}

function currentInputValue(input) {
  if (!input) {
    return "";
  }
  if (typeof input.value === "string") {
    return input.value;
  }
  if (input.valueInput && typeof input.valueInput.value === "string") {
    return input.valueInput.value;
  }
  return "";
}

async function submitControl(source, control, input, statusNode, applyButton) {
  const value = currentInputValue(input);
  setStatusText(statusNode, "Applying...", "warn");
  applyButton.disabled = true;

  try {
    const response = await fetch("/api/camera-control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: source.source, control: control.name, value }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `request failed: ${response.status}`);
    }
    setStatusText(statusNode, "Applied", "success");
    lastControlsFingerprint = "";
    await refresh();
  } catch (error) {
    setStatusText(statusNode, error.message, "error");
  } finally {
    applyButton.disabled = false;
  }
}

async function applyPreset(source, preset, statusNode, button) {
  setStatusText(statusNode, `Applying ${preset.label}...`, "warn");
  button.disabled = true;

  try {
    const response = await fetch("/api/camera-preset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: source.source, preset: preset.id }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `request failed: ${response.status}`);
    }

    const appliedCount = Array.isArray(payload.applied_controls) ? payload.applied_controls.length : 0;
    const missingCount = Array.isArray(payload.missing_controls) ? payload.missing_controls.length : 0;
    const missingNote = missingCount > 0 ? `, ${missingCount} skipped` : "";
    setStatusText(statusNode, `Applied ${appliedCount} control(s)${missingNote}`, "success");
    lastControlsFingerprint = "";
    await refresh();
  } catch (error) {
    setStatusText(statusNode, error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function renderSourceList(sources) {
  sourceList.innerHTML = "";
  if (!Array.isArray(sources) || sources.length === 0) {
    sourceList.textContent = "No camera sources configured.";
    return;
  }

  if (!selectedSource || !sources.some((source) => source.source === selectedSource)) {
    const preferred = sources.find((source) => source.available) || sources.find((source) => source.preview_url) || sources[0];
    selectedSource = preferred ? preferred.source : null;
  }

  for (const source of sources) {
    const entry = document.createElement("div");
    entry.className = "source-entry";

    const button = document.createElement("button");
    button.type = "button";
    button.className = `source-button${source.source === selectedSource ? " active" : ""}`;
    button.onclick = () => {
      selectedSource = source.source;
      lastControlsFingerprint = "";
      renderSourceList(sources);
      renderPreview(sources);
      renderDiagnostics(sources);
      renderControls(sources, latestPresets);
    };

    const topline = document.createElement("div");
    topline.className = "source-topline";

    const code = document.createElement("div");
    code.className = "source-code";
    code.textContent = `${source.source_label || source.source}`;
    topline.appendChild(code);

    const state = document.createElement("div");
    const hasPreview = Boolean(source.preview_url);
    state.className = `source-state${source.available || hasPreview ? "" : " offline"}`;
    state.textContent = source.available ? "ONLINE" : hasPreview ? "DIRECT" : "OFFLINE";
    topline.appendChild(state);

    const metrics = document.createElement("div");
    metrics.className = "source-metrics";
    metrics.appendChild(metricChip(`${Number(source.fps || 0).toFixed(1)} FPS`));
    if (source.frame_width && source.frame_height) {
      metrics.appendChild(metricChip(`${source.frame_width}x${source.frame_height}`));
    }
    metrics.appendChild(metricChip(`Frame ${source.last_frame_number ?? "-"}`));

    const detail = document.createElement("div");
    detail.className = "source-detail";
    detail.textContent = `Camera ID: ${source.process_source_name || "n/a"} | ${source.enabled ? "ENABLED" : "DISABLED"} | Last seen ${formatTime(source.last_seen_utc)}`;

    button.appendChild(topline);
    button.appendChild(metrics);
    button.appendChild(detail);

    if (source.device_path || source.camera_card) {
      const device = document.createElement("div");
      device.className = "source-device";
      device.textContent = `${source.device_path || ""}${source.camera_card ? ` | ${source.camera_card}` : ""}`;
      button.appendChild(device);
    }

    entry.appendChild(button);

    // Per-source management actions
    const actions = document.createElement("div");
    actions.className = "source-actions";

    const renameBtn = document.createElement("button");
    renameBtn.type = "button";
    renameBtn.className = "source-action-btn";
    renameBtn.textContent = "Rename";
    renameBtn.onclick = () => beginInlineRename(source.process_source_name);

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "source-action-btn source-action-remove";
    removeBtn.textContent = "Remove";
    removeBtn.onclick = () => removeCamera(source.process_source_name, source.source);

    const enableBtn = document.createElement("button");
    enableBtn.type = "button";
    enableBtn.className = "source-action-btn";
    enableBtn.textContent = source.enabled ? "Disable" : "Enable";
    enableBtn.onclick = () => setCameraEnabled(source.process_source_name, !source.enabled);

    actions.appendChild(renameBtn);
    actions.appendChild(enableBtn);
    actions.appendChild(removeBtn);
    entry.appendChild(actions);

    if (inlineRenameSourceKey === source.process_source_name) {
      const renameInline = document.createElement("div");
      renameInline.className = "source-inline-rename";

      const renameInput = document.createElement("input");
      renameInput.type = "text";
      renameInput.className = "source-inline-input";
      renameInput.maxLength = 64;
      renameInput.value = source.source;
      renameInput.placeholder = "Camera name";
      renameInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          submitInlineRename(source.process_source_name, renameInput.value.trim());
        }
        if (event.key === "Escape") {
          event.preventDefault();
          cancelInlineRename();
        }
      });

      const saveBtn = document.createElement("button");
      saveBtn.type = "button";
      saveBtn.className = "source-action-btn";
      saveBtn.textContent = "Save";
      saveBtn.onclick = () => submitInlineRename(source.process_source_name, renameInput.value.trim());

      const cancelBtn = document.createElement("button");
      cancelBtn.type = "button";
      cancelBtn.className = "source-action-btn";
      cancelBtn.textContent = "Cancel";
      cancelBtn.onclick = cancelInlineRename;

      renameInline.appendChild(renameInput);
      renameInline.appendChild(saveBtn);
      renameInline.appendChild(cancelBtn);
      entry.appendChild(renameInline);

      requestAnimationFrame(() => {
        renameInput.focus();
        renameInput.select();
      });
    }

    sourceList.appendChild(entry);
  }
}

function renderPreview(sources) {
  const source = Array.isArray(sources) ? sources.find((item) => item.source === selectedSource) : null;
  if (!source) {
    selectedSourceLabel.textContent = "No Camera Selected";
    selectedSourceMeta.textContent = "Waiting for runtime telemetry...";
    previewTimestamp.textContent = "No preview yet";
    renderCameraWarning(null);
    resetPreviewState("Waiting for preview frame...");
    return;
  }

  selectedSourceLabel.textContent = `${source.source_label || source.source} (${source.source})`;
  const previewModeLabel = source.preview_mode === "runtime"
    ? "DeepStream preview"
    : source.preview_mode === "direct"
      ? "Direct camera preview"
      : "No preview";
  const previewFps = Number(source.fps || source.capture_fps || 0).toFixed(1);
  const previewWidth = source.frame_width || source.capture_width || "-";
  const previewHeight = source.frame_height || source.capture_height || "-";
  selectedSourceMeta.textContent = `${previewModeLabel} | ${previewFps} FPS | ${previewWidth}x${previewHeight}`;
  previewTimestamp.textContent = source.preview_updated_utc
    ? `Preview ${formatTime(source.preview_updated_utc)}`
    : source.preview_mode === "direct"
      ? "Direct camera preview"
      : "No preview yet";
  renderCameraWarning(source);
  renderPreviewDetections(source);

  if (source.preview_url) {
    const separator = source.preview_url.includes("?") ? "&" : "?";
    const previewToken = source.preview_mode === "direct"
      ? Date.now()
      : [
          source.preview_updated_utc || source.last_seen_utc || "",
          source.preview_sequence || "",
          source.last_frame_number || "",
          Date.now(),
        ].join("-");
    const imagePath = absolutePreviewUrl(`${source.preview_url}${separator}t=${encodeURIComponent(previewToken)}`);
    if (source.preview_mode === "direct") {
      currentPreviewUrl = imagePath;
      commitPreviewImage(imagePath);
    } else if (imagePath !== currentPreviewUrl) {
      queuePreviewImage(imagePath);
    } else if (overlayState.zoom && displayedPreviewUrl) {
      zoomImage.classList.remove("hidden");
      zoomEmpty.classList.add("hidden");
    }

    if (!overlayState.zoom) {
      zoomImage.classList.add("hidden");
      zoomEmpty.classList.remove("hidden");
      zoomEmpty.textContent = "Center zoom disabled.";
    }
  } else {
    resetPreviewState("No preview image available yet.");
  }
}

function renderDiagnostics(sources) {
  const source = Array.isArray(sources) ? sources.find((item) => item.source === selectedSource) : null;
  diagnosticsPanel.innerHTML = "";

  if (!source) {
    diagnosticsPanel.textContent = "Waiting for source telemetry...";
    return;
  }

  const previewAgeSeconds = source.preview_updated_utc
    ? Math.max(0, Math.round((Date.now() - new Date(source.preview_updated_utc).getTime()) / 1000))
    : null;

  diagnosticsPanel.appendChild(diagnosticsRow("Preview source", source.preview_mode || "n/a"));
  diagnosticsPanel.appendChild(diagnosticsRow("Device", source.device_path || "n/a"));
  diagnosticsPanel.appendChild(diagnosticsRow("Camera", source.camera_card || "n/a"));
  diagnosticsPanel.appendChild(diagnosticsRow("Driver", source.camera_driver || "n/a"));
  diagnosticsPanel.appendChild(diagnosticsRow("Bus", source.camera_bus_info || "n/a"));
  diagnosticsPanel.appendChild(diagnosticsRow("Capture format", source.pixel_format || "n/a"));
  diagnosticsPanel.appendChild(diagnosticsRow(
    "Capture resolution",
    source.capture_width && source.capture_height ? `${source.capture_width}x${source.capture_height}` : "n/a",
  ));
  diagnosticsPanel.appendChild(diagnosticsRow(
    "Capture FPS",
    source.capture_fps ? `${Number(source.capture_fps).toFixed(1)} FPS` : "n/a",
  ));
  diagnosticsPanel.appendChild(diagnosticsRow(
    "Pipeline preview",
    source.frame_width && source.frame_height ? `${source.frame_width}x${source.frame_height}` : "n/a",
  ));
  diagnosticsPanel.appendChild(diagnosticsRow(
    "Preview age",
    source.preview_mode === "direct" ? "live" : previewAgeSeconds === null ? "n/a" : `${previewAgeSeconds}s`,
  ));
  diagnosticsPanel.appendChild(diagnosticsRow("Last seen", formatDateTime(source.last_seen_utc)));
}

function renderControls(sources, presets = DEFAULT_CAMERA_PRESETS) {
  const source = Array.isArray(sources) ? sources.find((item) => item.source === selectedSource) : null;

  if (!source || !source.device_path) {
    controlsPanel.textContent = "No configurable camera selected.";
    lastControlsFingerprint = "";
    return;
  }

  const fingerprint = controlFingerprint(source);
  if (fingerprint === lastControlsFingerprint) {
    return;
  }
  lastControlsFingerprint = fingerprint;

  controlsPanel.innerHTML = "";

  const presetWrap = document.createElement("div");
  presetWrap.className = "control-preset-wrap";

  const presetHeader = document.createElement("div");
  presetHeader.className = "control-preset-header";
  presetHeader.textContent = "Field Presets";
  presetWrap.appendChild(presetHeader);

  const presetButtons = document.createElement("div");
  presetButtons.className = "control-preset-buttons";
  const statusNode = document.createElement("div");
  setStatusText(statusNode, "Ready");

  for (const preset of presets) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "control-preset-button";
    button.textContent = preset.label;
    button.title = preset.description || "";
    button.onclick = () => applyPreset(source, preset, statusNode, button);
    presetButtons.appendChild(button);
  }

  presetWrap.appendChild(presetButtons);
  presetWrap.appendChild(statusNode);
  controlsPanel.appendChild(presetWrap);

  const controls = Array.isArray(source.camera_controls) ? source.camera_controls : [];
  if (controls.length === 0) {
    const emptyText = document.createElement("div");
    emptyText.textContent = "This device did not report any adjustable V4L2 controls.";
    controlsPanel.appendChild(emptyText);
    return;
  }

  for (const control of controls) {
    const row = document.createElement("div");
    row.className = "control-row";

    const head = document.createElement("div");
    head.className = "control-head";

    const label = document.createElement("div");
    label.className = "control-label";
    label.textContent = control.label || control.name;

    const meta = document.createElement("div");
    meta.className = "control-meta";
    meta.textContent = `Current ${control.value}${control.default !== undefined ? ` | Default ${control.default}` : ""}`;

    head.appendChild(label);
    head.appendChild(meta);
    row.appendChild(head);

    const input = createControlValueInput(control);
    const active = control.active !== false;
    if (input.valueInput) {
      input.valueInput.disabled = !active;
    }
    if (typeof input.disabled === "boolean") {
      input.disabled = !active;
    }
    const rangeInput = input.querySelector ? input.querySelector('input[type="range"]') : null;
    if (rangeInput) {
      rangeInput.disabled = !active;
    }
    row.appendChild(input);

    const actions = document.createElement("div");
    actions.className = "control-actions";

    const applyButton = document.createElement("button");
    applyButton.type = "button";
    applyButton.className = "control-apply";
    applyButton.textContent = "Apply";
    applyButton.disabled = !active;

    const statusNode = document.createElement("div");
    setStatusText(statusNode, active ? "Ready" : "Inactive while another mode is enabled", active ? "" : "warn");

    applyButton.onclick = () => submitControl(source, control, input, statusNode, applyButton);

    actions.appendChild(applyButton);
    actions.appendChild(statusNode);
    row.appendChild(actions);
    controlsPanel.appendChild(row);
  }
}

function renderRuntimeSummary(payload) {
  runtimeSummary.textContent = `Case ${payload.current_case_id || "n/a"} | ALPR ${payload.alpr_process_alive ? "running" : "offline"} | Runtime update ${formatTime(payload.runtime_status_utc)} | ${payload.sources.length} source(s)`;
}

async function refresh() {
  try {
    const response = await fetch("/api/camera-config", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`request failed: ${response.status}`);
    }

    const payload = await response.json();
    const sources = Array.isArray(payload.sources) ? payload.sources : [];
    currentSources = sources;
    latestPresets = Array.isArray(payload.presets) && payload.presets.length > 0
      ? payload.presets
      : DEFAULT_CAMERA_PRESETS;
    renderRuntimeSummary(payload);
    renderNetworkAccess(payload.network_access);
    renderSourceList(sources);
    renderPreview(sources);
    renderDiagnostics(sources);
    renderControls(sources, latestPresets);
  } catch (error) {
    runtimeSummary.textContent = `Preview unavailable: ${error.message}`;
  }
}

renderOverlayToolbar();
applyOverlayState();
refresh();
setInterval(refresh, 1000);

// ─── Camera management ───────────────────────────────────────────────────────

let renameModalSourceKey = null;

function showConfigChangesBanner() {
  document.getElementById("configChangesBanner").classList.remove("hidden");
}

function openAddCameraModal() {
  document.getElementById("addCameraUri").value = "";
  document.getElementById("addCameraName").value = "";
  setStatusText(document.getElementById("addCameraStatus"), "");
  document.getElementById("addCameraModal").classList.remove("hidden");
  document.getElementById("addCameraUri").focus();
}

function closeAddCameraModal() {
  document.getElementById("addCameraModal").classList.add("hidden");
}

function openRenameModal(sourceKey, currentName) {
  renameModalSourceKey = sourceKey;
  const nameInput = document.getElementById("renameCameraName");
  nameInput.value = currentName || "";
  setStatusText(document.getElementById("renameCameraStatus"), "");
  document.getElementById("renameCameraModal").classList.remove("hidden");
  nameInput.focus();
  nameInput.select();
}

function closeRenameModal() {
  document.getElementById("renameCameraModal").classList.add("hidden");
  renameModalSourceKey = null;
}

function beginInlineRename(sourceKey) {
  inlineRenameSourceKey = sourceKey;
  renderSourceList(currentSources);
}

function cancelInlineRename() {
  inlineRenameSourceKey = null;
  renderSourceList(currentSources);
}

async function submitInlineRename(sourceKey, nextName) {
  if (!sourceKey) return;
  const name = (nextName || "").trim();
  if (!name) {
    window.alert("Camera name is required.");
    return;
  }

  try {
    const response = await fetch(`/api/camera-source/${encodeURIComponent(sourceKey)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = payload.detail || `request failed: ${response.status}`;
      const msg = Array.isArray(detail) ? detail.map((e) => e.msg).join("; ") : String(detail);
      throw new Error(msg);
    }
    inlineRenameSourceKey = null;
    showConfigChangesBanner();
    await refresh();
  } catch (error) {
    window.alert(`Failed to rename camera: ${error.message}`);
  }
}

async function submitAddCamera() {
  const uri = document.getElementById("addCameraUri").value.trim();
  const name = document.getElementById("addCameraName").value.trim();
  const statusEl = document.getElementById("addCameraStatus");
  const submitBtn = document.getElementById("addCameraSubmit");

  setStatusText(statusEl, "Saving...", "warn");
  submitBtn.disabled = true;

  try {
    const response = await fetch("/api/camera-source", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ uri, name }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = payload.detail || `request failed: ${response.status}`;
      const msg = Array.isArray(detail) ? detail.map((e) => e.msg).join("; ") : String(detail);
      throw new Error(msg);
    }
    closeAddCameraModal();
    showConfigChangesBanner();
    await refresh();
  } catch (error) {
    setStatusText(statusEl, error.message, "error");
  } finally {
    submitBtn.disabled = false;
  }
}

async function submitRenameCamera() {
  if (!renameModalSourceKey) return;
  const name = document.getElementById("renameCameraName").value.trim();
  const statusEl = document.getElementById("renameCameraStatus");
  const submitBtn = document.getElementById("renameCameraSubmit");

  setStatusText(statusEl, "Saving...", "warn");
  submitBtn.disabled = true;

  try {
    const response = await fetch(`/api/camera-source/${encodeURIComponent(renameModalSourceKey)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = payload.detail || `request failed: ${response.status}`;
      const msg = Array.isArray(detail) ? detail.map((e) => e.msg).join("; ") : String(detail);
      throw new Error(msg);
    }
    closeRenameModal();
    showConfigChangesBanner();
    await refresh();
  } catch (error) {
    setStatusText(statusEl, error.message, "error");
  } finally {
    submitBtn.disabled = false;
  }
}

async function removeCamera(sourceKey, label) {
  const confirmed = window.confirm(
    `Remove camera "${label}" (${sourceKey}) from the config?\n\nThis will take effect after the ALPR pipeline is restarted.`
  );
  if (!confirmed) return;

  try {
    const response = await fetch(`/api/camera-source/${encodeURIComponent(sourceKey)}`, {
      method: "DELETE",
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `request failed: ${response.status}`);
    }
    showConfigChangesBanner();
    await refresh();
  } catch (error) {
    window.alert(`Failed to remove camera: ${error.message}`);
  }
}

async function setCameraEnabled(sourceKey, enabled) {
  try {
    const response = await fetch(`/api/camera-source/${encodeURIComponent(sourceKey)}/enabled`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `request failed: ${response.status}`);
    }
    showConfigChangesBanner();
    await refresh();
  } catch (error) {
    window.alert(`Failed to update camera state: ${error.message}`);
  }
}

async function restartAlprNow() {
  const restartBtn = document.getElementById("configChangesRestart");
  restartBtn.disabled = true;
  const previousText = restartBtn.textContent;
  restartBtn.textContent = "Restarting...";

  try {
    const response = await fetch("/api/alpr-restart", { method: "POST" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `request failed: ${response.status}`);
    }
    document.getElementById("configChangesBanner").classList.add("hidden");
    await refresh();
  } catch (error) {
    window.alert(`ALPR restart failed: ${error.message}`);
  } finally {
    restartBtn.disabled = false;
    restartBtn.textContent = previousText;
  }
}

// Modal event wiring
function bindClick(id, handler) {
  const element = document.getElementById(id);
  if (element) {
    element.addEventListener("click", handler);
  }
}

function bindKeydown(id, handler) {
  const element = document.getElementById(id);
  if (element) {
    element.addEventListener("keydown", handler);
  }
}

bindClick("addCameraBtn", openAddCameraModal);
bindClick("addCameraSubmit", submitAddCamera);
bindClick("addCameraCancel", closeAddCameraModal);
bindClick("addCameraModal", (e) => {
  if (e.target === e.currentTarget) closeAddCameraModal();
});
bindKeydown("addCameraUri", (e) => {
  if (e.key === "Enter") submitAddCamera();
  if (e.key === "Escape") closeAddCameraModal();
});
bindKeydown("addCameraName", (e) => {
  if (e.key === "Enter") submitAddCamera();
  if (e.key === "Escape") closeAddCameraModal();
});
bindClick("renameCameraSubmit", submitRenameCamera);
bindClick("renameCameraCancel", closeRenameModal);
bindClick("renameCameraModal", (e) => {
  if (e.target === e.currentTarget) closeRenameModal();
});
bindKeydown("renameCameraName", (e) => {
  if (e.key === "Enter") submitRenameCamera();
  if (e.key === "Escape") closeRenameModal();
});
bindClick("configChangesDismiss", () => {
  const banner = document.getElementById("configChangesBanner");
  if (banner) {
    banner.classList.add("hidden");
  }
});
bindClick("configChangesRestart", restartAlprNow);
