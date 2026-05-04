const feed = document.getElementById("feed");
const activeAlertsSection = document.getElementById("activeAlertsSection");
const activeAlerts = document.getElementById("activeAlerts");
const activeAlertsSummary = document.getElementById("activeAlertsSummary");
const searchInput = document.getElementById("search");
const sourceFilter = document.getElementById("sourceFilter");
const statusFilter = document.getElementById("statusFilter");
const hotlistOnlyBtn = document.getElementById("hotlistOnlyBtn");
const pauseBtn = document.getElementById("pauseBtn");
const pauseBadge = document.getElementById("pauseBadge");
const followBtn = document.getElementById("followBtn");
const audioBtn = document.getElementById("audioBtn");
const testAlertBtn = document.getElementById("testAlertBtn");
const audioArmBanner = document.getElementById("audioArmBanner");
const offlineBanner = document.getElementById("offlineBanner");
const pausedBanner = document.getElementById("pausedBanner");
const pausedCount = document.getElementById("pausedCount");
const alprAlive = document.getElementById("alprAlive");
const alprStatusBox = document.getElementById("alprStatusBox");
const networkAccessList = document.getElementById("networkAccessList");
const cameraStatusList = document.getElementById("cameraStatusList");
const gpsStatusChip = document.getElementById("gpsStatusChip");
const gpsStatusDetail = document.getElementById("gpsStatusDetail");
const imageViewer = document.getElementById("imageViewer");
const imageViewerImg = document.getElementById("imageViewerImg");
const imageViewerClose = document.getElementById("imageViewerClose");
const imageViewerPrev = document.getElementById("imageViewerPrev");
const imageViewerNext = document.getElementById("imageViewerNext");

let paused = false;
let followNew = true;
let audioEnabled = true;
let events = [];
let eventIds = new Set();
let socket = null;
let reconnectTimer = null;
let liveEventsRefreshInFlight = false;
let audioContext = null;
let audioKeepaliveOscillator = null;
let audioKeepaliveGain = null;
let audioArmed = false;
let bufferedWhilePaused = 0;
let liveProblem = false;
let liveFeedHealthy = false;
let lastKnownAlprAlive = null;
let viewerItems = [];
let viewerIndex = -1;
let viewerTouchStartX = null;
let viewerTouchStartY = null;
let suppressViewerClick = false;
let hotlistOnly = false;
let acknowledgingPlates = new Set();

const HOTLIST_PIN_MS = 60 * 1000;

function eventKey(event) {
  return (event && (event.display_id || event.event_id)) || "";
}

function updateAlprStatus(alive) {
  let text = "UNKNOWN";
  alprStatusBox.classList.remove("status-running", "status-paused", "status-stopped");

  if (paused) {
    text = "PAUSED";
    alprStatusBox.classList.add("status-paused");
  } else if (liveProblem) {
    text = "PROBLEM";
    alprStatusBox.classList.add("status-stopped");
  } else if (alive === true) {
    text = "RUNNING";
    alprStatusBox.classList.add("status-running");
  } else if (alive === false) {
    text = "STOPPED";
    alprStatusBox.classList.add("status-stopped");
  }

  alprAlive.textContent = text;
}

function syncPauseBadge() {
  pauseBadge.textContent = String(bufferedWhilePaused);
  pauseBadge.classList.toggle("hidden", bufferedWhilePaused <= 0);
}

function syncAudioBanner() {
  audioArmBanner.classList.toggle("hidden", audioArmed || !audioEnabled);
}

function currentViewerItem() {
  if (viewerIndex < 0 || viewerIndex >= viewerItems.length) {
    return null;
  }
  return viewerItems[viewerIndex];
}

function syncViewerNav() {
  const hasItems = viewerItems.length > 0;
  imageViewerPrev.disabled = !hasItems || viewerItems.length === 1;
  imageViewerNext.disabled = !hasItems || viewerItems.length === 1;
}

function showViewerIndex(index) {
  if (viewerItems.length === 0) {
    return;
  }

  viewerIndex = Math.max(0, Math.min(index, viewerItems.length - 1));
  const item = currentViewerItem();
  if (!item) {
    return;
  }

  imageViewerImg.src = item.url;
  imageViewerImg.alt = item.alt;
  imageViewer.classList.remove("hidden");
  imageViewer.setAttribute("aria-hidden", "false");
  syncViewerNav();
}

function openImageViewer(imageUrl, altText) {
  if (!imageUrl) {
    return;
  }
  viewerItems = [{ eventKey: "single", url: imageUrl, alt: altText || "ALPR read image" }];
  showViewerIndex(0);
}

function openImageViewerAt(index) {
  if (index < 0 || index >= viewerItems.length) {
    return;
  }
  showViewerIndex(index);
}

function showPreviousViewerItem() {
  if (viewerItems.length > 1) {
    showViewerIndex(viewerIndex - 1 < 0 ? viewerItems.length - 1 : viewerIndex - 1);
  }
}

function showNextViewerItem() {
  if (viewerItems.length > 1) {
    showViewerIndex((viewerIndex + 1) % viewerItems.length);
  }
}

function closeImageViewer() {
  imageViewer.classList.add("hidden");
  imageViewer.setAttribute("aria-hidden", "true");
  imageViewerImg.removeAttribute("src");
  viewerIndex = -1;
  viewerTouchStartX = null;
  viewerTouchStartY = null;
  suppressViewerClick = false;
}

imageViewerClose.addEventListener("click", closeImageViewer);
imageViewerPrev.addEventListener("click", (event) => {
  event.stopPropagation();
  showPreviousViewerItem();
});
imageViewerNext.addEventListener("click", (event) => {
  event.stopPropagation();
  showNextViewerItem();
});
imageViewer.addEventListener("touchstart", (event) => {
  if (event.touches.length !== 1) {
    viewerTouchStartX = null;
    viewerTouchStartY = null;
    return;
  }

  const touch = event.touches[0];
  viewerTouchStartX = touch.clientX;
  viewerTouchStartY = touch.clientY;
  suppressViewerClick = false;
}, { passive: true });
imageViewer.addEventListener("touchend", (event) => {
  if (viewerItems.length <= 1 || viewerTouchStartX === null || viewerTouchStartY === null || event.changedTouches.length !== 1) {
    viewerTouchStartX = null;
    viewerTouchStartY = null;
    return;
  }

  const touch = event.changedTouches[0];
  const deltaX = touch.clientX - viewerTouchStartX;
  const deltaY = touch.clientY - viewerTouchStartY;
  viewerTouchStartX = null;
  viewerTouchStartY = null;

  if (Math.abs(deltaX) < 40 || Math.abs(deltaX) <= Math.abs(deltaY) * 1.2) {
    return;
  }

  suppressViewerClick = true;
  if (deltaX > 0) {
    showPreviousViewerItem();
  } else {
    showNextViewerItem();
  }
});
imageViewer.addEventListener("click", (event) => {
  if (suppressViewerClick) {
    suppressViewerClick = false;
    return;
  }
  if (event.target === imageViewer) {
    closeImageViewer();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !imageViewer.classList.contains("hidden")) {
    closeImageViewer();
  } else if (event.key === "ArrowLeft" && !imageViewer.classList.contains("hidden") && viewerItems.length > 1) {
    showPreviousViewerItem();
  } else if (event.key === "ArrowRight" && !imageViewer.classList.contains("hidden") && viewerItems.length > 1) {
    showNextViewerItem();
  }
});

pauseBtn.onclick = () => {
  paused = !paused;
  pauseBtn.textContent = paused ? "Resume" : "Pause";
  pausedBanner.classList.toggle("hidden", !paused);
  updateAlprStatus(null);
  if (!paused) {
    bufferedWhilePaused = 0;
    pausedCount.textContent = "0";
    syncPauseBadge();
    render();
  }
};

followBtn.onclick = () => {
  followNew = !followNew;
  followBtn.textContent = followNew ? "Auto Scroll On" : "Auto Scroll Off";
  followBtn.classList.toggle("active", followNew);
};

hotlistOnlyBtn.onclick = () => {
  hotlistOnly = !hotlistOnly;
  hotlistOnlyBtn.classList.toggle("active", hotlistOnly);
  render();
};

audioBtn.onclick = async () => {
  audioEnabled = !audioEnabled;
  audioBtn.textContent = audioEnabled ? "Audio On" : "Audio Off";
  audioBtn.classList.toggle("active", audioEnabled);
  if (audioEnabled) {
    await armLiveAudio();
  } else {
    syncAudioBanner();
  }
};

async function armLiveAudio() {
  if (!audioEnabled) {
    audioEnabled = true;
    audioBtn.textContent = "Audio On";
    audioBtn.classList.add("active");
  }

  await primeAudio();
  await playLockedTone(true);
  audioArmed = true;
  syncAudioBanner();
}

audioArmBanner.onclick = async () => {
  try {
    await armLiveAudio();
  } catch (error) {
    console.error(error);
  }
};

testAlertBtn.onclick = async () => {
  if (!audioEnabled) {
    audioEnabled = true;
    audioBtn.textContent = "Audio On";
    audioBtn.classList.add("active");
  }

  testAlertBtn.textContent = "Testing...";
  testAlertBtn.classList.add("active");

  try {
    await primeAudio();
    await playHotlistTone(true);
    audioArmed = true;
    syncAudioBanner();
  } finally {
    window.setTimeout(() => {
      testAlertBtn.textContent = "Test Alert";
      testAlertBtn.classList.remove("active");
    }, 900);
  }
};

document.addEventListener("pointerdown", primeAudio, { once: true });
document.addEventListener("keydown", primeAudio, { once: true });

function buildImageUrl(path) {
  if (!path) return null;
  return `/media/${path}`;
}

function ensureAudioKeepalive(ctx) {
  if (!ctx || audioKeepaliveOscillator) {
    return;
  }

  audioKeepaliveGain = ctx.createGain();
  audioKeepaliveGain.gain.setValueAtTime(0.00001, ctx.currentTime);
  audioKeepaliveGain.connect(ctx.destination);

  audioKeepaliveOscillator = ctx.createOscillator();
  audioKeepaliveOscillator.type = "sine";
  audioKeepaliveOscillator.frequency.setValueAtTime(22, ctx.currentTime);
  audioKeepaliveOscillator.connect(audioKeepaliveGain);
  audioKeepaliveOscillator.start();
}

async function primeAudio() {
  if (!window.AudioContext && !window.webkitAudioContext) {
    return null;
  }

  if (!audioContext) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    audioContext = new Ctx();
  }

  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }

  ensureAudioKeepalive(audioContext);

  return audioContext;
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && audioEnabled) {
    primeAudio().catch(() => {});
  }
});

async function playLockedTone(force = false) {
  if (!audioEnabled && !force) return;

  const ctx = await primeAudio();
  if (!ctx) return;

  const start = ctx.currentTime;
  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(0.045, start + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.15);
  gain.connect(ctx.destination);

  const oscA = ctx.createOscillator();
  oscA.type = "sine";
  oscA.frequency.setValueAtTime(880, start);
  oscA.connect(gain);
  oscA.start(start);
  oscA.stop(start + 0.08);

  const oscB = ctx.createOscillator();
  oscB.type = "sine";
  oscB.frequency.setValueAtTime(1174, start + 0.06);
  oscB.connect(gain);
  oscB.start(start + 0.06);
  oscB.stop(start + 0.15);

  audioArmed = true;
  syncAudioBanner();
}

async function playHotlistTone(force = false) {
  if (!audioEnabled && !force) return;

  const ctx = await primeAudio();
  if (!ctx) return;

  const start = ctx.currentTime;
  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(0.065, start + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.34);
  gain.connect(ctx.destination);

  const oscA = ctx.createOscillator();
  oscA.type = "square";
  oscA.frequency.setValueAtTime(740, start);
  oscA.connect(gain);
  oscA.start(start);
  oscA.stop(start + 0.1);

  const oscB = ctx.createOscillator();
  oscB.type = "square";
  oscB.frequency.setValueAtTime(988, start + 0.11);
  oscB.connect(gain);
  oscB.start(start + 0.11);
  oscB.stop(start + 0.21);

  const oscC = ctx.createOscillator();
  oscC.type = "square";
  oscC.frequency.setValueAtTime(1318, start + 0.22);
  oscC.connect(gain);
  oscC.start(start + 0.22);
  oscC.stop(start + 0.34);

  audioArmed = true;
  syncAudioBanner();
}

function setOfflineBanner(visible, text) {
  offlineBanner.textContent = text;
  offlineBanner.classList.toggle("hidden", !visible);
}

function markLiveFeedHealthy() {
  liveFeedHealthy = true;
  liveProblem = false;
  setOfflineBanner(false, "");
  updateAlprStatus(lastKnownAlprAlive);
}

function markLiveFeedProblem(text) {
  liveFeedHealthy = false;
  liveProblem = true;
  setOfflineBanner(true, text);
  updateAlprStatus(lastKnownAlprAlive);
}

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], { hour12: false });
}

function formatStorage(gb) {
  if (gb === null || gb === undefined) return "-";
  return `${gb.toFixed ? gb.toFixed(1) : gb} GB`;
}

function formatGps(event) {
  if (!event || !event.gps_fix_valid) {
    return "";
  }

  const latitude = Number(event.gps_latitude);
  const longitude = Number(event.gps_longitude);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    return "";
  }

  const parts = [`${latitude.toFixed(6)}, ${longitude.toFixed(6)}`];
  const altitude = Number(event.gps_altitude_m);
  if (Number.isFinite(altitude)) {
    parts.push(`${altitude.toFixed(1)} m`);
  }
  return parts.join(" | ");
}

function sourceBadgeClass(source) {
  return `source-${String(source || "unknown").toLowerCase()}`;
}

function createBadge(text, className) {
  const badge = document.createElement("span");
  badge.className = className;
  badge.textContent = text;
  return badge;
}

function hotlistSummary(event) {
  if (!event.hotlist_hit) {
    return null;
  }

  const matchCount = Array.isArray(event.hotlist_entries) ? event.hotlist_entries.length : 0;
  const parts = [event.hotlist_highest_label || "Hotlist Hit"];
  if (matchCount > 1) {
    parts.push(`${matchCount} records`);
  }
  if (event.hotlist_alert && isPinnedHotlist(event)) {
    parts.push(`Pinned ${pinnedSecondsRemaining(event)}s`);
  } else if (event.hotlist_alert_reason === "locked-only") {
    parts.push("Waiting for LOCKED");
  } else {
    const suppressionText = suppressionSummary(event);
    if (suppressionText) {
      parts.push(suppressionText);
    }
  }
  return parts.join(" | ");
}

function suppressionSecondsRemaining(event) {
  const value = Date.parse(event && event.hotlist_alert_cooldown_until_utc ? event.hotlist_alert_cooldown_until_utc : "");
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.ceil((value - Date.now()) / 1000));
}

function formatSuppressionSeconds(totalSeconds) {
  if (totalSeconds <= 0) {
    return null;
  }

  if (totalSeconds >= 120) {
    return `${Math.ceil(totalSeconds / 60)}m`;
  }

  return `${totalSeconds}s`;
}

function suppressionSummary(event) {
  const remaining = formatSuppressionSeconds(suppressionSecondsRemaining(event));
  if (!remaining) {
    return null;
  }

  if (event.hotlist_alert_reason === "acknowledged") {
    return `Acknowledged ${remaining}`;
  }
  if (event.hotlist_alert_reason === "auto-snooze") {
    return `Auto Snooze ${remaining}`;
  }
  return null;
}

function eventTimestampMs(event) {
  const value = Date.parse(event && event.timestamp_utc ? event.timestamp_utc : "");
  return Number.isFinite(value) ? value : null;
}

function applyEventRuntimeState(event, fromLive) {
  if (!event || !event.hotlist_alert) {
    return event;
  }

  if (typeof event._pinUntilMs === "number") {
    return event;
  }

  const now = Date.now();
  if (fromLive) {
    event._pinUntilMs = now + HOTLIST_PIN_MS;
    return event;
  }

  const eventTime = eventTimestampMs(event);
  event._pinUntilMs = eventTime ? eventTime + HOTLIST_PIN_MS : now;
  return event;
}

function isPinnedHotlist(event) {
  return Boolean(
    event &&
    event.hotlist_alert &&
    typeof event._pinUntilMs === "number" &&
    event._pinUntilMs > Date.now()
  );
}

function pinnedSecondsRemaining(event) {
  if (!isPinnedHotlist(event)) {
    return 0;
  }
  return Math.max(1, Math.ceil((event._pinUntilMs - Date.now()) / 1000));
}

function hotlistBadgeClass(type) {
  switch (String(type || "").toUpperCase()) {
    case "SFR":
      return "badge hotlist-badge hotlist-sfr";
    case "SVS":
      return "badge hotlist-badge hotlist-svs";
    case "SLR":
      return "badge hotlist-badge hotlist-slr";
    default:
      return "badge hotlist-badge";
  }
}

function applyPlateSuppressionLocally(plate, reason, suppressedUntilUtc) {
  for (const event of events) {
    if (event.plate !== plate || !event.hotlist_hit) {
      continue;
    }
    event.hotlist_alert = false;
    event.hotlist_alert_reason = reason;
    event.hotlist_alert_cooldown_until_utc = suppressedUntilUtc;
    event._pinUntilMs = 0;
  }
}

async function acknowledgePlate(plate) {
  if (!plate || acknowledgingPlates.has(plate)) {
    return;
  }

  acknowledgingPlates.add(plate);
  render();

  try {
    const response = await fetch("/api/hotlist/acknowledge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plate, seconds: 30 * 60 }),
    });

    if (!response.ok) {
      throw new Error(`acknowledge failed: ${response.status}`);
    }

    const payload = await response.json();
    applyPlateSuppressionLocally(payload.plate, payload.reason, payload.suppressed_until_utc);
  } catch (error) {
    console.error(error);
  } finally {
    acknowledgingPlates.delete(plate);
    render();
  }
}

function createActiveAlertCard(event) {
  const card = document.createElement("div");
  card.className = "active-alert-card";

  const imagePath =
    event.annotated_frame_path ||
    event.full_frame_path ||
    event.plate_crop_path;

  const main = document.createElement("div");
  main.className = "active-alert-main";

  const topLine = document.createElement("div");
  topLine.className = "active-alert-topline";

  const plate = document.createElement("div");
  plate.className = "active-alert-plate";
  plate.textContent = event.plate;
  topLine.appendChild(plate);
  topLine.appendChild(createBadge(event.source, `source-badge ${sourceBadgeClass(event.source)}`));
  topLine.appendChild(createBadge(event.hotlist_highest_type || "HIT", hotlistBadgeClass(event.hotlist_highest_type)));

  const detail = document.createElement("div");
  detail.className = "active-alert-detail";
  detail.textContent = `${event.hotlist_highest_label || "Hotlist Hit"} | ${event.status} | ${formatTime(event.timestamp_utc)}`;

  const vehicleText = vehicleAttributesLabelText(event);
  let vehicleLine = null;
  if (vehicleText) {
    vehicleLine = document.createElement("div");
    vehicleLine.className = "active-alert-detail active-alert-vehicle";
    vehicleLine.textContent = vehicleText;
  }

  const footer = document.createElement("div");
  footer.className = "active-alert-footer";
  footer.textContent = `Pinned ${pinnedSecondsRemaining(event)}s | ${event.source_label || event.source} | Acknowledge to quiet 30m`;

  main.appendChild(topLine);
  main.appendChild(detail);
  if (vehicleLine) {
    main.appendChild(vehicleLine);
  }
  const gpsText = formatGps(event);
  if (gpsText) {
    const gpsLine = document.createElement("div");
    gpsLine.className = "active-alert-detail";
    gpsLine.textContent = `GPS: ${gpsText}`;
    main.appendChild(gpsLine);
  }
  main.appendChild(footer);

  card.appendChild(main);

  const dismissBtn = document.createElement("button");
  dismissBtn.className = "active-alert-dismiss";
  dismissBtn.textContent = acknowledgingPlates.has(event.plate) ? "Acking..." : "Acknowledge 30m";
  dismissBtn.disabled = acknowledgingPlates.has(event.plate);
  dismissBtn.onclick = (clickEvent) => {
    clickEvent.stopPropagation();
    acknowledgePlate(event.plate);
  };

  card.appendChild(dismissBtn);

  if (imagePath) {
    card.classList.add("clickable");
    card.onclick = () => {
      openImageViewer(buildImageUrl(imagePath), `${event.plate} ${event.status}`);
    };
  }

  return card;
}

function vehicleAttributesText(event) {
  const color = (event.vehicle_color || "").trim() || (((event.vehicle_type || "").trim() || (event.vehicle_make || "").trim()) ? "Unknown" : "");
  const parts = [event.vehicle_make, event.vehicle_type, color]
    .map((value) => (value || "").trim())
    .filter(Boolean);
  return parts.join(" | ");
}

function vehicleAttributesLabelText(event) {
  const vehicleText = vehicleAttributesText(event);
  return vehicleText ? `Vehicle: ${vehicleText}` : "";
}

function renderActiveAlerts() {
  activeAlerts.innerHTML = "";

  const pinnedAlerts = events.filter((event) => isPinnedHotlist(event));
  activeAlertsSection.classList.toggle("hidden", pinnedAlerts.length === 0);
  activeAlertsSummary.textContent = pinnedAlerts.length === 1 ? "1 alert" : `${pinnedAlerts.length} alerts`;

  for (const event of pinnedAlerts) {
    activeAlerts.appendChild(createActiveAlertCard(event));
  }
}

function renderStatus(status) {
  lastKnownAlprAlive = status.alpr_process_alive;
  updateAlprStatus(status.alpr_process_alive);

  networkAccessList.innerHTML = "";
  const networkAccess = Array.isArray(status.network_access) ? status.network_access : [];
  for (const item of networkAccess) {
    const link = document.createElement("a");
    link.className = `network-access-chip network-${String(item.medium || "network").toLowerCase()}`;
    link.href = item.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    const medium = String(item.medium || "network").toUpperCase();
    const interfaceText = item.interface ? ` ${item.interface}` : "";
    link.textContent = `${medium}${interfaceText} ${item.address}`;
    link.title = item.note ? `${item.note} | ${item.url}` : item.url;
    networkAccessList.appendChild(link);
  }

  const gpsStatus = status && status.gps_status ? status.gps_status : {};
  const gpsConnected = Boolean(gpsStatus.connected);
  const gpsFixValid = Boolean(gpsStatus.fix_valid);
  gpsStatusChip.className = "gps-status-chip";
  if (gpsConnected && gpsFixValid) {
    gpsStatusChip.classList.add("gps-status-connected");
    gpsStatusChip.textContent = "FIX";
  } else if (gpsConnected) {
    gpsStatusChip.classList.add("gps-status-warning");
    gpsStatusChip.textContent = "NO FIX";
  } else {
    gpsStatusChip.classList.add("gps-status-disconnected");
    gpsStatusChip.textContent = "DISCONNECTED";
  }

  const gpsParts = [];
  if (gpsStatus.source) {
    gpsParts.push(String(gpsStatus.source).toUpperCase());
  }
  if (gpsStatus.endpoint) {
    gpsParts.push(String(gpsStatus.endpoint));
  }
  if (gpsFixValid) {
    gpsParts.push(`${Number(gpsStatus.latitude).toFixed(6)}, ${Number(gpsStatus.longitude).toFixed(6)}`);
  }
  if (gpsStatus.last_received_utc) {
    gpsParts.push(`Last ${formatDateTime(gpsStatus.last_received_utc)}`);
  }
  gpsStatusDetail.textContent = gpsParts.join(" | ") || "MiFi GPS unavailable";

  cameraStatusList.innerHTML = "";
  const sourceHealth = Array.isArray(status.source_health) ? status.source_health : [];
  for (const source of sourceHealth) {
    const item = document.createElement("div");
    item.className = `camera-pill ${source.available ? "camera-online" : "camera-offline"}`;
    const lastSeen = formatTime(source.last_seen_utc);
    item.title = `${source.source_label || source.source || "Camera"} | ${source.available ? "Online" : "Offline"} | Last ${lastSeen}`;

    const dot = document.createElement("span");
    dot.className = "camera-pill-dot";

    const label = document.createElement("span");
    label.className = "camera-pill-label";
    label.textContent = source.source || "?";

    item.appendChild(dot);
    item.appendChild(label);
    cameraStatusList.appendChild(item);
  }
}

function createCard(event) {
  const card = document.createElement("div");
  card.className = "card";
  if (event.hotlist_hit) {
    card.classList.add("hotlist-hit");
  }
  if (isPinnedHotlist(event)) {
    card.classList.add("hotlist-pinned");
  }

  const imagePath =
    event.annotated_frame_path ||
    event.full_frame_path ||
    event.plate_crop_path;

  if (imagePath) {
    const img = document.createElement("img");
    img.src = buildImageUrl(imagePath);
    img.alt = `${event.plate} ${event.status}`;
    card.appendChild(img);
  } else {
    const placeholder = document.createElement("div");
    placeholder.className = "thumb-placeholder";
    placeholder.textContent = event.source;
    card.appendChild(placeholder);
  }

  const body = document.createElement("div");
  body.className = "card-body";

  const plate = document.createElement("div");
  plate.className = "plate";
  plate.textContent = event.plate;

  const meta = document.createElement("div");
  meta.className = "meta";

  const primaryLine = document.createElement("div");
  primaryLine.className = "meta-line primary-line";
  primaryLine.appendChild(createBadge(event.status, `badge ${event.status.toLowerCase()}`));
  primaryLine.appendChild(createBadge(event.source, `source-badge ${sourceBadgeClass(event.source)}`));
  if (event.hotlist_hit) {
    primaryLine.appendChild(createBadge(event.hotlist_highest_type || "HIT", hotlistBadgeClass(event.hotlist_highest_type)));
  }

  const confidence = document.createElement("span");
  confidence.className = "meta-inline";
  confidence.textContent = `${event.confidence}%`;
  primaryLine.appendChild(confidence);

  const secondaryLine = document.createElement("div");
  secondaryLine.className = "meta-line secondary-line";
  secondaryLine.textContent = `${formatTime(event.timestamp_utc)} | Track ${event.track_id_valid ? event.track_id : "-"}`;

  meta.appendChild(primaryLine);
  meta.appendChild(secondaryLine);

  const vehicleText = vehicleAttributesLabelText(event);
  if (vehicleText) {
    const vehicleLine = document.createElement("div");
    vehicleLine.className = "meta-line vehicle-line";
    vehicleLine.textContent = vehicleText;
    meta.appendChild(vehicleLine);
  }

  const gpsText = formatGps(event);
  if (gpsText) {
    const gpsLine = document.createElement("div");
    gpsLine.className = "meta-line vehicle-line";
    gpsLine.textContent = `GPS: ${gpsText}`;
    meta.appendChild(gpsLine);
  }

  const hotlistText = hotlistSummary(event);
  if (hotlistText) {
    const hotlistLine = document.createElement("div");
    hotlistLine.className = "meta-line hotlist-line";
    hotlistLine.textContent = hotlistText;
    meta.appendChild(hotlistLine);
  }

  body.appendChild(plate);
  body.appendChild(meta);

  card.appendChild(body);

  card.onclick = () => {
    if (imagePath) {
      const itemIndex = viewerItems.findIndex((item) => item.eventKey === eventKey(event));
      if (itemIndex >= 0) {
        openImageViewerAt(itemIndex);
      } else {
        openImageViewer(buildImageUrl(imagePath), `${event.plate} ${event.status}`);
      }
    }
  };

  return card;
}

function applyFilters(event) {
  const search = searchInput.value.toLowerCase();
  const source = sourceFilter.value;
  const status = statusFilter.value;

  if (search && !event.plate.toLowerCase().includes(search)) return false;
  if (source && event.source !== source) return false;
  if (status && event.status !== status) return false;
  if (hotlistOnly && !event.hotlist_hit) return false;

  return true;
}

function render() {
  renderActiveAlerts();
  feed.innerHTML = "";

  const filtered = events.filter(applyFilters);
  const ordered = filtered;

  viewerItems = ordered
    .map((event) => {
      const imagePath = event.annotated_frame_path || event.full_frame_path || event.plate_crop_path;
      if (!imagePath) {
        return null;
      }
      return {
        eventKey: eventKey(event),
        url: buildImageUrl(imagePath),
        alt: `${event.plate} ${event.status}`,
      };
    })
    .filter(Boolean);

  for (const event of ordered) {
    feed.appendChild(createCard(event));
  }
}

function eventSignature(event) {
  if (!event) {
    return "";
  }

  const normalized = { ...event };
  delete normalized._pinUntilMs;
  delete normalized.audio_cue;
  return JSON.stringify(normalized);
}

function addEvent(event, options = {}) {
  const playAudio = options.playAudio !== false;
  const renderNow = options.renderNow !== false;
  const followNow = options.followNow !== false;
  const key = eventKey(event);
  const existingIndex = events.findIndex((item) => eventKey(item) === key);
  const previousEvent = existingIndex >= 0 ? events[existingIndex] : null;

  if (previousEvent && eventSignature(previousEvent) === eventSignature(event)) {
    return false;
  }

  applyEventRuntimeState(event, true);
  if (existingIndex >= 0) {
    events.splice(existingIndex, 1);
  }

  eventIds.add(key);
  events.unshift(event);
  if (events.length > 200) {
    const removed = events.pop();
    if (removed) {
      eventIds.delete(eventKey(removed));
    }
  }

  if (playAudio && event.audio_cue === "hotlist") {
    playHotlistTone().catch(() => {});
  } else if (playAudio && event.audio_cue === "locked") {
    playLockedTone().catch(() => {});
  }

  if (paused) {
    bufferedWhilePaused += 1;
    pausedCount.textContent = String(bufferedWhilePaused);
    syncPauseBadge();
    return true;
  }

  if (renderNow) {
    render();
  }
  if (followNow && followNew) {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return true;
}

function scheduleReconnect() {
  if (reconnectTimer) {
    return;
  }
  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = null;
    connectWS();
  }, 2000);
}

function connectWS() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  if (!liveFeedHealthy) {
    setOfflineBanner(true, "Live feed connecting...");
  }
  socket = new WebSocket(`${protocol}://${location.host}/ws/live`);

  socket.onopen = () => {
    markLiveFeedHealthy();
  };

  socket.onmessage = (msg) => {
    const event = JSON.parse(msg.data);
    addEvent(event);
  };

  socket.onerror = () => {
    if (!liveFeedHealthy) {
      markLiveFeedProblem("Live feed offline. Reconnecting...");
    }
  };

  socket.onclose = () => {
    if (!liveFeedHealthy) {
      markLiveFeedProblem("Live feed offline. Reconnecting...");
    }
    scheduleReconnect();
  };
}

function loadInitialEvents() {
  fetch("/api/live-events", { cache: "no-store" })
    .then(r => r.json())
    .then(data => {
      markLiveFeedHealthy();
      events = [];
      eventIds = new Set();
      for (const event of data) {
        applyEventRuntimeState(event, false);
        const key = eventKey(event);
        if (!eventIds.has(key)) {
          eventIds.add(key);
          events.push(event);
        }
      }
      bufferedWhilePaused = 0;
      syncPauseBadge();
      render();
    })
    .catch(() => {
      markLiveFeedProblem("Live feed offline. Reconnecting...");
    });
}

function refreshLiveEvents() {
  if (liveEventsRefreshInFlight) {
    return;
  }

  liveEventsRefreshInFlight = true;
  fetch("/api/live-events?limit=100", { cache: "no-store" })
    .then(r => r.json())
    .then(data => {
      markLiveFeedHealthy();
      let changed = false;
      for (const event of [...data].reverse()) {
        const updated = addEvent(event, {
          renderNow: false,
          followNow: false,
        });
        changed = changed || updated;
      }

      if (changed && !paused) {
        render();
      }
    })
    .catch(() => {
      markLiveFeedProblem("Live feed offline. Reconnecting...");
    })
    .finally(() => {
      liveEventsRefreshInFlight = false;
    });
}

function refreshStatus() {
  fetch("/api/status")
    .then(r => r.json())
    .then(renderStatus)
    .catch(() => {
      liveProblem = true;
      updateAlprStatus(false);
    });
}

connectWS();
loadInitialEvents();
refreshStatus();
refreshLiveEvents();
window.setInterval(refreshStatus, 5000);
window.setInterval(refreshLiveEvents, 3000);
syncPauseBadge();
syncAudioBanner();

searchInput.oninput = render;
sourceFilter.onchange = render;
statusFilter.onchange = render;
window.setInterval(render, 1000);