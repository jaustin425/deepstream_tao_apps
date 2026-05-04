'use strict';

// ── state ──────────────────────────────────────────────────────────────────
let currentOffset = 0;
let currentTotal  = 0;
let currentLimit  = 50;

// ── DOM refs ──────────────────────────────────────────────────────────────
const fPlate       = document.getElementById('fPlate');
const fPlatePrefix = document.getElementById('fPlatePrefix');
const fMake        = document.getElementById('fMake');
const fType        = document.getElementById('fType');
const fColor       = document.getElementById('fColor');
const fSource      = document.getElementById('fSource');
const fStatus      = document.getElementById('fStatus');
const fAlert       = document.getElementById('fAlert');
const fAlertType   = document.getElementById('fAlertType');
const fStart       = document.getElementById('fStart');
const fEnd         = document.getElementById('fEnd');
const fLimit       = document.getElementById('fLimit');
const searchBtn    = document.getElementById('searchBtn');
const clearBtn     = document.getElementById('clearBtn');

const resultsCount    = document.getElementById('resultsCount');
const pagination      = document.getElementById('pagination');
const paginationBottom = document.getElementById('paginationBottom');
const errorMsg        = document.getElementById('errorMsg');
const loading         = document.getElementById('loading');
const resultsBody     = document.getElementById('resultsBody');
const noResults       = document.getElementById('noResults');

const dbRowCount  = document.getElementById('dbRowCount');
const dbLastEvent = document.getElementById('dbLastEvent');
const dbSize      = document.getElementById('dbSize');
const dbPath      = document.getElementById('dbPath');
const selfTestBtn    = document.getElementById('selfTestBtn');
const selfTestResult = document.getElementById('selfTestResult');

const imageModal   = document.getElementById('imageModal');
const modalBackdrop = document.getElementById('modalBackdrop');
const modalClose   = document.getElementById('modalClose');
const modalImg     = document.getElementById('modalImg');
const modalTitle   = document.getElementById('modalTitle');
const modalCaption = document.getElementById('modalCaption');

// ── DB status ─────────────────────────────────────────────────────────────
async function loadDbStatus() {
  try {
    const res  = await fetch('/api/events/db-status');
    const data = await res.json();
    dbRowCount.textContent  = data.row_count != null ? `${data.row_count.toLocaleString()} rows` : '— rows';
    dbLastEvent.textContent = data.last_event_timestamp_utc
      ? `last event ${fmtTimestamp(data.last_event_timestamp_utc)}`
      : 'no events yet';
    dbSize.textContent = data.db_size_mb != null ? `${data.db_size_mb} MB` : '—';
    dbPath.textContent = data.db_path || '';
  } catch (_) {
    dbRowCount.textContent = 'DB status unavailable';
  }
}

selfTestBtn.addEventListener('click', async () => {
  selfTestBtn.disabled = true;
  selfTestBtn.textContent = 'Running…';
  selfTestResult.className = 'selftest-result hidden';
  try {
    const res  = await fetch('/api/admin/db-selftest', { method: 'POST' });
    const data = await res.json();
    const pass = data.status === 'ok';
    selfTestResult.textContent = pass ? '✓ Pass' : '✗ Fail';
    selfTestResult.className   = `selftest-result ${pass ? 'pass' : 'fail'}`;
    if (!pass) {
      const failed = (data.steps || []).filter(s => !s.ok).map(s => `${s.step}: ${s.error || '?'}`).join('; ');
      selfTestResult.title = failed;
    }
  } catch (err) {
    selfTestResult.textContent = '✗ Error';
    selfTestResult.className   = 'selftest-result fail';
    selfTestResult.title       = String(err);
  } finally {
    selfTestBtn.disabled = false;
    selfTestBtn.textContent = 'Run Self-Test';
  }
});

// ── search ────────────────────────────────────────────────────────────────
function buildParams(offset) {
  const p = new URLSearchParams();
  const plate = fPlate.value.trim();
  if (plate) {
    p.set('plate', plate);
    if (fPlatePrefix.checked) p.set('plate_prefix', 'true');
  }
  if (fMake.value.trim())   p.set('vehicle_make',  fMake.value.trim());
  if (fType.value.trim())   p.set('vehicle_type',  fType.value.trim());
  if (fColor.value.trim())  p.set('vehicle_color', fColor.value.trim());
  if (fSource.value)        p.set('source',  fSource.value);
  if (fStatus.value)        p.set('status',  fStatus.value);
  const alertVal = fAlert.value;
  if (alertVal === 'true')  p.set('hotlist_alert', 'true');
  else if (alertVal === 'hit')  p.set('hotlist_hit', 'true');
  else if (alertVal === 'none') { p.set('hotlist_hit', 'false'); }
  if (fAlertType.value)     p.set('hotlist_type', fAlertType.value);
  if (fStart.value)         p.set('start_utc', toIso(fStart.value));
  if (fEnd.value)           p.set('end_utc',   toIso(fEnd.value));
  const lim = parseInt(fLimit.value, 10) || 50;
  p.set('limit',  String(lim));
  p.set('offset', String(offset));
  return p;
}

async function runSearch(offset) {
  currentOffset = offset;
  currentLimit  = parseInt(fLimit.value, 10) || 50;

  loading.classList.remove('hidden');
  errorMsg.classList.add('hidden');
  noResults.classList.add('hidden');
  resultsBody.innerHTML = '';
  resultsCount.textContent = '';
  renderPagination(0, 0);

  const params = buildParams(offset);
  try {
    const res = await fetch(`/api/events/search?${params}`);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    currentTotal = data.total;
    renderResults(data.items);
    renderPagination(data.total, offset);
    resultsCount.textContent = `${data.total.toLocaleString()} result${data.total === 1 ? '' : 's'}`;
    if (data.items.length === 0) noResults.classList.remove('hidden');
  } catch (err) {
    errorMsg.textContent = `Search error: ${err.message}`;
    errorMsg.classList.remove('hidden');
  } finally {
    loading.classList.add('hidden');
  }
}

// ── render rows ───────────────────────────────────────────────────────────
function renderResults(items) {
  resultsBody.innerHTML = '';
  for (const ev of items) {
    const tr = document.createElement('tr');
    if (ev.hotlist_alert) tr.classList.add('is-alert');

    const ts    = fmtTimestamp(ev.timestamp_utc);
    const make  = [ev.vehicle_make, ev.vehicle_type, ev.vehicle_color].filter(Boolean).join(' / ');
    const srcBadge   = `<span class="badge badge-src">${esc(ev.source)}</span>`;
    const statBadge  = ev.status === 'LOCKED'
      ? `<span class="badge badge-locked">LOCKED</span>`
      : `<span class="badge badge-confirmed">CONF</span>`;
    const alertBadge = ev.hotlist_alert
      ? `<span class="badge badge-alert">${esc(ev.hotlist_type || 'ALERT')}</span>`
      : ev.hotlist_hit
        ? `<span class="badge badge-hit">HIT</span>`
        : `<span style="color:#444">—</span>`;
    const gpsDot = `<span class="gps-dot ${ev.gps_fix_valid ? 'has-fix' : ''}" title="${ev.gps_fix_valid ? `${ev.gps_latitude?.toFixed(5)}, ${ev.gps_longitude?.toFixed(5)}` : 'No fix'}"></span>`;
    const imgLinks = buildImgLinks(ev);

    tr.innerHTML = `
      <td>${esc(ts)}</td>
      <td class="plate-cell">${esc(ev.plate)}</td>
      <td>${esc(String(ev.confidence))}%</td>
      <td>${srcBadge}</td>
      <td>${statBadge}</td>
      <td class="vehicle-detail">${make ? esc(make) : '<span style="color:#444">—</span>'}</td>
      <td>${alertBadge}</td>
      <td>${gpsDot}</td>
      <td>${imgLinks}</td>
    `;
    resultsBody.appendChild(tr);
  }
}

function buildImgLinks(ev) {
  const links = [];
  if (ev.plate_crop_path) {
    links.push(imgLinkHtml('/evidence/' + ev.plate_crop_path, 'Plate', ev.plate));
  }
  if (ev.vehicle_crop_path) {
    links.push(imgLinkHtml('/evidence/' + ev.vehicle_crop_path, 'Vehicle', ev.plate));
  }
  if (ev.annotated_frame_path) {
    links.push(imgLinkHtml('/evidence/' + ev.annotated_frame_path, 'Frame', ev.plate));
  }
  return links.join('') || '<span style="color:#444">—</span>';
}

function imgLinkHtml(url, label, plate) {
  const safeUrl   = esc(url);
  const safeLabel = esc(label);
  const safePlate = esc(plate);
  return `<span class="img-link" data-img-url="${safeUrl}" data-img-label="${safeLabel}" data-img-plate="${safePlate}">${safeLabel}</span>`;
}

// Delegated click for image links
document.getElementById('resultsBody').addEventListener('click', (e) => {
  const link = e.target.closest('.img-link');
  if (!link) return;
  openModal(link.dataset.imgUrl, link.dataset.imgLabel, link.dataset.imgPlate);
});

// ── pagination ────────────────────────────────────────────────────────────
function renderPagination(total, offset) {
  const html = buildPaginationHtml(total, offset);
  pagination.innerHTML = html;
  paginationBottom.innerHTML = html;
  pagination.querySelectorAll('.page-btn').forEach(bindPageBtn);
  paginationBottom.querySelectorAll('.page-btn').forEach(bindPageBtn);
}

function buildPaginationHtml(total, offset) {
  if (total === 0) return '';
  const limit     = currentLimit;
  const pageCount = Math.ceil(total / limit);
  const current   = Math.floor(offset / limit);
  if (pageCount <= 1) return '';

  const parts = [];
  parts.push(makePgBtn('‹ Prev', current > 0 ? (current - 1) * limit : null));

  const window = 2;
  for (let i = 0; i < pageCount; i++) {
    if (
      i === 0 ||
      i === pageCount - 1 ||
      (i >= current - window && i <= current + window)
    ) {
      parts.push(makePgBtn(String(i + 1), i * limit, i === current));
    } else if (
      i === current - window - 1 ||
      i === current + window + 1
    ) {
      parts.push('<span style="color:#444;padding:0 4px">…</span>');
    }
  }

  parts.push(makePgBtn('Next ›', current < pageCount - 1 ? (current + 1) * limit : null));
  return parts.join('');
}

function makePgBtn(label, offset, active = false) {
  if (offset === null) {
    return `<button class="page-btn" disabled>${esc(label)}</button>`;
  }
  return `<button class="page-btn${active ? ' active' : ''}" data-offset="${offset}">${esc(label)}</button>`;
}

function bindPageBtn(btn) {
  if (btn.disabled) return;
  btn.addEventListener('click', () => {
    const off = parseInt(btn.dataset.offset, 10);
    runSearch(off);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

// ── modal ─────────────────────────────────────────────────────────────────
function openModal(url, label, plate) {
  modalImg.src     = url;
  modalTitle.textContent   = `${plate} — ${label}`;
  modalCaption.textContent = url;
  imageModal.classList.remove('hidden');
}

function closeModal() {
  imageModal.classList.add('hidden');
  modalImg.src = '';
}

modalClose.addEventListener('click', closeModal);
modalBackdrop.addEventListener('click', closeModal);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

// ── event wiring ──────────────────────────────────────────────────────────
searchBtn.addEventListener('click', () => runSearch(0));
clearBtn.addEventListener('click', () => {
  fPlate.value       = '';
  fPlatePrefix.checked = false;
  fMake.value        = '';
  fType.value        = '';
  fColor.value       = '';
  fSource.value      = '';
  fStatus.value      = '';
  fAlert.value       = '';
  fAlertType.value   = '';
  fStart.value       = '';
  fEnd.value         = '';
  fLimit.value       = '50';
  resultsBody.innerHTML = '';
  resultsCount.textContent = '';
  noResults.classList.add('hidden');
  errorMsg.classList.add('hidden');
  renderPagination(0, 0);
});

[fPlate, fMake, fType, fColor].forEach(el => {
  el.addEventListener('keydown', (e) => { if (e.key === 'Enter') runSearch(0); });
});

// ── helpers ───────────────────────────────────────────────────────────────
function esc(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function toIso(localDatetime) {
  // datetime-local value is YYYY-MM-DDTHH:MM — treat as UTC
  if (!localDatetime) return '';
  return localDatetime.length === 16 ? localDatetime + ':00Z' : localDatetime + 'Z';
}

function fmtTimestamp(ts) {
  if (!ts) return '—';
  try {
    return new Date(ts).toISOString().replace('T', ' ').slice(0, 19);
  } catch (_) {
    return ts;
  }
}

// ── init ──────────────────────────────────────────────────────────────────
loadDbStatus();
runSearch(0);
