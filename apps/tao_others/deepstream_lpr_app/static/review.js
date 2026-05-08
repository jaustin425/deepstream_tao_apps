// review.js

(function () {
  'use strict';

  const QUEUE_SIZE = 50;

  let queue = [];        // array of PersistedPlateRead
  let queueIndex = 0;   // current position in queue
  let reviewed = 0;     // count reviewed this session

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const plateCropImg   = document.getElementById('plateCrop');
  const plateCropEmpty = document.getElementById('plateCropEmpty');
  const vehicleThumb   = document.getElementById('vehicleThumb');
  const plateText      = document.getElementById('plateText');
  const caPatternTag   = document.getElementById('caPattern');
  const metaSource     = document.getElementById('metaSource');
  const metaConf       = document.getElementById('metaConf');
  const metaTime       = document.getElementById('metaTime');
  const correctedPlate = document.getElementById('correctedPlate');
  const plateState     = document.getElementById('plateState');
  const plateType      = document.getElementById('plateType');
  const reviewNotes    = document.getElementById('reviewNotes');
  const btnCorrect     = document.getElementById('btnCorrect');
  const btnIncorrect   = document.getElementById('btnIncorrect');
  const btnSkip        = document.getElementById('btnSkip');
  const btnPrev        = document.getElementById('btnPrev');
  const progressText   = document.getElementById('progressText');
  const progressInner  = document.getElementById('progressInner');
  const statusMsg      = document.getElementById('statusMsg');

  // ── Load queue ────────────────────────────────────────────────────────────
  async function loadQueue() {
    // If a specific uuid was passed in the URL, fetch just that one first
    const params = new URLSearchParams(window.location.search);
    const targetUuid = params.get('uuid');

    try {
      const r = await fetch(`/api/review/queue?limit=${QUEUE_SIZE}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      queue = await r.json();
      queueIndex = 0;

      // If a uuid was requested, move it to the front of the queue
      if (targetUuid) {
        const idx = queue.findIndex(ev => ev.event_uuid === targetUuid);
        if (idx > 0) {
          const [item] = queue.splice(idx, 1);
          queue.unshift(item);
        } else if (idx === -1) {
          // Not in queue (already reviewed or missing) — fetch it directly
          try {
            const r2 = await fetch(`/api/events/${encodeURIComponent(targetUuid)}`);
            if (r2.ok) { const ev = await r2.json(); queue.unshift(ev); }
          } catch (_) {}
        }
      }

      if (queue.length === 0) {
        showStatus('Queue empty — nothing left to review!', 'ok');
        clearDisplay();
      } else {
        renderCurrent();
      }
    } catch (e) {
      showStatus(`Failed to load queue: ${e.message}`, 'err');
    }
  }

  // ── Render current item ───────────────────────────────────────────────────
  function renderCurrent() {
    if (queueIndex < 0 || queueIndex >= queue.length) {
      clearDisplay();
      showStatus('End of queue. Reload to fetch more.', 'ok');
      return;
    }
    const ev = queue[queueIndex];
    hideStatus();

    // Plate text
    plateText.textContent = ev.plate || '—';

    // CA pattern tag
    if (ev.ca_pattern && ev.ca_pattern !== 'CA_UNKNOWN' && ev.ca_pattern !== '') {
      caPatternTag.textContent = ev.ca_pattern.replace('CA_', '').replace('_', ' ');
      caPatternTag.classList.remove('hidden');
    } else {
      caPatternTag.classList.add('hidden');
    }

    // Meta chips
    metaSource.textContent = ev.source || '—';
    metaConf.textContent   = ev.confidence != null ? `conf ${ev.confidence}` : '';
    metaTime.textContent   = ev.timestamp_utc ? ev.timestamp_utc.replace('T', ' ').slice(0, 19) + ' UTC' : '';

    // Plate crop
    if (ev.plate_crop_path) {
      const src = `/evidence/${encodeEvPath(ev.plate_crop_path)}`;
      plateCropImg.src = src;
      plateCropImg.classList.remove('hidden');
      plateCropEmpty.classList.add('hidden');
    } else {
      plateCropImg.classList.add('hidden');
      plateCropEmpty.classList.remove('hidden');
    }

    // Vehicle thumbnail
    if (ev.vehicle_crop_path) {
      const src = `/evidence/${encodeEvPath(ev.vehicle_crop_path)}`;
      vehicleThumb.src = src;
      vehicleThumb.classList.remove('hidden');
    } else {
      vehicleThumb.classList.add('hidden');
    }

    // Form reset
    correctedPlate.value = '';
    plateState.value     = ev.plate_state || '';
    plateType.value      = ev.plate_type  || '';
    reviewNotes.value    = '';

    // Progress
    updateProgress();
  }

  function encodeEvPath(p) {
    // p is like "cases/xxx/frames/yyy.jpg" — encode each segment
    return p.split('/').map(encodeURIComponent).join('/');
  }

  function updateProgress() {
    const total = queue.length;
    const pos   = queueIndex + 1;
    progressText.textContent = `${pos} / ${total}  (${reviewed} reviewed this session)`;
    progressInner.style.width = total > 0 ? `${Math.round((queueIndex / total) * 100)}%` : '0%';
  }

  function clearDisplay() {
    plateText.textContent = '—';
    caPatternTag.classList.add('hidden');
    plateCropImg.classList.add('hidden');
    plateCropEmpty.classList.remove('hidden');
    vehicleThumb.classList.add('hidden');
    metaSource.textContent = '';
    metaConf.textContent   = '';
    metaTime.textContent   = '';
    progressText.textContent = `(${reviewed} reviewed this session)`;
    progressInner.style.width = '0%';
  }

  // ── Submit review ─────────────────────────────────────────────────────────
  async function submitReview(status) {
    if (queueIndex >= queue.length) return;
    const ev = queue[queueIndex];

    const body = {
      review_status: status,
    };
    const corr = correctedPlate.value.trim().toUpperCase();
    if (corr) body.corrected_plate = corr;
    if (plateState.value)  body.plate_state  = plateState.value;
    if (plateType.value)   body.plate_type   = plateType.value;
    if (reviewNotes.value.trim()) body.review_notes = reviewNotes.value.trim();

    try {
      const r = await fetch(`/api/events/${ev.event_uuid}/review`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const txt = await r.text();
        showStatus(`Save failed (${r.status}): ${txt}`, 'err');
        return;
      }
      reviewed++;
      queueIndex++;
      renderCurrent();
    } catch (e) {
      showStatus(`Network error: ${e.message}`, 'err');
    }
  }

  function skip() {
    queueIndex++;
    renderCurrent();
  }

  function prev() {
    if (queueIndex > 0) {
      queueIndex--;
      renderCurrent();
    }
  }

  // ── Keyboard ──────────────────────────────────────────────────────────────
  document.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
    if (e.key === 'c' || e.key === 'C') { e.preventDefault(); submitReview('correct'); }
    if (e.key === 'i' || e.key === 'I') { e.preventDefault(); submitReview('incorrect'); }
    if (e.key === 'ArrowRight') { e.preventDefault(); skip(); }
    if (e.key === 'ArrowLeft')  { e.preventDefault(); prev(); }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && document.activeElement === correctedPlate) {
      e.preventDefault();
      submitReview('correct');
    }
  });

  // ── Button wiring ─────────────────────────────────────────────────────────
  btnCorrect.addEventListener('click',   () => submitReview('correct'));
  btnIncorrect.addEventListener('click', () => submitReview('incorrect'));
  btnSkip.addEventListener('click',      skip);
  btnPrev.addEventListener('click',      prev);

  // ── Status helpers ────────────────────────────────────────────────────────
  function showStatus(msg, type) {
    statusMsg.textContent = msg;
    statusMsg.className   = `status-msg ${type}`;
    statusMsg.classList.remove('hidden');
  }
  function hideStatus() {
    statusMsg.classList.add('hidden');
  }

  // ── Export dataset ────────────────────────────────────────────────────────
  const btnExport    = document.getElementById('btnExport');
  const exportResult = document.getElementById('exportResult');

  if (btnExport) {
    btnExport.addEventListener('click', async () => {
      const body = {
        review_status: document.getElementById('exportStatus').value || 'correct',
        limit: parseInt(document.getElementById('exportLimit').value, 10) || 2000,
      };
      const st = document.getElementById('exportState').value;
      const ty = document.getElementById('exportType').value;
      if (st) body.plate_state = st;
      if (ty) body.plate_type  = ty;

      btnExport.disabled = true;
      btnExport.textContent = 'Exporting…';
      exportResult.className = 'export-result hidden';

      try {
        const r = await fetch('/api/dataset/export', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
        exportResult.textContent =
          `✓ Exported ${data.exported} record${data.exported !== 1 ? 's' : ''}`
          + (data.skipped ? ` (${data.skipped} skipped — missing crop)` : '')
          + `  →  ${data.output_dir}`;
        exportResult.className = 'export-result ok';
      } catch (e) {
        exportResult.textContent = `Export failed: ${e.message}`;
        exportResult.className = 'export-result err';
      } finally {
        btnExport.disabled = false;
        btnExport.textContent = '⇩ Export';
      }
    });
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  loadQueue();
})();
