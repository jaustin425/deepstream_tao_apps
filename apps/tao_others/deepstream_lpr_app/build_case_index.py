#!/usr/bin/env python3
import csv
import html
import json
import sys
from collections import defaultdict
from pathlib import Path


def _safe_str(value) -> str:
    if value is None:
        return ""
    return str(value)


def _html(value) -> str:
    return html.escape(_safe_str(value))


def _load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {exc}") from exc
    return rows


def _sort_rows(rows):
    def _frame_number(row):
        try:
            return int(row.get("frame_number", 0))
        except (TypeError, ValueError):
            return 0

    return sorted(
        rows,
        key=lambda row: (
            _safe_str(row.get("timestamp_utc")),
            _frame_number(row),
            _safe_str(row.get("event_id")),
        ),
    )


def _compact_event_id(event_id: str) -> str:
    event_id = _safe_str(event_id)
    pos = event_id.rfind("_evt_")
    if pos == -1:
        return event_id
    return event_id[pos + 5 :]


def _preview_path(row) -> str:
    return (
        _safe_str(row.get("annotated_frame_path"))
        or _safe_str(row.get("full_frame_path"))
        or _safe_str(row.get("plate_crop_path"))
    )


def _gps_display(row) -> str:
  if row.get("gps_fix_valid") is not True:
    return ""

  latitude = row.get("gps_latitude")
  longitude = row.get("gps_longitude")
  if latitude is None or longitude is None:
    return ""

  parts = [f"{float(latitude):.6f}, {float(longitude):.6f}"]
  altitude = row.get("gps_altitude_m")
  if altitude is not None:
    parts.append(f"{float(altitude):.1f} m")
  return " | ".join(parts)


def _merged_rows(case_dir: Path):
    rows = _load_jsonl(case_dir / "events.jsonl")
    rows.extend(_load_jsonl(case_dir / "debug" / "debug.jsonl"))
    return _sort_rows(rows)


def _summary_counts(rows):
    counts = {
        "total": len(rows),
        "debug": 0,
        "confirmed": 0,
        "locked": 0,
    }
    for row in rows:
        event_type = _safe_str(row.get("event_type", "")).strip().lower()
        if event_type in counts:
            counts[event_type] += 1
    return counts


def _state_rank(event_type: str) -> int:
    event_type = _safe_str(event_type).strip().upper()
    if event_type == "LOCKED":
        return 3
    if event_type == "CONFIRMED":
        return 2
    if event_type == "DEBUG":
        return 1
    return 0


def _timestamp_sort_value(timestamp: str) -> int:
    digits = "".join(ch for ch in _safe_str(timestamp) if ch.isdigit())
    if not digits:
        return 0
    return int(digits)


def _group_rows_by_plate(rows):
    grouped = defaultdict(list)
    for row in rows:
        plate = _safe_str(row.get("plate")).strip() or "(unknown)"
        grouped[plate].append(row)

    for plate_rows in grouped.values():
        plate_rows.sort(
            key=lambda row: (
                _safe_str(row.get("timestamp_utc")),
                int(row.get("frame_number", 0) or 0),
                _safe_str(row.get("event_id")),
            )
        )

    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _summarize_plate_group(plate, rows):
    best_row = max(
        rows,
        key=lambda row: (
            _state_rank(row.get("event_type")),
            int(row.get("confidence", 0) or 0),
        ),
    )

    preferred_track_row = next(
        (
            row
            for row in sorted(
                rows,
                key=lambda row: (
                    _state_rank(row.get("event_type")),
                    int(row.get("confidence", 0) or 0),
                    _timestamp_sort_value(row.get("timestamp_utc")),
                ),
                reverse=True,
            )
            if row.get("track_id_valid") is True
        ),
        None,
    )
    preview_path = _preview_path(best_row)
    gps_row = next((row for row in reversed(rows) if row.get("gps_fix_valid") is True), None)

    return {
        "plate": plate,
        "best_state": _safe_str(best_row.get("event_type")).strip() or "UNKNOWN",
        "best_conf": int(best_row.get("confidence", 0) or 0),
      "track_display": _safe_str(preferred_track_row.get("track_id")) if preferred_track_row else "-",
        "first_seen": _safe_str(rows[0].get("timestamp_utc")),
        "last_seen": _safe_str(rows[-1].get("timestamp_utc")),
        "event_count": len(rows),
        "preview_path": preview_path,
      "gps_display": _gps_display(gps_row or best_row),
    }


def _sorted_plate_groups(rows):
    grouped = _group_rows_by_plate(rows)
    summaries = []
    for plate, plate_rows in grouped.items():
        summary = _summarize_plate_group(plate, plate_rows)
        summaries.append((summary, plate_rows))

    summaries.sort(
        key=lambda item: (
            -_state_rank(item[0]["best_state"]),
            -_timestamp_sort_value(item[0]["last_seen"]),
            -item[0]["best_conf"],
            -item[0]["event_count"],
            item[0]["plate"].lower(),
        )
    )
    return summaries


def _plate_summary_counts(grouped_summaries):
    counts = {
        "total": len(grouped_summaries),
        "locked": 0,
        "confirmed_only": 0,
        "debug_only": 0,
    }
    for summary, _plate_rows in grouped_summaries:
        best_state = _safe_str(summary.get("best_state")).strip().upper()
        if best_state == "LOCKED":
            counts["locked"] += 1
        elif best_state == "CONFIRMED":
            counts["confirmed_only"] += 1
        elif best_state == "DEBUG":
            counts["debug_only"] += 1
    return counts


def write_csv(case_dir: Path, rows):
    out_path = case_dir / "index.csv"
    fieldnames = [
        "event_id",
        "event_type",
        "plate",
        "confidence",
        "track_id",
        "track_id_valid",
        "video_source",
        "timestamp_utc",
        "frame_number",
        "gps_fix_valid",
        "gps_latitude",
        "gps_longitude",
        "gps_altitude_m",
        "gps_speed_knots",
        "gps_timestamp_utc",
        "preview_path",
        "full_frame_path",
        "plate_crop_path",
        "annotated_frame_path",
        "full_frame_sha256",
        "plate_crop_sha256",
        "annotated_frame_sha256",
        "model_version",
        "notes",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key, "") for key in fieldnames}
            output["preview_path"] = _preview_path(row)
            writer.writerow(output)

    return out_path


def write_html(case_dir: Path, rows):
    out_path = case_dir / "index.html"
    event_counts = _summary_counts(rows)
    grouped_summaries = _sorted_plate_groups(rows)
    plate_counts = _plate_summary_counts(grouped_summaries)
    group_blocks = []

    for summary, plate_rows in grouped_summaries:
        plate = summary["plate"]
        badge_class = _safe_str(summary["best_state"]).lower()
        details_open_attr = " open" if badge_class == "locked" else ""
        preview_html = ""
        if summary["preview_path"]:
            preview_html = (
          f'<a href="{_html(summary["preview_path"])}" class="preview-link">'
          f'<img src="{_html(summary["preview_path"])}" class="thumb" loading="lazy" alt="{_html(plate)}">'
          f'<div class="preview-label">Open Image</div></a>'
            )

        event_rows = []
        for row in plate_rows:
            event_id = _html(row.get("event_id"))
            event_type_raw = _safe_str(row.get("event_type", "")).strip()
            event_type = _html(event_type_raw)
            event_badge = event_type_raw.lower()
            conf = _html(row.get("confidence"))
            track_display = _html(row.get("track_id")) if row.get("track_id_valid", False) else "&mdash;"
            timestamp = _html(row.get("timestamp_utc"))
            frame_number = _html(row.get("frame_number"))
            gps_display = _html(_gps_display(row)) or "&mdash;"

            full_frame_path = _safe_str(row.get("full_frame_path", ""))
            plate_crop_path = _safe_str(row.get("plate_crop_path", ""))
            annotated_frame_path = _safe_str(row.get("annotated_frame_path", ""))

            links = []
            if full_frame_path:
                links.append(f'<a href="{_html(full_frame_path)}">full</a>')
            if plate_crop_path:
                links.append(f'<a href="{_html(plate_crop_path)}">plate</a>')
            if annotated_frame_path:
                links.append(f'<a href="{_html(annotated_frame_path)}">annotated</a>')

            event_rows.append(
                f"""
                <tr>
                  <td>{event_id}</td>
                  <td><span class=\"badge {event_badge}\">{event_type}</span></td>
                  <td>{conf}</td>
                  <td>{track_display}</td>
                  <td>{timestamp}</td>
                  <td>{frame_number}</td>
                  <td>{gps_display}</td>
                  <td>{' | '.join(links) if links else '&mdash;'}</td>
                </tr>
                """
            )

        group_blocks.append(
            f"""
            <details class=\"plate-group\" data-plate=\"{_html(plate.lower())}\" data-best-type=\"{_html(badge_class)}\"{details_open_attr}>
              <summary>
                <div class=\"group-summary\">
                  <div class=\"group-main\">
                    <strong class=\"plate-title\">{_html(plate)}</strong>
                    <span class=\"badge {badge_class}\">{_html(summary['best_state'])}</span>
                    <span>Best Conf: {summary['best_conf']}</span>
                    <span>Track: {_html(summary['track_display'])}</span>
                    <span>Events: {summary['event_count']}</span>
                    <span>First: {_html(summary['first_seen'])}</span>
                    <span>Last: {_html(summary['last_seen'])}</span>
                    <span>GPS: {_html(summary['gps_display']) or '&mdash;'}</span>
                  </div>
                  <div class=\"group-preview\">{preview_html}</div>
                </div>
              </summary>
              <table class=\"nested-table\">
                <thead>
                  <tr>
                    <th>Event</th>
                    <th>Type</th>
                    <th>Conf</th>
                    <th>Track</th>
                    <th>UTC</th>
                    <th>Frame</th>
                    <th>GPS</th>
                    <th>Files</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(event_rows)}
                </tbody>
              </table>
            </details>
            """
        )

    page = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>Case Review - {_html(case_dir.name)}</title>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<style>
:root {{
  color-scheme: dark;
}}
body {{
  font-family: Arial, sans-serif;
  margin: 20px;
  background: #111;
  color: #eee;
}}
h1 {{
  margin: 0 0 8px 0;
}}
.meta {{
  color: #bbb;
  margin-bottom: 16px;
}}
.toolbar {{
  display: flex;
  gap: 16px;
  align-items: end;
  flex-wrap: wrap;
  margin-bottom: 16px;
}}
.summary {{
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}}
.summary-group {{
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}}
.summary-group-title {{
  width: 100%;
  color: #bbb;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.summary-card {{
  min-width: 120px;
  padding: 10px 12px;
  border: 1px solid #333;
  background: #1a1a1a;
  border-radius: 10px;
}}
.summary-filter-card {{
  appearance: none;
  text-align: left;
  color: inherit;
  cursor: pointer;
}}
.summary-filter-card:hover {{
  background: #222;
}}
.summary-filter-card.is-active {{
  border-color: #8ecbff;
  background: #163042;
  color: #d7ecff;
}}
.summary-filter-card:focus-visible {{
  outline: 2px solid #8ecbff;
  outline-offset: 2px;
}}
.summary-card .label {{
  display: block;
  color: #bbb;
  font-size: 12px;
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.summary-card .value {{
  font-size: 22px;
  font-weight: bold;
}}
.filter-box label {{
  display: block;
  color: #bbb;
  font-size: 12px;
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.filter-box input {{
  min-width: 220px;
  padding: 10px 12px;
  border: 1px solid #333;
  border-radius: 8px;
  background: #171717;
  color: #eee;
}}
.type-filters {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}}
.filter-chip {{
  padding: 8px 12px;
  border: 1px solid #3a3a3a;
  border-radius: 999px;
  background: #1a1a1a;
  color: #ddd;
  cursor: pointer;
  font-size: 13px;
}}
.filter-chip.is-active {{
  border-color: #8ecbff;
  background: #163042;
  color: #d7ecff;
}}
.filter-reset {{
  padding: 8px 12px;
  border: 1px solid #555;
  border-radius: 8px;
  background: #242424;
  color: #eee;
  cursor: pointer;
  font-size: 13px;
}}
.filter-reset:hover {{
  background: #2e2e2e;
}}
.group-controls {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}}
.action-controls {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}}
.filter-status {{
  min-width: 140px;
  padding: 10px 12px;
  border: 1px solid #333;
  border-radius: 8px;
  background: #171717;
  color: #eee;
  font-weight: bold;
  cursor: pointer;
  text-align: left;
}}
.filter-status:hover {{
  background: #212121;
}}
.filter-status:focus-visible {{
  outline: 2px solid #8ecbff;
  outline-offset: 2px;
}}
.thumb {{
  max-width: 240px;
  max-height: 150px;
  border: 1px solid #444;
}}
a {{
  color: #8ecbff;
  text-decoration: none;
}}
a:hover {{
  text-decoration: underline;
}}
.badge {{
  display: inline-block;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: bold;
}}
.badge.confirmed {{
  background: #7a6a00;
  color: #fff3a0;
}}
.badge.locked {{
  background: #0c5f1e;
  color: #a8ffb8;
}}
.badge.debug {{
  background: #7a1212;
  color: #ffb3b3;
}}
.plate-group {{
  margin-bottom: 14px;
  border: 1px solid #333;
  border-radius: 12px;
  background: #1a1a1a;
  overflow: hidden;
}}
.plate-group summary {{
  list-style: none;
  cursor: pointer;
  padding: 12px;
  background: #181818;
}}
.plate-group[open] summary {{
  background: #1d1d1d;
}}
.plate-group summary::-webkit-details-marker {{
  display: none;
}}
.group-summary {{
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  flex-wrap: wrap;
}}
.group-main {{
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  align-items: center;
  cursor: pointer;
}}
.plate-title {{
  font-size: 20px;
}}
.group-preview {{
  border-radius: 8px;
}}
.group-preview:hover {{
  outline: 2px solid #8ecbff;
  border-radius: 8px;
}}
.group-preview .preview-link {{
  display: block;
  cursor: pointer;
}}
.group-preview .thumb {{
  max-width: 200px;
  max-height: 120px;
}}
.preview-label {{
  margin-top: 6px;
  font-size: 12px;
  color: #d7ecff;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.nested-table {{
  width: 100%;
  border-collapse: collapse;
  background: #141414;
}}
.nested-table th,
.nested-table td {{
  border: 1px solid #333;
  padding: 8px;
  text-align: left;
  vertical-align: top;
}}
.nested-table th {{
  background: #202020;
}}
</style>
</head>
<body>
<h1>Case Review: {_html(case_dir.name)}</h1>
<div class=\"meta\">{len(rows)} events</div>
<div class=\"toolbar\">
  <div class="summary">
    <div class="summary-group">
      <div class="summary-group-title">Event Counts</div>
      <div class="summary-card"><span class="label">Total Events</span><span class="value">{event_counts['total']}</span></div>
      <div class="summary-card"><span class="label">Debug</span><span class="value">{event_counts['debug']}</span></div>
      <div class="summary-card"><span class="label">Confirmed</span><span class="value">{event_counts['confirmed']}</span></div>
      <div class="summary-card"><span class="label">Locked</span><span class="value">{event_counts['locked']}</span></div>
    </div>
    <div class="summary-group">
      <div class="summary-group-title">Plate Counts</div>
      <button type="button" class="summary-card summary-filter-card" data-summary-filter="all"><span class="label">Total Plates</span><span class="value">{plate_counts['total']}</span></button>
      <button type="button" class="summary-card summary-filter-card" data-summary-filter="locked"><span class="label">Locked Plates</span><span class="value">{plate_counts['locked']}</span></button>
      <button type="button" class="summary-card summary-filter-card" data-summary-filter="confirmed"><span class="label">Confirmed-Only</span><span class="value">{plate_counts['confirmed_only']}</span></button>
      <button type="button" class="summary-card summary-filter-card" data-summary-filter="debug"><span class="label">Debug-Only</span><span class="value">{plate_counts['debug_only']}</span></button>
    </div>
  </div>
  <div class="filter-box">
    <label for="plateFilter">Filter Plate</label>
    <input id="plateFilter" type="search" placeholder="Type plate text" autocomplete="off">
  </div>
  <div class="filter-box">
    <label>Filter Plate Status</label>
    <div class="type-filters">
      <button type="button" class="filter-chip" data-filter-type="debug">Debug-Only</button>
      <button type="button" class="filter-chip" data-filter-type="confirmed">Confirmed-Only</button>
      <button type="button" class="filter-chip" data-filter-type="locked">Locked Plates</button>
    </div>
  </div>
  <div class="filter-box">
    <label>Groups</label>
    <div class="group-controls">
      <button type="button" id="openLockedOnly" class="filter-reset">Open Locked Only</button>
      <button type="button" id="expandVisible" class="filter-reset">Expand Visible</button>
      <button type="button" id="collapseVisible" class="filter-reset">Collapse Visible</button>
    </div>
  </div>
  <div class="filter-box">
    <label>Visible Groups</label>
    <button type="button" id="visibleGroupCount" class="filter-status">{plate_counts['total']} / {plate_counts['total']}</button>
  </div>
  <div class="filter-box">
    <label>&nbsp;</label>
    <div class="action-controls">
      <button type="button" id="copyFilteredLink" class="filter-reset">Copy Filtered Link</button>
      <button type="button" id="clearFilters" class="filter-reset">Clear Filters</button>
    </div>
  </div>
</div>
{''.join(group_blocks)}
<script>
const plateFilter = document.getElementById('plateFilter');
const groups = Array.from(document.querySelectorAll('.plate-group'));
const statusButtons = Array.from(document.querySelectorAll('.filter-chip'));
const summaryFilterButtons = Array.from(document.querySelectorAll('.summary-filter-card'));
const copyFilteredLinkButton = document.getElementById('copyFilteredLink');
const clearFiltersButton = document.getElementById('clearFilters');
const openLockedOnlyButton = document.getElementById('openLockedOnly');
const expandVisibleButton = document.getElementById('expandVisible');
const collapseVisibleButton = document.getElementById('collapseVisible');
const visibleGroupCount = document.getElementById('visibleGroupCount');
const activeStatuses = new Set();
let restoredOpenGroups = null;
let suppressHashWrite = false;
let copyLinkResetTimer = null;

function setCopyLinkLabel(label) {{
  copyFilteredLinkButton.textContent = label;
  if (copyLinkResetTimer !== null) {{
    clearTimeout(copyLinkResetTimer);
  }}
  if (label !== 'Copy Filtered Link') {{
    copyLinkResetTimer = window.setTimeout(() => {{
      copyFilteredLinkButton.textContent = 'Copy Filtered Link';
      copyLinkResetTimer = null;
    }}, 1500);
  }}
}}

async function copyCurrentLink() {{
  const url = window.location.href;
  if (navigator.clipboard && window.isSecureContext) {{
    await navigator.clipboard.writeText(url);
    return;
  }}

  const textArea = document.createElement('textarea');
  textArea.value = url;
  textArea.setAttribute('readonly', '');
  textArea.style.position = 'absolute';
  textArea.style.left = '-9999px';
  document.body.appendChild(textArea);
  textArea.select();
  const copied = document.execCommand('copy');
  document.body.removeChild(textArea);
  if (!copied) {{
    throw new Error('copy failed');
  }}
}}

function syncStatusButtons() {{
  for (const button of statusButtons) {{
    const type = button.dataset.filterType;
    button.classList.toggle('is-active', activeStatuses.has(type));
  }}

  for (const button of summaryFilterButtons) {{
    const type = button.dataset.summaryFilter;
    if (type === 'all') {{
      button.classList.toggle('is-active', activeStatuses.size === 0);
    }} else {{
      button.classList.toggle('is-active', activeStatuses.has(type));
    }}
  }}
}}

function writeHashState() {{
  if (suppressHashWrite) {{
    return;
  }}

  const params = new URLSearchParams();
  const query = plateFilter.value.trim();
  if (query) {{
    params.set('plate', query);
  }}

  const statuses = Array.from(activeStatuses).sort();
  if (statuses.length > 0) {{
    params.set('status', statuses.join(','));
  }}

  const visiblePlateIds = visibleGroups()
    .map((group) => group.dataset.plate || '')
    .filter(Boolean);
  if (visiblePlateIds.length > 0) {{
    const openPlateIds = visibleGroups()
      .filter((group) => group.open)
      .map((group) => group.dataset.plate || '')
      .filter(Boolean)
      .sort();
    params.set('open', openPlateIds.join(','));
  }}

  const hash = params.toString();
  history.replaceState(null, '', hash ? `#${{hash}}` : `${{window.location.pathname}}${{window.location.search}}`);
}}

function restoreHashState() {{
  const hash = window.location.hash.startsWith('#') ? window.location.hash.slice(1) : '';
  if (!hash) {{
    syncStatusButtons();
    return;
  }}

  const params = new URLSearchParams(hash);
  plateFilter.value = params.get('plate') || '';
  activeStatuses.clear();
  restoredOpenGroups = params.has('open')
    ? new Set((params.get('open') || '').split(',').filter(Boolean))
    : null;

  const statuses = (params.get('status') || '').split(',').filter(Boolean);
  for (const status of statuses) {{
    if (statusButtons.some((button) => button.dataset.filterType === status)) {{
      activeStatuses.add(status);
    }}
  }}

  syncStatusButtons();
}}

function visibleGroups() {{
  return groups.filter((group) => group.style.display !== 'none');
}}

function applyRestoredOpenState() {{
  if (restoredOpenGroups === null) {{
    return;
  }}

  for (const group of visibleGroups()) {{
    const plate = group.dataset.plate || '';
    group.open = restoredOpenGroups.has(plate);
  }}
}}

function updateVisibleGroupCount() {{
  const visibleCount = visibleGroups().length;
  visibleGroupCount.textContent = `${{visibleCount}} / ${{groups.length}}`;
}}

function resetFilters() {{
  plateFilter.value = '';
  activeStatuses.clear();
  syncStatusButtons();
  applyFilters();
}}

function applyFilters() {{
  const query = plateFilter.value.trim().toLowerCase();
  for (const group of groups) {{
    const plate = group.dataset.plate || '';
    const plateStatus = group.dataset.bestType || '';
    const matchesPlate = !query || plate.includes(query);
    const matchesStatus = activeStatuses.size === 0 || activeStatuses.has(plateStatus);
    group.style.display = matchesPlate && matchesStatus ? '' : 'none';
  }}
  updateVisibleGroupCount();
  writeHashState();
}}

plateFilter.addEventListener('input', () => {{
  applyFilters();
}});

for (const button of statusButtons) {{
  button.addEventListener('click', () => {{
    const type = button.dataset.filterType;
    if (activeStatuses.has(type)) {{
      activeStatuses.delete(type);
    }} else {{
      activeStatuses.add(type);
    }}
    syncStatusButtons();
    applyFilters();
  }});
}}

for (const preview of document.querySelectorAll('.group-preview')) {{
  preview.addEventListener('click', (event) => {{
    event.stopPropagation();
  }});
}}

for (const button of summaryFilterButtons) {{
  button.addEventListener('click', () => {{
    const type = button.dataset.summaryFilter;
    if (type === 'all') {{
      activeStatuses.clear();
    }} else if (activeStatuses.has(type)) {{
      activeStatuses.delete(type);
    }} else {{
      activeStatuses.add(type);
    }}
    syncStatusButtons();
    applyFilters();
  }});
}}

openLockedOnlyButton.addEventListener('click', () => {{
  suppressHashWrite = true;
  for (const group of visibleGroups()) {{
    group.open = group.dataset.bestType === 'locked';
  }}
  suppressHashWrite = false;
  writeHashState();
}});

expandVisibleButton.addEventListener('click', () => {{
  suppressHashWrite = true;
  for (const group of visibleGroups()) {{
    group.open = true;
  }}
  suppressHashWrite = false;
  writeHashState();
}});

collapseVisibleButton.addEventListener('click', () => {{
  suppressHashWrite = true;
  for (const group of visibleGroups()) {{
    group.open = false;
  }}
  suppressHashWrite = false;
  writeHashState();
}});

for (const group of groups) {{
  group.addEventListener('toggle', () => {{
    writeHashState();
  }});
}}

clearFiltersButton.addEventListener('click', () => {{
  resetFilters();
}});

visibleGroupCount.addEventListener('click', () => {{
  resetFilters();
}});

copyFilteredLinkButton.addEventListener('click', async () => {{
  try {{
    await copyCurrentLink();
    setCopyLinkLabel('Link Copied');
  }} catch (_error) {{
    setCopyLinkLabel('Copy Failed');
  }}
}});

restoreHashState();
suppressHashWrite = true;
applyFilters();
applyRestoredOpenState();
suppressHashWrite = false;
writeHashState();
</script>
</body>
</html>
"""

    out_path.write_text(page, encoding="utf-8")
    return out_path


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 build_case_index.py /path/to/case_dir")
        return 1

    case_dir = Path(sys.argv[1]).expanduser().resolve()
    if not case_dir.is_dir():
        print(f"Error: case directory not found: {case_dir}")
        return 1

    rows = _merged_rows(case_dir)
    csv_path = write_csv(case_dir, rows)
    html_path = write_html(case_dir, rows)

    print(f"Wrote {csv_path}")
    print(f"Wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())