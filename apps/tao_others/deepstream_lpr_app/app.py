import asyncio
import csv
import json
import math
import mimetypes
import os
import re
import signal
import sqlite3
import threading
import uuid
from collections import defaultdict
from collections import deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
import shutil
import subprocess
import time
from typing import Deque, Dict, List, Literal, Optional, Set

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator


APP_TITLE = "ALPR Field Unit"
MAX_LIVE_EVENTS = 500
EVIDENCE_ROOT = Path("./evidence").resolve()
STATIC_ROOT = Path("./static").resolve()
RUNTIME_ROOT = Path("./runtime").resolve()
LOG_ROOT = Path("./logs").resolve()
HOTLIST_ROOT = Path("./hotlists").resolve()
APP_ROOT = Path(__file__).resolve().parent
ALPR_PID_PATH = RUNTIME_ROOT / "alpr.pid"
BACKEND_PID_PATH = RUNTIME_ROOT / "backend.pid"
ALPR_STATUS_PATH = RUNTIME_ROOT / "alpr_status.json"
CAMERA_REGISTRY_PATH = RUNTIME_ROOT / "camera_registry.json"
SOURCE_STALE_SECONDS = 5
ALPR_START_GRACE_SECONDS = 15
ALPR_RUNTIME_STALE_SECONDS = 15
ALPR_EVENT_STALE_SECONDS = 45
ALPR_WATCHDOG_INTERVAL_SECONDS = max(2.0, float(os.getenv("ALPR_WATCHDOG_INTERVAL_SECONDS", "5")))
ALPR_WATCHDOG_COOLDOWN_SECONDS = max(20.0, float(os.getenv("ALPR_WATCHDOG_COOLDOWN_SECONDS", "45")))
ACTIVE_CONFIG_ENV = "ALPR_ACTIVE_CONFIG_PATH"
V4L2_CACHE_TTL_SECONDS = 2.0
DIRECT_PREVIEW_CACHE_TTL_SECONDS = 0.5
DIRECT_PREVIEW_WIDTH = 960
DIRECT_PREVIEW_HEIGHT = 600
DIRECT_PREVIEW_FRAMERATE = 30
ALPR_DAY_WHITE_BALANCE_TEMPERATURE = 4600
GPS_MIFI_HOST = os.getenv("GPS_MIFI_HOST", "192.168.1.1")
GPS_MIFI_PORT = int(os.getenv("GPS_MIFI_PORT", "11010"))
GPS_RECONNECT_DELAY_SECONDS = 5.0
ALPR_NIGHT_WHITE_BALANCE_TEMPERATURE = 4200
UNIT_ID = os.getenv("ALPR_UNIT_ID", os.getenv("HOSTNAME", "unit-unknown")).strip() or "unit-unknown"
EVENT_DB_PATH = EVIDENCE_ROOT / "alpr_events.db"


class EventStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    LOCKED = "LOCKED"


class CameraSource(str, Enum):
    LF = "LF"
    RF = "RF"
    LR = "LR"
    RR = "RR"


SOURCE_LABELS = {
    CameraSource.LF: "Left Front",
    CameraSource.RF: "RF camera",
    CameraSource.LR: "Left Rear",
    CameraSource.RR: "Right Rear",
}

HOTLIST_LIST_INFO = {
    "svs.tbl": {"type": "SVS", "label": "Stolen Vehicle", "priority": 3},
    "slr.tbl": {"type": "SLR", "label": "Stolen Plate", "priority": 2},
    "sfr.tbl": {"type": "SFR", "label": "Felony Vehicle", "priority": 4},
}

HOTLIST_FILENAMES = tuple(HOTLIST_LIST_INFO.keys())
HOTLIST_TYPE_INFO = {
    info["type"]: {
        "filename": filename,
        "label": info["label"],
        "priority": info["priority"],
    }
    for filename, info in HOTLIST_LIST_INFO.items()
}

HOTLIST_PRIORITY = {
    "SLR": 2,
    "SVS": 3,
    "SFR": 4,
}

HOTLIST_LOCAL_OVERLAY_PATH = HOTLIST_ROOT / "local_hotlist_entries.json"
HOTLIST_MANIFEST_PATH = HOTLIST_ROOT / "hotlist_manifest.json"
HOTLIST_UPLOAD_MAX_BYTES = max(
    2 * 1024 * 1024,
    int(os.getenv("HOTLIST_UPLOAD_MAX_BYTES", str(10 * 1024 * 1024))),
)

HOTLIST_ACKNOWLEDGE_SECONDS = max(60, int(os.getenv("HOTLIST_ACKNOWLEDGE_SECONDS", "1800")))
HOTLIST_AUTO_SNOOZE_COUNT = max(2, int(os.getenv("HOTLIST_AUTO_SNOOZE_COUNT", "5")))
HOTLIST_AUTO_SNOOZE_WINDOW_SECONDS = max(10, int(os.getenv("HOTLIST_AUTO_SNOOZE_WINDOW_SECONDS", "60")))
HOTLIST_RUNTIME_PLATE_TRANSLATION = str.maketrans({"O": "0"})


def normalize_plate_text(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("plate cannot be empty")
    return normalized


def hotlist_runtime_plate_key(plate: str) -> str:
    return normalize_plate_text(plate).translate(HOTLIST_RUNTIME_PLATE_TRANSLATION)


class HotlistEntry(BaseModel):
    entry_id: Optional[str] = None
    plate: str = Field(..., min_length=1)
    state: str = Field(..., min_length=2, max_length=2)
    county_code: str = Field(..., min_length=2, max_length=2)
    entry_date: str
    list_type: str = Field(..., min_length=3, max_length=3)
    list_label: str = Field(..., min_length=1)
    source_file: str = Field(..., min_length=1)
    source_kind: str = Field(default="agency", min_length=1)
    created_utc: Optional[str] = None

    @field_validator("plate")
    @classmethod
    def normalize_hotlist_plate(cls, value: str) -> str:
        return normalize_plate_text(value)

    @field_validator("state")
    @classmethod
    def normalize_state(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 2:
            raise ValueError("state must be 2 characters")
        return normalized

    @field_validator("county_code")
    @classmethod
    def normalize_county_code(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) != 2:
            raise ValueError("county_code must be 2 characters")
        return normalized


class LiveEvent(BaseModel):
    event_id: str = Field(..., min_length=1)
    display_id: Optional[str] = None
    case_id: str = Field(..., min_length=1)
    plate: str = Field(..., min_length=1)
    audio_cue: Optional[str] = None
    vehicle_make: Optional[str] = None
    vehicle_type: Optional[str] = None
    vehicle_color: Optional[str] = None
    status: EventStatus
    confidence: int = Field(..., ge=0, le=100)
    source: CameraSource
    source_label: Optional[str] = None
    timestamp_utc: str
    frame_number: int = Field(..., ge=0)
    track_id: int = Field(default=0, ge=0)
    track_id_valid: bool = False
    gps_fix_valid: bool = False
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude_m: Optional[float] = None
    gps_speed_knots: Optional[float] = None
    gps_timestamp_utc: Optional[str] = None
    full_frame_path: Optional[str] = None
    annotated_frame_path: Optional[str] = None
    plate_crop_path: Optional[str] = None
    vehicle_crop_path: Optional[str] = None
    hotlist_hit: bool = False
    hotlist_entries: List[HotlistEntry] = Field(default_factory=list)
    hotlist_highest_label: Optional[str] = None
    hotlist_highest_type: Optional[str] = None
    hotlist_highest_priority: Optional[int] = None
    hotlist_alert: bool = False
    hotlist_alert_reason: Optional[str] = None
    hotlist_alert_cooldown_until_utc: Optional[str] = None

    @field_validator("plate")
    @classmethod
    def normalize_plate(cls, value: str) -> str:
        return normalize_plate_text(value)

    @field_validator("timestamp_utc")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp_utc must be ISO-8601 UTC") from exc
        return value

    @model_validator(mode="after")
    def fill_source_label(self) -> "LiveEvent":
        if not self.source_label:
            self.source_label = SOURCE_LABELS.get(self.source)
        return self


class StatusResponse(BaseModel):
    service: Literal["alpr-field-unit"] = "alpr-field-unit"
    status: Literal["ok"] = "ok"
    now_utc: str
    live_event_count: int
    current_case_id: Optional[str] = None
    current_case_event_count: int = 0
    last_event_time_utc: Optional[str] = None
    alpr_process_alive: bool
    alpr_pid: Optional[int] = None
    backend_pid: int
    runtime_status_utc: Optional[str] = None
    storage_free_bytes: int
    storage_free_gb: float
    evidence_root: str
    available_sources: List[str]
    source_health: List[dict]
    network_access: List[dict] = Field(default_factory=list)
    gps_status: dict = Field(default_factory=dict)
    backend_log_path: str
    alpr_log_path: str


class HotlistStatusResponse(BaseModel):
    status: Literal["ok"] = "ok"
    unique_plates: int
    total_entries: int
    sources: List[str]
    hotlist_root: str
    last_reload_utc: Optional[str] = None
    local_entry_count: int = 0
    source_details: List[dict] = Field(default_factory=list)


class HotlistAcknowledgeRequest(BaseModel):
    plate: str = Field(..., min_length=1)
    seconds: int = Field(default=HOTLIST_ACKNOWLEDGE_SECONDS, ge=60, le=24 * 60 * 60)

    @field_validator("plate")
    @classmethod
    def normalize_plate(cls, value: str) -> str:
        return normalize_plate_text(value)


class HotlistAcknowledgeResponse(BaseModel):
    status: Literal["ok"] = "ok"
    plate: str
    suppressed_until_utc: str
    reason: Literal["acknowledged", "auto-snooze"] = "acknowledged"


class PersistedPlateRead(BaseModel):
    event_uuid: str
    event_id: str
    unit_id: str
    case_id: str
    status: EventStatus
    source: CameraSource
    timestamp_utc: str
    plate: str
    confidence: int
    frame_number: int
    track_id_valid: bool
    track_id: int
    vehicle_make: Optional[str] = None
    vehicle_type: Optional[str] = None
    vehicle_color: Optional[str] = None
    gps_fix_valid: bool = False
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude_m: Optional[float] = None
    gps_speed_knots: Optional[float] = None
    gps_timestamp_utc: Optional[str] = None
    full_frame_path: Optional[str] = None
    annotated_frame_path: Optional[str] = None
    plate_crop_path: Optional[str] = None
    vehicle_crop_path: Optional[str] = None
    hotlist_hit: bool = False
    hotlist_alert: bool = False
    hotlist_alert_reason: Optional[str] = None
    hotlist_type: Optional[str] = None
    hotlist_label: Optional[str] = None
    hotlist_priority: Optional[int] = None
    hotlist_entries: List[HotlistEntry] = Field(default_factory=list)
    created_at_utc: Optional[str] = None
    updated_at_utc: Optional[str] = None


class PersistedPlateReadSearchResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[PersistedPlateRead]


class HotlistResetRequest(BaseModel):
    plate: Optional[str] = None

    @field_validator("plate")
    @classmethod
    def normalize_plate(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().upper()
        return normalized or None


class HotlistLocalEntryRequest(BaseModel):
    plate: str = Field(..., min_length=1)
    list_type: str = Field(default="SLR", min_length=3, max_length=3)
    state: str = Field(default="CA", min_length=2, max_length=2)
    county_code: str = Field(default="00", min_length=2, max_length=2)
    entry_date: Optional[str] = None

    @field_validator("plate")
    @classmethod
    def normalize_local_plate(cls, value: str) -> str:
        return normalize_plate_text(value)

    @field_validator("list_type")
    @classmethod
    def normalize_list_type(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in HOTLIST_TYPE_INFO:
            raise ValueError("list_type must be one of SLR, SVS, SFR")
        return normalized

    @field_validator("state")
    @classmethod
    def normalize_local_state(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 2:
            raise ValueError("state must be 2 characters")
        return normalized

    @field_validator("county_code")
    @classmethod
    def normalize_local_county(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) != 2:
            raise ValueError("county_code must be 2 characters")
        return normalized

    @field_validator("entry_date")
    @classmethod
    def normalize_entry_date(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        try:
            return datetime.strptime(normalized, "%Y-%m-%d").date().isoformat()
        except ValueError as exc:
            raise ValueError("entry_date must be YYYY-MM-DD") from exc


class CameraControlRequest(BaseModel):
    source: str = Field(..., min_length=1)
    control: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)

    @field_validator("source", "control", "value")
    @classmethod
    def normalize_camera_control_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field cannot be empty")
        return normalized


class CameraPresetRequest(BaseModel):
    source: str = Field(..., min_length=1)
    preset: str = Field(..., min_length=1)

    @field_validator("source", "preset")
    @classmethod
    def normalize_camera_preset_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field cannot be empty")
        return normalized


_CAMERA_NAME_RE = re.compile(r'^[A-Z0-9_\-]+$')
_CAMERA_URI_SCHEMES = ("rtsp://", "rtsps://", "v4l2://", "/dev/", "file://", "http://", "https://")


class AddCameraRequest(BaseModel):
    uri: str = Field(..., min_length=1, max_length=1024)
    name: str = Field(..., min_length=1, max_length=64)

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, v: str) -> str:
        v = v.strip()
        if not any(v.startswith(s) for s in _CAMERA_URI_SCHEMES):
            raise ValueError(
                "uri must start with rtsp://, rtsps://, v4l2://, /dev/, file://, http://, or https://"
            )
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("name cannot be empty")
        if not _CAMERA_NAME_RE.match(v):
            raise ValueError("name may only contain letters, digits, hyphens, and underscores")
        return v


class RenameCameraRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("name cannot be empty")
        if not _CAMERA_NAME_RE.match(v):
            raise ValueError("name may only contain letters, digits, hyphens, and underscores")
        return v


class ToggleCameraEnabledRequest(BaseModel):
    enabled: bool


app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

live_events: List[LiveEvent] = []
clients: Set[WebSocket] = set()
hotlist_by_plate: Dict[str, List[HotlistEntry]] = {}
hotlist_by_runtime_plate: Dict[str, List[HotlistEntry]] = {}
local_hotlist_entries: List[HotlistEntry] = []
hotlist_status_snapshot: dict = {}
suppressed_hotlist_plates: Dict[str, tuple] = {}
recent_hotlist_sightings: Dict[str, Deque[datetime]] = defaultdict(deque)
v4l2_device_cache: Dict[str, dict] = {}
direct_preview_cache: Dict[str, dict] = {}
direct_preview_health: Dict[str, dict] = {}
backend_started_at = time.monotonic()
alpr_watchdog_task: Optional[asyncio.Task] = None
alpr_watchdog_last_restart_monotonic = 0.0
gps_reader_task: Optional[asyncio.Task] = None
latest_gps_fix: Dict[str, object] = {}
event_db_lock = threading.Lock()

if STATIC_ROOT.exists():
    app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


CAMERA_PRESETS = [
    {
        "id": "alpr_day",
        "label": "ALPR Day",
        "description": "Manual exposure and locked white balance for daytime traffic.",
        "controls": [
            ("auto_exposure", "1"),
            ("exposure_time_absolute", "5"),
            ("gain", "100"),
            ("white_balance_automatic", "0"),
            ("white_balance_temperature", str(ALPR_DAY_WHITE_BALANCE_TEMPERATURE)),
        ],
    },
    {
        "id": "alpr_night",
        "label": "ALPR Night",
        "description": "Longer manual exposure, more gain, and locked white balance for low light.",
        "controls": [
            ("auto_exposure", "1"),
            ("exposure_time_absolute", "120"),
            ("gain", "350"),
            ("white_balance_automatic", "0"),
            ("white_balance_temperature", str(ALPR_NIGHT_WHITE_BALANCE_TEMPERATURE)),
        ],
    },
    {
        "id": "factory_auto",
        "label": "Factory Auto",
        "description": "Restore automatic exposure and white balance.",
        "controls": [
            ("auto_exposure", "0"),
            ("white_balance_automatic", "1"),
        ],
    },
]


CAMERA_CONTROL_RE = re.compile(
    r"^\s*(?P<name>[a-zA-Z0-9_]+)\s+0x[0-9a-f]+\s+\((?P<type>[a-z0-9_]+)\)\s*:\s*(?P<rest>.+)$"
)
CAMERA_MENU_RE = re.compile(r"^\s*(?P<value>-?\d+)\s*:\s*(?P<label>.+)$")
CAMERA_WIDTH_HEIGHT_RE = re.compile(r"(?P<width>\d+)\/(?P<height>\d+)")
CAMERA_FPS_RE = re.compile(r"(?P<fps>\d+(?:\.\d+)?)")


def _int_if_possible(value: Optional[str]):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def active_config_path() -> Optional[Path]:
    raw_path = os.getenv(ACTIVE_CONFIG_ENV, "").strip()
    if not raw_path:
        return None

    path = Path(raw_path).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        return None

    if not resolved.exists() or not resolved.is_file():
        return None
    return resolved


def split_source_values(raw_value: str) -> List[str]:
    cleaned = raw_value.strip().strip("[]")
    if not cleaned:
        return []
    return [
        item.strip().strip("\"").strip("'")
        for item in re.split(r"\s*[;,]\s*", cleaned)
        if item.strip()
    ]


def load_config_source_uris(config_path: Optional[Path] = None) -> Dict[str, str]:
    path = config_path or active_config_path()
    if path is None:
        return {}

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}

    source_values: List[str] = []
    in_source_list = False
    collecting_list_items = False

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if indent == 0 and stripped.endswith(":"):
            in_source_list = stripped == "source-list:"
            collecting_list_items = False
            continue

        if not in_source_list:
            continue

        if indent == 0:
            in_source_list = False
            collecting_list_items = False
            continue

        if stripped.startswith("list:"):
            value = stripped.split(":", 1)[1].strip()
            if value:
                source_values.extend(split_source_values(value))
                collecting_list_items = False
            else:
                collecting_list_items = True
            continue

        if collecting_list_items and stripped.startswith("- "):
            source_values.append(stripped[2:].strip().strip("\"").strip("'"))
            continue

        if collecting_list_items and not stripped.startswith("- "):
            collecting_list_items = False

    return {
        f"source_{index}": uri
        for index, uri in enumerate(source_values)
        if uri
    }


def load_config_source_map(config_path: Optional[Path] = None) -> Dict[str, str]:
    path = config_path or active_config_path()
    if path is None:
        return {}

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}

    mapping = {}
    in_live_dashboard = False
    in_source_map = False

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if indent == 0 and stripped.endswith(":"):
            in_live_dashboard = stripped == "live-dashboard:"
            in_source_map = False
            continue

        if not in_live_dashboard:
            continue

        if indent == 0:
            in_live_dashboard = False
            in_source_map = False
            continue

        if indent == 2 and stripped == "source-map:":
            in_source_map = True
            continue

        if in_source_map and indent <= 2:
            in_source_map = False

        if in_source_map and ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'").upper()
            if key and value:
                mapping[key] = value

    return mapping


def rewrite_config_yaml_sources(
    new_uris: List[str],
    new_source_map: Dict[str, str],
    config_path: Optional[Path] = None,
) -> None:
    """Atomically rewrite source-list.list and live-dashboard.source-map in the active YAML config."""
    path = config_path or active_config_path()
    if path is None:
        raise HTTPException(status_code=503, detail="No active config file is set; cannot modify sources.")

    try:
        original = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read config: {exc}")

    lines = original.splitlines(keepends=True)
    output: List[str] = []
    in_source_list = False
    in_live_dashboard = False
    in_source_map = False

    for raw_line in lines:
        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if not stripped:
            if not in_source_map:
                output.append(raw_line)
            continue

        if stripped.startswith("#"):
            output.append(raw_line)
            continue

        if indent == 0:
            in_source_list = (stripped == "source-list:")
            in_live_dashboard = (stripped == "live-dashboard:")
            in_source_map = False
            output.append(raw_line)
            continue

        if in_source_list and indent == 2 and stripped.startswith("list:"):
            output.append(f"  list: {';'.join(new_uris)}\n")
            continue

        if in_live_dashboard and indent == 2 and stripped == "source-map:":
            output.append("  source-map:\n")
            for key, label in new_source_map.items():
                output.append(f"    {key}: {label}\n")
            in_source_map = True
            continue

        if in_source_map:
            if indent >= 4:
                continue  # skip old source_N entries
            else:
                in_source_map = False  # fall through to output

        output.append(raw_line)

    try:
        temp_path = path.with_name(f".{path.name}.tmp")
        temp_path.write_text("".join(output), encoding="utf-8")
        temp_path.replace(path)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not write config: {exc}")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _registry_default_sources(config_path: Path) -> List[dict]:
    source_uris = load_config_source_uris(config_path)
    source_map = load_config_source_map(config_path)
    entries: List[dict] = []

    for raw_source in sorted(source_uris.keys(), key=lambda item: int(item.split("_", 1)[1]) if "_" in item else 9999):
        uri = source_uris.get(raw_source)
        if not uri:
            continue
        name = str(source_map.get(raw_source, raw_source)).strip().upper() or raw_source
        entries.append(
            {
                "camera_id": raw_source,
                "uri": uri,
                "name": name,
                "enabled": True,
            }
        )

    return entries


def _normalize_registry_entries(raw_entries: List[dict]) -> List[dict]:
    normalized: List[dict] = []
    for index, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            continue
        camera_id = str(item.get("camera_id") or f"camera_{index}").strip()
        uri = str(item.get("uri") or "").strip()
        name = str(item.get("name") or camera_id).strip().upper()
        enabled = bool(item.get("enabled", True))
        if not camera_id or not uri or not name:
            continue
        normalized.append(
            {
                "camera_id": camera_id,
                "uri": uri,
                "name": name,
                "enabled": enabled,
            }
        )
    return normalized


def load_camera_registry(config_path: Optional[Path] = None) -> List[dict]:
    active_path = config_path or active_config_path()
    if active_path is None:
        return []

    if CAMERA_REGISTRY_PATH.exists() and CAMERA_REGISTRY_PATH.is_file():
        try:
            with CAMERA_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict) and payload.get("config_path") == str(active_path):
                entries = _normalize_registry_entries(payload.get("sources", []))
                if entries:
                    return entries
        except (OSError, ValueError, TypeError):
            pass

    entries = _registry_default_sources(active_path)
    _write_json_atomic(
        CAMERA_REGISTRY_PATH,
        {
            "config_path": str(active_path),
            "sources": entries,
        },
    )
    return entries


def save_camera_registry(entries: List[dict], config_path: Optional[Path] = None) -> List[dict]:
    active_path = config_path or active_config_path()
    if active_path is None:
        raise HTTPException(status_code=503, detail="No active config file is set.")

    normalized = _normalize_registry_entries(entries)
    _write_json_atomic(
        CAMERA_REGISTRY_PATH,
        {
            "config_path": str(active_path),
            "sources": normalized,
        },
    )
    return normalized


def apply_registry_to_active_config(entries: List[dict], config_path: Optional[Path] = None) -> None:
    active_path = config_path or active_config_path()
    if active_path is None:
        raise HTTPException(status_code=503, detail="No active config file is set.")

    enabled_entries = [item for item in entries if item.get("enabled")]
    new_uris = [str(item.get("uri") or "").strip() for item in enabled_entries if str(item.get("uri") or "").strip()]
    new_map = {
        f"source_{index}": str(item.get("name") or f"source_{index}").strip().upper()
        for index, item in enumerate(enabled_entries)
    }
    rewrite_config_yaml_sources(new_uris, new_map, active_path)


def device_path_from_source_uri(source_uri: Optional[str]) -> Optional[str]:
    if not source_uri:
        return None
    if source_uri.startswith("v4l2://"):
        return source_uri[len("v4l2://") :]
    if source_uri.startswith("/dev/video"):
        return source_uri
    return None


def canonical_device_path(device_path: Optional[str]) -> Optional[str]:
    if not device_path:
        return None

    path = Path(device_path)
    if not path.exists():
        return device_path

    try:
        return str(path.resolve())
    except OSError:
        return device_path


def runtime_preview_info(raw_source: str, runtime_item: dict) -> tuple:
    preview_path = runtime_item.get("preview_path")
    preview_updated_utc = runtime_item.get("preview_updated_utc")

    if isinstance(preview_path, str) and preview_path.strip():
        return preview_path, preview_updated_utc

    if not raw_source:
        return None, preview_updated_utc

    candidate = RUNTIME_ROOT / "previews" / f"{raw_source}.jpg"
    if not candidate.exists():
        return None, preview_updated_utc

    try:
        updated = datetime.fromtimestamp(candidate.stat().st_mtime, tz=timezone.utc)
        preview_updated_utc = updated.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except OSError:
        preview_updated_utc = preview_updated_utc or None

    return f"previews/{candidate.name}", preview_updated_utc


def runtime_preview_detections(runtime_item: dict) -> List[dict]:
    raw_items = runtime_item.get("preview_detections")
    if not isinstance(raw_items, list):
        return []

    normalized = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            left = max(0, int(item.get("left", 0)))
            top = max(0, int(item.get("top", 0)))
            width = int(item.get("width", 0))
            height = int(item.get("height", 0))
            confidence = int(item.get("confidence", 0))
        except (TypeError, ValueError):
            continue

        if width <= 0 or height <= 0:
            continue

        focus_state = str(item.get("focus_state") or "out_of_focus").strip()
        if focus_state not in {"out_of_focus", "approaching_focus", "in_focus"}:
            focus_state = "out_of_focus"

        plate = item.get("plate")
        normalized.append(
            {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "confidence": max(0, min(100, confidence)),
                "plate": str(plate).strip().upper() if isinstance(plate, str) else "",
                "focus_state": focus_state,
            }
        )

    return normalized


def source_order_value(source_code: str) -> int:
    ordered = [source.value for source in CameraSource]
    try:
        return ordered.index(source_code)
    except ValueError:
        return len(ordered)


def resolve_raw_source_name(source_name: str, runtime_status: dict, source_uris: Dict[str, str]) -> Optional[str]:
    if source_name in source_uris:
        return source_name

    runtime_source_names = []
    for item in runtime_status.get("sources", []):
        if isinstance(item, dict):
            raw_source = str(item.get("source") or item.get("name") or "").strip()
            if raw_source:
                runtime_source_names.append(raw_source)

    candidates = list(source_uris.keys()) + runtime_source_names
    for raw_source in candidates:
        source_code, _ = source_display(raw_source)
        if source_code == source_name:
            return raw_source
    return None


def run_v4l2_command(device_path: str, *args: str) -> str:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--device", device_path, *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        return result.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def parse_v4l2_controls(control_text: str) -> List[dict]:
    controls: List[dict] = []
    current_control = None

    for raw_line in control_text.splitlines():
        line = raw_line.rstrip()
        control_match = CAMERA_CONTROL_RE.match(line)
        if control_match:
            rest = control_match.group("rest")
            attributes = dict(re.findall(r"([a-z_]+)=([^\s]+)", rest))
            flags = [flag for flag in attributes.get("flags", "").split(",") if flag]
            current_control = {
                "name": control_match.group("name"),
                "label": control_match.group("name").replace("_", " ").title(),
                "type": control_match.group("type"),
                "min": _int_if_possible(attributes.get("min")),
                "max": _int_if_possible(attributes.get("max")),
                "step": _int_if_possible(attributes.get("step")) or 1,
                "default": _int_if_possible(attributes.get("default")),
                "value": _int_if_possible(attributes.get("value")),
                "flags": flags,
                "active": "inactive" not in flags,
                "options": [],
            }
            controls.append(current_control)
            continue

        if current_control is None:
            continue

        menu_match = CAMERA_MENU_RE.match(line)
        if menu_match:
            current_control["options"].append(
                {
                    "value": _int_if_possible(menu_match.group("value")),
                    "label": menu_match.group("label").strip(),
                }
            )

    return controls


def parse_v4l2_device_info(all_text: str) -> dict:
    info = {
        "camera_driver": None,
        "driver_version": None,
        "camera_card": None,
        "camera_bus_info": None,
        "pixel_format": None,
        "capture_width": None,
        "capture_height": None,
        "capture_fps": None,
    }

    for raw_line in all_text.splitlines():
        line = raw_line.strip()
        if line.startswith("Driver name"):
            info["camera_driver"] = line.split(":", 1)[1].strip()
        elif line.startswith("Driver version"):
            info["driver_version"] = line.split(":", 1)[1].strip()
        elif line.startswith("Card type"):
            info["camera_card"] = line.split(":", 1)[1].strip()
        elif line.startswith("Bus info"):
            info["camera_bus_info"] = line.split(":", 1)[1].strip()
        elif line.startswith("Width/Height"):
            match = CAMERA_WIDTH_HEIGHT_RE.search(line)
            if match:
                info["capture_width"] = int(match.group("width"))
                info["capture_height"] = int(match.group("height"))
        elif line.startswith("Pixel Format"):
            info["pixel_format"] = line.split(":", 1)[1].strip()
        elif line.startswith("Frames per second"):
            match = CAMERA_FPS_RE.search(line)
            if match:
                info["capture_fps"] = float(match.group("fps"))

    return info


def invalidate_v4l2_cache(device_path: Optional[str]) -> None:
    canonical_path = canonical_device_path(device_path)
    if canonical_path:
        v4l2_device_cache.pop(canonical_path, None)


def preview_health_key(device_path: Optional[str]) -> Optional[str]:
    return canonical_device_path(device_path) or device_path


def record_preview_success(device_path: str) -> None:
    key = preview_health_key(device_path)
    if not key:
        return
    direct_preview_health[key] = {
        "consecutive_failures": 0,
        "last_error": None,
        "last_success_utc": utc_now_iso(),
    }


def record_preview_failure(device_path: str, detail: str) -> None:
    key = preview_health_key(device_path)
    if not key:
        return
    current = direct_preview_health.get(key, {})
    direct_preview_health[key] = {
        "consecutive_failures": int(current.get("consecutive_failures", 0)) + 1,
        "last_error": detail,
        "last_failure_utc": utc_now_iso(),
        "last_success_utc": current.get("last_success_utc"),
    }


def preview_warning_for_device(device_path: Optional[str]) -> Optional[str]:
    if not device_path:
        return None

    if not Path(device_path).exists():
        return "Camera disconnected. Check the USB connection."

    state = direct_preview_health.get(preview_health_key(device_path) or "", {})
    failures = int(state.get("consecutive_failures", 0))
    if failures < 2:
        return None

    detail = str(state.get("last_error") or "").lower()
    if "input/output error" in detail or "protocol error" in detail or "uvc" in detail:
        return "Camera link unstable. Check the USB cable, port, or power."
    if "device path not found" in detail or "no such file" in detail:
        return "Camera disconnected. Check the USB connection."
    return "Camera preview is unstable. Check the camera link and reconnect the device if needed."


def get_v4l2_device_state(device_path: Optional[str]) -> dict:
    canonical_path = canonical_device_path(device_path)
    if not canonical_path:
        return {}

    cached = v4l2_device_cache.get(canonical_path)
    now = time.monotonic()
    if cached and cached.get("expires_at", 0.0) > now:
        return cached.get("data", {})

    all_output = run_v4l2_command(canonical_path, "--all")
    controls_output = run_v4l2_command(canonical_path, "--list-ctrls-menus")

    data = parse_v4l2_device_info(all_output)
    data["camera_controls"] = parse_v4l2_controls(controls_output)

    v4l2_device_cache[canonical_path] = {
        "expires_at": now + V4L2_CACHE_TTL_SECONDS,
        "data": data,
    }
    return data


def apply_v4l2_control(device_path: str, control_name: str, value: str) -> None:
    canonical_path = canonical_device_path(device_path) or device_path
    subprocess.run(
        ["v4l2-ctl", "--device", canonical_path, "--set-ctrl", f"{control_name}={value}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=3,
    )
    invalidate_v4l2_cache(device_path)


def capture_direct_preview(device_path: str) -> bytes:
    canonical_path = canonical_device_path(device_path) or device_path
    cached = direct_preview_cache.get(canonical_path)
    now = time.monotonic()
    if cached and cached.get("expires_at", 0.0) > now:
        return cached.get("data", b"")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "video4linux2",
                "-input_format",
                "yuyv422",
                "-video_size",
                f"{DIRECT_PREVIEW_WIDTH}x{DIRECT_PREVIEW_HEIGHT}",
                "-framerate",
                str(DIRECT_PREVIEW_FRAMERATE),
                "-i",
                canonical_path,
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=8,
        )
    except OSError as exc:
        detail = f"failed to launch ffmpeg: {exc}"
        record_preview_failure(canonical_path, detail)
        raise HTTPException(status_code=500, detail=detail)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip() or "failed to capture camera preview"
        record_preview_failure(canonical_path, detail)
        raise HTTPException(status_code=502, detail=detail)
    except subprocess.TimeoutExpired:
        record_preview_failure(canonical_path, "camera preview capture timed out")
        raise HTTPException(status_code=504, detail="camera preview capture timed out")

    if not result.stdout:
        record_preview_failure(canonical_path, "camera preview capture returned no image data")
        raise HTTPException(status_code=502, detail="camera preview capture returned no image data")

    direct_preview_cache[canonical_path] = {
        "expires_at": now + DIRECT_PREVIEW_CACHE_TTL_SECONDS,
        "data": result.stdout,
    }
    record_preview_success(canonical_path)
    return result.stdout


def camera_config_sources(runtime_status: dict, alpr_process_alive: bool) -> List[dict]:
    now = datetime.now(timezone.utc)
    alpr_pid = read_pid(ALPR_PID_PATH)
    direct_preview_blocked = alpr_startup_grace_active(alpr_pid)
    registry_entries = load_camera_registry()
    runtime_sources = {}

    for item in runtime_status.get("sources", []):
        if not isinstance(item, dict):
            continue
        raw_source = str(item.get("source") or item.get("name") or "").strip()
        if raw_source:
            runtime_sources[raw_source] = item

    merged_sources = []

    enabled_entries = [entry for entry in registry_entries if entry.get("enabled")]
    runtime_name_by_camera_id: Dict[str, Optional[str]] = {}
    for index, entry in enumerate(enabled_entries):
        runtime_name_by_camera_id[str(entry.get("camera_id"))] = f"source_{index}"

    for entry in registry_entries:
        camera_id = str(entry.get("camera_id") or "").strip()
        if not camera_id:
            continue

        runtime_name = runtime_name_by_camera_id.get(camera_id)
        runtime_item = runtime_sources.get(runtime_name or "", {})
        source_code = camera_id
        label = str(entry.get("name") or camera_id).strip()
        source_uri = str(entry.get("uri") or "").strip()
        enabled = bool(entry.get("enabled", True))
        device_path = device_path_from_source_uri(source_uri)
        canonical_path = canonical_device_path(device_path)
        seen_at = parse_utc_iso(runtime_item.get("last_seen_utc"))
        runtime_item_fresh = runtime_item_is_fresh(runtime_item)
        available = bool(
            alpr_process_alive
            and enabled
            and seen_at is not None
            and (now - seen_at).total_seconds() <= SOURCE_STALE_SECONDS
        )
        runtime_preview_path, runtime_preview_updated_utc = runtime_preview_info(runtime_name or "", runtime_item)
        preview_detections = runtime_preview_detections(runtime_item)
        preview_overlay_width = runtime_item.get("preview_overlay_width")
        preview_overlay_height = runtime_item.get("preview_overlay_height")
        if not isinstance(preview_overlay_width, int) or preview_overlay_width <= 0:
            preview_overlay_width = None
        if not isinstance(preview_overlay_height, int) or preview_overlay_height <= 0:
            preview_overlay_height = None
        preview_url = None
        preview_mode = "unavailable"
        if runtime_preview_path and alpr_process_alive and enabled and runtime_item_fresh:
            preview_url = f"/runtime-media/{runtime_preview_path}"
            preview_mode = "runtime"
        elif device_path and (not alpr_process_alive or not runtime_item_fresh) and not direct_preview_blocked:
            preview_url = f"/api/camera-preview/{camera_id}"
            preview_mode = "direct"

        v4l2_state = get_v4l2_device_state(canonical_path or device_path)
        preview_warning = preview_warning_for_device(canonical_path or device_path)
        if direct_preview_blocked and device_path and preview_mode == "unavailable":
            preview_warning = (
                "Waiting for the ALPR pipeline to claim the camera before direct preview is allowed."
            )
        if alpr_process_alive and preview_mode == "unavailable" and device_path:
            preview_warning = (
                "Direct preview is paused while ALPR is running so the pipeline can keep control of the camera."
            )
        if alpr_process_alive and not runtime_item_fresh and preview_mode == "direct":
            preview_warning = (
                "Runtime preview stalled, so the config page switched to direct camera preview."
            )

        merged_sources.append(
            {
                "camera_id": camera_id,
                "enabled": enabled,
                "source": source_code,
                "source_label": label,
                "process_source_name": camera_id,
                "runtime_source_name": runtime_name,
                "source_uri": source_uri,
                "device_path": canonical_path or device_path,
                "available": available,
                "last_seen_utc": runtime_item.get("last_seen_utc"),
                "last_frame_number": runtime_item.get("last_frame_number"),
                "fps": runtime_item.get("fps"),
                "frame_width": runtime_item.get("frame_width"),
                "frame_height": runtime_item.get("frame_height"),
                "preview_path": runtime_preview_path,
                "preview_url": preview_url,
                "preview_mode": preview_mode,
                "preview_warning": preview_warning,
                "preview_updated_utc": runtime_preview_updated_utc,
                "preview_overlay_width": preview_overlay_width,
                "preview_overlay_height": preview_overlay_height,
                "preview_detections": preview_detections,
                **v4l2_state,
            }
        )

    # Include orphan runtime sources if the running pipeline has entries not present in registry.
    for raw_runtime_name, runtime_item in runtime_sources.items():
        if raw_runtime_name in runtime_name_by_camera_id.values():
            continue
        source_code, label = source_display(raw_runtime_name)
        merged_sources.append(
            {
                "camera_id": raw_runtime_name,
                "enabled": True,
                "source": source_code,
                "source_label": label,
                "process_source_name": raw_runtime_name,
                "runtime_source_name": raw_runtime_name,
                "source_uri": None,
                "device_path": None,
                "available": bool(alpr_process_alive and runtime_item_is_fresh(runtime_item)),
                "last_seen_utc": runtime_item.get("last_seen_utc"),
                "last_frame_number": runtime_item.get("last_frame_number"),
                "fps": runtime_item.get("fps"),
                "frame_width": runtime_item.get("frame_width"),
                "frame_height": runtime_item.get("frame_height"),
                "preview_path": None,
                "preview_url": None,
                "preview_mode": "unavailable",
                "preview_warning": "Runtime source is active but not found in camera registry.",
                "preview_updated_utc": runtime_item.get("preview_updated_utc"),
                "preview_overlay_width": None,
                "preview_overlay_height": None,
                "preview_detections": runtime_preview_detections(runtime_item),
            }
        )

    return sorted(
        merged_sources,
        key=lambda item: (0 if item.get("enabled") else 1, item.get("source_label") or item["process_source_name"]),
    )


def resolve_camera_device(source_name: str, runtime_status: dict) -> tuple:
    registry_entries = load_camera_registry()

    selected_entry = None
    for entry in registry_entries:
        camera_id = str(entry.get("camera_id") or "").strip()
        camera_name = str(entry.get("name") or "").strip().upper()
        if source_name == camera_id or source_name.upper() == camera_name:
            selected_entry = entry
            break

    if selected_entry is not None:
        raw_source = str(selected_entry.get("camera_id") or "").strip()
        source_uri = str(selected_entry.get("uri") or "").strip()
    else:
        source_uris = load_config_source_uris()
        raw_source = resolve_raw_source_name(source_name, runtime_status, source_uris)
        if raw_source is None:
            raise HTTPException(status_code=404, detail="camera source not found")
        source_uri = source_uris.get(raw_source)

    device_path = device_path_from_source_uri(source_uri)
    if not device_path:
        raise HTTPException(status_code=400, detail="camera source is not backed by a V4L2 device")

    canonical_path = canonical_device_path(device_path)
    if not canonical_path or not Path(canonical_path).exists():
        raise HTTPException(status_code=404, detail="camera device path not found")

    return raw_source, canonical_path


def parse_utc_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def event_uuid_for(event: LiveEvent, unit_id: str = UNIT_ID) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{unit_id}:{event.event_id}"))


def ensure_event_db() -> None:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(EVENT_DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plate_reads (
                event_uuid TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                unit_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                status TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                plate TEXT NOT NULL,
                plate_key TEXT NOT NULL,
                confidence INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                track_id_valid INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                vehicle_make TEXT,
                vehicle_type TEXT,
                vehicle_color TEXT,
                gps_fix_valid INTEGER NOT NULL,
                gps_latitude REAL,
                gps_longitude REAL,
                gps_altitude_m REAL,
                gps_speed_knots REAL,
                gps_timestamp_utc TEXT,
                full_frame_path TEXT,
                annotated_frame_path TEXT,
                plate_crop_path TEXT,
                vehicle_crop_path TEXT,
                hotlist_hit INTEGER NOT NULL,
                hotlist_alert INTEGER NOT NULL,
                hotlist_alert_reason TEXT,
                hotlist_type TEXT,
                hotlist_label TEXT,
                hotlist_priority INTEGER,
                hotlist_entries_json TEXT NOT NULL,
                raw_event_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_plate_key ON plate_reads(plate_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_plate ON plate_reads(plate)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_ts ON plate_reads(timestamp_utc)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_source_ts ON plate_reads(source, timestamp_utc)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_vehicle_make ON plate_reads(vehicle_make)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_vehicle_type ON plate_reads(vehicle_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_vehicle_color ON plate_reads(vehicle_color)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_alert_type ON plate_reads(hotlist_alert, hotlist_type)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_plate_reads_geo ON plate_reads(gps_fix_valid, gps_latitude, gps_longitude)"
        )


def persist_plate_read(event: LiveEvent) -> None:
    ensure_event_db()
    now_utc = utc_now_iso()
    event_uuid = event_uuid_for(event)
    plate_key = hotlist_runtime_plate_key(event.plate)
    hotlist_entries_json = json.dumps([entry.model_dump(mode="json") for entry in event.hotlist_entries])
    raw_event_json = event.model_dump_json()

    with event_db_lock, sqlite3.connect(EVENT_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO plate_reads (
                event_uuid, event_id, unit_id, case_id, status, source, timestamp_utc,
                plate, plate_key, confidence, frame_number, track_id_valid, track_id,
                vehicle_make, vehicle_type, vehicle_color,
                gps_fix_valid, gps_latitude, gps_longitude, gps_altitude_m, gps_speed_knots, gps_timestamp_utc,
                full_frame_path, annotated_frame_path, plate_crop_path, vehicle_crop_path,
                hotlist_hit, hotlist_alert, hotlist_alert_reason, hotlist_type, hotlist_label, hotlist_priority,
                hotlist_entries_json, raw_event_json, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_uuid) DO UPDATE SET
                event_id = excluded.event_id,
                case_id = excluded.case_id,
                status = excluded.status,
                source = excluded.source,
                timestamp_utc = excluded.timestamp_utc,
                plate = excluded.plate,
                plate_key = excluded.plate_key,
                confidence = excluded.confidence,
                frame_number = excluded.frame_number,
                track_id_valid = excluded.track_id_valid,
                track_id = excluded.track_id,
                vehicle_make = excluded.vehicle_make,
                vehicle_type = excluded.vehicle_type,
                vehicle_color = excluded.vehicle_color,
                gps_fix_valid = excluded.gps_fix_valid,
                gps_latitude = excluded.gps_latitude,
                gps_longitude = excluded.gps_longitude,
                gps_altitude_m = excluded.gps_altitude_m,
                gps_speed_knots = excluded.gps_speed_knots,
                gps_timestamp_utc = excluded.gps_timestamp_utc,
                full_frame_path = excluded.full_frame_path,
                annotated_frame_path = excluded.annotated_frame_path,
                plate_crop_path = excluded.plate_crop_path,
                vehicle_crop_path = excluded.vehicle_crop_path,
                hotlist_hit = excluded.hotlist_hit,
                hotlist_alert = excluded.hotlist_alert,
                hotlist_alert_reason = excluded.hotlist_alert_reason,
                hotlist_type = excluded.hotlist_type,
                hotlist_label = excluded.hotlist_label,
                hotlist_priority = excluded.hotlist_priority,
                hotlist_entries_json = excluded.hotlist_entries_json,
                raw_event_json = excluded.raw_event_json,
                updated_at_utc = excluded.updated_at_utc
            """,
            (
                event_uuid,
                event.event_id,
                UNIT_ID,
                event.case_id,
                event.status.value,
                event.source.value,
                event.timestamp_utc,
                event.plate,
                plate_key,
                event.confidence,
                event.frame_number,
                1 if event.track_id_valid else 0,
                event.track_id,
                event.vehicle_make,
                event.vehicle_type,
                event.vehicle_color,
                1 if event.gps_fix_valid else 0,
                event.gps_latitude,
                event.gps_longitude,
                event.gps_altitude_m,
                event.gps_speed_knots,
                event.gps_timestamp_utc,
                event.full_frame_path,
                event.annotated_frame_path,
                event.plate_crop_path,
                event.vehicle_crop_path,
                1 if event.hotlist_hit else 0,
                1 if event.hotlist_alert else 0,
                event.hotlist_alert_reason,
                event.hotlist_highest_type,
                event.hotlist_highest_label,
                event.hotlist_highest_priority,
                hotlist_entries_json,
                raw_event_json,
                now_utc,
                now_utc,
            ),
        )


def escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def parse_hotlist_entries_json(raw: str) -> List[HotlistEntry]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    entries: List[HotlistEntry] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            entries.append(HotlistEntry.model_validate(item))
        except Exception:
            continue
    return entries


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


def row_to_persisted_plate_read(row: sqlite3.Row) -> PersistedPlateRead:
    return PersistedPlateRead(
        event_uuid=row["event_uuid"],
        event_id=row["event_id"],
        unit_id=row["unit_id"],
        case_id=row["case_id"],
        status=EventStatus(row["status"]),
        source=CameraSource(row["source"]),
        timestamp_utc=row["timestamp_utc"],
        plate=row["plate"],
        confidence=row["confidence"],
        frame_number=row["frame_number"],
        track_id_valid=bool(row["track_id_valid"]),
        track_id=row["track_id"],
        vehicle_make=row["vehicle_make"],
        vehicle_type=row["vehicle_type"],
        vehicle_color=row["vehicle_color"],
        gps_fix_valid=bool(row["gps_fix_valid"]),
        gps_latitude=row["gps_latitude"],
        gps_longitude=row["gps_longitude"],
        gps_altitude_m=row["gps_altitude_m"],
        gps_speed_knots=row["gps_speed_knots"],
        gps_timestamp_utc=row["gps_timestamp_utc"],
        full_frame_path=row["full_frame_path"],
        annotated_frame_path=row["annotated_frame_path"],
        plate_crop_path=row["plate_crop_path"],
        vehicle_crop_path=row["vehicle_crop_path"],
        hotlist_hit=bool(row["hotlist_hit"]),
        hotlist_alert=bool(row["hotlist_alert"]),
        hotlist_alert_reason=row["hotlist_alert_reason"],
        hotlist_type=row["hotlist_type"],
        hotlist_label=row["hotlist_label"],
        hotlist_priority=row["hotlist_priority"],
        hotlist_entries=parse_hotlist_entries_json(row["hotlist_entries_json"]),
        created_at_utc=row["created_at_utc"],
        updated_at_utc=row["updated_at_utc"],
    )


def to_case_relative_media_path(case_id: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().replace("\\", "/")
    if not normalized:
        return None
    if normalized.startswith(f"{case_id}/"):
        return normalized
    return f"{case_id}/{normalized.lstrip('/')}"


def parse_source_from_video_source(raw_source: Optional[str]) -> Optional[CameraSource]:
    if not raw_source:
        return None
    source_code, _ = source_display(str(raw_source))
    try:
        return CameraSource(source_code)
    except Exception:
        return None


def live_event_from_jsonl_record(record: dict, case_id: str) -> Optional[LiveEvent]:
    if not isinstance(record, dict):
        return None

    event_type = str(record.get("event_type") or "").strip().upper()
    if event_type not in {EventStatus.CONFIRMED.value, EventStatus.LOCKED.value}:
        return None

    source = parse_source_from_video_source(record.get("video_source"))
    if source is None:
        return None

    plate = str(record.get("plate") or "").strip().upper()
    if not plate:
        return None

    event_id = str(record.get("event_id") or "").strip()
    if not event_id:
        return None

    timestamp_utc = str(record.get("timestamp_utc") or "").strip()
    if not timestamp_utc or parse_utc_iso(timestamp_utc) is None:
        return None

    try:
        frame_number = int(record.get("frame_number") or 0)
        confidence = int(record.get("confidence") or 0)
        track_id = int(record.get("track_id") or 0)
    except (TypeError, ValueError):
        return None

    event = LiveEvent(
        event_id=event_id,
        case_id=case_id,
        plate=plate,
        status=EventStatus(event_type),
        confidence=max(0, min(100, confidence)),
        source=source,
        timestamp_utc=timestamp_utc,
        frame_number=max(0, frame_number),
        track_id=max(0, track_id),
        track_id_valid=bool(record.get("track_id_valid", False)),
        gps_fix_valid=bool(record.get("gps_fix_valid", False)),
        gps_latitude=record.get("gps_latitude"),
        gps_longitude=record.get("gps_longitude"),
        gps_altitude_m=record.get("gps_altitude_m"),
        gps_speed_knots=record.get("gps_speed_knots"),
        gps_timestamp_utc=record.get("gps_timestamp_utc"),
        full_frame_path=to_case_relative_media_path(case_id, record.get("full_frame_path")),
        annotated_frame_path=to_case_relative_media_path(case_id, record.get("annotated_frame_path")),
        plate_crop_path=to_case_relative_media_path(case_id, record.get("plate_crop_path")),
        vehicle_crop_path=to_case_relative_media_path(case_id, record.get("vehicle_crop_path")),
        vehicle_make=(str(record.get("vehicle_make") or "").strip() or None),
        vehicle_type=(str(record.get("vehicle_type") or "").strip() or None),
        vehicle_color=(str(record.get("vehicle_color") or "").strip() or None),
    )
    return event


def iter_jsonl_records(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload
    except (FileNotFoundError, OSError):
        return


def build_backfill_event(record: dict, case_id: str) -> Optional[LiveEvent]:
    event = live_event_from_jsonl_record(record, case_id)
    if event is None:
        return None
    event = enrich_with_hotlist(event)
    if event.hotlist_hit and event.status == EventStatus.LOCKED:
        event.hotlist_alert = True
        event.hotlist_alert_reason = "backfill-locked-hit"
    else:
        event.hotlist_alert = False
        event.hotlist_alert_reason = None
    event.hotlist_alert_cooldown_until_utc = None
    return event


def backfill_events_from_jsonl(case_filter: Optional[str] = None, limit_cases: Optional[int] = None) -> dict:
    ensure_event_db()
    if not EVIDENCE_ROOT.exists():
        return {
            "status": "ok",
            "db_path": str(EVENT_DB_PATH),
            "cases_scanned": 0,
            "records_scanned": 0,
            "records_imported": 0,
            "records_skipped": 0,
            "case_filter": case_filter,
            "message": "evidence root does not exist",
        }

    case_dirs = sorted(
        [path for path in EVIDENCE_ROOT.iterdir() if path.is_dir() and path.name.startswith("case_")],
        key=lambda path: path.name,
    )
    if case_filter:
        case_dirs = [path for path in case_dirs if path.name == case_filter]

    if limit_cases is not None and limit_cases > 0:
        case_dirs = case_dirs[:limit_cases]

    cases_scanned = 0
    records_scanned = 0
    records_imported = 0
    records_skipped = 0

    for case_dir in case_dirs:
        jsonl_path = case_dir / "events.jsonl"
        if not jsonl_path.exists():
            continue
        cases_scanned += 1

        for record in iter_jsonl_records(jsonl_path):
            records_scanned += 1
            event = build_backfill_event(record, case_dir.name)
            if event is None:
                records_skipped += 1
                continue
            try:
                persist_plate_read(event)
                records_imported += 1
            except Exception:
                records_skipped += 1

    return {
        "status": "ok",
        "db_path": str(EVENT_DB_PATH),
        "cases_scanned": cases_scanned,
        "records_scanned": records_scanned,
        "records_imported": records_imported,
        "records_skipped": records_skipped,
        "case_filter": case_filter,
    }


def event_db_status() -> dict:
    ensure_event_db()
    size_bytes = EVENT_DB_PATH.stat().st_size if EVENT_DB_PATH.exists() else 0
    wal_path = Path(str(EVENT_DB_PATH) + "-wal")
    shm_path = Path(str(EVENT_DB_PATH) + "-shm")
    wal_size_bytes = wal_path.stat().st_size if wal_path.exists() else 0
    shm_size_bytes = shm_path.stat().st_size if shm_path.exists() else 0

    with event_db_lock, sqlite3.connect(EVENT_DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS row_count, MAX(timestamp_utc) AS max_timestamp_utc, "
            "MAX(updated_at_utc) AS max_updated_utc FROM plate_reads"
        ).fetchone()

    row_count = int(row[0]) if row and row[0] is not None else 0
    max_timestamp_utc = row[1] if row else None
    max_updated_utc = row[2] if row else None

    return {
        "status": "ok",
        "db_path": str(EVENT_DB_PATH),
        "unit_id": UNIT_ID,
        "row_count": row_count,
        "last_event_timestamp_utc": max_timestamp_utc,
        "last_write_utc": max_updated_utc,
        "db_size_bytes": size_bytes,
        "db_size_mb": round(size_bytes / (1024 * 1024), 2),
        "wal_size_bytes": wal_size_bytes,
        "shm_size_bytes": shm_size_bytes,
    }


def event_db_selftest() -> dict:
    """Insert a clearly-marked synthetic row, read it back, then delete it.  Returns a report dict."""
    import time as _time
    test_event_id = f"_selftest_{uuid.uuid4().hex}"
    test_case_id = "_selftest_case"
    ts = utc_now_iso()
    event = LiveEvent(
        event_id=test_event_id,
        case_id=test_case_id,
        plate="SELFTEST",
        status=EventStatus.CONFIRMED,
        confidence=99,
        source=CameraSource.LF,
        timestamp_utc=ts,
        frame_number=0,
    )
    event_uuid = event_uuid_for(event)

    steps: list[dict] = []
    ok = True

    # --- insert ---
    t0 = _time.monotonic()
    try:
        persist_plate_read(event)
        elapsed_ms = round((_time.monotonic() - t0) * 1000, 2)
        steps.append({"step": "insert", "ok": True, "elapsed_ms": elapsed_ms})
    except Exception as exc:
        steps.append({"step": "insert", "ok": False, "error": str(exc)})
        ok = False

    # --- read-back ---
    if ok:
        t0 = _time.monotonic()
        try:
            with event_db_lock, sqlite3.connect(EVENT_DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM plate_reads WHERE event_uuid = ?", (event_uuid,)
                ).fetchone()
            elapsed_ms = round((_time.monotonic() - t0) * 1000, 2)
            if row is None:
                steps.append({"step": "read", "ok": False, "error": "row not found after insert"})
                ok = False
            else:
                verified = {
                    "event_uuid": row["event_uuid"] == event_uuid,
                    "plate": row["plate"] == "SELFTEST",
                    "case_id": row["case_id"] == test_case_id,
                    "status": row["status"] == EventStatus.CONFIRMED.value,
                    "confidence": row["confidence"] == 99,
                    "source": row["source"] == CameraSource.LF.value,
                }
                all_ok = all(verified.values())
                steps.append({"step": "read", "ok": all_ok, "elapsed_ms": elapsed_ms, "verified_fields": verified})
                if not all_ok:
                    ok = False
        except Exception as exc:
            steps.append({"step": "read", "ok": False, "error": str(exc)})
            ok = False

    # --- delete ---
    t0 = _time.monotonic()
    try:
        with event_db_lock, sqlite3.connect(EVENT_DB_PATH) as conn:
            conn.execute("DELETE FROM plate_reads WHERE event_uuid = ?", (event_uuid,))
        elapsed_ms = round((_time.monotonic() - t0) * 1000, 2)
        steps.append({"step": "delete", "ok": True, "elapsed_ms": elapsed_ms})
    except Exception as exc:
        steps.append({"step": "delete", "ok": False, "error": str(exc)})
        ok = False

    return {
        "status": "ok" if ok else "fail",
        "db_path": str(EVENT_DB_PATH),
        "unit_id": UNIT_ID,
        "event_uuid": event_uuid,
        "steps": steps,
    }


def runtime_item_is_fresh(runtime_item: dict, max_age_seconds: int = ALPR_RUNTIME_STALE_SECONDS) -> bool:
    now = datetime.now(timezone.utc)
    for field_name in ("last_seen_utc", "preview_updated_utc"):
        seen_at = parse_utc_iso(runtime_item.get(field_name))
        if seen_at is not None and (now - seen_at).total_seconds() <= max_age_seconds:
            return True
    return False


def hotlist_label_for_type(list_type: str) -> str:
    return HOTLIST_TYPE_INFO.get(list_type, {}).get("label", list_type)


def hotlist_manifest_default() -> dict:
    return {
        "last_reload_utc": None,
        "agency": {
            filename: {"last_import_utc": None}
            for filename in HOTLIST_FILENAMES
        },
        "local_overlay": {"last_updated_utc": None},
    }


def ensure_hotlist_root() -> None:
    HOTLIST_ROOT.mkdir(parents=True, exist_ok=True)


def write_text_atomic(path: Path, content: str) -> None:
    ensure_hotlist_root()
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def write_json_atomic(path: Path, payload: dict) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))


def load_hotlist_manifest() -> dict:
    default_manifest = hotlist_manifest_default()
    if not HOTLIST_MANIFEST_PATH.exists() or not HOTLIST_MANIFEST_PATH.is_file():
        return default_manifest

    try:
        with HOTLIST_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError, TypeError):
        return default_manifest

    if not isinstance(loaded, dict):
        return default_manifest

    manifest = default_manifest
    manifest.update({key: value for key, value in loaded.items() if key in manifest})
    if isinstance(loaded.get("agency"), dict):
        for filename in HOTLIST_FILENAMES:
            if isinstance(loaded["agency"].get(filename), dict):
                manifest["agency"][filename].update(loaded["agency"][filename])
    if isinstance(loaded.get("local_overlay"), dict):
        manifest["local_overlay"].update(loaded["local_overlay"])
    return manifest


def save_hotlist_manifest(manifest: dict) -> None:
    write_json_atomic(HOTLIST_MANIFEST_PATH, manifest)


def build_local_hotlist_entry(
    plate: str,
    list_type: str,
    state: str = "CA",
    county_code: str = "00",
    entry_date: Optional[str] = None,
    source_file: str = "local-overlay",
    entry_id: Optional[str] = None,
    created_utc: Optional[str] = None,
) -> HotlistEntry:
    entry_date_value = entry_date or datetime.now(timezone.utc).date().isoformat()
    return HotlistEntry(
        entry_id=entry_id or uuid.uuid4().hex[:12],
        plate=plate,
        state=state,
        county_code=county_code,
        entry_date=entry_date_value,
        list_type=list_type,
        list_label=hotlist_label_for_type(list_type),
        source_file=source_file,
        source_kind="local",
        created_utc=created_utc or utc_now_iso(),
    )


def entry_identity(entry: HotlistEntry) -> tuple:
    return (
        entry.plate,
        entry.list_type,
        entry.state,
        entry.county_code,
        entry.entry_date,
        entry.source_kind,
        entry.source_file,
    )


def dedupe_hotlist_entries(entries: List[HotlistEntry]) -> List[HotlistEntry]:
    seen = set()
    unique_entries = []
    for entry in entries:
        identity = entry_identity(entry)
        if identity in seen:
            continue
        seen.add(identity)
        unique_entries.append(entry)
    return unique_entries


def load_local_hotlist_entries() -> List[HotlistEntry]:
    if not HOTLIST_LOCAL_OVERLAY_PATH.exists() or not HOTLIST_LOCAL_OVERLAY_PATH.is_file():
        return []

    try:
        with HOTLIST_LOCAL_OVERLAY_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return []

    raw_entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_entries, list):
        return []

    entries: List[HotlistEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        try:
            record = dict(item)
            record["source_kind"] = "local"
            entries.append(HotlistEntry(**record))
        except Exception:
            continue
    return dedupe_hotlist_entries(entries)


def save_local_hotlist_entries(entries: List[HotlistEntry]) -> None:
    ensure_hotlist_root()
    payload = {
        "entries": [entry.model_dump() for entry in entries],
    }
    write_json_atomic(HOTLIST_LOCAL_OVERLAY_PATH, payload)


def hotlist_file_updated_utc(path: Path) -> Optional[str]:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None
    return utc_iso(modified)


def validate_hotlist_text(text: str, source_file: str) -> tuple:
    entries: List[HotlistEntry] = []
    issues: List[dict] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        record = raw_line.strip()
        if not record or record.startswith("DATE"):
            continue

        entry = parse_hotlist_line(record, source_file)
        if entry is None:
            issues.append({"line": line_number, "detail": "invalid hotlist row"})
            continue
        entries.append(entry)

    return entries, issues


def parse_local_hotlist_csv(text: str, default_list_type: str) -> tuple:
    rows: List[HotlistEntry] = []
    issues: List[dict] = []
    stripped_lines = [line for line in text.splitlines() if line.strip()]
    if not stripped_lines:
        return rows, issues

    header = stripped_lines[0].lower()
    has_header = "plate" in header

    if has_header:
        reader = csv.DictReader(stripped_lines)
        for row_number, row in enumerate(reader, start=2):
            if not isinstance(row, dict):
                continue
            plate = str(row.get("plate") or "").strip().upper()
            if not plate:
                issues.append({"line": row_number, "detail": "missing plate"})
                continue
            list_type = str(row.get("list_type") or default_list_type).strip().upper()
            state = str(row.get("state") or "CA").strip().upper()
            county_code = str(row.get("county_code") or "00").strip()
            entry_date = str(row.get("entry_date") or "").strip() or None
            try:
                rows.append(
                    build_local_hotlist_entry(
                        plate=plate,
                        list_type=list_type,
                        state=state,
                        county_code=county_code,
                        entry_date=entry_date,
                        source_file="local-csv",
                    )
                )
            except Exception as exc:
                issues.append({"line": row_number, "detail": str(exc)})
    else:
        reader = csv.reader(stripped_lines)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            plate = str(row[0] or "").strip().upper()
            if not plate:
                issues.append({"line": row_number, "detail": "missing plate"})
                continue
            try:
                rows.append(
                    build_local_hotlist_entry(
                        plate=plate,
                        list_type=default_list_type,
                        source_file="local-csv",
                    )
                )
            except Exception as exc:
                issues.append({"line": row_number, "detail": str(exc)})

    return dedupe_hotlist_entries(rows), issues


def parse_hotlist_line(line: str, source_file: str) -> Optional[HotlistEntry]:
    record = line.strip()
    if not record or record.startswith("DATE"):
        return None

    parts = record.split()
    list_info = HOTLIST_LIST_INFO.get(source_file)
    if not list_info:
        return None

    plate = ""
    raw_meta = ""
    if len(parts) >= 2:
        plate = parts[0].strip().upper()
        raw_meta = parts[1].strip()
    elif len(record) > 12:
        plate = record[:-12].strip().upper()
        raw_meta = record[-12:].strip()

    if not plate or len(raw_meta) != 12:
        return None
    state = raw_meta[0:2]
    county_code = raw_meta[2:4]
    date_raw = raw_meta[4:12]

    try:
        entry_date = datetime.strptime(date_raw, "%Y%m%d").date().isoformat()
    except ValueError:
        entry_date = date_raw

    return HotlistEntry(
        plate=plate,
        state=state,
        county_code=county_code,
        entry_date=entry_date,
        list_type=list_info["type"],
        list_label=list_info["label"],
        source_file=source_file,
        source_kind="agency",
    )

def build_hotlist_state() -> tuple:
    grouped = defaultdict(list)
    source_details = []
    manifest = load_hotlist_manifest()
    agency_entries = 0

    for filename in HOTLIST_FILENAMES:
        path = HOTLIST_ROOT / filename
        entries_for_source: List[HotlistEntry] = []
        if path.exists() and path.is_file():
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        entry = parse_hotlist_line(line, filename)
                        if entry is not None:
                            grouped[entry.plate].append(entry)
                            entries_for_source.append(entry)
            except OSError:
                entries_for_source = []

        agency_entries += len(entries_for_source)
        source_details.append(
            {
                "source_id": filename,
                "source_kind": "agency",
                "list_type": HOTLIST_LIST_INFO[filename]["type"],
                "label": HOTLIST_LIST_INFO[filename]["label"],
                "file_name": filename,
                "exists": path.exists() and path.is_file(),
                "updated_utc": hotlist_file_updated_utc(path) if path.exists() and path.is_file() else None,
                "last_import_utc": manifest["agency"].get(filename, {}).get("last_import_utc"),
                "record_count": len(entries_for_source),
                "unique_plates": len({entry.plate for entry in entries_for_source}),
            }
        )

    local_entries = load_local_hotlist_entries()
    for entry in local_entries:
        grouped[entry.plate].append(entry)

    source_details.append(
        {
            "source_id": "local-overlay",
            "source_kind": "local",
            "list_type": None,
            "label": "Local Overlay",
            "file_name": HOTLIST_LOCAL_OVERLAY_PATH.name,
            "exists": HOTLIST_LOCAL_OVERLAY_PATH.exists() and HOTLIST_LOCAL_OVERLAY_PATH.is_file(),
            "updated_utc": hotlist_file_updated_utc(HOTLIST_LOCAL_OVERLAY_PATH)
            if HOTLIST_LOCAL_OVERLAY_PATH.exists() and HOTLIST_LOCAL_OVERLAY_PATH.is_file()
            else None,
            "last_import_utc": manifest.get("local_overlay", {}).get("last_updated_utc"),
            "record_count": len(local_entries),
            "unique_plates": len({entry.plate for entry in local_entries}),
        }
    )

    hotlists = {}
    runtime_hotlists: Dict[str, List[HotlistEntry]] = {}
    total_entries = 0
    for plate, entries in grouped.items():
        deduped = dedupe_hotlist_entries(entries)
        hotlists[plate] = sorted(
            deduped,
            key=lambda item: (
                HOTLIST_PRIORITY.get(item.list_type, 0),
                1 if item.source_kind == "agency" else 0,
                item.created_utc or "",
            ),
            reverse=True,
        )
        total_entries += len(hotlists[plate])

        runtime_key = hotlist_runtime_plate_key(plate)
        runtime_hotlists.setdefault(runtime_key, []).extend(hotlists[plate])

    for runtime_key, entries in list(runtime_hotlists.items()):
        runtime_hotlists[runtime_key] = sorted(
            dedupe_hotlist_entries(entries),
            key=lambda item: (
                HOTLIST_PRIORITY.get(item.list_type, 0),
                1 if item.source_kind == "agency" else 0,
                item.created_utc or "",
            ),
            reverse=True,
        )

    snapshot = {
        "status": "ok",
        "unique_plates": len(hotlists),
        "total_entries": total_entries,
        "agency_entry_count": agency_entries,
        "local_entry_count": len(local_entries),
        "sources": list(HOTLIST_FILENAMES),
        "hotlist_root": str(HOTLIST_ROOT),
        "last_reload_utc": manifest.get("last_reload_utc"),
        "source_details": source_details,
    }
    return hotlists, runtime_hotlists, local_entries, snapshot


def refresh_hotlist_state(update_manifest_reload: bool = False) -> dict:
    global hotlist_by_plate, hotlist_by_runtime_plate, local_hotlist_entries, hotlist_status_snapshot

    manifest = load_hotlist_manifest()
    if update_manifest_reload:
        manifest["last_reload_utc"] = utc_now_iso()
        save_hotlist_manifest(manifest)

    hotlist_by_plate, hotlist_by_runtime_plate, local_hotlist_entries, hotlist_status_snapshot = build_hotlist_state()
    return hotlist_status_snapshot


def load_hotlists() -> Dict[str, List[HotlistEntry]]:
    hotlists, _, _, _ = build_hotlist_state()
    return hotlists


def list_hotlist_entries(source_kind: str = "all") -> List[HotlistEntry]:
    entries: List[HotlistEntry] = []
    for plate_entries in hotlist_by_plate.values():
        for entry in plate_entries:
            if source_kind != "all" and entry.source_kind != source_kind:
                continue
            entries.append(entry)

    return sorted(
        dedupe_hotlist_entries(entries),
        key=lambda item: (
            item.plate,
            -HOTLIST_PRIORITY.get(item.list_type, 0),
            item.source_kind,
            item.entry_date,
        ),
    )


def hotlist_totals() -> tuple:
    total_entries = hotlist_status_snapshot.get("total_entries")
    unique_plates = hotlist_status_snapshot.get("unique_plates")
    if isinstance(unique_plates, int) and isinstance(total_entries, int):
        return unique_plates, total_entries
    total_entries = sum(len(entries) for entries in hotlist_by_plate.values())
    return len(hotlist_by_plate), total_entries


def store_hotlist_upload(filename: str, text: str) -> None:
    ensure_hotlist_root()
    write_text_atomic(HOTLIST_ROOT / filename, text)


def read_upload_text(upload: UploadFile) -> str:
    data = upload.file.read(HOTLIST_UPLOAD_MAX_BYTES + 1)
    if len(data) > HOTLIST_UPLOAD_MAX_BYTES:
        limit_mb = HOTLIST_UPLOAD_MAX_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"{upload.filename or 'upload'} exceeds size limit ({limit_mb:.1f} MiB)",
        )
    return data.decode("utf-8", errors="replace")


def save_local_entries_and_refresh(entries: List[HotlistEntry]) -> dict:
    manifest = load_hotlist_manifest()
    manifest["local_overlay"]["last_updated_utc"] = utc_now_iso()
    save_local_hotlist_entries(entries)
    save_hotlist_manifest(manifest)
    return refresh_hotlist_state(update_manifest_reload=True)


def upsert_local_hotlist_entry(entry: HotlistEntry) -> tuple:
    current_entries = load_local_hotlist_entries()
    identity = entry_identity(entry)
    for existing in current_entries:
        if entry_identity(existing) == identity:
            return existing, False
    current_entries.append(entry)
    save_local_entries_and_refresh(current_entries)
    return entry, True


def hotlist_matches_for_plate(plate: str) -> List[HotlistEntry]:
    matches = hotlist_by_plate.get(plate, [])
    if matches:
        return matches
    return hotlist_by_runtime_plate.get(hotlist_runtime_plate_key(plate), [])


def hotlist_alert_plate_key(event: LiveEvent) -> str:
    if event.hotlist_entries:
        best_entry = max(event.hotlist_entries, key=lambda item: HOTLIST_PRIORITY.get(item.list_type, 0))
        return hotlist_runtime_plate_key(best_entry.plate)
    return hotlist_runtime_plate_key(event.plate)


def enrich_with_hotlist(event: LiveEvent) -> LiveEvent:
    matches = hotlist_matches_for_plate(event.plate)

    if not matches:
        event.hotlist_hit = False
        event.hotlist_entries = []
        event.hotlist_highest_label = None
        event.hotlist_highest_type = None
        event.hotlist_highest_priority = None
        event.hotlist_alert = False
        event.hotlist_alert_reason = None
        event.hotlist_alert_cooldown_until_utc = None
        return event

    event.hotlist_hit = True
    event.hotlist_entries = list(matches)

    best = max(matches, key=lambda item: HOTLIST_PRIORITY.get(item.list_type, 0))
    event.hotlist_highest_label = best.list_label
    event.hotlist_highest_type = best.list_type
    event.hotlist_highest_priority = HOTLIST_PRIORITY.get(best.list_type, 0)
    return event


def utc_iso(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def live_display_id(event: LiveEvent) -> str:
    if event.track_id_valid and event.track_id > 0:
        return f"{event.case_id}:{event.source.value}:{event.track_id}"
    if event.plate:
        return f"{event.case_id}:{event.source.value}:{event.plate}"
    return event.event_id


def event_sort_value(event: LiveEvent) -> tuple:
    status_rank = 2 if event.status == EventStatus.LOCKED else 1
    timestamp = parse_utc_iso(event.timestamp_utc) or datetime.fromtimestamp(0, tz=timezone.utc)
    return (status_rank, event.confidence, timestamp, event.frame_number)


def merge_live_event(existing: Optional[LiveEvent], incoming: LiveEvent) -> LiveEvent:
    incoming.display_id = live_display_id(incoming)
    if existing is None:
        return incoming

    best = incoming if event_sort_value(incoming) >= event_sort_value(existing) else existing
    merged = best.model_copy(deep=True)
    merged.display_id = incoming.display_id

    if not merged.full_frame_path:
        merged.full_frame_path = incoming.full_frame_path or existing.full_frame_path
    if not merged.annotated_frame_path:
        merged.annotated_frame_path = incoming.annotated_frame_path or existing.annotated_frame_path
    if not merged.plate_crop_path:
        merged.plate_crop_path = incoming.plate_crop_path or existing.plate_crop_path
    if not merged.vehicle_crop_path:
        merged.vehicle_crop_path = incoming.vehicle_crop_path or existing.vehicle_crop_path
    if not merged.vehicle_make:
        merged.vehicle_make = incoming.vehicle_make or existing.vehicle_make
    if not merged.vehicle_type:
        merged.vehicle_type = incoming.vehicle_type or existing.vehicle_type
    if not merged.vehicle_color:
        merged.vehicle_color = incoming.vehicle_color or existing.vehicle_color
    if not merged.vehicle_color and (merged.vehicle_make or merged.vehicle_type):
        merged.vehicle_color = "Unknown"
    if not merged.gps_fix_valid:
        preferred = incoming if incoming.gps_fix_valid else existing
        if preferred.gps_fix_valid:
            merged.gps_fix_valid = True
            merged.gps_latitude = preferred.gps_latitude
            merged.gps_longitude = preferred.gps_longitude
            merged.gps_altitude_m = preferred.gps_altitude_m
            merged.gps_speed_knots = preferred.gps_speed_knots
            merged.gps_timestamp_utc = preferred.gps_timestamp_utc
    if merged.gps_fix_valid and not merged.gps_timestamp_utc:
        merged.gps_timestamp_utc = incoming.gps_timestamp_utc or existing.gps_timestamp_utc

    return merged


def upsert_live_event(event: LiveEvent) -> None:
    identity = event.display_id or event.event_id
    filtered = [item for item in live_events if (item.display_id or item.event_id) != identity]
    filtered.append(event)
    if len(filtered) > MAX_LIVE_EVENTS:
        filtered = filtered[-MAX_LIVE_EVENTS:]
    live_events[:] = filtered


def prune_hotlist_runtime_state(now: datetime) -> None:
    stale_plates = [plate for plate, state in suppressed_hotlist_plates.items() if state[0] <= now]
    for plate in stale_plates:
        suppressed_hotlist_plates.pop(plate, None)

    window_start = now - timedelta(seconds=HOTLIST_AUTO_SNOOZE_WINDOW_SECONDS)
    stale_sighting_plates = []
    for plate, sightings in recent_hotlist_sightings.items():
        while sightings and sightings[0] < window_start:
            sightings.popleft()
        if not sightings:
            stale_sighting_plates.append(plate)

    for plate in stale_sighting_plates:
        recent_hotlist_sightings.pop(plate, None)


def suppress_plate(plate: str, reason: str, seconds: int) -> datetime:
    until = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    suppressed_hotlist_plates[plate] = (until, reason)
    recent_hotlist_sightings.pop(plate, None)
    return until


def apply_hotlist_alert_policy(event: LiveEvent) -> LiveEvent:
    if not event.hotlist_hit:
        event.hotlist_alert = False
        event.hotlist_alert_reason = None
        event.hotlist_alert_cooldown_until_utc = None
        return event

    now = datetime.now(timezone.utc)
    prune_hotlist_runtime_state(now)
    alert_plate_key = hotlist_alert_plate_key(event)

    if event.status != EventStatus.LOCKED:
        event.hotlist_alert = False
        event.hotlist_alert_reason = "locked-only"
        event.hotlist_alert_cooldown_until_utc = None
        return event

    suppression_state = suppressed_hotlist_plates.get(alert_plate_key)
    if suppression_state and suppression_state[0] > now:
        event.hotlist_alert = False
        event.hotlist_alert_reason = suppression_state[1]
        event.hotlist_alert_cooldown_until_utc = utc_iso(suppression_state[0])
        return event

    sightings = recent_hotlist_sightings[alert_plate_key]
    sightings.append(now)
    window_start = now - timedelta(seconds=HOTLIST_AUTO_SNOOZE_WINDOW_SECONDS)
    while sightings and sightings[0] < window_start:
        sightings.popleft()

    if len(sightings) >= HOTLIST_AUTO_SNOOZE_COUNT:
        suppression_until = suppress_plate(alert_plate_key, "auto-snooze", HOTLIST_ACKNOWLEDGE_SECONDS)
        event.hotlist_alert = False
        event.hotlist_alert_reason = "auto-snooze"
        event.hotlist_alert_cooldown_until_utc = utc_iso(suppression_until)
        return event

    event.hotlist_alert = True
    event.hotlist_alert_reason = "new-alert"
    event.hotlist_alert_cooldown_until_utc = None
    return event


refresh_hotlist_state(update_manifest_reload=False)


def read_pid(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def pid_is_alive(pid: Optional[int]) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def alpr_startup_grace_active(alpr_pid: Optional[int]) -> bool:
    if time.monotonic() - backend_started_at <= ALPR_START_GRACE_SECONDS:
        return True

    try:
        pid_mtime = ALPR_PID_PATH.stat().st_mtime
    except OSError:
        return False

    return (time.time() - pid_mtime) <= ALPR_START_GRACE_SECONDS


def read_runtime_status() -> dict:
    try:
        with ALPR_STATUS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def default_source_map() -> dict:
    mapping = {
        "source_0": "LF",
        "source_1": "RF",
        "source_2": "LR",
        "source_3": "RR",
    }
    mapping.update(load_config_source_map())
    raw_map = os.getenv("ALPR_LIVE_SOURCE_MAP", "")
    for entry in raw_map.split(","):
        item = entry.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip().upper()
    return mapping


def source_display(raw_source: str) -> tuple:
    code = default_source_map().get(raw_source, raw_source)
    try:
        label = SOURCE_LABELS[CameraSource(code)]
    except Exception:
        label = raw_source or code
    return code, label


def dashboard_bind_host() -> str:
    value = os.getenv("ALPR_DASHBOARD_HOST", "0.0.0.0").strip()
    return value or "0.0.0.0"


def dashboard_bind_port() -> int:
    raw_value = os.getenv("ALPR_DASHBOARD_PORT", "8080").strip()
    try:
        value = int(raw_value)
    except ValueError:
        return 8080
    return value if value > 0 else 8080


def run_command_lines(args: List[str]) -> List[str]:
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def network_medium_for_interface(interface_name: str) -> str:
    lowered = (interface_name or "").lower()
    if lowered.startswith(("wl", "wlan")):
        return "wifi"
    if lowered.startswith(("en", "eth")):
        return "ethernet"
    return "network"


def ignore_network_interface(interface_name: str) -> bool:
    lowered = (interface_name or "").lower()
    return lowered.startswith(("lo", "docker", "br-", "veth", "virbr", "zt", "tailscale", "tun", "tap"))


def nmcli_connection_field(connection_name: str, field_name: str) -> Optional[str]:
    if not connection_name:
        return None
    lines = run_command_lines(["nmcli", "-g", field_name, "connection", "show", connection_name])
    if not lines:
        return None
    value = lines[0].strip()
    return value or None


def describe_network_connection(interface_name: str) -> Optional[str]:
    lines = run_command_lines(["nmcli", "-g", "GENERAL.CONNECTION", "device", "show", interface_name])
    if not lines:
        return None

    connection_name = lines[0].strip()
    if not connection_name or connection_name == "--":
        return None

    medium = network_medium_for_interface(interface_name)
    if medium == "wifi":
        mode = (nmcli_connection_field(connection_name, "802-11-wireless.mode") or "").strip().lower()
        ssid = nmcli_connection_field(connection_name, "802-11-wireless.ssid")
        if mode == "ap":
            return f"Hotspot {ssid}" if ssid else "Hotspot"
        if ssid:
            return f"WiFi {ssid}"
        return f"WiFi {connection_name}"

    if medium == "ethernet":
        return f"Ethernet {connection_name}"

    return connection_name


def collect_network_access() -> List[dict]:
    port = dashboard_bind_port()
    bind_host = dashboard_bind_host()
    entries: List[dict] = []
    seen = set()

    def add_entry(interface_name: Optional[str], medium: str, address: str, note: Optional[str], priority: int) -> None:
        normalized_address = (address or "").strip()
        if not normalized_address:
            return
        key = (interface_name or "", normalized_address)
        if key in seen:
            return
        seen.add(key)
        entries.append(
            {
                "interface": interface_name,
                "medium": medium,
                "address": normalized_address,
                "url": f"http://{normalized_address}:{port}",
                "note": note,
                "priority": priority,
            }
        )

    add_entry("lo", "local", "127.0.0.1", "Local device only", 0)

    if bind_host not in {"", "0.0.0.0", "::", "*"} and bind_host.lower() not in {"localhost"}:
        add_entry(None, "bound", bind_host, "Configured dashboard bind address", 5)
        return [
            {key: value for key, value in entry.items() if key != "priority"}
            for entry in sorted(entries, key=lambda item: (item["priority"], item["interface"] or "", item["address"]))
        ]

    try:
        result = subprocess.run(
            ["ip", "-j", "-4", "addr", "show", "up"],
            check=True,
            capture_output=True,
            text=True,
        )
        addresses = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        addresses = []

    for item in addresses:
        if not isinstance(item, dict):
            continue
        interface_name = str(item.get("ifname") or "").strip()
        if not interface_name or ignore_network_interface(interface_name):
            continue

        medium = network_medium_for_interface(interface_name)
        note = describe_network_connection(interface_name)
        priority = 10 if medium == "wifi" else 20 if medium == "ethernet" else 30
        for addr in item.get("addr_info", []):
            if not isinstance(addr, dict) or addr.get("family") != "inet":
                continue
            address = str(addr.get("local") or "").strip()
            if not address or address.startswith("127."):
                continue
            add_entry(interface_name, medium, address, note, priority)

    return [
        {key: value for key, value in entry.items() if key != "priority"}
        for entry in sorted(entries, key=lambda item: (item["priority"], item["interface"] or "", item["address"]))
    ]


def latest_case_dir() -> Optional[Path]:
    if not EVIDENCE_ROOT.exists():
        return None
    case_dirs = [path for path in EVIDENCE_ROOT.iterdir() if path.is_dir() and path.name.startswith("case_")]
    if not case_dirs:
        return None
    return max(case_dirs, key=lambda path: path.stat().st_mtime)


def count_jsonl_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except (FileNotFoundError, OSError):
        return 0


def read_last_jsonl_record(path: Path) -> Optional[dict]:
    try:
        last_line = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_line = line
        return json.loads(last_line) if last_line else None
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def collect_source_health(runtime_status: dict, alpr_process_alive: bool) -> List[dict]:
    now = datetime.now(timezone.utc)
    health_by_source = {}

    for source in CameraSource:
        health_by_source[source.value] = {
            "source": source.value,
            "source_label": SOURCE_LABELS[source],
            "process_source_name": None,
            "available": False,
            "last_seen_utc": None,
            "last_frame_number": None,
            "live_event_count": 0,
            "live_last_event_time_utc": None,
        }

    for event in live_events:
        source_key = event.source.value
        entry = health_by_source.setdefault(
            source_key,
            {
                "source": source_key,
                "source_label": event.source_label or source_key,
                "process_source_name": None,
                "available": False,
                "last_seen_utc": None,
                "last_frame_number": None,
                "live_event_count": 0,
                "live_last_event_time_utc": None,
            },
        )
        entry["live_event_count"] += 1
        if not entry["live_last_event_time_utc"] or event.timestamp_utc > entry["live_last_event_time_utc"]:
            entry["live_last_event_time_utc"] = event.timestamp_utc

    for item in runtime_status.get("sources", []):
        if not isinstance(item, dict):
            continue
        raw_source = str(item.get("source") or item.get("name") or "").strip()
        if not raw_source:
            continue
        source_code, label = source_display(raw_source)
        entry = health_by_source.setdefault(
            source_code,
            {
                "source": source_code,
                "source_label": label,
                "process_source_name": raw_source,
                "available": False,
                "last_seen_utc": None,
                "last_frame_number": None,
                "live_event_count": 0,
                "live_last_event_time_utc": None,
            },
        )
        entry["source_label"] = label
        entry["process_source_name"] = raw_source
        entry["last_seen_utc"] = item.get("last_seen_utc")
        entry["last_frame_number"] = item.get("last_frame_number")
        seen_at = parse_utc_iso(entry["last_seen_utc"])
        entry["available"] = bool(
            alpr_process_alive
            and seen_at is not None
            and (now - seen_at).total_seconds() <= SOURCE_STALE_SECONDS
        )

    return sorted(health_by_source.values(), key=lambda entry: entry["source"])


def collect_gps_status() -> dict:
    snapshot = dict(latest_gps_fix) if isinstance(latest_gps_fix, dict) else {}
    received_utc = snapshot.get("gps_received_utc")
    received_at = parse_utc_iso(received_utc if isinstance(received_utc, str) else None)
    fresh = bool(
        received_at is not None
        and (datetime.now(timezone.utc) - received_at).total_seconds() <= 15
    )
    latitude = snapshot.get("gps_latitude")
    longitude = snapshot.get("gps_longitude")
    fix_valid = bool(
        snapshot.get("gps_fix_valid")
        and isinstance(latitude, (int, float))
        and isinstance(longitude, (int, float))
    )
    return {
        "source": snapshot.get("gps_source") or "mifi-tcp",
        "endpoint": snapshot.get("gps_endpoint") or f"{GPS_MIFI_HOST}:{GPS_MIFI_PORT}",
        "connected": fresh,
        "fix_valid": fix_valid,
        "last_received_utc": received_utc,
        "gps_timestamp_utc": snapshot.get("gps_timestamp_utc"),
        "latitude": latitude if fix_valid else None,
        "longitude": longitude if fix_valid else None,
        "speed_knots": snapshot.get("gps_speed_knots") if fix_valid else None,
    }


def resolve_current_case_id(runtime_status: dict) -> Optional[str]:
    current_case = runtime_status.get("current_case_id")
    if isinstance(current_case, str) and current_case:
        return current_case
    case_dir = latest_case_dir()
    return case_dir.name if case_dir else None


def resolve_last_event_time(case_id: Optional[str]) -> Optional[str]:
    latest_time = None
    for event in live_events:
        if latest_time is None or event.timestamp_utc > latest_time:
            latest_time = event.timestamp_utc

    if case_id:
        record = read_last_jsonl_record(EVIDENCE_ROOT / case_id / "events.jsonl")
        if record:
            disk_time = record.get("timestamp_utc")
            if isinstance(disk_time, str) and (latest_time is None or disk_time > latest_time):
                latest_time = disk_time

    return latest_time


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def runtime_status_is_fresh(runtime_status: dict) -> bool:
    updated_at = parse_utc_iso(runtime_status.get("updated_utc"))
    if updated_at is None:
        return False
    return (datetime.now(timezone.utc) - updated_at).total_seconds() <= ALPR_RUNTIME_STALE_SECONDS


def runtime_sources_are_fresh(runtime_status: dict) -> bool:
    now = datetime.now(timezone.utc)
    for item in runtime_status.get("sources", []):
        if not isinstance(item, dict):
            continue
        seen_at = parse_utc_iso(item.get("last_seen_utc"))
        if seen_at is not None and (now - seen_at).total_seconds() <= ALPR_RUNTIME_STALE_SECONDS:
            return True
    return False


def live_events_are_fresh() -> bool:
    now = datetime.now(timezone.utc)
    for event in reversed(live_events):
        seen_at = parse_utc_iso(event.timestamp_utc)
        if seen_at is not None and (now - seen_at).total_seconds() <= ALPR_EVENT_STALE_SECONDS:
            return True
    return False


def log_watchdog(message: str) -> None:
    print(f"[{utc_now_iso()}] {message}", flush=True)


def remove_pid_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def stop_alpr_process(pid: Optional[int], timeout_seconds: float = 10.0) -> None:
    if pid and pid > 0 and pid_is_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if not pid_is_alive(pid):
                break
            time.sleep(0.25)

        if pid_is_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    remove_pid_file(ALPR_PID_PATH)
    try:
        ALPR_STATUS_PATH.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def start_alpr_process() -> Optional[int]:
    config_path = active_config_path()
    if config_path is None:
        log_watchdog("ALPR watchdog skipped restart because no active config path is available")
        return None

    binary_path = APP_ROOT / "deepstream-lpr-app"
    if not binary_path.exists():
        log_watchdog(f"ALPR watchdog skipped restart because binary is missing: {binary_path}")
        return None

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("ALPR_LIVE_ENDPOINT", f"http://127.0.0.1:{dashboard_bind_port()}/api/live-event")

    with (LOG_ROOT / "alpr.log").open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{utc_now_iso()}] ALPR watchdog restarting worker\n")
        log_handle.flush()
        process = subprocess.Popen(
            [str(binary_path), str(config_path)],
            cwd=str(APP_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    ALPR_PID_PATH.write_text(str(process.pid), encoding="utf-8")
    return process.pid


# ── MiFi GPS reader ──────────────────────────────────────────────────────────

def _parse_nmea_dddmm(raw: str, hemi: str) -> Optional[float]:
    """Convert NMEA DDDMM.mmmmm + hemisphere letter to signed decimal degrees."""
    try:
        raw = raw.strip()
        if not raw:
            return None
        dot = raw.index(".")
        degrees = float(raw[: dot - 2])
        minutes = float(raw[dot - 2 :])
        decimal = degrees + minutes / 60.0
        if hemi.upper() in ("S", "W"):
            decimal = -decimal
        return round(decimal, 7)
    except (ValueError, IndexError):
        return None


def _parse_gprmc(sentence: str) -> Optional[dict]:
    """
    Parse a $GPRMC sentence and return a GPS fix dict, or None if invalid/no-fix.
    Fields: lat, lon, speed_knots, timestamp_utc.
    """
    try:
        # Strip checksum if present
        if "*" in sentence:
            sentence = sentence[: sentence.index("*")]
        parts = sentence.strip().split(",")
        # $GPRMC,hhmmss.ss,A,ddmm.mmmm,N,dddmm.mmmm,W,speed,course,ddmmyy,,,
        if len(parts) < 8:
            return None
        status = parts[2].strip().upper()
        if status != "A":
            return None  # void / no fix

        lat = _parse_nmea_dddmm(parts[3], parts[4])
        lon = _parse_nmea_dddmm(parts[5], parts[6])
        if lat is None or lon is None:
            return None

        speed_str = parts[7].strip()
        speed_knots = float(speed_str) if speed_str else None

        # Build UTC timestamp from time+date fields
        gps_ts = None
        time_field = parts[1].strip()
        date_field = parts[9].strip() if len(parts) > 9 else ""
        if len(time_field) >= 6 and len(date_field) == 6:
            try:
                hh, mm, ss = int(time_field[0:2]), int(time_field[2:4]), int(time_field[4:6])
                day, month, year = int(date_field[0:2]), int(date_field[2:4]), int(date_field[4:6]) + 2000
                gps_ts = datetime(year, month, day, hh, mm, ss, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            except (ValueError, OverflowError):
                pass

        return {
            "gps_fix_valid": True,
            "gps_latitude": lat,
            "gps_longitude": lon,
            "gps_speed_knots": speed_knots,
            "gps_timestamp_utc": gps_ts,
        }
    except Exception:
        return None


async def gps_mifi_reader_loop() -> None:
    """Background task: stream NMEA from MiFi TCP port and cache the latest fix."""
    global latest_gps_fix
    while True:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(GPS_MIFI_HOST, GPS_MIFI_PORT),
                timeout=10.0,
            )
            try:
                buffer = ""
                while True:
                    raw_chunk = await asyncio.wait_for(reader.read(4096), timeout=15.0)
                    if not raw_chunk:
                        break
                    buffer += raw_chunk.decode("ascii", errors="replace")
                    lines = re.split(r"\r\n|\n|\r", buffer)
                    if not lines:
                        continue
                    buffer = lines.pop()
                    for line in lines:
                        line = line.replace("\x00", "").strip()
                        if not line.startswith("$GPRMC"):
                            continue
                        fix = _parse_gprmc(line)
                        if fix:
                            latest_gps_fix = {
                                **fix,
                                "gps_received_utc": utc_now_iso(),
                                "gps_source": "mifi-tcp",
                                "gps_endpoint": f"{GPS_MIFI_HOST}:{GPS_MIFI_PORT}",
                            }
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        await asyncio.sleep(GPS_RECONNECT_DELAY_SECONDS)


async def alpr_watchdog_loop() -> None:
    global alpr_watchdog_last_restart_monotonic

    while True:
        await asyncio.sleep(ALPR_WATCHDOG_INTERVAL_SECONDS)

        try:
            runtime_status = read_runtime_status()
            alpr_pid = read_pid(ALPR_PID_PATH)

            if alpr_startup_grace_active(alpr_pid):
                continue

            if effective_alpr_process_alive(alpr_pid, runtime_status):
                continue

            if time.monotonic() - alpr_watchdog_last_restart_monotonic < ALPR_WATCHDOG_COOLDOWN_SECONDS:
                continue

            alpr_watchdog_last_restart_monotonic = time.monotonic()
            log_watchdog("ALPR watchdog detected stale or dead runtime, restarting worker")
            await asyncio.to_thread(stop_alpr_process, alpr_pid)
            restarted_pid = await asyncio.to_thread(start_alpr_process)
            if restarted_pid:
                log_watchdog(f"ALPR watchdog restarted worker with pid {restarted_pid}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_watchdog(f"ALPR watchdog error: {exc}")


@app.on_event("startup")
async def start_alpr_watchdog() -> None:
    global alpr_watchdog_task
    ensure_event_db()
    if alpr_watchdog_task is None or alpr_watchdog_task.done():
        alpr_watchdog_task = asyncio.create_task(alpr_watchdog_loop())


@app.on_event("startup")
async def start_gps_reader() -> None:
    global gps_reader_task
    if gps_reader_task is None or gps_reader_task.done():
        gps_reader_task = asyncio.create_task(gps_mifi_reader_loop())


@app.on_event("shutdown")
async def stop_alpr_watchdog() -> None:
    global alpr_watchdog_task
    if alpr_watchdog_task is None:
        return
    alpr_watchdog_task.cancel()
    try:
        await alpr_watchdog_task
    except asyncio.CancelledError:
        pass
    alpr_watchdog_task = None


@app.on_event("shutdown")
async def stop_gps_reader() -> None:
    global gps_reader_task
    if gps_reader_task is None:
        return
    gps_reader_task.cancel()
    try:
        await gps_reader_task
    except asyncio.CancelledError:
        pass
    gps_reader_task = None


def effective_alpr_process_alive(alpr_pid: Optional[int], runtime_status: dict) -> bool:
    runtime_fresh = (
        runtime_status_is_fresh(runtime_status)
        or runtime_sources_are_fresh(runtime_status)
        or live_events_are_fresh()
    )
    process_alive = pid_is_alive(alpr_pid)

    if process_alive:
        return runtime_fresh or alpr_startup_grace_active(alpr_pid)

    return runtime_fresh


async def broadcast_event(event: LiveEvent, audio_cue: Optional[str] = None) -> None:
    dead_clients: List[WebSocket] = []
    payload = event.model_dump()
    if audio_cue:
        payload["audio_cue"] = audio_cue

    for websocket in clients:
        try:
            await websocket.send_json(payload)
        except Exception:
            dead_clients.append(websocket)

    for websocket in dead_clients:
        clients.discard(websocket)


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    index_path = STATIC_ROOT / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return """
    <!doctype html>
    <html>
      <head><meta charset="utf-8"><title>ALPR Field Unit</title></head>
      <body>
        <h1>ALPR Field Unit</h1>
        <p>Backend is running.</p>
        <p>Create static/index.html next.</p>
      </body>
    </html>
    """


@app.get("/config", response_class=HTMLResponse)
async def config_page() -> str:
    config_path = STATIC_ROOT / "config.html"
    if config_path.exists():
        return config_path.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="config page not found")


@app.get("/hotlists", response_class=HTMLResponse)
async def hotlists_page() -> str:
    hotlists_path = STATIC_ROOT / "hotlists.html"
    if hotlists_path.exists():
        return hotlists_path.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="hotlists page not found")


@app.get("/events", response_class=HTMLResponse)
async def events_page() -> str:
    events_path = STATIC_ROOT / "events.html"
    if events_path.exists():
        return events_path.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="events page not found")


@app.get("/api/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    runtime_status = read_runtime_status()
    alpr_pid = read_pid(ALPR_PID_PATH)
    alpr_process_alive = effective_alpr_process_alive(alpr_pid, runtime_status)
    backend_pid = os.getpid()
    try:
        BACKEND_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
        BACKEND_PID_PATH.write_text(str(backend_pid), encoding="utf-8")
    except OSError:
        pass

    current_case_id = resolve_current_case_id(runtime_status)
    current_case_event_count = 0
    if current_case_id:
        current_case_event_count = count_jsonl_rows(EVIDENCE_ROOT / current_case_id / "events.jsonl")

    disk = shutil.disk_usage(EVIDENCE_ROOT if EVIDENCE_ROOT.exists() else Path("."))

    return StatusResponse(
        now_utc=utc_now_iso(),
        live_event_count=len(live_events),
        current_case_id=current_case_id,
        current_case_event_count=current_case_event_count,
        last_event_time_utc=resolve_last_event_time(current_case_id),
        alpr_process_alive=alpr_process_alive,
        alpr_pid=alpr_pid,
        backend_pid=backend_pid,
        runtime_status_utc=runtime_status.get("updated_utc"),
        storage_free_bytes=disk.free,
        storage_free_gb=round(disk.free / (1024 ** 3), 2),
        evidence_root=str(EVIDENCE_ROOT),
        available_sources=[source.value for source in CameraSource],
        source_health=collect_source_health(runtime_status, alpr_process_alive),
        network_access=collect_network_access(),
        gps_status=collect_gps_status(),
        backend_log_path=str(LOG_ROOT / "backend.log"),
        alpr_log_path=str(LOG_ROOT / "alpr.log"),
    )


@app.get("/api/camera-config")
async def get_camera_config() -> dict:
    runtime_status = read_runtime_status()
    alpr_pid = read_pid(ALPR_PID_PATH)
    alpr_process_alive = effective_alpr_process_alive(alpr_pid, runtime_status)

    config_path = active_config_path()
    return {
        "status": "ok",
        "current_case_id": resolve_current_case_id(runtime_status),
        "runtime_status_utc": runtime_status.get("updated_utc"),
        "config_path": str(config_path) if config_path else None,
        "alpr_process_alive": alpr_process_alive,
        "network_access": collect_network_access(),
        "sources": camera_config_sources(runtime_status, alpr_process_alive),
        "presets": [
            {
                "id": preset["id"],
                "label": preset["label"],
                "description": preset["description"],
            }
            for preset in CAMERA_PRESETS
        ],
    }


@app.post("/api/camera-control")
async def set_camera_control(request: CameraControlRequest) -> dict:
    runtime_status = read_runtime_status()
    _, device_path = resolve_camera_device(request.source, runtime_status)

    try:
        apply_v4l2_control(device_path, request.control, request.value)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "failed to set camera control"
        raise HTTPException(status_code=400, detail=detail)

    return {
        "status": "ok",
        "source": request.source,
        "device_path": device_path,
        "control": request.control,
        "value": request.value,
    }


@app.post("/api/camera-preset")
async def apply_camera_preset(request: CameraPresetRequest) -> dict:
    runtime_status = read_runtime_status()
    _, device_path = resolve_camera_device(request.source, runtime_status)

    preset = next((item for item in CAMERA_PRESETS if item["id"] == request.preset), None)
    if preset is None:
        raise HTTPException(status_code=404, detail="camera preset not found")

    device_state = get_v4l2_device_state(device_path)
    available_controls = {item["name"] for item in device_state.get("camera_controls", [])}
    applied_controls = []
    missing_controls = []

    try:
        for control_name, value in preset["controls"]:
            if control_name not in available_controls:
                missing_controls.append(control_name)
                continue
            apply_v4l2_control(device_path, control_name, value)
            applied_controls.append({"name": control_name, "value": value})
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "failed to apply camera preset"
        raise HTTPException(status_code=400, detail=detail)

    return {
        "status": "ok",
        "source": request.source,
        "device_path": device_path,
        "preset": request.preset,
        "applied_controls": applied_controls,
        "missing_controls": missing_controls,
    }


@app.get("/api/camera-preview/{source_name}")
async def get_camera_preview(source_name: str):
    runtime_status = read_runtime_status()
    alpr_pid = read_pid(ALPR_PID_PATH)
    source_uris = load_config_source_uris()
    raw_source = resolve_raw_source_name(source_name, runtime_status, source_uris)
    runtime_source_item = {}
    if raw_source is not None:
        for item in runtime_status.get("sources", []):
            if isinstance(item, dict) and str(item.get("source") or item.get("name") or "").strip() == raw_source:
                runtime_source_item = item
                break

    if pid_is_alive(alpr_pid) and runtime_item_is_fresh(runtime_source_item):
        raise HTTPException(
            status_code=409,
            detail="direct camera preview is unavailable while ALPR runtime preview is healthy",
        )
    if alpr_startup_grace_active(alpr_pid):
        raise HTTPException(
            status_code=409,
            detail="direct camera preview is unavailable while ALPR is starting",
        )
    _, device_path = resolve_camera_device(source_name, runtime_status)
    jpeg_bytes = capture_direct_preview(device_path)
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, max-age=0"},
    )


@app.post("/api/camera-source")
async def add_camera_source(request: AddCameraRequest) -> dict:
    config_path = active_config_path()
    if config_path is None:
        raise HTTPException(status_code=503, detail="No active config file is set.")

    registry_entries = load_camera_registry(config_path)
    uri_exists = any(str(item.get("uri") or "").strip().lower() == request.uri.strip().lower() for item in registry_entries)
    if uri_exists:
        raise HTTPException(status_code=409, detail="camera URI already exists")

    name_exists = any(str(item.get("name") or "").strip().upper() == request.name.strip().upper() for item in registry_entries)
    if name_exists:
        raise HTTPException(status_code=409, detail="camera name already exists")

    new_key = f"camera_{uuid.uuid4().hex[:8]}"
    registry_entries.append(
        {
            "camera_id": new_key,
            "uri": request.uri,
            "name": request.name,
            "enabled": True,
        }
    )
    saved = save_camera_registry(registry_entries, config_path)
    apply_registry_to_active_config(saved, config_path)

    return {
        "status": "ok",
        "source_key": new_key,
        "uri": request.uri,
        "name": request.name,
        "restart_required": True,
    }


@app.delete("/api/camera-source/{source_key}")
async def remove_camera_source(source_key: str) -> dict:
    config_path = active_config_path()
    if config_path is None:
        raise HTTPException(status_code=503, detail="No active config file is set.")

    registry_entries = load_camera_registry(config_path)
    remaining = [item for item in registry_entries if str(item.get("camera_id") or "") != source_key]
    if len(remaining) == len(registry_entries):
        raise HTTPException(status_code=404, detail=f"Source '{source_key}' not found in config.")

    saved = save_camera_registry(remaining, config_path)
    apply_registry_to_active_config(saved, config_path)

    return {
        "status": "ok",
        "removed": source_key,
        "restart_required": True,
    }


@app.patch("/api/camera-source/{source_key}")
async def rename_camera_source(source_key: str, request: RenameCameraRequest) -> dict:
    config_path = active_config_path()
    if config_path is None:
        raise HTTPException(status_code=503, detail="No active config file is set.")

    registry_entries = load_camera_registry(config_path)
    found = False
    requested_name = request.name.strip().upper()

    for item in registry_entries:
        if str(item.get("camera_id") or "") != source_key:
            continue
        found = True
        continue

    if not found:
        raise HTTPException(status_code=404, detail=f"Source '{source_key}' not found in config.")

    name_exists = any(
        str(item.get("camera_id") or "") != source_key
        and str(item.get("name") or "").strip().upper() == requested_name
        for item in registry_entries
    )
    if name_exists:
        raise HTTPException(status_code=409, detail="camera name already exists")

    for item in registry_entries:
        if str(item.get("camera_id") or "") == source_key:
            item["name"] = requested_name
            break

    saved = save_camera_registry(registry_entries, config_path)
    apply_registry_to_active_config(saved, config_path)

    return {
        "status": "ok",
        "source_key": source_key,
        "name": request.name,
        "restart_required": True,
    }


@app.patch("/api/camera-source/{source_key}/enabled")
async def set_camera_source_enabled(source_key: str, request: ToggleCameraEnabledRequest) -> dict:
    config_path = active_config_path()
    if config_path is None:
        raise HTTPException(status_code=503, detail="No active config file is set.")

    registry_entries = load_camera_registry(config_path)
    found = False
    for item in registry_entries:
        if str(item.get("camera_id") or "") == source_key:
            item["enabled"] = bool(request.enabled)
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"Source '{source_key}' not found in config.")

    saved = save_camera_registry(registry_entries, config_path)
    apply_registry_to_active_config(saved, config_path)

    return {
        "status": "ok",
        "source_key": source_key,
        "enabled": bool(request.enabled),
        "restart_required": True,
    }


@app.post("/api/alpr-restart")
async def restart_alpr_runtime() -> dict:
    pid = read_pid(ALPR_PID_PATH)
    stop_alpr_process(pid)
    new_pid = start_alpr_process()
    if not new_pid:
        raise HTTPException(status_code=500, detail="ALPR restart failed")
    return {
        "status": "ok",
        "pid": new_pid,
    }


@app.get("/api/live-events", response_model=List[LiveEvent])
async def get_live_events(
    response: Response,
    limit: int = 100,
    status: Optional[EventStatus] = None,
    source: Optional[CameraSource] = None,
    plate: Optional[str] = None,
) -> List[LiveEvent]:
    response.headers["Cache-Control"] = "no-store, no-cache, max-age=0"
    safe_limit = max(1, min(limit, MAX_LIVE_EVENTS))
    plate_query = plate.strip().upper() if plate else None

    items = list(live_events)
    items.reverse()

    filtered: List[LiveEvent] = []
    for event in items:
        if status and event.status != status:
            continue
        if source and event.source != source:
            continue
        if plate_query and plate_query not in event.plate:
            continue
        filtered.append(event)
        if len(filtered) >= safe_limit:
            break

    return filtered


@app.get("/api/events/search", response_model=PersistedPlateReadSearchResponse)
async def search_persisted_events(
    response: Response,
    limit: int = 100,
    offset: int = 0,
    plate: Optional[str] = None,
    plate_prefix: bool = False,
    vehicle_make: Optional[str] = None,
    vehicle_type: Optional[str] = None,
    vehicle_color: Optional[str] = None,
    status: Optional[EventStatus] = None,
    source: Optional[CameraSource] = None,
    hotlist_alert: Optional[bool] = None,
    hotlist_hit: Optional[bool] = None,
    hotlist_type: Optional[str] = None,
    start_utc: Optional[str] = None,
    end_utc: Optional[str] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    near_lat: Optional[float] = None,
    near_lon: Optional[float] = None,
    radius_km: Optional[float] = None,
) -> PersistedPlateReadSearchResponse:
    response.headers["Cache-Control"] = "no-store, no-cache, max-age=0"
    ensure_event_db()

    safe_limit = max(1, min(limit, 500))
    safe_offset = max(0, offset)
    normalized_plate = plate.strip().upper() if plate else None
    normalized_hotlist_type = hotlist_type.strip().upper() if hotlist_type else None

    start_dt = parse_utc_iso(start_utc) if start_utc else None
    end_dt = parse_utc_iso(end_utc) if end_utc else None
    if start_utc and start_dt is None:
        raise HTTPException(status_code=400, detail="start_utc must be ISO-8601")
    if end_utc and end_dt is None:
        raise HTTPException(status_code=400, detail="end_utc must be ISO-8601")
    if start_dt and end_dt and end_dt < start_dt:
        raise HTTPException(status_code=400, detail="end_utc must be greater than or equal to start_utc")

    have_bbox = all(value is not None for value in (lat_min, lat_max, lon_min, lon_max))
    if any(value is not None for value in (lat_min, lat_max, lon_min, lon_max)) and not have_bbox:
        raise HTTPException(status_code=400, detail="lat/lon bounding box requires lat_min, lat_max, lon_min, lon_max")
    if have_bbox and (lat_min > lat_max or lon_min > lon_max):
        raise HTTPException(status_code=400, detail="invalid location bounding box")

    have_radius = all(value is not None for value in (near_lat, near_lon, radius_km))
    if any(value is not None for value in (near_lat, near_lon, radius_km)) and not have_radius:
        raise HTTPException(status_code=400, detail="radius search requires near_lat, near_lon, and radius_km")
    if have_radius and radius_km <= 0:
        raise HTTPException(status_code=400, detail="radius_km must be greater than zero")

    if have_radius:
        lat_delta = radius_km / 111.0
        lon_scale = max(0.1, abs(math.cos(math.radians(near_lat))))
        lon_delta = radius_km / (111.320 * lon_scale)
        radius_lat_min = near_lat - lat_delta
        radius_lat_max = near_lat + lat_delta
        radius_lon_min = near_lon - lon_delta
        radius_lon_max = near_lon + lon_delta
    else:
        radius_lat_min = radius_lat_max = radius_lon_min = radius_lon_max = None

    conditions: List[str] = []
    args: List[object] = []

    if normalized_plate:
        if plate_prefix:
            conditions.append("plate LIKE ? ESCAPE '\\\\'")
            args.append(f"{escape_like(normalized_plate)}%")
        else:
            conditions.append("plate_key LIKE ? ESCAPE '\\\\'")
            args.append(f"%{escape_like(hotlist_runtime_plate_key(normalized_plate))}%")

    if vehicle_make:
        conditions.append("vehicle_make LIKE ? ESCAPE '\\\\'")
        args.append(f"%{escape_like(vehicle_make.strip())}%")
    if vehicle_type:
        conditions.append("vehicle_type LIKE ? ESCAPE '\\\\'")
        args.append(f"%{escape_like(vehicle_type.strip())}%")
    if vehicle_color:
        conditions.append("vehicle_color LIKE ? ESCAPE '\\\\'")
        args.append(f"%{escape_like(vehicle_color.strip())}%")
    if status:
        conditions.append("status = ?")
        args.append(status.value)
    if source:
        conditions.append("source = ?")
        args.append(source.value)
    if hotlist_alert is not None:
        conditions.append("hotlist_alert = ?")
        args.append(1 if hotlist_alert else 0)
    if hotlist_hit is not None:
        conditions.append("hotlist_hit = ?")
        args.append(1 if hotlist_hit else 0)
    if normalized_hotlist_type:
        conditions.append("hotlist_type = ?")
        args.append(normalized_hotlist_type)
    if start_dt:
        conditions.append("timestamp_utc >= ?")
        args.append(utc_iso(start_dt.astimezone(timezone.utc)))
    if end_dt:
        conditions.append("timestamp_utc <= ?")
        args.append(utc_iso(end_dt.astimezone(timezone.utc)))

    if have_bbox:
        conditions.extend(
            [
                "gps_fix_valid = 1",
                "gps_latitude >= ?",
                "gps_latitude <= ?",
                "gps_longitude >= ?",
                "gps_longitude <= ?",
            ]
        )
        args.extend([lat_min, lat_max, lon_min, lon_max])

    if have_radius:
        conditions.extend(
            [
                "gps_fix_valid = 1",
                "gps_latitude >= ?",
                "gps_latitude <= ?",
                "gps_longitude >= ?",
                "gps_longitude <= ?",
            ]
        )
        args.extend([radius_lat_min, radius_lat_max, radius_lon_min, radius_lon_max])

    where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    total_query = f"SELECT COUNT(*) AS count FROM plate_reads {where_sql}"
    select_query = (
        "SELECT * FROM plate_reads "
        f"{where_sql} "
        "ORDER BY timestamp_utc DESC, event_uuid DESC "
        "LIMIT ? OFFSET ?"
    )

    with event_db_lock, sqlite3.connect(EVENT_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        total_row = conn.execute(total_query, args).fetchone()
        rows = conn.execute(select_query, [*args, safe_limit, safe_offset]).fetchall()

    items = [row_to_persisted_plate_read(row) for row in rows]

    if have_radius:
        filtered_items: List[PersistedPlateRead] = []
        for item in items:
            if item.gps_latitude is None or item.gps_longitude is None:
                continue
            if distance_km(near_lat, near_lon, item.gps_latitude, item.gps_longitude) <= radius_km:
                filtered_items.append(item)
        items = filtered_items

    return PersistedPlateReadSearchResponse(
        total=int(total_row["count"]) if total_row else 0,
        limit=safe_limit,
        offset=safe_offset,
        items=items,
    )


@app.get("/api/events/db-status")
async def get_events_db_status() -> dict:
    return event_db_status()


@app.post("/api/admin/db-selftest")
async def run_db_selftest() -> dict:
    return await asyncio.to_thread(event_db_selftest)


@app.post("/api/events/backfill")
async def backfill_events(case_id: Optional[str] = None, limit_cases: int = 0) -> dict:
    safe_limit_cases = max(0, min(limit_cases, 10000))
    result = await asyncio.to_thread(
        backfill_events_from_jsonl,
        case_id,
        safe_limit_cases if safe_limit_cases > 0 else None,
    )
    result["limit_cases"] = safe_limit_cases if safe_limit_cases > 0 else None
    return result


@app.post("/api/live-event", response_model=LiveEvent)
async def post_live_event(event: LiveEvent) -> LiveEvent:
    # Stamp GPS from MiFi if the incoming event has no fix
    if not event.gps_fix_valid and latest_gps_fix:
        event.gps_fix_valid = latest_gps_fix.get("gps_fix_valid", False)
        event.gps_latitude = latest_gps_fix.get("gps_latitude")
        event.gps_longitude = latest_gps_fix.get("gps_longitude")
        event.gps_speed_knots = latest_gps_fix.get("gps_speed_knots")
        event.gps_timestamp_utc = latest_gps_fix.get("gps_timestamp_utc")

    display_id = live_display_id(event)
    incoming_status = event.status
    existing = next(
        (item for item in reversed(live_events) if (item.display_id or item.event_id) == display_id),
        None,
    )
    event = merge_live_event(existing, event)
    event = enrich_with_hotlist(event)
    event = apply_hotlist_alert_policy(event)

    audio_cue = None
    if incoming_status == EventStatus.LOCKED:
        if event.hotlist_hit:
            if event.hotlist_alert:
                audio_cue = "hotlist"
        else:
            audio_cue = "locked"

    event.audio_cue = audio_cue
    upsert_live_event(event)

    try:
        await asyncio.to_thread(persist_plate_read, event)
    except Exception as exc:
        print(f"[WARN] failed to persist plate read to SQLite: {exc}")

    await broadcast_event(event, audio_cue)
    return event


@app.post("/api/hotlist/acknowledge", response_model=HotlistAcknowledgeResponse)
async def acknowledge_hotlist(request: HotlistAcknowledgeRequest) -> HotlistAcknowledgeResponse:
    suppressed_until = suppress_plate(hotlist_runtime_plate_key(request.plate), "acknowledged", request.seconds)
    return HotlistAcknowledgeResponse(
        plate=request.plate,
        suppressed_until_utc=utc_iso(suppressed_until),
        reason="acknowledged",
    )


@app.post("/api/hotlist/reset")
async def reset_hotlist_suppression(request: HotlistResetRequest) -> dict:
    if request.plate:
        plate_key = hotlist_runtime_plate_key(request.plate)
        suppressed_hotlist_plates.pop(plate_key, None)
        recent_hotlist_sightings.pop(plate_key, None)
        return {"status": "ok", "plate": request.plate, "reset": True}

    suppressed_hotlist_plates.clear()
    recent_hotlist_sightings.clear()
    return {"status": "ok", "plate": None, "reset": True}


@app.post("/api/hotlist/reload")
async def reload_hotlist() -> dict:
    snapshot = refresh_hotlist_state(update_manifest_reload=True)
    return dict(snapshot)


@app.post("/api/hotlist/upload")
async def upload_hotlists(
    svs_file: Optional[UploadFile] = File(default=None),
    slr_file: Optional[UploadFile] = File(default=None),
    sfr_file: Optional[UploadFile] = File(default=None),
) -> dict:
    uploads = {
        "svs.tbl": svs_file,
        "slr.tbl": slr_file,
        "sfr.tbl": sfr_file,
    }
    selected_uploads = {filename: upload for filename, upload in uploads.items() if upload is not None}
    if not selected_uploads:
        raise HTTPException(status_code=400, detail="at least one hotlist file is required")

    validated_uploads = {}
    errors = {}
    warnings = {}
    for filename, upload in selected_uploads.items():
        text = read_upload_text(upload)
        entries, issues = validate_hotlist_text(text, filename)
        if not entries:
            errors[filename] = issues
            continue
        if issues:
            warnings[filename] = issues
        validated_uploads[filename] = text

    if errors:
        raise HTTPException(status_code=400, detail={"message": "hotlist upload validation failed", "errors": errors})

    for filename, text in validated_uploads.items():
        store_hotlist_upload(filename, text)

    manifest = load_hotlist_manifest()
    imported_utc = utc_now_iso()
    for filename in validated_uploads:
        manifest["agency"][filename]["last_import_utc"] = imported_utc
    save_hotlist_manifest(manifest)
    snapshot = refresh_hotlist_state(update_manifest_reload=True)
    return {
        "status": "ok",
        "updated_sources": sorted(validated_uploads.keys()),
        "imported_at_utc": imported_utc,
        "warnings": warnings,
        **snapshot,
    }


@app.get("/api/hotlist/entries")
async def hotlist_entries(source_kind: str = "all", limit: int = 500) -> dict:
    normalized_kind = source_kind.strip().lower()
    if normalized_kind not in {"all", "agency", "local"}:
        raise HTTPException(status_code=400, detail="source_kind must be all, agency, or local")

    safe_limit = max(1, min(limit, 2000))
    entries = list_hotlist_entries(normalized_kind)
    return {
        "status": "ok",
        "source_kind": normalized_kind,
        "total_available": len(entries),
        "entries": [entry.model_dump() for entry in entries[:safe_limit]],
    }


@app.post("/api/hotlist/local-entry")
async def add_local_hotlist_entry(request: HotlistLocalEntryRequest) -> dict:
    entry = build_local_hotlist_entry(
        plate=request.plate,
        list_type=request.list_type,
        state=request.state,
        county_code=request.county_code,
        entry_date=request.entry_date,
        source_file="local-manual",
    )
    saved_entry, created = upsert_local_hotlist_entry(entry)
    return {
        "status": "ok",
        "created": created,
        "entry": saved_entry.model_dump(),
        "summary": dict(hotlist_status_snapshot),
    }


@app.post("/api/hotlist/local-csv")
async def upload_local_hotlist_csv(
    csv_file: UploadFile = File(...),
    default_list_type: str = Form("SLR"),
) -> dict:
    normalized_list_type = default_list_type.strip().upper()
    if normalized_list_type not in HOTLIST_TYPE_INFO:
        raise HTTPException(status_code=400, detail="default_list_type must be one of SLR, SVS, SFR")

    text = read_upload_text(csv_file)
    imported_entries, issues = parse_local_hotlist_csv(text, normalized_list_type)
    if not imported_entries and issues:
        raise HTTPException(status_code=400, detail={"message": "csv import failed", "issues": issues})

    current_entries = load_local_hotlist_entries()
    existing_identities = {entry_identity(entry) for entry in current_entries}
    new_entries = [entry for entry in imported_entries if entry_identity(entry) not in existing_identities]

    if new_entries:
        current_entries.extend(new_entries)
        save_local_entries_and_refresh(dedupe_hotlist_entries(current_entries))

    return {
        "status": "ok",
        "imported_count": len(new_entries),
        "skipped_duplicates": len(imported_entries) - len(new_entries),
        "issues": issues,
        "summary": dict(hotlist_status_snapshot),
    }


@app.delete("/api/hotlist/local-entry/{entry_id}")
async def delete_local_hotlist_entry(entry_id: str) -> dict:
    normalized_entry_id = entry_id.strip()
    if not normalized_entry_id:
        raise HTTPException(status_code=400, detail="entry_id is required")

    current_entries = load_local_hotlist_entries()
    remaining_entries = [entry for entry in current_entries if entry.entry_id != normalized_entry_id]
    if len(remaining_entries) == len(current_entries):
        raise HTTPException(status_code=404, detail="local hotlist entry not found")

    save_local_entries_and_refresh(remaining_entries)
    return {
        "status": "ok",
        "removed": True,
        "entry_id": normalized_entry_id,
        "summary": dict(hotlist_status_snapshot),
    }


@app.get("/api/hotlist/status", response_model=HotlistStatusResponse)
async def hotlist_status() -> HotlistStatusResponse:
    return HotlistStatusResponse(
        unique_plates=hotlist_status_snapshot.get("unique_plates", 0),
        total_entries=hotlist_status_snapshot.get("total_entries", 0),
        sources=list(HOTLIST_FILENAMES),
        hotlist_root=str(HOTLIST_ROOT),
        last_reload_utc=hotlist_status_snapshot.get("last_reload_utc"),
        local_entry_count=hotlist_status_snapshot.get("local_entry_count", 0),
        source_details=hotlist_status_snapshot.get("source_details", []),
    )


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    await websocket.accept()
    clients.add(websocket)

    try:
        snapshot = list(live_events)
        snapshot.reverse()
        for event in snapshot[:50]:
            await websocket.send_json(event.model_dump())

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception:
        clients.discard(websocket)


@app.get("/media/{relative_path:path}")
async def get_media(relative_path: str):
    full_path = (EVIDENCE_ROOT / relative_path).resolve()

    if not str(full_path).startswith(str(EVIDENCE_ROOT)):
        raise HTTPException(status_code=400, detail="invalid media path")

    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    return FileResponse(full_path)


@app.get("/runtime-media/{relative_path:path}")
async def get_runtime_media(relative_path: str):
    full_path = (RUNTIME_ROOT / relative_path).resolve()

    if not str(full_path).startswith(str(RUNTIME_ROOT)):
        raise HTTPException(status_code=400, detail="invalid runtime media path")

    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    try:
        content = full_path.read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"failed to read runtime media: {exc}") from exc

    media_type = mimetypes.guess_type(str(full_path))[0] or "application/octet-stream"
    return Response(content=content, media_type=media_type)