#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$APP_DIR/logs"
RUNTIME_DIR="$APP_DIR/runtime"
NETWORK_HELPER="$APP_DIR/scripts/configure_network_access.sh"
BACKEND_PID_FILE="$RUNTIME_DIR/backend.pid"
ALPR_PID_FILE="$RUNTIME_DIR/alpr.pid"
START_LOCK_FILE="$RUNTIME_DIR/field_unit_start.lock"
BACKEND_LOG="$LOG_DIR/backend.log"
ALPR_LOG="$LOG_DIR/alpr.log"
DEFAULT_CONFIG="$APP_DIR/../../../configs/app/lpr_app_us_config.yml"

BACKEND_HOST="${ALPR_DASHBOARD_HOST:-0.0.0.0}"
BACKEND_PORT="${ALPR_DASHBOARD_PORT:-8080}"
export ALPR_DASHBOARD_HOST="$BACKEND_HOST"
export ALPR_DASHBOARD_PORT="$BACKEND_PORT"
ALPR_CONFIG="${1:-$DEFAULT_CONFIG}"
ALPR_PROFILE_OVERRIDE="${2:-}"

if [[ "$ALPR_CONFIG" != /* ]]; then
  ALPR_CONFIG="$(realpath -m "$ALPR_CONFIG")"
fi

if [[ ! -f "$ALPR_CONFIG" ]]; then
  echo "ALPR config not found: $ALPR_CONFIG"
  exit 1
fi

export ALPR_ACTIVE_CONFIG_PATH="$ALPR_CONFIG"
if [[ -n "$ALPR_PROFILE_OVERRIDE" ]]; then
  export ALPR_RUNTIME_PROFILE="$ALPR_PROFILE_OVERRIDE"
else
  unset ALPR_RUNTIME_PROFILE
fi

resolve_backend_python() {
  local candidates=()

  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    candidates+=("${VIRTUAL_ENV}/bin/python")
  fi

  if [[ -x "$APP_DIR/.venv/bin/python" ]]; then
    candidates+=("$APP_DIR/.venv/bin/python")
  fi

  candidates+=("/usr/bin/python3" "python3")

  for candidate in "${candidates[@]}"; do
    if "$candidate" -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  printf '%s\n' "/usr/bin/python3"
}

BACKEND_PYTHON="$(resolve_backend_python)"

print_dashboard_urls() {
  echo "Dashboard URLs:"
  echo "  http://127.0.0.1:${BACKEND_PORT}"
  ip -o -4 addr show up | awk -v port="$BACKEND_PORT" '$2 != "lo" {split($4, cidr, "/"); printf "  %s -> http://%s:%s\n", $2, cidr[1], port}' || true
}

maybe_enable_network_access() {
  if [[ ! -x "$NETWORK_HELPER" ]]; then
    return
  fi

  if [[ "${ALPR_WIFI_AP_ENABLE:-0}" == "1" ]]; then
    if [[ $(id -u) -ne 0 ]]; then
      echo "Skipping WiFi hotspot enable because root privileges are required."
    else
      "$NETWORK_HELPER" hotspot-up \
        "${ALPR_WIFI_AP_INTERFACE:-}" \
        "${ALPR_WIFI_AP_SSID:-ALPR-Field-Unit}" \
        "${ALPR_WIFI_AP_PASSWORD:-ALPRAccess123}"
    fi
  fi

  if [[ "${ALPR_ETHERNET_SHARE_ENABLE:-0}" == "1" ]]; then
    if [[ $(id -u) -ne 0 ]]; then
      echo "Skipping Ethernet shared-link enable because root privileges are required."
    else
      "$NETWORK_HELPER" ethernet-up "${ALPR_ETHERNET_INTERFACE:-}"
    fi
  fi
}

resolve_live_source_map() {
  local config_path="$1"
  [[ -f "$config_path" ]] || return 0

  awk '
    /^live-dashboard:[[:space:]]*$/ {
      in_live = 1
      in_map = 0
      next
    }
    in_live && /^[^[:space:]]/ {
      in_live = 0
      in_map = 0
    }
    in_live && /^[[:space:]]+source-map:[[:space:]]*$/ {
      in_map = 1
      next
    }
    in_map && /^    [A-Za-z0-9_-]+:[[:space:]]*[A-Za-z0-9_-]+[[:space:]]*$/ {
      line = $0
      sub(/^[[:space:]]+/, "", line)
      split(line, parts, ":")
      key = parts[1]
      value = parts[2]
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      if (out != "") {
        out = out ","
      }
      out = out key "=" toupper(value)
      next
    }
    in_map {
      in_map = 0
    }
    END {
      print out
    }
  ' "$config_path"
}

mkdir -p "$LOG_DIR" "$RUNTIME_DIR"

exec 9>"$START_LOCK_FILE"
if ! flock -n 9; then
  echo "Field unit start is already in progress."
  exit 1
fi

maybe_enable_network_access

if [[ -z "${ALPR_LIVE_SOURCE_MAP:-}" ]]; then
  ALPR_LIVE_SOURCE_MAP="$(resolve_live_source_map "$ALPR_CONFIG")"
  export ALPR_LIVE_SOURCE_MAP
fi

is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 1

  local pid
  pid="$(cat "$pid_file")"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

stop_pid_file() {
  local pid_file="$1"
  if is_running "$pid_file"; then
    kill "$(cat "$pid_file")" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}

find_running_alpr_pids() {
  local binary_path="$APP_DIR/deepstream-lpr-app"
  local pids=()
  mapfile -t pids < <(pgrep -f "$binary_path" 2>/dev/null || true)

  for pid in "${pids[@]}"; do
    [[ -z "$pid" || "$pid" == "$$" ]] && continue
    local cmdline
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    [[ "$cmdline" == *"$binary_path"* ]] || continue
    printf '%s\n' "$pid"
  done
}

emit_alpr_failure_hint() {
  local log_file="$1"
  local log_offset="${2:-}"
  local recent_log

  if [[ -n "$log_offset" && "$log_offset" =~ ^[0-9]+$ ]]; then
    recent_log="$(tail -c +"$((log_offset + 1))" "$log_file" 2>/dev/null || true)"
  else
    recent_log=""
  fi

  if [[ -z "$recent_log" ]]; then
    recent_log="$(tail -n 120 "$log_file" 2>/dev/null || true)"
  fi

  if grep -Eq 'Cannot identify device|No such file or directory' <<<"$recent_log"; then
    echo "Detected missing camera device while starting DeepStream."
    echo "Check that the Arducam is enumerated: lsusb"
    echo "Check that a V4L2 node exists: ls -l /dev/video* /dev/v4l/by-path"
    echo "If the camera was unplugged or reset, reseat the USB cable or power-cycle the camera."
    return
  fi

  if grep -Eq 'Could not open device|Device or resource busy' <<<"$recent_log"; then
    echo "Detected camera device contention while starting DeepStream."
    echo "Check which process owns the camera: fuser -v /dev/video0"
    echo "Close any direct-preview or test process that is still holding the camera, then restart the service."
    return
  fi

  if grep -Eq 'nvbufsurftransform:cuInit failed|cudaErrorDevicesUnavailable|RuntimeException|cudaErrorIllegalAddress|GPUassert failed' <<<"$recent_log"; then
    echo "Detected CUDA initialization failure while starting DeepStream."
    echo "This usually means the NVIDIA driver/GPU is in a bad state, not that the dashboard config page broke startup."
    echo "Check: nvidia-smi"
    echo "Check kernel log for Xid faults: journalctl -k -n 120 --no-pager | grep -iE 'nvrm|xid|nvidia'"
    echo "Typical recovery is to reboot, or reload the NVIDIA kernel modules if you manage the host locally."
    return
  fi

  echo "Fresh ALPR log tail:"
  printf '%s\n' "$recent_log"
}

if is_running "$BACKEND_PID_FILE"; then
  echo "Backend already running with PID $(cat "$BACKEND_PID_FILE")"
  exit 1
fi

if is_running "$ALPR_PID_FILE"; then
  echo "ALPR already running with PID $(cat "$ALPR_PID_FILE")"
  exit 1
fi

RUNNING_ALPR_PIDS="$(find_running_alpr_pids | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
if [[ -n "$RUNNING_ALPR_PIDS" ]]; then
  echo "ALPR already running with PID(s): $RUNNING_ALPR_PIDS"
  echo "Run ./stop_field_unit.sh first to clear stale or orphaned ALPR processes."
  exit 1
fi

rm -f "$RUNTIME_DIR/alpr_status.json"

printf '\n[%s] starting backend\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BACKEND_LOG"
(
  cd "$APP_DIR"
  nohup setsid "$BACKEND_PYTHON" -m uvicorn app:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" >> "$BACKEND_LOG" 2>&1 &
  echo $! > "$BACKEND_PID_FILE"
)

sleep 1
if ! is_running "$BACKEND_PID_FILE"; then
  echo "Backend failed to start. Check $BACKEND_LOG"
  exit 1
fi

ALPR_LOG_OFFSET=0
if [[ -f "$ALPR_LOG" ]]; then
  ALPR_LOG_OFFSET="$(wc -c < "$ALPR_LOG" | tr -d '[:space:]')"
fi

printf '\n[%s] starting alpr\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ALPR_LOG"
(
  cd "$APP_DIR"
  export ALPR_LIVE_ENDPOINT="${ALPR_LIVE_ENDPOINT:-http://127.0.0.1:${BACKEND_PORT}/api/live-event}"
  nohup setsid "$APP_DIR/deepstream-lpr-app" "$ALPR_CONFIG" >> "$ALPR_LOG" 2>&1 &
  echo $! > "$ALPR_PID_FILE"
)

sleep 1
if ! is_running "$ALPR_PID_FILE"; then
  echo "ALPR failed to start. Stopping backend. Check $ALPR_LOG"
  emit_alpr_failure_hint "$ALPR_LOG" "$ALPR_LOG_OFFSET"
  stop_pid_file "$BACKEND_PID_FILE"
  exit 1
fi

echo "Backend PID: $(cat "$BACKEND_PID_FILE")"
echo "ALPR PID: $(cat "$ALPR_PID_FILE")"
if [[ -n "${ALPR_RUNTIME_PROFILE:-}" ]]; then
  echo "ALPR runtime profile override: $ALPR_RUNTIME_PROFILE"
fi
print_dashboard_urls
echo "Backend log: $BACKEND_LOG"
echo "ALPR log: $ALPR_LOG"
