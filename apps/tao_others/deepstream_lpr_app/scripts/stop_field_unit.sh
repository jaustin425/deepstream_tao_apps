#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="$APP_DIR/runtime"
BACKEND_PID_FILE="$RUNTIME_DIR/backend.pid"
ALPR_PID_FILE="$RUNTIME_DIR/alpr.pid"

stop_pid_file() {
  local pid_file="$1"
  local label="$2"
  local expected_cmd="${3:-}"

  if [[ ! -f "$pid_file" ]]; then
    echo "$label not running"
    return 0
  fi

  local pid
  pid="$(cat "$pid_file")"
  if [[ -z "$pid" ]]; then
    rm -f "$pid_file"
    echo "$label pid file was empty"
    return 0
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file"
    echo "$label already stopped"
    return 0
  fi

  if [[ -n "$expected_cmd" ]]; then
    local cmdline
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" != *"$expected_cmd"* ]]; then
      rm -f "$pid_file"
      echo "$label pid file was stale or pointed at another process"
      return 0
    fi
  fi

  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pid_file"
      echo "$label stopped"
      return 0
    fi
    sleep 1
  done

  kill -9 "$pid" 2>/dev/null || true
  rm -f "$pid_file"
  echo "$label force stopped"
}

stop_orphan_alpr_processes() {
  local binary_path="$APP_DIR/deepstream-lpr-app"
  local pids=()
  mapfile -t pids < <(pgrep -f "$binary_path" 2>/dev/null || true)

  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi

  local stopped=0
  for pid in "${pids[@]}"; do
    [[ -z "$pid" || "$pid" == "$$" ]] && continue

    local cmdline
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    [[ "$cmdline" == *"$binary_path"* ]] || continue

    kill "$pid" 2>/dev/null || true
    stopped=$((stopped + 1))
  done

  for pid in "${pids[@]}"; do
    [[ -z "$pid" || "$pid" == "$$" ]] && continue
    for _ in $(seq 1 5); do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done

  if [[ "$stopped" -gt 0 ]]; then
    echo "Stopped $stopped orphan ALPR process(es)"
  fi
}

stop_pid_file "$ALPR_PID_FILE" "ALPR" "deepstream-lpr-app"
stop_orphan_alpr_processes
stop_pid_file "$BACKEND_PID_FILE" "Backend" "uvicorn app:app"
rm -f "$RUNTIME_DIR/alpr_status.json"
