#!/usr/bin/env bash

set -euo pipefail

ACTION="${1:-up}"
KERNEL_RELEASE="$(uname -r)"
KERNEL_BUILD="/lib/modules/${KERNEL_RELEASE}/build"
MT76_REF="${ALPR_MT76_REF:-v5.15.148}"
MT76_REPO="${ALPR_MT76_REPO:-gregkh/linux}"
MT76_BUILD_DIR="${ALPR_MT76_BUILD_DIR:-/tmp/mt76-${MT76_REF}}"
MT76_INSTALL_DIR="/lib/modules/${KERNEL_RELEASE}/updates/alpr-mt76"

VERIZON_CONN="${ALPR_VERIZON_WIFI_PROFILE:-Verizon-MiFi8800L-AA46}"
VERIZON_SSID="${ALPR_VERIZON_WIFI_SSID:-$VERIZON_CONN}"
VERIZON_PASSWORD="${ALPR_VERIZON_WIFI_PASSWORD:-}"
CLIENT_IFACE="${ALPR_VERIZON_WIFI_INTERFACE:-}"
AP_IFACE="${ALPR_WIFI_AP_INTERFACE:-alpr_ap0}"
AP_CONN="${ALPR_WIFI_AP_PROFILE:-ALPR-Field-Unit-AP}"
AP_SSID="${ALPR_WIFI_AP_SSID:-ALPR-Field-Unit}"
AP_PASSWORD="${ALPR_WIFI_AP_PASSWORD:-ALPRAccess123}"

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run this script with sudo so it can install/load the Panda WiFi driver and create NetworkManager profiles." >&2
    exit 1
  fi
}

require_tools() {
  local missing=()
  for tool in git make gcc python3 nmcli iw modprobe depmod; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      missing+=("$tool")
    fi
  done

  if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "Missing required tool(s): ${missing[*]}" >&2
    exit 1
  fi

  if [[ ! -d "$KERNEL_BUILD" ]]; then
    echo "Kernel build headers not found: $KERNEL_BUILD" >&2
    exit 1
  fi
}

module_loaded() {
  lsmod | awk '{print $1}' | grep -Fxq "$1"
}

download_mt76_sources() {
  if [[ -f "$MT76_BUILD_DIR/Makefile" && -f "$MT76_BUILD_DIR/mt76x0/Makefile" ]]; then
    return
  fi

  rm -rf "$MT76_BUILD_DIR"
  mkdir -p "$MT76_BUILD_DIR"

  python3 - "$MT76_REPO" "$MT76_REF" "$MT76_BUILD_DIR" <<'PY'
import json
import os
import sys
import urllib.request

repo, ref, out_root = sys.argv[1:4]
root_path = "drivers/net/wireless/mediatek/mt76"
api_base = f"https://api.github.com/repos/{repo}/contents"
headers = {"User-Agent": "alpr-field-unit-panda-wifi"}

def fetch_json(url):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)

def download(url, dest):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response, open(dest, "wb") as output:
        output.write(response.read())

def walk(path):
    for item in fetch_json(f"{api_base}/{path}?ref={ref}"):
        rel = os.path.relpath(item["path"], root_path)
        dest = os.path.join(out_root, rel)
        if item["type"] == "dir":
            os.makedirs(dest, exist_ok=True)
            walk(item["path"])
        elif item["type"] == "file":
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            download(item["download_url"], dest)

walk(root_path)
PY
}

build_mt76_modules() {
  if [[ -f "$MT76_BUILD_DIR/mt76x0/mt76x0u.ko" ]]; then
    return
  fi

  make -C "$KERNEL_BUILD" M="$MT76_BUILD_DIR" \
    CONFIG_MT76_CORE=m \
    CONFIG_MT76_USB=m \
    CONFIG_MT76x02_LIB=m \
    CONFIG_MT76x02_USB=m \
    CONFIG_MT76x0_COMMON=m \
    CONFIG_MT76x0U=m \
    modules
}

install_mt76_modules() {
  install -d "$MT76_INSTALL_DIR" "$MT76_INSTALL_DIR/mt76x0"
  install -m 0644 \
    "$MT76_BUILD_DIR/mt76.ko" \
    "$MT76_BUILD_DIR/mt76-usb.ko" \
    "$MT76_BUILD_DIR/mt76x02-lib.ko" \
    "$MT76_BUILD_DIR/mt76x02-usb.ko" \
    "$MT76_INSTALL_DIR/"
  install -m 0644 \
    "$MT76_BUILD_DIR/mt76x0/mt76x0-common.ko" \
    "$MT76_BUILD_DIR/mt76x0/mt76x0u.ko" \
    "$MT76_INSTALL_DIR/mt76x0/"
  depmod "$KERNEL_RELEASE"
}

load_mt76_modules() {
  modprobe mt76x0u
  sleep 2
}

wifi_ifaces() {
  nmcli -t -f DEVICE,TYPE device status | awk -F: '$2 == "wifi" {print $1}'
}

first_wifi_iface() {
  local iface
  while read -r iface; do
    [[ -n "$iface" ]] || continue
    if [[ "$iface" != "$AP_IFACE" ]]; then
      printf '%s\n' "$iface"
      return
    fi
  done < <(wifi_ifaces)
}

first_ap_candidate_iface() {
  local iface
  while read -r iface; do
    [[ -n "$iface" ]] || continue
    [[ "$iface" == "$CLIENT_IFACE" ]] && continue
    [[ "$iface" == p2p-* ]] && continue
    printf '%s\n' "$iface"
    return
  done < <(wifi_ifaces)
}

connection_exists() {
  nmcli -t -f NAME connection show | grep -Fxq "$1"
}

ensure_client_connection() {
  if [[ -z "$CLIENT_IFACE" ]]; then
    CLIENT_IFACE="$(first_wifi_iface || true)"
  fi

  if [[ -z "$CLIENT_IFACE" ]]; then
    echo "No WiFi client interface found after loading mt76x0u." >&2
    exit 1
  fi

  nmcli radio wifi on

  if connection_exists "$VERIZON_CONN"; then
    nmcli connection modify "$VERIZON_CONN" \
      connection.autoconnect yes \
      connection.interface-name "" \
      802-11-wireless.mac-address "" >/dev/null
    nmcli connection up "$VERIZON_CONN" ifname "$CLIENT_IFACE" >/dev/null
  else
    if [[ -z "$VERIZON_PASSWORD" ]]; then
      echo "NetworkManager profile '$VERIZON_CONN' does not exist and ALPR_VERIZON_WIFI_PASSWORD is not set." >&2
      exit 1
    fi

    nmcli device wifi connect "$VERIZON_SSID" \
      password "$VERIZON_PASSWORD" \
      ifname "$CLIENT_IFACE" \
      name "$VERIZON_CONN" >/dev/null
    nmcli connection modify "$VERIZON_CONN" connection.autoconnect yes >/dev/null
  fi

  echo "Connected WiFi client '$CLIENT_IFACE' using profile '$VERIZON_CONN'."
}

client_phy() {
  iw dev "$CLIENT_IFACE" info | awk '/wiphy/ {print "phy" $2; exit}'
}

client_channel() {
  local freq
  freq="$(iw dev "$CLIENT_IFACE" link | awk '/freq:/ {print $2; exit}')"
  [[ -n "$freq" ]] || return 0

  awk -v freq="$freq" 'BEGIN {
    if (freq == 2484) print 14;
    else if (freq < 2484) print int((freq - 2407) / 5);
    else print int((freq - 5000) / 5);
  }'
}

client_band() {
  local freq
  freq="$(iw dev "$CLIENT_IFACE" link | awk '/freq:/ {print $2; exit}')"
  [[ -n "$freq" ]] || return 0

  if [[ "$freq" -lt 3000 ]]; then
    echo "bg"
  else
    echo "a"
  fi
}

ensure_ap_interface() {
  if ip link show "$AP_IFACE" >/dev/null 2>&1; then
    return
  fi

  local existing_ap_iface
  existing_ap_iface="$(first_ap_candidate_iface || true)"
  if [[ -n "$existing_ap_iface" ]]; then
    AP_IFACE="$existing_ap_iface"
    nmcli device set "$AP_IFACE" managed yes >/dev/null || true
    return
  fi

  local phy
  phy="$(client_phy)"
  if [[ -z "$phy" ]]; then
    echo "Could not resolve WiFi phy for $CLIENT_IFACE." >&2
    exit 1
  fi

  iw phy "$phy" interface add "$AP_IFACE" type __ap
  nmcli device set "$AP_IFACE" managed yes >/dev/null || true
  sleep 1

  if ! ip link show "$AP_IFACE" >/dev/null 2>&1; then
    existing_ap_iface="$(first_ap_candidate_iface || true)"
    if [[ -n "$existing_ap_iface" ]]; then
      AP_IFACE="$existing_ap_iface"
      nmcli device set "$AP_IFACE" managed yes >/dev/null || true
      return
    fi

    echo "Could not create or find an AP-capable interface on $phy." >&2
    exit 1
  fi
}

ensure_ap_connection() {
  if [[ ${#AP_PASSWORD} -lt 8 ]]; then
    echo "ALPR WiFi AP password must be at least 8 characters." >&2
    exit 1
  fi

  ensure_ap_interface

  if connection_exists "$AP_CONN"; then
    nmcli connection down "$AP_CONN" >/dev/null 2>&1 || true
    nmcli connection delete "$AP_CONN" >/dev/null 2>&1 || true
  fi

  nmcli connection add \
    type wifi \
    ifname "$AP_IFACE" \
    con-name "$AP_CONN" \
    autoconnect yes \
    ssid "$AP_SSID" >/dev/null

  nmcli connection modify "$AP_CONN" \
    connection.interface-name "$AP_IFACE" \
    802-11-wireless.mode ap \
    802-11-wireless-security.key-mgmt wpa-psk \
    802-11-wireless-security.psk "$AP_PASSWORD" \
    ipv4.method shared \
    ipv6.method ignore >/dev/null

  local channel band
  channel="$(client_channel || true)"
  band="$(client_band || true)"
  if [[ -n "$channel" && -n "$band" ]]; then
    nmcli connection modify "$AP_CONN" \
      802-11-wireless.band "$band" \
      802-11-wireless.channel "$channel" >/dev/null
  fi

  nmcli connection up "$AP_CONN" ifname "$AP_IFACE" >/dev/null

  echo "ALPR access point enabled on $AP_IFACE."
  echo "  SSID: $AP_SSID"
  echo "  Password: $AP_PASSWORD"
}

show_status() {
  nmcli device status
  echo
  iw dev || true
  echo
  ip -o -4 addr show up | awk '$2 != "lo" {split($4, cidr, "/"); printf "%s -> http://%s:%s\n", $2, cidr[1], ENVIRON["ALPR_DASHBOARD_PORT"] ? ENVIRON["ALPR_DASHBOARD_PORT"] : "8080"}'
}

driver_up() {
  require_root
  require_tools
  if modinfo mt76x0u >/dev/null 2>&1; then
    load_mt76_modules
    return
  fi
  download_mt76_sources
  build_mt76_modules
  install_mt76_modules
  load_mt76_modules
}

wifi_up() {
  require_root
  require_tools
  load_mt76_modules
  ensure_client_connection
  ensure_ap_connection
}

down() {
  require_root
  nmcli connection down "$AP_CONN" >/dev/null 2>&1 || true
  nmcli connection delete "$AP_CONN" >/dev/null 2>&1 || true
  if ip link show "$AP_IFACE" >/dev/null 2>&1; then
    ip link set "$AP_IFACE" down >/dev/null 2>&1 || true
    iw dev "$AP_IFACE" del >/dev/null 2>&1 || true
  fi
  echo "ALPR Panda WiFi AP removed."
}

case "$ACTION" in
  up)
    driver_up
    wifi_up
    show_status
    ;;
  driver-up)
    driver_up
    show_status
    ;;
  wifi-up)
    wifi_up
    show_status
    ;;
  down)
    down
    ;;
  status)
    show_status
    ;;
  *)
    echo "Usage: $0 {up|driver-up|wifi-up|down|status}" >&2
    exit 1
    ;;
esac
