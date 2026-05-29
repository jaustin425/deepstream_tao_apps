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
VERIZON_BAND="${ALPR_VERIZON_WIFI_BAND:-}"
VERIZON_CHANNEL="${ALPR_VERIZON_WIFI_CHANNEL:-}"
VERIZON_ROUTE_METRIC="${ALPR_VERIZON_WIFI_ROUTE_METRIC:-700}"
VERIZON_CONNECT_TIMEOUT="${ALPR_VERIZON_WIFI_CONNECT_TIMEOUT:-35}"
VERIZON_REQUIRED="${ALPR_VERIZON_WIFI_REQUIRED:-0}"
VERIZON_ENABLE="${ALPR_VERIZON_WIFI_ENABLE:-1}"
CLIENT_IFACE="${ALPR_VERIZON_WIFI_INTERFACE:-}"
AP_IFACE="${ALPR_WIFI_AP_INTERFACE:-alpr_ap0}"
AP_CONN="${ALPR_WIFI_AP_PROFILE:-ALPR-Field-Unit-AP}"
AP_SSID="${ALPR_WIFI_AP_SSID:-ALPR-Field-Unit}"
AP_PASSWORD="${ALPR_WIFI_AP_PASSWORD:-ALPRAccess123}"
AP_SUBNET="${ALPR_WIFI_AP_SUBNET:-10.42.0.0/24}"
AP_POLICY_TABLE="${ALPR_WIFI_AP_POLICY_TABLE:-142}"
AP_POLICY_PRIORITY="${ALPR_WIFI_AP_POLICY_PRIORITY:-142}"
INTERNET_UPSTREAM_IFACE="${ALPR_INTERNET_UPSTREAM_INTERFACE:-}"
EXPLICIT_NAT="${ALPR_WIFI_AP_EXPLICIT_NAT:-0}"
SYNC_AP_CHANNEL="${ALPR_WIFI_AP_SYNC_CHANNEL:-0}"
CLIENT_CONNECTED=0
CLIENT_AP_BAND=""
CLIENT_AP_CHANNEL=""

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

connection_device() {
  local connection_name="$1"
  nmcli -t -f NAME,DEVICE connection show --active | awk -F: -v name="$connection_name" '$1 == name {print $2; exit}'
}

device_connection() {
  local interface_name="$1"
  nmcli -t -f DEVICE,CONNECTION device status | awk -F: -v iface="$interface_name" '$1 == iface {print $2; exit}'
}

device_exists() {
  [[ -e "/sys/class/net/$1" ]]
}

wifi_iface_mode() {
  local interface_name="$1"
  if [[ "$(device_connection "$interface_name")" == "$AP_CONN" ]]; then
    echo "AP"
  fi
}

first_wifi_iface() {
  local active_client
  active_client="$(connection_device "$VERIZON_CONN")"
  if [[ -n "$active_client" ]]; then
    printf '%s\n' "$active_client"
    return
  fi

  local iface
  while read -r iface; do
    [[ -n "$iface" ]] || continue
    [[ "$iface" == "$AP_IFACE" ]] && continue
    [[ "$(device_connection "$iface")" == "$AP_CONN" ]] && continue
    [[ "$(wifi_iface_mode "$iface")" == "AP" ]] && continue
    printf '%s\n' "$iface"
    return
  done < <(wifi_ifaces)
}

first_ap_candidate_iface() {
  local iface
  while read -r iface; do
    [[ -n "$iface" ]] || continue
    [[ "$iface" == "$CLIENT_IFACE" ]] && continue
    [[ "$iface" == p2p-* ]] && continue
    [[ "$(device_connection "$iface")" == "$VERIZON_CONN" ]] && continue
    printf '%s\n' "$iface"
    return
  done < <(wifi_ifaces)
}

connection_exists() {
  nmcli -t -f NAME connection show | grep -Fxq "$1"
}

stop_ap_connection() {
  local active_ap_iface
  active_ap_iface="$(connection_device "$AP_CONN")"
  if [[ -n "$active_ap_iface" ]]; then
    AP_IFACE="$active_ap_iface"
  fi
  nmcli connection down "$AP_CONN" >/dev/null 2>&1 || true
}

stop_client_connection() {
  nmcli connection down "$VERIZON_CONN" >/dev/null 2>&1 || true
}

set_client_ap_channel_from_config() {
  if [[ -n "$VERIZON_BAND" && -n "$VERIZON_CHANNEL" ]]; then
    CLIENT_AP_BAND="$VERIZON_BAND"
    CLIENT_AP_CHANNEL="$VERIZON_CHANNEL"
  fi
}

discover_client_ap_channel() {
  set_client_ap_channel_from_config
  if [[ -n "$CLIENT_AP_BAND" && -n "$CLIENT_AP_CHANNEL" ]]; then
    return
  fi

  local selected
  selected="$(nmcli -t -f SSID,CHAN,SIGNAL device wifi list ifname "$CLIENT_IFACE" --rescan yes 2>/dev/null | \
    awk -F: -v ssid="$VERIZON_SSID" '
      $1 == ssid && $2 ~ /^[0-9]+$/ && $2 <= 14 {
        if ($3 + 0 >= best_signal) {
          best_signal=$3 + 0
          best_channel=$2
        }
      }
      END {
        if (best_channel != "") {
          print "bg " best_channel
        }
      }')"

  if [[ -z "$selected" ]]; then
    selected="$(nmcli -t -f SSID,CHAN,SIGNAL device wifi list ifname "$CLIENT_IFACE" --rescan no 2>/dev/null | \
      awk -F: -v ssid="$VERIZON_SSID" '
        $1 == ssid && $2 ~ /^[0-9]+$/ {
          if ($3 + 0 >= best_signal) {
            best_signal=$3 + 0
            best_channel=$2
          }
        }
        END {
          if (best_channel != "") {
            band=(best_channel <= 14) ? "bg" : "a"
            print band " " best_channel
          }
        }')"
  fi

  if [[ -n "$selected" ]]; then
    read -r CLIENT_AP_BAND CLIENT_AP_CHANNEL <<< "$selected"
    echo "Found '$VERIZON_SSID' on ${CLIENT_AP_BAND}/${CLIENT_AP_CHANNEL}; matching AP channel for STA+AP coexistence."
  fi
}

ensure_client_connection() {
  if [[ "$VERIZON_ENABLE" != "1" ]]; then
    CLIENT_IFACE="$(connection_device "$VERIZON_CONN")"
    if [[ -n "$CLIENT_IFACE" ]]; then
      CLIENT_CONNECTED=1
      echo "Using existing WiFi client '$CLIENT_IFACE' from profile '$VERIZON_CONN'."
    else
      CLIENT_CONNECTED=0
      echo "Skipping Verizon WiFi connection; local AP will still be enabled."
    fi
    return 0
  fi

  if [[ -z "$CLIENT_IFACE" ]]; then
    CLIENT_IFACE="$(first_wifi_iface || true)"
  fi

  if [[ -z "$CLIENT_IFACE" ]]; then
    echo "No WiFi client interface found after loading mt76x0u." >&2
    exit 1
  fi

  nmcli radio wifi on
  discover_client_ap_channel

  if connection_exists "$VERIZON_CONN"; then
    nmcli connection modify "$VERIZON_CONN" \
      connection.autoconnect yes \
      ipv4.never-default no \
      ipv4.route-metric "$VERIZON_ROUTE_METRIC" \
      ipv6.never-default no \
      ipv6.route-metric "$VERIZON_ROUTE_METRIC" \
      connection.interface-name "" \
      802-11-wireless.mac-address "" >/dev/null
    if [[ -n "$CLIENT_AP_BAND" ]]; then
      nmcli connection modify "$VERIZON_CONN" 802-11-wireless.band "$CLIENT_AP_BAND" >/dev/null
    fi
    if [[ -n "$CLIENT_AP_CHANNEL" ]]; then
      nmcli connection modify "$VERIZON_CONN" 802-11-wireless.channel "$CLIENT_AP_CHANNEL" >/dev/null
    fi
    if ! timeout "$VERIZON_CONNECT_TIMEOUT" nmcli connection up "$VERIZON_CONN" ifname "$CLIENT_IFACE" >/dev/null; then
      echo "Verizon WiFi profile '$VERIZON_CONN' did not connect on $CLIENT_IFACE; continuing with local AP only." >&2
      stop_client_connection
      if [[ "$VERIZON_REQUIRED" == "1" ]]; then
        exit 1
      fi
      CLIENT_IFACE=""
      CLIENT_CONNECTED=0
      return 0
    fi
  else
    if [[ -z "$VERIZON_PASSWORD" ]]; then
      echo "NetworkManager profile '$VERIZON_CONN' does not exist and ALPR_VERIZON_WIFI_PASSWORD is not set; continuing with local AP only." >&2
      if [[ "$VERIZON_REQUIRED" == "1" ]]; then
        exit 1
      fi
      CLIENT_IFACE=""
      CLIENT_CONNECTED=0
      return 0
    fi

    if ! timeout "$VERIZON_CONNECT_TIMEOUT" nmcli device wifi connect "$VERIZON_SSID" \
      password "$VERIZON_PASSWORD" \
      ifname "$CLIENT_IFACE" \
      name "$VERIZON_CONN" >/dev/null; then
      echo "Verizon SSID '$VERIZON_SSID' did not connect on $CLIENT_IFACE; continuing with local AP only." >&2
      stop_client_connection
      if [[ "$VERIZON_REQUIRED" == "1" ]]; then
        exit 1
      fi
      CLIENT_IFACE=""
      CLIENT_CONNECTED=0
      return 0
    fi
    nmcli connection modify "$VERIZON_CONN" \
      connection.autoconnect yes \
      ipv4.never-default no \
      ipv4.route-metric "$VERIZON_ROUTE_METRIC" \
      ipv6.never-default no \
      ipv6.route-metric "$VERIZON_ROUTE_METRIC" >/dev/null
    if [[ -n "$CLIENT_AP_BAND" ]]; then
      nmcli connection modify "$VERIZON_CONN" 802-11-wireless.band "$CLIENT_AP_BAND" >/dev/null
    fi
    if [[ -n "$CLIENT_AP_CHANNEL" ]]; then
      nmcli connection modify "$VERIZON_CONN" 802-11-wireless.channel "$CLIENT_AP_CHANNEL" >/dev/null
    fi
  fi

  echo "Connected WiFi client '$CLIENT_IFACE' using profile '$VERIZON_CONN'."
  CLIENT_CONNECTED=1
}

client_gateway() {
  nmcli -g IP4.GATEWAY device show "$CLIENT_IFACE" | awk 'NF {print; exit}'
}

client_phy() {
  local phy_path="/sys/class/net/${CLIENT_IFACE}/phy80211"
  if [[ -e "$phy_path" ]]; then
    basename "$(readlink -f "$phy_path")"
    return
  fi

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
  if device_exists "$AP_IFACE"; then
    return
  fi

  local existing_ap_iface
  existing_ap_iface="$(first_ap_candidate_iface || true)"
  if [[ -n "$existing_ap_iface" ]]; then
    AP_IFACE="$existing_ap_iface"
    nmcli device set "$AP_IFACE" managed yes >/dev/null || true
    return
  fi

  if [[ "$CLIENT_CONNECTED" != "1" && -n "$CLIENT_IFACE" ]]; then
    AP_IFACE="$CLIENT_IFACE"
    CLIENT_IFACE=""
    nmcli device set "$AP_IFACE" managed yes >/dev/null || true
    return
  fi

  local phy
  phy="$(client_phy)"
  if [[ -z "$phy" ]]; then
    echo "Could not resolve WiFi phy for AP creation." >&2
    exit 1
  fi

  iw phy "$phy" interface add "$AP_IFACE" type __ap
  nmcli device set "$AP_IFACE" managed yes >/dev/null || true
  sleep 1

  if ! device_exists "$AP_IFACE"; then
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
    802-11-wireless-security.proto rsn \
    802-11-wireless-security.pairwise ccmp \
    802-11-wireless-security.group ccmp \
    802-11-wireless-security.pmf 1 \
    802-11-wireless-security.psk "$AP_PASSWORD" \
    ipv4.method shared \
    ipv6.method ignore >/dev/null

  local channel band
  if [[ "$SYNC_AP_CHANNEL" == "1" && "$CLIENT_CONNECTED" == "1" ]]; then
    channel="$(client_channel || true)"
    band="$(client_band || true)"
  elif [[ -n "$CLIENT_AP_BAND" && -n "$CLIENT_AP_CHANNEL" ]]; then
    channel="$CLIENT_AP_CHANNEL"
    band="$CLIENT_AP_BAND"
  else
    channel=""
    band=""
  fi
  if [[ -n "${channel:-}" && -n "${band:-}" ]]; then
    nmcli connection modify "$AP_CONN" \
      802-11-wireless.band "$band" \
      802-11-wireless.channel "$channel" >/dev/null
  fi

  nmcli connection up "$AP_CONN" ifname "$AP_IFACE" >/dev/null

  echo "ALPR access point enabled on $AP_IFACE."
  echo "  SSID: $AP_SSID"
  echo "  Password: $AP_PASSWORD"
}

ensure_internet_sharing() {
  if [[ "$EXPLICIT_NAT" != "1" ]]; then
    sysctl -w net.ipv4.ip_forward=1 >/dev/null
    echo "NetworkManager shared-mode AP enabled. Explicit policy NAT is disabled."
    return
  fi

  if [[ "$CLIENT_CONNECTED" != "1" ]]; then
    sysctl -w net.ipv4.ip_forward=1 >/dev/null
    echo "Local AP enabled without Verizon upstream. Internet sharing will use the system default route if NetworkManager provides one."
    return
  fi

  local upstream_iface="${INTERNET_UPSTREAM_IFACE:-$CLIENT_IFACE}"
  local gateway

  gateway="$(client_gateway || true)"
  if [[ -z "$gateway" ]]; then
    echo "No IPv4 gateway found for upstream interface $CLIENT_IFACE; skipping explicit NAT setup." >&2
    return
  fi

  sysctl -w net.ipv4.ip_forward=1 >/dev/null

  ip route replace default via "$gateway" dev "$upstream_iface" table "$AP_POLICY_TABLE"
  ip rule del from "$AP_SUBNET" lookup "$AP_POLICY_TABLE" priority "$AP_POLICY_PRIORITY" >/dev/null 2>&1 || true
  ip rule add from "$AP_SUBNET" lookup "$AP_POLICY_TABLE" priority "$AP_POLICY_PRIORITY"

  iptables -t nat -C POSTROUTING -s "$AP_SUBNET" -o "$upstream_iface" -j MASQUERADE >/dev/null 2>&1 || \
    iptables -t nat -A POSTROUTING -s "$AP_SUBNET" -o "$upstream_iface" -j MASQUERADE
  iptables -C FORWARD -i "$AP_IFACE" -o "$upstream_iface" -j ACCEPT >/dev/null 2>&1 || \
    iptables -A FORWARD -i "$AP_IFACE" -o "$upstream_iface" -j ACCEPT
  iptables -C FORWARD -i "$upstream_iface" -o "$AP_IFACE" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT >/dev/null 2>&1 || \
    iptables -A FORWARD -i "$upstream_iface" -o "$AP_IFACE" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT

  echo "Explicit AP internet sharing enabled: $AP_SUBNET -> $upstream_iface via $gateway"
}

show_status() {
  nmcli device status
  echo
  if [[ "${ALPR_WIFI_VERBOSE_STATUS:-0}" == "1" ]]; then
    iw dev || true
    echo
  fi
  nmcli -t -f GENERAL.DEVICE,IP4.ADDRESS device show | awk -F: '
    $1 == "GENERAL.DEVICE" {dev=$2}
    $1 ~ /^IP4.ADDRESS/ && dev != "lo" {
      split($2, cidr, "/")
      printf "%s -> http://%s:%s\n", dev, cidr[1], ENVIRON["ALPR_DASHBOARD_PORT"] ? ENVIRON["ALPR_DASHBOARD_PORT"] : "8080"
    }'
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
  if [[ "$VERIZON_ENABLE" == "1" ]]; then
    stop_ap_connection
  fi
  ensure_client_connection
  ensure_ap_connection
  ensure_internet_sharing
}

down() {
  require_root
  nmcli connection down "$AP_CONN" >/dev/null 2>&1 || true
  nmcli connection delete "$AP_CONN" >/dev/null 2>&1 || true
  if device_exists "$AP_IFACE"; then
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
  ap-up)
    VERIZON_ENABLE=0
    driver_up
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
    echo "Usage: $0 {up|driver-up|wifi-up|ap-up|down|status}" >&2
    exit 1
    ;;
esac
