#!/usr/bin/env bash

set -euo pipefail

ACTION="${1:-status}"
if [[ $# -gt 0 ]]; then
  shift
fi

HOTSPOT_CONN="${ALPR_WIFI_AP_PROFILE:-ALPR-Field-Unit-AP}"
ETHERNET_CONN="${ALPR_ETHERNET_PROFILE:-ALPR-Field-Unit-Ethernet}"
DEFAULT_SSID="${ALPR_WIFI_AP_SSID:-ALPR-Field-Unit}"
DEFAULT_PASSWORD="${ALPR_WIFI_AP_PASSWORD:-ALPRAccess123}"
DEFAULT_PORT="${ALPR_DASHBOARD_PORT:-8080}"

require_nmcli() {
  if ! command -v nmcli >/dev/null 2>&1; then
    echo "nmcli is required for network access setup." >&2
    exit 1
  fi
}

connection_exists() {
  local connection_name="$1"
  nmcli -t -f NAME connection show | grep -Fxq "$connection_name"
}

delete_connection() {
  local connection_name="$1"
  if connection_exists "$connection_name"; then
    nmcli connection delete "$connection_name" >/dev/null 2>&1 || true
  fi
}

device_type() {
  local interface_name="$1"
  nmcli -t -f DEVICE,TYPE device status | awk -F: -v iface="$interface_name" '$1 == iface { print $2; exit }'
}

require_device_type() {
  local interface_name="$1"
  local expected_type="$2"
  local actual_type
  actual_type="$(device_type "$interface_name")"
  if [[ -z "$actual_type" ]]; then
    echo "Interface $interface_name was not found by NetworkManager." >&2
    exit 1
  fi
  if [[ "$actual_type" != "$expected_type" ]]; then
    echo "Interface $interface_name is type $actual_type, expected $expected_type." >&2
    exit 1
  fi
}

show_urls() {
  local interface_name="${1:-}"
  local port="${2:-$DEFAULT_PORT}"

  echo "Dashboard URLs:"
  echo "  http://127.0.0.1:${port}"

  if [[ -n "$interface_name" ]]; then
    ip -o -4 addr show dev "$interface_name" up | awk -v port="$port" '{split($4, cidr, "/"); printf "  http://%s:%s\n", cidr[1], port}'
    return
  fi

  ip -o -4 addr show up | awk -v port="$port" '$2 != "lo" {split($4, cidr, "/"); printf "  %s -> http://%s:%s\n", $2, cidr[1], port}'
}

hotspot_up() {
  local interface_name="${1:-${ALPR_WIFI_AP_INTERFACE:-}}"
  local ssid="${2:-$DEFAULT_SSID}"
  local password="${3:-$DEFAULT_PASSWORD}"

  if [[ -z "$interface_name" ]]; then
    echo "Usage: $0 hotspot-up <wifi-interface> [ssid] [password]" >&2
    exit 1
  fi
  if [[ ${#password} -lt 8 ]]; then
    echo "Hotspot password must be at least 8 characters." >&2
    exit 1
  fi

  require_device_type "$interface_name" "wifi"
  delete_connection "$HOTSPOT_CONN"

  nmcli device wifi hotspot \
    ifname "$interface_name" \
    con-name "$HOTSPOT_CONN" \
    ssid "$ssid" \
    password "$password" >/dev/null

  nmcli connection modify "$HOTSPOT_CONN" connection.autoconnect yes ipv6.method ignore >/dev/null

  echo "Hotspot enabled on $interface_name"
  echo "  SSID: $ssid"
  echo "  Password: $password"
  show_urls "$interface_name"
}

hotspot_down() {
  if connection_exists "$HOTSPOT_CONN"; then
    nmcli connection down "$HOTSPOT_CONN" >/dev/null 2>&1 || true
    nmcli connection delete "$HOTSPOT_CONN" >/dev/null 2>&1 || true
  fi
  echo "Hotspot profile removed: $HOTSPOT_CONN"
}

ethernet_up() {
  local interface_name="${1:-${ALPR_ETHERNET_INTERFACE:-}}"

  if [[ -z "$interface_name" ]]; then
    echo "Usage: $0 ethernet-up <ethernet-interface>" >&2
    exit 1
  fi

  require_device_type "$interface_name" "ethernet"
  delete_connection "$ETHERNET_CONN"

  nmcli connection add \
    type ethernet \
    ifname "$interface_name" \
    con-name "$ETHERNET_CONN" \
    autoconnect yes \
    ipv4.method shared \
    ipv6.method ignore >/dev/null

  nmcli connection up "$ETHERNET_CONN" >/dev/null

  echo "Ethernet shared link enabled on $interface_name"
  show_urls "$interface_name"
}

ethernet_down() {
  if connection_exists "$ETHERNET_CONN"; then
    nmcli connection down "$ETHERNET_CONN" >/dev/null 2>&1 || true
    nmcli connection delete "$ETHERNET_CONN" >/dev/null 2>&1 || true
  fi
  echo "Ethernet shared profile removed: $ETHERNET_CONN"
}

status() {
  nmcli device status
  echo
  show_urls
}

require_nmcli

case "$ACTION" in
  hotspot-up)
    hotspot_up "$@"
    ;;
  hotspot-down)
    hotspot_down
    ;;
  ethernet-up)
    ethernet_up "$@"
    ;;
  ethernet-down)
    ethernet_down
    ;;
  status)
    status
    ;;
  *)
    echo "Usage: $0 {status|hotspot-up <wifi-iface> [ssid] [password]|hotspot-down|ethernet-up <eth-iface>|ethernet-down}" >&2
    exit 1
    ;;
esac