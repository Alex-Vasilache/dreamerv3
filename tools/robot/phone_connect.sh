#!/usr/bin/env bash
# Find the phone's wireless-debugging port and `adb connect` to it.
#
#   tools/robot/phone_connect.sh [ip]      # default: PHONE_IP from robot.env
#
# Android picks a new port every time Wireless debugging is switched on, and
# switches it off by itself after a reboot or a WiFi change. Nothing open in
# 30000-50000 means it is off: Settings -> Developer options -> Wireless debugging.
set -euo pipefail
source "$(dirname "$0")/common.sh"
IP=${1:-${PHONE_IP:-}}
[ -n "$IP" ] || robot_die "pass the phone's IP, or set PHONE_IP in tools/robot/robot.env"
adb devices | grep -q "^$IP:.*device$" && { echo "already connected"; exit 0; }
PORT=$(python3 - "$IP" <<'PY'
import concurrent.futures as cf, socket, sys
def probe(port):
  s = socket.socket(); s.settimeout(0.5)
  try:
    s.connect((sys.argv[1], port)); return port
  except OSError:
    return None
  finally:
    s.close()
with cf.ThreadPoolExecutor(512) as ex:
  print(next((p for p in ex.map(probe, range(30000, 50001)) if p), ''))
PY
)
[ -n "$PORT" ] || { echo "no adb port open on $IP: turn on Wireless debugging"; exit 1; }
adb connect "$IP:$PORT"
