#!/usr/bin/env bash
# Flash one robot base (RP2040) from the Mac and check that it answers.
#
#   tools/robot/flash_base.sh            # flash the base plugged into the Mac
#   tools/robot/flash_base.sh --build    # rebuild the firmware first (needs colima)
#
# Uses the firmware checkout at ROBOT_FIRMWARE_DIR (robot.env). Needs
# picotool, and docker for --build (on macOS through colima).
#
# The base's External/Phone switch must be DOWN (External) and the cable in
# the base's external USB port. A running base is rebooted into its
# bootloader automatically; only a base that shows nothing at all needs
# BOOTSEL held while plugging in.
set -euo pipefail
source "$(dirname "$0")/common.sh"
robot_need ROBOT_FIRMWARE_DIR
FW=$ROBOT_FIRMWARE_DIR
UF2=$FW/build/robot.uf2

if [ "${1:-}" = --build ]; then
  if [ "$(uname)" = Darwin ]; then
    colima status >/dev/null 2>&1 || colima start --cpu 4 --memory 4
  fi
  case $(uname -m) in arm64|aarch64) TAG=latest-arm64 ;; *) TAG=latest ;; esac
  docker run --rm -v "$FW":/project -w /project \
    topher217/smartphone-robot-firmware:$TAG bash -c \
    "rm -rf build && mkdir build && cd build && \
     cmake .. -DLOGGER=USB -DBOOT_TESTS=OFF >/dev/null && make -j4 2>&1 | tail -1"
fi
[ -f "$UF2" ] || { echo "no $UF2; run with --build"; exit 1; }
echo "firmware: $(git -C "$FW" log --oneline -1)"

picotool load -f -x "$UF2" 2>&1 | tr '\r' '\n' | grep -E 'OK|rebooted|ERROR|No accessible' | tail -2
# The base's USB serial: /dev/cu.usbmodem* on macOS, /dev/ttyACM* on Linux.
PAT='^(cu\.usbmodem|ttyACM)'
for _ in $(seq 1 100); do ls /dev | grep -qE "$PAT" && break; sleep 0.2; done
dev=/dev/$(ls /dev | grep -E "$PAT" | head -1)
[ "$dev" != /dev/ ] || robot_die "the base did not come back as a serial device"
python3 - "$dev" <<'EOF'
import os, select, sys, time, tty
fd = os.open(sys.argv[1], os.O_RDWR | os.O_NOCTTY); tty.setraw(fd)
def read(t):
    b, end = b'', time.time() + t
    while time.time() < end:
        if select.select([fd], [], [], 0.01)[0]:
            b += os.read(fd, 65536)
    return b
t0 = time.time()
while time.time() - t0 < 30:
    os.write(fd, bytes([0xFE, 0x01, 0, 0, 0xFF]))  # SET_MOTOR_LEVEL 0, 0
    if read(0.2).startswith(b'\xfe\x01'):
        print('OK: base answers, %.1f s after boot' % (time.time() - t0))
        sys.exit(0)
print('FAIL: no answer in 30 s'); sys.exit(1)
EOF
