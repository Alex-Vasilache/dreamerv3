#!/usr/bin/env bash
# Tap OK on the phone's "Allow <app> to access Pico?" dialog, if it is showing.
#
# An app started from the Mac (adb am start) has to request USB permission,
# and that dialog has no "always" box, so it appears on every launch. Run this
# right after launching dreamerBridge or pidBalancer. Waits up to $1 seconds
# (default 10) for the dialog; exits 0 whether or not one appeared.
set -euo pipefail
wait_s=${1:-10}
for _ in $(seq 1 "$wait_s"); do
  if adb shell dumpsys window | grep -q 'mCurrentFocus=.*UsbPermissionActivity'; then
    adb shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1
    xy=$(adb shell cat /sdcard/ui.xml | python3 -c '
import re, sys
x = sys.stdin.read()
m = re.search(r"text=\"OK\"[^>]*bounds=\"\[(\d+),(\d+)\]\[(\d+),(\d+)\]\"", x)
if m:
    print((int(m[1]) + int(m[3])) // 2, (int(m[2]) + int(m[4])) // 2)')
    if [ -n "$xy" ]; then
      adb shell input tap $xy
      echo "allowed USB access"
      exit 0
    fi
  fi
  sleep 1
done
echo "no USB permission dialog"
