#!/usr/bin/env bash
# Put Python modules on the phone so the dreamerBridge app imports them
# instead of its own copies (it puts files/python first on sys.path), then
# restart the app. A Python change then costs seconds, not a reinstall.
#
#   tools/robot/dev/push_python.sh path/to/module.py [...]   # push and restart
#   tools/robot/dev/push_python.sh --list                    # what is overriding
#   tools/robot/dev/push_python.sh --clear                   # back to the app's own
#
# Restarting the app ends a running session; the demo supervisor starts the
# next one. Overrides outlive app updates: --clear after installing a build
# that already contains the change.
set -euo pipefail
source "$(dirname "$0")/../common.sh"
DIR=$PHONE_FILES/python
case "${1:-}" in
  --list) adb shell ls -la "$DIR" 2>/dev/null || echo "no overrides"; exit 0 ;;
  --clear) adb shell rm -rf "$DIR"; echo "overrides cleared"; RESTART=1 ;;
  '') robot_die "usage: push_python.sh FILE.py [...] | --list | --clear" ;;
  *)
    adb shell mkdir -p "$DIR"
    for f in "$@"; do
      [ -f "$f" ] || robot_die "no such file: $f"
      adb push "$f" "$DIR/" >/dev/null && echo "pushed $(basename "$f")"
    done ;;
esac
adb shell am force-stop "$APP"
adb shell monkey -p "$APP" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1
echo "restarted $APP"
