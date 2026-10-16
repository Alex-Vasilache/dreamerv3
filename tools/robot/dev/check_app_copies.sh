#!/usr/bin/env bash
# The phone app runs three modules of dreamerv3/deploy/ as verbatim copies
# (Chaquopy cannot import from this repo). Check they still match; with --fix,
# copy this repo's versions over the app's.
#
#   tools/robot/dev/check_app_copies.sh [--fix]
set -uo pipefail
source "$(dirname "$0")/../common.sh"
robot_need ROBOT_ANDROID_DIR
APP_PY=$ROBOT_ANDROID_DIR/apps/dreamerBridge/src/main/python
status=0
for f in bridge_link.py numpy_policy.py policy_runner.py; do
  if cmp -s "$ROBOT_ROOT/dreamerv3/deploy/$f" "$APP_PY/$f"; then
    echo "same     $f"
  elif [ "${1:-}" = --fix ]; then
    cp "$ROBOT_ROOT/dreamerv3/deploy/$f" "$APP_PY/$f"; echo "copied   $f"
  else
    echo "DIFFERS  $f   (diff $ROBOT_ROOT/dreamerv3/deploy/$f $APP_PY/$f)"; status=1
  fi
done
exit $status
