#!/usr/bin/env bash
# Fetch what saved policies did on the robot. The Dreamer Player (and PLAY on
# the dashboard) writes every step to the phone's recordings/<name>_<time>.csv
# with a .json of the policy's settings beside it; no trainer is involved.
#
#   tools/robot/pull_recordings.sh            # floor from each policy's sim run
#   tools/robot/pull_recordings.sh table      # the robot was on the table
#
# Copies new recordings to ~/logdir/robot_recordings/ and adds each to
# ~/logdir/robot_recordings/sysid/<name, - for _>_<floor>/balance.csv, the layout
# sysid_fit.py and the sim comparisons read (tools/robot/dev/
# recording_to_sysid.py). With a Slurm trainer (robot.env), they are also
# copied to $ROBOT_RUNS/sysid/recordings/ on the trainer host.
set -euo pipefail
source "$(dirname "$0")/common.sh"
FLOOR=${1:-}
OUT="$HOME/logdir/robot_recordings"
mkdir -p "$OUT/sysid"
cd "$ROBOT_ROOT"

for f in $(adb shell ls "$PHONE_FILES/recordings/" 2>/dev/null | tr -d '\r' | grep '\.csv$'); do
  base=${f%.csv}
  [ -f "$OUT/$f" ] && continue
  adb pull "$PHONE_FILES/recordings/$f" "$PHONE_FILES/recordings/$base.json" "$OUT/" >/dev/null
  name=$("$(robot_python)" -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$OUT/$base.json")
  floor=${FLOOR:-$("$(robot_python)" -c "import json,sys; print(json.load(open(sys.argv[1])).get('sim_surface', 'carpet'))" "$OUT/$base.json")}
  "$(robot_python)" tools/robot/dev/recording_to_sysid.py "$OUT/$f" "$OUT/sysid/${name//_/-}_$floor"
done

if [ "${ROBOT_TRAINER:-}" = slurm ] && [ -n "${ROBOT_SSH:-}" ]; then
  ssh "$ROBOT_SSH" "mkdir -p $ROBOT_RUNS/sysid/recordings"
  rsync -a "$OUT/" "$ROBOT_SSH:$ROBOT_RUNS/sysid/recordings/"
  echo "copied to $ROBOT_SSH:$ROBOT_RUNS/sysid/recordings/"
fi
