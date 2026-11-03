#!/usr/bin/env bash
# Stop a robot run: the learner (Slurm job or local process) and the phone app.
#
#   tools/robot/stop.sh          # the latest run
#   tools/robot/stop.sh <job>    # a given Slurm job id, or local<pid>
#
# The base stops the wheels by itself 250 ms after commands stop.
set -uo pipefail
source "$(dirname "$0")/common.sh"

JOB=${1:-$(trainer_sh "sed -n 's/^JOB=//p' '$ROBOT_RUNS/robot_endpoints/latest' 2>/dev/null")}
if [ -z "$JOB" ]; then
  echo "no running robot job found in $ROBOT_RUNS/robot_endpoints"
elif [[ $JOB == local* ]]; then
  kill "${JOB#local}" 2>/dev/null && echo "stopped learner pid ${JOB#local}" \
    || echo "learner pid ${JOB#local} is not running"
else
  trainer_sh "scancel $JOB" && echo "cancelled job $JOB"
fi
[ -f "$RELAY_PID" ] && relay_stop && echo "stopped the relay"
if command -v adb >/dev/null && adb get-state >/dev/null 2>&1; then
  adb shell am force-stop $APP && echo "stopped the phone app"
  # A NO_BASE link test leaves the app in no-base mode, where saved policies
  # run with the wheels off; put it back.
  if adb shell cat $PHONE_FILES/trainer.json 2>/dev/null | grep -q '"no_base": true'; then
    adb shell "sed -i 's/, \"no_base\": true//' $PHONE_FILES/trainer.json" \
      && echo "phone back in base mode"
  fi
fi
