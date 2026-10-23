#!/usr/bin/env bash
# A training job for live demos: start it once, then restart training from
# scratch from the phone's dashboard (http://<phone>:8080/dash, RESTART) as
# often as you like, without giving up the GPU.
#
#   tools/robot/demo.sh                     # e<N>_demo, robot defaults, 65 Hz
#   tools/robot/demo.sh demo_carpet [main.py flags]
#   CONFIGS="robot robot_command" tools/robot/demo.sh demo_cmd
#
# The job runs the learner under tools/robot/demo_supervisor.py, which keeps
# the next session started and parked beside the live one, so a restart is
# live again in a few seconds, from random weights, with a fresh dashboard.
# Sessions go in <run dir>/s01, s02, ...; <run dir>/sessions.tsv lists them.
# Stop with tools/robot/stop.sh. Otherwise as tools/robot/train.sh.
set -euo pipefail
export DEMO=1
# Each session runs until restarted, not to a step count.
export STEPS=${STEPS:-100000000}
if [ $# -eq 0 ] || [[ $1 == -* ]]; then
  set -- demo "$@"
fi
exec "$(dirname "$0")/train.sh" "$@"
