#!/usr/bin/env bash
# One command to start a robot training session. Connect the phone, run this.
#
#   tools/start_robot.sh            # Saion learner if reachable, else local
#   MODE=local tools/start_robot.sh # force the learner onto this Mac
#   MODE=saion tools/start_robot.sh # insist on Saion, fail if it is not there
#
# Either way the actor runs here beside the robot at ~45Hz and TensorBoard comes
# up on http://localhost:6006. The difference is only where the gradients are
# computed.
#
# Why the fallback exists: the OIST jump host stalls for minutes at a time, and
# a robot session is worth more than the extra throughput -- measured, the M5
# sustains ~1.6k samples/s against a train_ratio target that wants 23k, so the
# local run learns more slowly but it does learn.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-auto}"
LOCAL="${LOGDIR:-$HOME/logdir/robot_local}"
CONFIGS="${CONFIGS:-robot_daydreamer}"
SSH="ssh -o BatchMode=yes -o ControlPath=none -o ConnectTimeout=15"
# saion when the internal name resolves, saion-ext otherwise -- see
# docs/SAION_BRIDGE_FINDINGS.md: the login.oist.jp path is policed to ~10 KB/s
# aggregate and cannot carry the bridge, the direct host does 2.3 MB/s.
if [ -z "${REMOTE:-}" ]; then
  if [ -n "$(dig +short saion.oist.jp 2>/dev/null)" ]; then
    REMOTE=saion
  else
    REMOTE=saion-ext
  fi
fi
LOGS="${LOGS:-$HOME/logdir/.robot_session}"
mkdir -p "$LOGS"

stop_stale() {
  pkill -f 'dreamerv3/main.py' 2>/dev/null
  pkill -f robot_bridge_sync 2>/dev/null
  pkill -f fake_robot_phone 2>/dev/null
  sleep 2
}

remote_run_dir() {
  # The newest robot learner job that is actually running, or nothing.
  #
  # Sorted by job id and filtered on RUNNING, and stale endpoint files are
  # deleted as they are found: a cancelled job's file can outlive it (the
  # sbatch EXIT trap does not fire on SIGKILL), and picking one silently sends
  # every chunk to a directory nobody is reading -- the robot looks perfect
  # while nothing learns.
  $SSH "$REMOTE" '
    for f in $(ls -1 /work/DoyaU/vasilache/work/robot_endpoints/job_* 2>/dev/null | sort -t_ -k2 -n); do
      [ -f "$f" ] || continue
      ( . "$f"
        if squeue -j "$JOB" -h -o %T 2>/dev/null | grep -q RUNNING; then
          echo "$RUN_DIR"
        else
          rm -f "$f"
        fi )
    done | tail -1' 2>/dev/null
}

stop_stale
rm -rf "$LOCAL"; mkdir -p "$LOCAL"

REMOTE_DIR=""
if [ "$MODE" != local ]; then
  echo "checking Saion..."
  REMOTE_DIR=$(remote_run_dir)
fi

if [ -n "$REMOTE_DIR" ]; then
  echo "Learner: Saion V100, $REMOTE_DIR"
  nohup "$ROOT/tools/robot_bridge_sync.sh" "$LOCAL" "$REMOTE_DIR/logdir" \
    > "$LOGS/bridge.log" 2>&1 &
elif [ "$MODE" = saion ]; then
  echo "No running Saion learner job. Submit one with:" >&2
  echo "  ssh $REMOTE 'cd /apps/unit/DoyaU/vasilache/apps/code/robot/dreamerv3 && \\" >&2
  echo "    CODE=\$PWD SCRIPT=online_learner sbatch sbatch/run_robot_v100.sbatch'" >&2
  exit 1
else
  echo "Learner: this Mac (Saion unreachable or no job running)"
  nohup "$ROOT/.venv/bin/python" -u "$ROOT/dreamerv3/main.py" --logdir "$LOCAL" \
    --configs $CONFIGS --script online_learner --run.steps 1000000 \
    > "$LOGS/learner.log" 2>&1 &
fi

nohup "$ROOT/tools/robot_tensorboard.sh" "$LOCAL" > "$LOGS/tensorboard.log" 2>&1 &
sleep 2
LOGDIR="$LOCAL" SPLIT=actor CONFIGS=$CONFIGS "$ROOT/tools/run_robot.sh" \
  --run.steps 1000000 > "$LOGS/actor.log" 2>&1 &

cat <<EOF

Logdir:      $LOCAL
TensorBoard: http://localhost:6006
Actor log:   tail -f $LOGS/actor.log
Progress:    tail -f $LOCAL/env0/timing.csv

Connect the phone now; the actor is listening on port 3000.
Stop everything with: tools/stop_robot.sh
EOF
