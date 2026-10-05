#!/usr/bin/env bash
# Launch a robot training run with a stable log path, so another terminal can
# follow it live without knowing the timestamped directory name.
#
# By default the learner and actor run as separate processes so they can sit on
# different devices: the learner keeps the GPU while the actor's policy runs on
# CPU, where it does not queue behind a training batch. Sharing one device
# costs the robot real time -- measured 2026-09-02, the phone sat blocked up to
# 282ms waiting for an action, against 7ms of actual network. On macOS
# DREAMERV3_PLATFORM is the only thing that overrides the configured platform,
# which is why it is set here and not in the config.
#
#   tools/run_robot.sh [extra main.py flags...]
#   CONFIGS=robot tools/run_robot.sh --run.steps 3000
#   SPLIT=0 tools/run_robot.sh            # one process, both on the GPU
#   ACTOR_PLATFORM=metal tools/run_robot.sh   # actor on the GPU too
#
# Then, from any other terminal:
#
#   tail -f ~/logdir/latest/train.log
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${DREAMER_LOGDIR:-$HOME/logdir}"
# Overridable: the Saion bridge needs a stable path to sync, not a new
# timestamped directory per launch.
LOGDIR="${LOGDIR:-$BASE/robot_$(date +%Y%m%dT%H%M%S)}"
CONFIGS="${CONFIGS:-robot_balance}"
# 1 = learner and actor here; 0 = one process; actor = actor only, for when the
# learner is on Saion and tools/robot_bridge_sync.sh carries the shared state.
SPLIT="${SPLIT:-1}"
# Which device the actor's policy runs on. Measured on hardware 2026-09-03 at
# repeat 1, both with the learner idled (cpu 33.2Hz against METAL 33.3Hz) and
# with it training (METAL 44.2Hz then cpu 44.7Hz back to back): the same either
# way. Compare only back to back -- the WiFi drifts enough between runs to
# invent a 10Hz difference that is not the device. The actor's whole step path
# costs ~1ms of a 20ms budget, so cpu stays the default: it leaves the GPU
# entirely to the learner and costs nothing.
ACTOR_PLATFORM="${ACTOR_PLATFORM:-cpu}"
# jax registers the Apple backend as 'METAL' and rejects 'metal', so accept the
# spelling a person would type.
case "$ACTOR_PLATFORM" in metal|gpu|Metal|GPU) ACTOR_PLATFORM=METAL;; esac
PY="$ROOT/.venv/bin/python"

mkdir -p "$LOGDIR"
# Replace rather than nest, so `latest` always names a run and never a symlink
# living inside the previous one.
ln -sfn "$LOGDIR" "$BASE/latest"
: > "$LOGDIR/train.log"

mode='single process, learner and actor share the GPU'
[ "$SPLIT" = 1 ] && mode="split, learner on GPU and actor on $ACTOR_PLATFORM"
[ "$SPLIT" = actor ] && mode="actor only on $ACTOR_PLATFORM (learner elsewhere)"
echo "Logdir:  $LOGDIR"
echo "Configs: $CONFIGS"
echo "Mode:    $mode"
echo "Watch:   tail -f $BASE/latest/train.log"
echo

# Tag each process's output and fold it into one combined log, so a single
# tail -f shows both halves interleaved.
run() {  # run <tag> <logfile> <command...>
  local tag=$1 file=$2
  shift 2
  "$@" 2>&1 \
    | awk -v t="[$tag] " '{print t $0; fflush()}' \
    | tee "$LOGDIR/$file" >> "$LOGDIR/train.log"
}

pids=()
trap 'kill "${pids[@]}" 2>/dev/null' INT TERM

if [ "$SPLIT" = actor ]; then
  (
    export DREAMERV3_PLATFORM=$ACTOR_PLATFORM
    run actor actor.log \
      "$PY" -u "$ROOT/dreamerv3/main.py" --logdir "$LOGDIR" \
      --configs $CONFIGS --script online_actor "$@"
  ) &
  pids+=($!)
elif [ "$SPLIT" = 1 ]; then
  run learner learner.log \
    "$PY" -u "$ROOT/dreamerv3/main.py" --logdir "$LOGDIR" \
    --configs $CONFIGS --script online_learner "$@" &
  pids+=($!)
  # The actor blocks on the learner_ready file, so it can start immediately;
  # the subshell keeps the platform override off the learner.
  (
    export DREAMERV3_PLATFORM=$ACTOR_PLATFORM
    run actor actor.log \
      "$PY" -u "$ROOT/dreamerv3/main.py" --logdir "$LOGDIR" \
      --configs $CONFIGS --script online_actor "$@"
  ) &
  pids+=($!)
else
  run train single.log \
    "$PY" -u "$ROOT/dreamerv3/main.py" --logdir "$LOGDIR" \
    --configs $CONFIGS "$@" &
  pids+=($!)
fi

wait
