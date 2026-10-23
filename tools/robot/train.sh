#!/usr/bin/env bash
# Start a robot training run and point the phone at it.
#
#   tools/robot/train.sh                          # 65 Hz balancing, default name
#   HZ=25 tools/robot/train.sh [name] [extra main.py flags]
#   CONFIGS="robot robot_command" tools/robot/train.sh cmd
#   RUN_DIR=<previous run dir> tools/robot/train.sh resume   # continue a run
#   NO_BASE=1 HZ=100 tools/robot/train.sh linktest           # phone alone, no wheels
#   DEMO=1 tools/robot/train.sh demo   # restartable from the dashboard (demo.sh)
#   GPU=p100 tools/robot/train.sh                 # Slurm: use ROBOT_SLURM_FLAGS_p100
#
# Where the learner runs (this computer or a Slurm cluster) and where runs are
# kept is set in tools/robot/robot.env; see tools/robot/robot.env.example.
# Needs the phone on adb (tools/robot/phone_connect.sh) and, unless NO_BASE,
# on a switched-on base. Stop with tools/robot/stop.sh.
set -euo pipefail
source "$(dirname "$0")/common.sh"

# The control rate, one number for both sides: the phone gets it as max_hz in
# trainer.json, the learner as --env.robot.hz (which sets the step-counted
# horizon and episode length; configs.yaml, `robot`). HZ, else an
# --env.robot.hz among the flags, else 65.
prev=
for a in "$@"; do
  case $a in --env.robot.hz=*) FLAG_HZ=${a#*=} ;; esac
  [ "$prev" = --env.robot.hz ] && FLAG_HZ=$a
  prev=$a
done
if [ -n "${FLAG_HZ:-}" ] && [ -n "${HZ:-}" ] && [ "${FLAG_HZ%.*}" != "${HZ%.*}" ]; then
  robot_die "HZ=$HZ but --env.robot.hz $FLAG_HZ; give the rate once"
fi
HZ=${HZ:-${FLAG_HZ:-65}}
HZ=${HZ%.*}
[[ $HZ =~ ^[0-9]+$ ]] || robot_die "HZ must be a whole number of Hz, not '$HZ'"
[ -n "${FLAG_HZ:-}" ] || set -- "$@" --env.robot.hz "$HZ"
CONFIGS=${CONFIGS:-robot}
STEPS=${STEPS:-100000}
NO_BASE=${NO_BASE:-}

NAME=robot_${HZ}hz
if [ $# -gt 0 ] && [[ $1 != -* ]]; then NAME=$1; shift; fi
# Optional experiment numbering: e<highest + 1>_<name>. A name that already
# starts with e<digits>_ is kept, and link tests are never numbered.
if [ -n "$ROBOT_NUMBERS_CMD" ] && [[ ! $NAME =~ ^e[0-9]+_ ]] && [ -z "$NO_BASE" ]; then
  LAST=$(trainer_sh "$ROBOT_NUMBERS_CMD" 2>/dev/null \
    | tr -c '[:alnum:]' '\n' | grep -xE 'e[0-9]{3,5}' | tr -d e | sort -n | tail -1 || true)
  [ -n "$LAST" ] || robot_die "ROBOT_NUMBERS_CMD printed no e<N> names"
  NAME=e$((LAST + 1))_$NAME
fi
[[ $NAME =~ ^[A-Za-z0-9_.-]+$ ]] || robot_die "name may only use letters, digits, _ . -"

command -v adb >/dev/null || robot_die "adb not found (Android platform-tools)"
adb get-state >/dev/null 2>&1 || robot_die "no phone on adb: run tools/robot/phone_connect.sh"
if [ -z "$NO_BASE" ]; then
  # dumpsys usb can throw while dumping its connection records and never print
  # host_connected, so a device node besides the root hubs (00N/001) also counts.
  { adb shell dumpsys usb 2>&1 | grep 'host_connected=true' >/dev/null \
    || adb shell 'ls -d /dev/bus/usb/*/*' 2>/dev/null | grep -v '/001$' >/dev/null; } \
    || robot_die "no base on the phone: seat it and switch the base on (or NO_BASE=1)"
fi
# A dozing phone blocks the app's network even with the app on screen.
adb shell input keyevent KEYCODE_WAKEUP; adb shell wm dismiss-keyguard

# Variables the job reads. Quoted for the remote shell.
q() { printf '%q' "$1"; }
JOB_ENV="ROBOT_RUNS=$(q "$ROBOT_RUNS") ROBOT_SETUP=$(q "$ROBOT_SETUP") CONFIGS=$(q "$CONFIGS") STEPS=$STEPS${DEMO:+ DEMO=1}"
[ -n "${RUN_DIR:-}" ] && JOB_ENV+=" RUN_DIR=$(q "$RUN_DIR")"
ARGS=""; for a in "$@"; do ARGS+=" $(q "$a")"; done
# Leftovers of a crashed or stopped run that block a resume: the learner exits
# at once on an error file, and waits for a stale actor_step to be reached.
CLEAN=${RUN_DIR:+rm -f $(q "$RUN_DIR/logdir/error_learner") $(q "$RUN_DIR/logdir/online_shared/actor_step");}

echo "== starting $NAME ($ROBOT_TRAINER)"
if [ "$ROBOT_TRAINER" = slurm ]; then
  if [ -n "${GPU:-}" ]; then
    var=ROBOT_SLURM_FLAGS_$GPU
    [ -n "${!var:-}" ] || robot_die "GPU=$GPU but $var is not set in robot.env"
    FLAGS=${!var}
  else
    FLAGS=${ROBOT_SLURM_FLAGS:-}
  fi
  LOGS=$ROBOT_RUNS/slurm_logs
  JOB=$(trainer_sh "cd $(q "$ROBOT_REMOTE_CODE") && git pull -q && mkdir -p $(q "$LOGS") && $CLEAN \
    $JOB_ENV CODE=$(q "$ROBOT_REMOTE_CODE") \
    sbatch --parsable -J $NAME -o $(q "$LOGS")/%x-%j.out -e $(q "$LOGS")/%x-%j.err $FLAGS \
    tools/robot/trainer_job.sh$ARGS")
  JOB=${JOB%%;*}
  echo "job $JOB, log: $LOGS/$NAME-$JOB.out"
else
  mkdir -p "$ROBOT_RUNS"
  LOG=$ROBOT_RUNS/$NAME-$(date +%Y%m%d_%H%M%S).log
  # The learner outlives this script: stop it with tools/robot/stop.sh.
  # exec, so $! is the job's own pid, which names its endpoint file.
  bash -c "$CLEAN $JOB_ENV NAME=$NAME exec nohup bash $(q "$ROBOT_TOOLS/trainer_job.sh")$ARGS" \
    >"$LOG" 2>&1 </dev/null &
  JOB=local$!
  echo "learner pid ${JOB#local}, log: $LOG"
fi

echo "== waiting for the learner to start"
EP=$ROBOT_RUNS/robot_endpoints/job_$JOB
until trainer_sh "test -f $(q "$EP")"; do
  if [ "$ROBOT_TRAINER" = local ] && ! kill -0 "${JOB#local}" 2>/dev/null; then
    tail -20 "$LOG"; robot_die "the learner exited; log above"
  fi
  sleep 5
done
eval "$(trainer_sh "cat $(q "$EP")")"   # NODE, PORT, RUN_DIR
if [ "$ROBOT_TRAINER" = slurm ]; then
  IP=$(trainer_sh "getent hosts $NODE" | awk '{print $1; exit}')
else
  IP=$(local_ip)
fi
[ -n "$IP" ] || robot_die "could not find an address for $NODE"
echo "node $NODE ($IP), port $PORT, run dir $RUN_DIR"

echo "== pointing the phone at it"
# What the running app (if any) was started with, read before it is replaced.
OLD=$(adb shell cat $PHONE_FILES/trainer.json 2>/dev/null || true)
tmp=$(mktemp -d)
echo "{\"ip\": \"$IP\", \"port\": $PORT, \"max_hz\": $HZ${NO_BASE:+, \"no_base\": true}}" \
  > "$tmp/trainer.json"
adb shell mkdir -p $PHONE_FILES
adb push "$tmp/trainer.json" $PHONE_FILES/ >/dev/null
# A running app rereads trainer.json on every reconnect attempt, so leave it
# be: killing it can make the phone stop powering the base (a replug fixes it).
# Except across a no_base change, which the app reads only at start.
old_nobase=; [[ $OLD == *'"no_base": true'* ]] && old_nobase=1
if adb shell pidof $APP >/dev/null && [ "$old_nobase" != "$NO_BASE" ]; then
  echo "restarting the app: it is in the other base mode (no_base=${old_nobase:-0})"
  adb shell am force-stop $APP
fi
if ! adb shell pidof $APP >/dev/null; then
  adb shell monkey -p $APP -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 \
    || robot_die "could not start $APP: is the dreamerBridge app installed? (ROBOT.md, step 3)"
  [ -n "$NO_BASE" ] || "$ROBOT_TOOLS/phone_allow_usb.sh" 10
fi

echo "== waiting for the phone to connect (the learner takes ~1 min to start)"
until adb logcat -d | grep -q "dreamerBridge: connected to $IP:$PORT"; do sleep 5; done
echo "connected."
echo "scores:      $( [ "$ROBOT_TRAINER" = slurm ] && echo "ssh $ROBOT_SSH ")tail -f $RUN_DIR/logdir/scores.jsonl"
echo "tensorboard: tools/robot/tensorboard.sh"
echo "stop:        tools/robot/stop.sh"
