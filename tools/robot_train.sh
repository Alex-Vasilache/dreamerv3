#!/usr/bin/env bash
# Start a robot training run on Saion and point the phone at it.
#
#   tools/robot_train.sh                                # -> e<next>_robot_25hz
#   HZ=50 CONFIGS="robot_daydreamer robot_fast robot_50hz" \
#     tools/robot_train.sh [name] [extra main.py flags]  # -> e<next>_<name>
#   tools/robot_train.sh e1300_mine ...                 # an explicit number is kept
#   RUN_DIR=<a previous run dir on /work> tools/robot_train.sh resume   # resume it
#   NO_BASE=1 HZ=100 tools/robot_train.sh linktest_100hz    # phone alone, no number
#
# The experiment number is the highest e<N> found on Saion in EXPERIMENTS*.md,
# the slurm log names and the last month's job names, plus one. A name that
# starts with e<digits>_ is used as given.
#
# Needs: `ssh saion` working, the phone on adb, a base on the phone.
# Stop with:  ssh saion scancel <job>; adb shell am force-stop jp.oist.abcvlib.dreamerBridge
set -euo pipefail
CONFIGS=${CONFIGS:-robot_daydreamer robot_fast robot_25hz}
STEPS=${STEPS:-100000}
HZ=${HZ:-25}
NAME=robot_${HZ}hz
if [ $# -gt 0 ] && [[ $1 != -* ]]; then NAME=$1; shift; fi
if [[ ! $NAME =~ ^e[0-9]+_ ]] && [ -z "${NO_BASE:-}" ]; then
  LAST=$(ssh saion 'cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3
    { cat EXPERIMENTS*.md
      ls /work/DoyaU/vasilache/work/slurm_logs
      sacct -n -X -S "$(date -d "-30 days" +%F)" -o JobName%60
    } 2>/dev/null | tr -c "[:alnum:]" "\n" | grep -xE "e[0-9]{3,5}" | tr -d e | sort -n | tail -1')
  [ -n "$LAST" ] || { echo "could not read the experiment numbers on Saion"; exit 1; }
  NAME=e$((LAST + 1))_$NAME
fi
RUN_DIR_ENV=${RUN_DIR:+RUN_DIR=$RUN_DIR}
CODE=/apps/unit/DoyaU/vasilache/apps/code/robot/dreamerv3
APP=jp.oist.abcvlib.dreamerBridge
HERE=$(cd "$(dirname "$0")" && pwd)

NO_BASE=${NO_BASE:-}
if [ -z "$NO_BASE" ]; then
  adb shell dumpsys usb | grep -q 'host_connected=true' \
    || { echo "no base on the phone (or adb not connected)"; exit 1; }
fi
# A dozing phone blocks the app's network even with the app on screen.
adb shell input keyevent KEYCODE_WAKEUP; adb shell wm dismiss-keyguard

echo "== submitting $NAME"
JOB=$(ssh saion "cd $CODE && git pull -q && \
  ${RUN_DIR:+rm -f $RUN_DIR/logdir/error_learner $RUN_DIR/logdir/online_shared/actor_step;} \
  CODE=$CODE SCRIPT=train STEPS=$STEPS CONFIGS='$CONFIGS' $RUN_DIR_ENV \
  sbatch --parsable -J $NAME sbatch/run_robot_v100.sbatch --env.robot.onboard True $*")
echo "job $JOB"

echo "== waiting for it to start"
EP=/work/DoyaU/vasilache/work/robot_endpoints/job_$JOB
until ssh saion "test -f $EP"; do sleep 10; done
eval "$(ssh saion "cat $EP")"   # NODE, PORT, RUN_DIR
IP=$(ssh saion "getent hosts $NODE" | awk '{print $1}')
echo "node $NODE ($IP), port $PORT, run dir $RUN_DIR"

echo "== pointing the phone at it"
# What the running app (if any) was started with, read before it is replaced.
OLD=$(adb shell cat /sdcard/Android/data/$APP/files/trainer.json 2>/dev/null || true)
tmp=$(mktemp -d)
echo "{\"ip\": \"$IP\", \"port\": $PORT, \"max_hz\": $HZ${NO_BASE:+, \"no_base\": true}}" \
  > "$tmp/trainer.json"
adb push "$tmp/trainer.json" /sdcard/Android/data/$APP/files/ >/dev/null
# A running app rereads trainer.json on every reconnect attempt, so leave it
# be: killing it can make the phone stop powering the base (a replug fixes it).
# Except across a no_base change, which the app reads only at start: one left
# in a link test's no-base mode never drives the wheels (e1290 sat still until
# it was restarted).
old_nobase=; [[ $OLD == *'"no_base": true'* ]] && old_nobase=1
if adb shell pidof $APP >/dev/null && [ "$old_nobase" != "$NO_BASE" ]; then
  echo "restarting the app: it is in the other base mode (no_base=${old_nobase:-0})"
  adb shell am force-stop $APP
fi
if ! adb shell pidof $APP >/dev/null; then
  adb shell monkey -p $APP -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1
  [ -n "$NO_BASE" ] || "$HERE/phone_allow_usb.sh" 10
fi

echo "== waiting for the phone to connect (the trainer takes ~1 min to start)"
until adb logcat -d | grep -q "dreamerBridge: connected to $IP:$PORT"; do sleep 5; done
echo "connected. Scores: ssh saion tail -f $RUN_DIR/logdir/scores.jsonl"
echo "stop:      ssh saion scancel $JOB; adb shell am force-stop $APP"
