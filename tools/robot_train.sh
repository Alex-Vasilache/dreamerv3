#!/usr/bin/env bash
# Start a robot training run on Saion and point the phone at it.
#
#   tools/robot_train.sh e1224_robot_25hz                     # default configs
#   CONFIGS="robot_daydreamer robot_fast robot_25hz" STEPS=100000 \
#     tools/robot_train.sh e1224_robot_25hz [extra main.py flags]
#   RUN_DIR=<a previous run dir on /work> tools/robot_train.sh e1225_...   # resume it
#
# Needs: `ssh saion` working, the phone on adb, a base on the phone.
# Stop with:  ssh saion scancel <job>; adb shell am force-stop jp.oist.abcvlib.dreamerBridge
set -euo pipefail
NAME=${1:?usage: tools/robot_train.sh e<N>_<name> [flags]}; shift
CONFIGS=${CONFIGS:-robot_daydreamer robot_fast robot_25hz}
STEPS=${STEPS:-100000}
HZ=${HZ:-25}
RUN_DIR_ENV=${RUN_DIR:+RUN_DIR=$RUN_DIR}
CODE=/apps/unit/DoyaU/vasilache/apps/code/robot/dreamerv3
APP=jp.oist.abcvlib.dreamerBridge
HERE=$(cd "$(dirname "$0")" && pwd)

adb shell dumpsys usb | grep -q 'host_connected=true' \
  || { echo "no base on the phone (or adb not connected)"; exit 1; }

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
tmp=$(mktemp -d)
echo "{\"ip\": \"$IP\", \"port\": $PORT, \"max_hz\": $HZ}" > "$tmp/trainer.json"
adb push "$tmp/trainer.json" /sdcard/Android/data/$APP/files/ >/dev/null
# A running app rereads trainer.json on every reconnect attempt, so leave it
# be: killing it can make the phone stop powering the base (a replug fixes it).
if ! adb shell pidof $APP >/dev/null; then
  adb shell monkey -p $APP -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1
  "$HERE/phone_allow_usb.sh" 10
fi

echo "== waiting for the phone to connect (the trainer takes ~1 min to start)"
until adb logcat -d | grep -q "dreamerBridge: connected to $IP:$PORT"; do sleep 5; done
echo "connected. Scores: ssh saion tail -f $RUN_DIR/logdir/scores.jsonl"
echo "stop:      ssh saion scancel $JOB; adb shell am force-stop $APP"
