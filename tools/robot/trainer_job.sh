#!/bin/bash -l
#SBATCH -J robot
#SBATCH -t 0-8:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G

# The learner for a robot run. tools/robot/train.sh starts it, either with
# sbatch on a cluster (partition, GPU and log paths come from its command
# line) or as a plain background process on this computer. Not usually run
# by hand.
#
# It listens on a port; the phone connects to it directly, runs the policy
# itself and streams every observation back, and the learner pushes new
# weights to the phone every few seconds (configs.yaml: online_publish_every).
#
# Before starting it writes $ROBOT_RUNS/robot_endpoints/job_<id> (NODE, PORT,
# RUN_DIR, JOB) so train.sh knows where to point the phone; `latest` links to
# the newest one.
#
# Environment: CODE (the dreamerv3 checkout), ROBOT_RUNS, ROBOT_SETUP,
# CONFIGS, STEPS, RUN_DIR (resume that run), PORT, DEMO (1: run the learner
# under tools/robot/demo_supervisor.py, which restarts training from scratch
# whenever the phone's dashboard asks, inside this job; tools/robot/demo.sh).
# Extra arguments go to dreamerv3/main.py.
set -uo pipefail

# Under sbatch the script runs from a spool copy, so its own path says
# nothing; train.sh submits from the checkout, which is SLURM_SUBMIT_DIR.
CODE="${CODE:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
ROBOT_RUNS="${ROBOT_RUNS:-$HOME/logdir/robot}"
CONFIGS="${CONFIGS:-robot}"
STEPS="${STEPS:-100000}"
JOB="${SLURM_JOB_ID:-local$$}"
NAME="${SLURM_JOB_NAME:-${NAME:-robot}}"
# Derived from the job id so two robot jobs on one node cannot collide.
PORT="${PORT:-$((3000 + ${SLURM_JOB_ID:-$$} % 500))}"

if [ -n "${ROBOT_SETUP:-}" ]; then
  eval "$ROBOT_SETUP"
fi
PY=python
if [ -z "${ROBOT_SETUP:-}" ] && [ -x "$CODE/.venv/bin/python" ]; then
  PY="$CODE/.venv/bin/python"
fi

RUN_DIR="${RUN_DIR:-$ROBOT_RUNS/${NAME}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

ENDPOINT_DIR="$ROBOT_RUNS/robot_endpoints"
mkdir -p "$ENDPOINT_DIR"
cat > "$ENDPOINT_DIR/job_$JOB" <<EOF
NODE=$(hostname)
PORT=$PORT
JOB=$JOB
PID=$$
RUN_DIR=$RUN_DIR
STARTED=$(date '+%Y-%m-%dT%H:%M:%S')
EOF
ln -sfn "$ENDPOINT_DIR/job_$JOB" "$ENDPOINT_DIR/latest"
# A stale endpoint would send the next phone at a dead node, which looks
# exactly like a robot that will not connect.
trap 'rm -f "$ENDPOINT_DIR/job_$JOB"' EXIT

echo "[robot] node=$(hostname) port=$PORT job=$JOB run_dir=$RUN_DIR"
echo "[robot] configs=$CONFIGS steps=$STEPS args=$*"

cd "$RUN_DIR"
# --script train runs learner and actor in this one process; with
# env.robot.onboard the "actor" is only the link to the phone, which runs the
# policy itself. A run dir with a checkpoint resumes from it. Under DEMO the
# supervisor runs it once per session, each with its own --logdir.
if [ -n "${DEMO:-}" ]; then
  LAUNCH=("$PY" -u "$CODE/tools/robot/demo_supervisor.py"
          --run_dir "$RUN_DIR" --name "$NAME" -- "$PY" -u "$CODE/dreamerv3/main.py")
else
  LAUNCH=("$PY" -u "$CODE/dreamerv3/main.py" --logdir "$RUN_DIR/logdir")
fi
"${LAUNCH[@]}" \
  --configs $CONFIGS \
  --script train \
  --env.robot.host 0.0.0.0 \
  --env.robot.port "$PORT" \
  --env.robot.onboard True \
  --run.steps "$STEPS" \
  --jax.prealloc False \
  "$@" &
# In the background so a plain `kill <PID>` (tools/robot/stop.sh on a local
# run) reaches python too, not only this shell.
child=$!
trap 'kill $child 2>/dev/null' TERM INT
wait $child; status=$?
wait $child 2>/dev/null
exit $status
