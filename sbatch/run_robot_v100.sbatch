#!/bin/bash -l
#SBATCH -J robot_v100
#SBATCH -p gpu-v100
#SBATCH -t 0-8:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# DreamerV3 against the physical smartphone robot, with the learner on a V100.
#
# The robot is not on the cluster -- it is a phone on the lab WiFi -- so the job
# listens on a port of its compute node and the phone reaches it through an SSH
# tunnel opened from the MacBook:
#
#   sbatch sbatch/run_robot_v100.sbatch      # on saion
#   tools/robot_tunnel.sh                    # on the MacBook, keeps running
#
# The tunnel helper reads the endpoint this script publishes, so it needs no
# arguments. The phone keeps pointing at the MacBook's LAN address on port 3000
# exactly as it does for a local run; only what sits behind that port changes.
#
# Why this is viable at all: the bridge protocol is pipelined (the phone free-
# runs at 50Hz and never blocks on the trainer), so the extra WiFi + SSH round
# trip does not slow the control loop down. It shows up as a fixed action delay
# of ceil(rtt / 20ms) ticks instead. In lock-step mode the same tunnel would cap
# the loop at 1/rtt and be useless.
#
#   CONFIGS=robot_daydreamer sbatch sbatch/run_robot_v100.sbatch
#   PORT=3007 STEPS=200000 sbatch sbatch/run_robot_v100.sbatch
set -uo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

# Overridable so a working copy can be staged under /work without touching the
# shared checkout or pushing a branch first:
#   CODE=/apps/unit/DoyaU/vasilache/apps/code/robot/dreamerv3 sbatch ...
CODE="${CODE:-/apps/unit/DoyaU/vasilache/apps/code/dreamerv3}"
CONFIGS="${CONFIGS:-robot_daydreamer}"
# `train` runs both halves here and expects the robot to reach this node, which
# only works over a tunnel -- measured at 0.45Hz with 95% of frames dropped, so
# not in practice. `online_learner` is the useful one: the actor stays on the
# MacBook beside the robot and tools/robot_bridge_sync.sh carries chunks and
# weights across, neither of which cares about latency.
SCRIPT="${SCRIPT:-online_learner}"
STEPS="${STEPS:-100000}"
# Derived from the job id so two robot jobs on one node cannot collide. The
# tunnel helper reads it back rather than guessing.
PORT="${PORT:-$((3000 + SLURM_JOB_ID % 500))}"

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir robot_v100)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

# Published for tools/robot_tunnel.sh. The `latest` symlink is what makes the
# helper argument-free; the per-job copy is what survives the next launch.
ENDPOINT_DIR=/work/DoyaU/vasilache/work/robot_endpoints
mkdir -p "$ENDPOINT_DIR"
cat > "$ENDPOINT_DIR/job_${SLURM_JOB_ID}" <<EOF
NODE=$(hostname)
PORT=$PORT
JOB=$SLURM_JOB_ID
RUN_DIR=$RUN_DIR
STARTED=$(date -Is)
EOF
ln -sfn "$ENDPOINT_DIR/job_${SLURM_JOB_ID}" "$ENDPOINT_DIR/latest"
# Leaving a stale endpoint behind sends the next tunnel at a dead node, which
# looks exactly like a robot that will not connect.
trap 'rm -f "$ENDPOINT_DIR/job_${SLURM_JOB_ID}"' EXIT

echo "[robot] node=$(hostname) port=$PORT job=$SLURM_JOB_ID run_dir=$RUN_DIR"
if [ "$SCRIPT" = online_learner ]; then
  echo "[robot] learner only; the actor runs on the MacBook. Start there:"
  echo "[robot]   tools/robot_bridge_sync.sh ~/logdir/robot_local $RUN_DIR/logdir"
else
  echo "[robot] from the MacBook:  tools/robot_tunnel.sh"
  echo "[robot] or by hand:        ssh -N -L 0.0.0.0:3000:$(hostname):$PORT saion"
fi

cd "$RUN_DIR"

# One process: online_learning launches the learner and the actor itself, and
# online_actor_cpu (default True) keeps the actor's policy off the GPU so it
# never queues behind a training batch. That split is what holds the phone's
# 20ms control period; measured on the MacBook, sharing one device cost up to
# 282ms of blocked robot time per step.
python -u "$CODE/dreamerv3/main.py" \
  --logdir "$RUN_DIR/logdir" \
  --configs $CONFIGS \
  --script "$SCRIPT" \
  --env.robot.host 0.0.0.0 \
  --env.robot.port "$PORT" \
  --run.steps "$STEPS" \
  --jax.platform cuda \
  --jax.prealloc False \
  "$@"

exit $?
