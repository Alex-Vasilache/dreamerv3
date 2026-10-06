#!/bin/bash -l
#SBATCH -J sim_phone
#SBATCH -p gpu-v100
#SBATCH -t 0-2:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# The phone robot without the phone: cartpole swingup paced to wall-clock time
# through the robot's actor/learner split (config block `sim_phone`). Training
# speed is judged in seconds of real time, so the wall limit is the budget.
#
#   sbatch sbatch/run_sim_phone.sbatch
#   CONFIGS="sim_phone" sbatch sbatch/run_sim_phone.sbatch --env.dmc.repeat 4 --env.realtime_hz 25
#   sbatch -p gpu-p100 --gres=gpu:p100:1 --exclude=saion-gpu[01-10] sbatch/run_sim_phone.sbatch
set -uo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

CODE="${CODE:-/apps/unit/DoyaU/vasilache/apps/code/robot/dreamerv3}"
CONFIGS="${CONFIGS:-sim_phone}"
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir "${SLURM_JOB_NAME:-sim_phone}")}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"
echo "ARGS=$*" >> "$RUN_DIR/job.env"
echo "[sim] node=$(hostname) job=${SLURM_JOB_ID:-} run_dir=$RUN_DIR configs=$CONFIGS args=$*"

cd "$RUN_DIR"
python -u "$CODE/dreamerv3/main.py" \
  --logdir "$RUN_DIR/logdir" \
  --configs $CONFIGS \
  --jax.platform cuda \
  "$@"
exit $?
