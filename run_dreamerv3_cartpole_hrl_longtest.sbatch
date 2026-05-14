#!/bin/bash -l
#SBATCH -J dreamerv3_hrl_cp_long
#SBATCH -p gpu
#SBATCH -t 0-8:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#
# Longer cartpole swingup run with Director HRL (same stack as smoke, more steps).
# Submit: sbatch run_dreamerv3_cartpole_hrl_longtest.sbatch
# Optional: STEPS=500000 RUN_DIR=/work/DoyaU/vasilache/work/my_run sbatch ...
#
set -euo pipefail

mkdir -p /work/DoyaU/vasilache/work/slurm_logs

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

STEPS="${STEPS:-300000}"
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir dreamerv3_hrl_cartpole_long)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"
echo "STEPS=$STEPS" >> "$RUN_DIR/job.env"

export MUJOCO_GL=egl

cd "$RUN_DIR"

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs defaults debug \
  --task dmc_cartpole_swingup \
  --env.dmc.image False \
  --run.envs 4 \
  --run.steps "$STEPS" \
  --run.log_every 500 \
  --run.report_every 5000 \
  --run.save_every 20000 \
  --logger.outputs jsonl \
  --jax.platform cuda \
  --jax.prealloc False \
  --agent.use_director_hrl True

exit $?
