#!/bin/bash -l
#SBATCH -J dreamerv3_hrl_smoke
#SBATCH -p gpu
#SBATCH -t 0-0:45:00
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#
# Lightweight train with Director-style HRL enabled (use_director_hrl).
# Submit: sbatch run_dreamerv3_smoke_director_hrl.sbatch
# Optional: RUN_DIR=/work/DoyaU/vasilache/work/my_run sbatch run_dreamerv3_smoke_director_hrl.sbatch
#
set -euo pipefail

mkdir -p /work/DoyaU/vasilache/work/slurm_logs

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir dreamerv3_hrl_smoke)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

export MUJOCO_GL=egl

cd "$RUN_DIR"

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs defaults debug \
  --task dmc_cartpole_swingup \
  --env.dmc.image False \
  --run.envs 4 \
  --run.steps 2500 \
  --run.log_every 50 \
  --run.report_every 200 \
  --run.save_every 800 \
  --logger.outputs jsonl \
  --jax.platform cuda \
  --jax.prealloc False \
  --agent.use_director_hrl True

exit $?
