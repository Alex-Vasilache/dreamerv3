#!/bin/bash -l
#SBATCH -J dreamerv3_gifs
#SBATCH -p gpu
#SBATCH -t 0-1:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH -o /home/g/george-vasilache/slurm-%j.out
#SBATCH -e /home/g/george-vasilache/slurm-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/source_dreamerv3_env.sh

LOGDIR="/work/DoyaU/vasilache/dreamerv3_run_05l9Sm/logdir"
export MUJOCO_GL=egl

cd /work/DoyaU/vasilache/dreamerv3_run_05l9Sm
python -u /apps/unit/DoyaU/vasilache/code/dreamerv3/dreamerv3/main.py \
  --logdir "$LOGDIR" \
  --configs size1m dmc_vision \
  --task dmc_acrobot_swingup \
  --batch_size 8 \
  --run.envs 4 \
  --run.steps 405500 \
  --run.report_every -1 \
  --run.save_every 0 \
  --logger.outputs jsonl,scope,tensorboard \
  --jax.platform cuda \
  --jax.prealloc False \
  --jax.compute_dtype float32

exit $?
