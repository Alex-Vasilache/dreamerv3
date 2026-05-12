#!/bin/bash -l
#SBATCH -J dreamerv3_online
#SBATCH -p gpu
#SBATCH -t 0-4:00:00
#SBATCH -c 16
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /home/g/george-vasilache/slurm-%j.out
#SBATCH -e /home/g/george-vasilache/slurm-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/source_dreamerv3_env.sh

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir dreamerv3_online)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

export MUJOCO_GL=egl

cd "$RUN_DIR"
python -u /apps/unit/DoyaU/vasilache/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size12m \
  --task dmc_cartpole_swingup \
  --env.dmc.image False \
  --online_learning True \
  --online_actor_cpu True \
  --run.envs 4 \
  --run.steps 1100000 \
  --logger.outputs jsonl,scope,tensorboard \
  --jax.platform cuda \
  --jax.prealloc False

exit $?
