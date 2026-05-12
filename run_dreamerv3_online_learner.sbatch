#!/bin/bash -l
#SBATCH -J dv3_online_learner
#SBATCH -p gpu
#SBATCH -t 0-4:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH -o /home/g/george-vasilache/slurm-%j.out
#SBATCH -e /home/g/george-vasilache/slurm-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/source_dreamerv3_env.sh

if [ -z "${LOGDIR:-}" ]; then
  RUN_DIR="$(mktemp -d /work/DoyaU/vasilache/dreamerv3_online_XXXXXX)"
  LOGDIR="$RUN_DIR/logdir"
  mkdir -p "$LOGDIR"
  echo "LOGDIR=$LOGDIR" > "$RUN_DIR/job.env"
  echo "Created LOGDIR=$LOGDIR"
else
  mkdir -p "$LOGDIR"
fi

export MUJOCO_GL=egl

python -u /apps/unit/DoyaU/vasilache/code/dreamerv3/dreamerv3/main.py \
  --logdir "$LOGDIR" \
  --script online_learner \
  --configs size12m dmc_vision \
  --task dmc_cartpole_swingup \
  --online_learning True \
  --batch_size 4 \
  --run.steps 1100000 \
  --logger.outputs jsonl,scope,tensorboard \
  --jax.platform cuda \
  --jax.prealloc False \
  --jax.compute_dtype float32

exit $?
