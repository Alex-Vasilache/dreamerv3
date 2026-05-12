#!/bin/bash -l
#SBATCH -J dv3_online_actor
#SBATCH -p intel
#SBATCH -t 0-4:00:00
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -o /home/g/george-vasilache/slurm-%j.out
#SBATCH -e /home/g/george-vasilache/slurm-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/source_dreamerv3_env.sh

if [ -z "${LOGDIR:-}" ]; then
  echo "LOGDIR must point at the learner logdir (export LOGDIR=...)" >&2
  exit 1
fi

export MUJOCO_GL=egl

python -u /apps/unit/DoyaU/vasilache/code/dreamerv3/dreamerv3/main.py \
  --logdir "$LOGDIR" \
  --script online_actor \
  --configs size12m dmc_vision \
  --task dmc_cartpole_swingup \
  --online_learning True \
  --batch_size 4 \
  --run.envs 4 \
  --run.steps 1100000 \
  --logger.outputs jsonl,scope \
  --jax.platform cpu \
  --jax.prealloc False \
  --jax.precompile False \
  --jax.compute_dtype float32

exit $?
