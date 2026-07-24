#!/bin/bash -l
#SBATCH -J v3_cartpole_flat_exp7_6m
#SBATCH -p gpu-v100
#SBATCH -t 0-24:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir v3_cartpole_flat_exp7_6m)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

export MUJOCO_GL=egl

cd "$RUN_DIR"
# Exp7: HRL turned off (--agent.use_hrl False) -> flat DreamerV3 AC (single
# pol/val over WM features, no manager / goal-VAE / skill aggregation). Size 6m,
# environment cartpole swingup. Manager/goal flags omitted (unused in flat mode).
AGENT_FLAGS=(
  --agent.use_hrl False
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size6m \
  --task dmc_cartpole_swingup \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 32 32 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 16 \
  --run.steps 4000000 \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}"

exit $?
