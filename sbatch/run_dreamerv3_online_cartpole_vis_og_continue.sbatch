#!/bin/bash -l
#SBATCH -J dreamerv3_vis1m32e1_og_cont
#SBATCH -p gpu-v100
#SBATCH -t 0-24:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

RUN_DIR="/work/DoyaU/vasilache/work/dreamerv3_online_vis1m32_1env_og_20260527_133543_lUzGFP"
echo "RUN_DIR=$RUN_DIR (continuation)" >> "$RUN_DIR/job.env"

export MUJOCO_GL=egl

cd "$RUN_DIR"
AGENT_FLAGS=(
  --agent.use_single_rollout False
  --agent.use_rms_loss_norm False
  --agent.imag_loss.use_pmpo_actor False
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size1m \
  --task dmc_cartpole_swingup \
  --env.dmc.image True \
  --env.dmc.size 32 32 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --online_learning True \
  --online_actor_cpu True \
  --run.envs 1 \
  --run.steps 4000000 \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}"

exit $?
