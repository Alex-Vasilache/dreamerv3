#!/bin/bash -l
#SBATCH -J dreamerv3_vis1m32e1_v4
#SBATCH -p gpu
#SBATCH -t 0-4:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#
# Same as run_dreamerv3_online_cartpole_vis.sbatch plus DreamerV4-style toggles:
# use_single_rollout, use_rms_loss_norm, imag_loss.use_pmpo_actor (all three on).
# Single-flag A/B runs: run_dreamerv3_online_cartpole_vis_ab_rollout|rms|pmpo.sbatch
# Submit: sbatch run_dreamerv3_online_cartpole_vis_v4.sbatch

set -euo pipefail

mkdir -p /work/DoyaU/vasilache/work/slurm_logs

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir dreamerv3_online_vis1m32_1env_v4)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

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
  --batch_size 32 \
  --batch_length 16 \
  --online_learning True \
  --online_actor_cpu True \
  --run.envs 1 \
  --run.steps 1100000 \
  --logger.outputs jsonl,scope,tensorboard \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}"

exit $?
