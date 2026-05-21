#!/bin/bash -l
#SBATCH -J dreamerv3_fast_v4
#SBATCH -p gpu
#SBATCH -t 0-8:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail
mkdir -p /work/DoyaU/vasilache/work/slurm_logs
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir dreamerv3_online_fast_v4)}"
mkdir -p "$RUN_DIR"
export MUJOCO_GL=egl
cd "$RUN_DIR"

AGENT_FLAGS=(
  --agent.use_single_rollout True
  --agent.use_rms_loss_norm True
  --agent.imag_loss.use_pmpo_actor True
  --agent.report False
  --jax.precompile False
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size1m \
  --task dmc_cartpole_swingup \
  --env.dmc.image True \
  --env.dmc.size 32 32 \
  --batch_size 32 \
  --batch_length 16 \
  --online_learning True \
  --online_actor_cpu True \
  --run.envs 1 \
  --run.steps 2200000 \
  --run.report_every 600 \
  --logger.outputs jsonl,tensorboard \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}"
