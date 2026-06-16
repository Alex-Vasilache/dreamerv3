#!/bin/bash -l
#SBATCH -J smoke_mask
#SBATCH -p gpu
#SBATCH -t 0-00:30:00
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

MASK_MODE="${MASK_MODE:-prob}"
MGR_FREQ="${MGR_FREQ:-1}"
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir smoke_${MASK_MODE}_k${MGR_FREQ})}"
mkdir -p "$RUN_DIR"
export MUJOCO_GL=egl

# Fast CPU smoke test (debug config: deter=8, batch 8x10) of the new masked
# sparsity paths. Runs a few hundred steps to force JAX to trace+compile loss()
# (manager imagination, B2 prob penalty / C reinforce reward-shaping) and
# report(). Success = no trace/shape/grad errors and metrics logged.
cd "$RUN_DIR"
python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs debug masked_goals \
  --task dmc_cartpole_swingup \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 32 32 \
  --agent.mask_topk 0 \
  --agent.goal_struct_weight 200.0 \
  --agent.manager_sample_freq "$MGR_FREQ" \
  --agent.mask_sparsity_mode "$MASK_MODE" \
  --agent.mask_sparsity_max 5.0 \
  --run.steps 400 \
  --logger.outputs jsonl \
  --jax.platform cpu \
  --jax.prealloc False

echo "SMOKE_EXIT=$? mode=$MASK_MODE freq=$MGR_FREQ"
