#!/bin/bash -l
#SBATCH -J smoke_joint
#SBATCH -p gpu-v100
#SBATCH -t 0-00:20:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir smoke_joint_edit)}"
mkdir -p "$RUN_DIR"
export MUJOCO_GL=egl

# Fast GPU smoke test of the single-head ("joint") masked manager
# (agent.mask_joint_edit): the skill head widens to (L, C+1) and absorbs the
# per-block mask (class 0 = abstain). Runs a few hundred steps on the small
# `debug` config (deter=8, batch 8x10) to force JAX to trace+compile loss()
# (manager imagination with the widened C+1 categorical, clean per-block
# REINFORCE, goal-code edit) and report() with the joint viz path. Success =
# no trace/shape/grad errors and goal/mask_frac_mean logged in metrics.jsonl.
cd "$RUN_DIR"
python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs debug masked_goals \
  --task dmc_cartpole_swingup \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 64 64 \
  --agent.mask_joint_edit True \
  --agent.manager_sample_freq 8 \
  --agent.mask_sparsity_mode prob \
  --agent.mask_sparsity_target 0.3 \
  --agent.mask_sparsity_target_init 1.0 \
  --agent.mask_sparsity_target_vel 0.02 \
  --agent.mask_sparsity_max 5.0 \
  --run.steps 400 \
  --logger.outputs jsonl \
  --jax.platform cuda \
  --jax.prealloc False

echo "SMOKE_EXIT=$? joint_edit=True"
