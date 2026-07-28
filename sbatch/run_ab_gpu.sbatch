#!/bin/bash -l
# Short GPU A/B of one production recipe against a chosen repo tree.
#
# Used to validate the 2026-07-28 modularization on the path that actually
# matters: real cuDNN conv, real jit/sharding, real replay -- none of which the
# CPU parity smoke exercises. Launch the same RECIPE twice, once with
# REPO=<pre-refactor tree> and once with REPO=<refactored tree>, then compare
# the metrics.jsonl curves.
#
# Usage:
#   REPO=<tree> RUN_TAG=<name> [RECIPE=plain_vark|director] [SEED=0]
#     [RUN_STEPS=30000] [TASK=dmc_cartpole_swingup] sbatch sbatch/run_ab_gpu.sbatch
#SBATCH -J ab_gpu
#SBATCH -p gpu-v100
#SBATCH -t 0-04:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

REPO="${REPO:?set REPO to the repo tree under test}"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
RECIPE="${RECIPE:-plain_vark}"
SEED="${SEED:-0}"
RUN_STEPS="${RUN_STEPS:-30000}"
TASK="${TASK:-dmc_cartpole_swingup}"

RUN_DIR="/work/DoyaU/vasilache/work/ab_${RUN_TAG}"
rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"; cd "$RUN_DIR"
echo "REPO=$REPO RECIPE=$RECIPE SEED=$SEED STEPS=$RUN_STEPS"

# ``variable_goal_block_rew`` only exists on the pre-refactor tree (block pooling
# is unconditional after the refactor), so pass it only where the flag is real --
# otherwise the flag parser rejects the unknown key.
EXTRA=()
if grep -qE '^ *variable_goal_block_rew:' "$REPO/dreamerv3/configs.yaml"; then
  EXTRA+=( --agent.variable_goal_block_rew True )
fi

case "$RECIPE" in
  plain_vark)
    CONFIGS="size1m variable_goals"
    RECIPE_FLAGS=(
      --agent.variable_goal_length True
      --agent.goal_duration_min 1
      --agent.goal_duration_max 16
      --agent.goal_duration_target 8.0
      --agent.goal_duration_reg 0.0
      --agent.goal_duration_lagrange True
      --agent.goal_duration_lagrange_max 5.0
      --agent.mgr_reward_agg sum
      --agent.goal_duration_relabel_truncated False
      "${EXTRA[@]}"
    )
    ;;
  director)
    CONFIGS="size1m"
    RECIPE_FLAGS=( --agent.use_hrl True --agent.manager_sample_freq 8 )
    ;;
  *) echo "unknown RECIPE=$RECIPE" >&2; exit 1 ;;
esac

python -u "$REPO/dreamerv3/main.py" \
  --logdir "$RUN_DIR/logdir" \
  --seed "$SEED" \
  --configs $CONFIGS \
  --task "$TASK" \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 32 32 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.imag_length 16 \
  --agent.mgr_expl_weight 0.1 \
  --agent.mgr_retnorm.impl meanstd \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 4 \
  --run.steps "$RUN_STEPS" \
  --run.log_every 20 \
  --logger.outputs jsonl \
  --jax.platform cuda \
  --jax.prealloc False \
  "${RECIPE_FLAGS[@]}"

exit $?
