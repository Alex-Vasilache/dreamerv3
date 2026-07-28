#!/bin/bash -l
# Full 4M-step V100 A/B validating the 2026-07-28 modularization.
#
# Runs ONE repo tree with the current production recipe. Launch it twice per
# task -- once REPO=<pre-refactor tree>, once REPO=<refactored tree> -- at the
# same seed, and compare the learning curves.
#
# Why not just reuse run_v3_prior_vargoal_small.sbatch: that script predates
# ``variable_goal_block_rew`` and cannot express block pooling, so the
# pre-refactor side would silently run the FULL-RESOLUTION var-K path (the one
# the refactor deleted) against the refactored side's block-pooled path. That
# is a real behavioral difference by construction, not an A/B. This script
# instead builds one identical flag list for both trees, and adds the
# block-pooling flag only on the tree where it still exists (after the refactor
# block pooling is unconditional under variable_goal_length, so the flag is
# gone).
#
# Recipe = the e363/e364 research cell: plain var-K, block-pooled manager
# credit, Lagrangian duration prior at target 8, SMDP ``sum`` reward
# aggregation, hindsight relabel OFF, implicit sparsity (goal_soft_reuse_adapt)
# ON so the retained masking mechanism is actually exercised.
#
# Usage:
#   REPO=<tree> EXP_TAG=e372 TASK=dmc_cheetah_run ARM=pre \
#     sbatch -J e372_cheetah_pre sbatch/run_refactor_ab_4m.sbatch
#SBATCH -p gpu-v100
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

REPO="${REPO:?set REPO to the repo tree under test}"
EXP_TAG="${EXP_TAG:?set EXP_TAG}"
ARM="${ARM:?set ARM (pre|post)}"
TASK="${TASK:-dmc_cheetah_run}"
SEED="${SEED:-0}"
RUN_STEPS="${RUN_STEPS:-4000000}"
DUR_TARGET="${DUR_TARGET:-8.0}"
DUR_MIN="${DUR_MIN:-1}"
DUR_MAX="${DUR_MAX:-16}"
MGR_REWARD_AGG="${MGR_REWARD_AGG:-sum}"
RELABEL="${RELABEL:-False}"
SOFT_REUSE="${SOFT_REUSE:-True}"
SOFT_REUSE_TARGET="${SOFT_REUSE_TARGET:-0.5}"
# Small-scale calibration of the ratchet velocity: project convention is the BIG
# value (3.9e-6) scaled by the small/BIG env-steps-per-train-call ratio.
SOFT_REUSE_TARGET_INIT="${SOFT_REUSE_TARGET_INIT:-0.0}"
SOFT_REUSE_TARGET_VEL="${SOFT_REUSE_TARGET_VEL:-1.0e-6}"

# Requeue-safe RUN_DIR: requeues keep the same SLURM_JOB_ID, so a preempted or
# self-requeued job resumes from its checkpoint in the same dir.
RUN_SUFFIX="${EXP_TAG}_${TASK}_refab_${ARM}"
RUN_DIR="${RUN_DIR:-/work/DoyaU/vasilache/work/${RUN_SUFFIX}_j${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"

# Block pooling is opt-in before the refactor and unconditional after it, so the
# flag is passed only where the config key still exists. Everything else below
# is byte-identical between the two arms.
BLOCK_REW=()
if grep -qE '^ *variable_goal_block_rew:' "$REPO/dreamerv3/configs.yaml"; then
  BLOCK_REW=( --agent.variable_goal_block_rew True )
fi

cat > "$RUN_DIR/job.env" <<EOF
RUN_DIR=$RUN_DIR
EXP_TAG=$EXP_TAG ARM=$ARM REPO=$REPO
GIT_HEAD=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
TASK=$TASK SEED=$SEED STEPS=$RUN_STEPS
RECIPE=plain_vark BLOCK_POOLED=True DUR_MODE=lagrangian DUR_TARGET=$DUR_TARGET
MGR_REWARD_AGG=$MGR_REWARD_AGG RELABEL=$RELABEL
SOFT_REUSE=$SOFT_REUSE TARGET=$SOFT_REUSE_TARGET INIT=$SOFT_REUSE_TARGET_INIT VEL=$SOFT_REUSE_TARGET_VEL
NOTE=4M A/B for the agent.py modularization (pre vs post refactor)
EOF
echo "ARM=$ARM REPO=$REPO HEAD=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"

on_usr1() {
  echo "[self-requeue] USR1 near walltime; requeueing $SLURM_JOB_ID to resume in $RUN_DIR"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap on_usr1 USR1

python -u "$REPO/dreamerv3/main.py" \
  --logdir "$RUN_DIR/logdir" \
  --seed "$SEED" \
  --configs size6m variable_goals \
  --task "$TASK" \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 32 32 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.goal_autoencoder_beta 0.25 \
  --agent.goal_rec_loss_agg sum \
  --agent.goal_opt.lr 4e-5 \
  --agent.goal_opt.wd 0.0 \
  --agent.goal_opt.eps 1e-20 \
  --agent.manager_actent_impl mult \
  --agent.manager_actent_target 0.5 \
  --agent.goal_struct_weight 0.0 \
  --agent.goal_struct_adapt False \
  --agent.mgr_expl_weight 0.1 \
  --agent.mgr_reward_agg "$MGR_REWARD_AGG" \
  --agent.mgr_retnorm.impl meanstd \
  --agent.variable_goal_length True \
  --agent.goal_duration_min "$DUR_MIN" \
  --agent.goal_duration_max "$DUR_MAX" \
  --agent.goal_duration_target "$DUR_TARGET" \
  --agent.goal_duration_reg 0.0 \
  --agent.goal_duration_lagrange True \
  --agent.goal_duration_lagrange_max 5.0 \
  --agent.goal_duration_fixed 0 \
  --agent.goal_duration_relabel_truncated "$RELABEL" \
  --agent.goal_soft_reuse_adapt "$SOFT_REUSE" \
  --agent.goal_soft_reuse_target "$SOFT_REUSE_TARGET" \
  --agent.goal_soft_reuse_target_init "$SOFT_REUSE_TARGET_INIT" \
  --agent.goal_soft_reuse_target_vel "$SOFT_REUSE_TARGET_VEL" \
  --agent.report False \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 16 \
  --run.steps "$RUN_STEPS" \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  "${BLOCK_REW[@]}" &
TRAIN_PID=$!
# Backgrounded + wait so the USR1 trap can fire while training is in flight
# (bash defers traps until a foreground command returns).
wait "$TRAIN_PID"

exit $?
