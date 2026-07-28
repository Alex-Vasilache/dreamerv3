#!/bin/bash -l
#SBATCH -p gpu-v100
#SBATCH -t 2-00:00:00
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

# Masked + variable-K + soft fixed duration prior (e57 recipe), generalized across TASK.
# Small/V100-scale template for the current recipe (variable-K with block-pooled
# manager credit); ports to any DMC task via TASK.
# Submit with EXP_TAG, TASK, DUR_TARGET, and optional DUR_MAX (must be >= DUR_TARGET).
# DUR_MODE: fixed (default, e57 goal_duration_reg) | lagrangian (dual-ascent
#   Lagrangian on |E[dur]-target|, folded out into its own independently-scaled loss key).
# RECIPE: plain_vark (default: variable-K with block-pooled manager credit) |
#   director (pure Director baseline: fixed manager_sample_freq=MGR_FREQ, no
#     duration head -- isolates the var-K machinery).
# SEED: replicate index (default 0).
# Example:
#   sbatch -J e139_cheetah --export=ALL,EXP_TAG=e139,TASK=dmc_cheetah_run,DUR_TARGET=4.0 sbatch/run_v3_prior_vargoal_small.sbatch
EXP_TAG="${EXP_TAG:?set EXP_TAG}"
TASK="${TASK:-dmc_cartpole_swingup}"
DUR_TARGET="${DUR_TARGET:-4.0}"
DUR_MIN="${DUR_MIN:-1}"
DUR_MAX="${DUR_MAX:-16}"
if awk "BEGIN {exit !($DUR_TARGET > $DUR_MAX)}"; then
  DUR_MAX="$DUR_TARGET"
fi

RECIPE="${RECIPE:-plain_vark}"   # plain_vark | director
MGR_FREQ="${MGR_FREQ:-8}"

RUN_SUFFIX="${EXP_TAG}_${TASK}_${RECIPE}"
# Requeue-safe RUN_DIR: keyed to SLURM_JOB_ID (stable across requeues) so a
# requeued job resumes from checkpoint instead of restarting in a fresh mktemp dir.
RUN_DIR="${RUN_DIR:-/work/DoyaU/vasilache/work/${RUN_SUFFIX}_j${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"

export MUJOCO_GL=egl

DUR_REG="${DUR_REG:-0.01}"
DUR_MODE="${DUR_MODE:-fixed}"
DUR_LAGRANGE_MAX="${DUR_LAGRANGE_MAX:-5.0}"
SEED="${SEED:-0}"
WORKER_TIMED_GOALS="${WORKER_TIMED_GOALS:-False}"   # HiTS-style worker countdown conditioning
STRUCT_ADAPT="${STRUCT_ADAPT:-False}"   # AutoAdapt multiplier toward struct-loss ceiling (goal_struct_adapt)
STRUCT_ADAPT_TARGET="${STRUCT_ADAPT_TARGET:-0.005}"  # struct-loss setpoint (dual-ascent unless one_sided)
MGR_REWARD_AGG="${MGR_REWARD_AGG:-mean}"   # mean (default) | sum (couples mgr return to hold length)
# Condition the manager on the previous goal code so it can deliberately re-emit
# (reuse) blocks -- required for implicit sparsity to be learnable, not incidental.
# Valid on plain Director too.
MGR_COND_GOALCODE="${MGR_COND_GOALCODE:-False}"
# Direct (REINFORCE-free) implicit sparsity (07-16, F18 follow-up #2): leaves
# plain/direct sampling untouched (mutually exclusive with GOAL_DELTA_MODE) and
# instead adds an ordinary loss on the overlap between this decision's own
# softmax and the previous decision's soft distribution, plus feeds that same
# soft distribution to the manager as an input -- see agent.py
# goal_soft_reuse_adapt / EXPERIMENTS.md §7. GOAL_SOFT_REUSE_ADAPT=True drives
# an adaptive Lagrange multiplier holding mean overlap at GOAL_SOFT_REUSE_TARGET.
# Implies mgr_cond_goalcode (forced in agent.py).
GOAL_SOFT_REUSE_ADAPT="${GOAL_SOFT_REUSE_ADAPT:-False}"
GOAL_SOFT_REUSE_TARGET="${GOAL_SOFT_REUSE_TARGET:-0.7}"
# Open-loop anneal of the target itself (agent.Ratchet), same idea as
# IMPL_TARGET_INIT/_VEL. Default init==target -> no-op.
GOAL_SOFT_REUSE_TARGET_INIT="${GOAL_SOFT_REUSE_TARGET_INIT:-$GOAL_SOFT_REUSE_TARGET}"
GOAL_SOFT_REUSE_TARGET_VEL="${GOAL_SOFT_REUSE_TARGET_VEL:-0.01}"

if [ "$DUR_MODE" = "lagrangian" ]; then
  DUR_FLAGS=(
    --agent.goal_duration_reg 0.0
    --agent.goal_duration_lagrange True
    --agent.goal_duration_lagrange_max "$DUR_LAGRANGE_MAX"
  )
else
  DUR_FLAGS=(
    --agent.goal_duration_reg "$DUR_REG"
    --agent.goal_duration_lagrange False
  )
fi

VARK_FLAGS=(
  --agent.variable_goal_length True
  --agent.goal_duration_min "$DUR_MIN"
  --agent.goal_duration_max "$DUR_MAX"
  --agent.goal_duration_target "$DUR_TARGET"
  "${DUR_FLAGS[@]}"
)

case "$RECIPE" in
  plain_vark)
    CONFIGS="size6m variable_goals"
    STRUCT_W="${STRUCT_W:-0.0}"
    RECIPE_FLAGS=( "${VARK_FLAGS[@]}" )
    ;;
  director)
    # Pure Director baseline: fixed-K goals, no variable-K.
    CONFIGS="size6m"
    STRUCT_W="${STRUCT_W:-0.0}"
    RECIPE_FLAGS=( --agent.manager_sample_freq "$MGR_FREQ" )
    ;;
  *)
    echo "unknown RECIPE=$RECIPE" >&2; exit 1
    ;;
esac

echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"
echo "EXP_TAG=$EXP_TAG TASK=$TASK RECIPE=$RECIPE DUR_TARGET=$DUR_TARGET DUR_MODE=$DUR_MODE DUR_REG=$DUR_REG DUR_MAX=$DUR_MAX STRUCT_W=$STRUCT_W STRUCT_ADAPT=$STRUCT_ADAPT STRUCT_ADAPT_TARGET=$STRUCT_ADAPT_TARGET WORKER_TIMED_GOALS=$WORKER_TIMED_GOALS MGR_REWARD_AGG=$MGR_REWARD_AGG MGR_FREQ=$MGR_FREQ MGR_COND_GOALCODE=$MGR_COND_GOALCODE GOAL_SOFT_REUSE_ADAPT=$GOAL_SOFT_REUSE_ADAPT GOAL_SOFT_REUSE_TARGET=$GOAL_SOFT_REUSE_TARGET GOAL_SOFT_REUSE_TARGET_INIT=$GOAL_SOFT_REUSE_TARGET_INIT GOAL_SOFT_REUSE_TARGET_VEL=$GOAL_SOFT_REUSE_TARGET_VEL SEED=$SEED" >> "$RUN_DIR/job.env"
RUN_STEPS="${RUN_STEPS:-4000000}"

cd "$RUN_DIR"

# Self-requeue near walltime (USR1 600s before the limit): requeue keeps the job
# id -> same RUN_DIR -> checkpoint resume, so 4M-step runs survive the wall limit.
on_usr1() {
  echo "[self-requeue] USR1 near walltime; requeueing job $SLURM_JOB_ID to resume in $RUN_DIR"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap on_usr1 USR1

AGENT_FLAGS=(
  --agent.goal_autoencoder_beta 0.25
  --agent.goal_rec_loss_agg sum
  --agent.goal_opt.lr 4e-5
  --agent.goal_opt.wd 0.0
  --agent.goal_opt.eps 1e-20
  --agent.manager_actent_impl mult
  --agent.manager_actent_target 0.5
  --agent.goal_struct_weight "$STRUCT_W"
  --agent.worker_timed_goals "$WORKER_TIMED_GOALS"
  --agent.goal_struct_adapt "$STRUCT_ADAPT"
  --agent.goal_struct_adapt_target "$STRUCT_ADAPT_TARGET"
  --agent.mgr_cond_goalcode "$MGR_COND_GOALCODE"
  --agent.goal_soft_reuse_adapt "$GOAL_SOFT_REUSE_ADAPT"
  --agent.goal_soft_reuse_target "$GOAL_SOFT_REUSE_TARGET"
  --agent.goal_soft_reuse_target_init "$GOAL_SOFT_REUSE_TARGET_INIT"
  --agent.goal_soft_reuse_target_vel "$GOAL_SOFT_REUSE_TARGET_VEL"
  "${RECIPE_FLAGS[@]}"
  --agent.report False
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --seed "$SEED" \
  --configs $CONFIGS \
  --task "$TASK" \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 32 32 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.mgr_expl_weight 0.1 \
  --agent.mgr_reward_agg "$MGR_REWARD_AGG" \
  --agent.mgr_retnorm.impl meanstd \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 16 \
  --run.steps "$RUN_STEPS" \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}" &
TRAIN_PID=$!
# Backgrounded + wait so the USR1 trap can run while training is in flight.
wait "$TRAIN_PID"

exit $?
