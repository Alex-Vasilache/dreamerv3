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
# Same recipe as run_v3_cartpole_vargoal_masked_durreg_soft.sbatch (e57: dur_target=4,
# mask_target=0.3 -> best-15 839, max 854 on cartpole); this script ports it to any DMC task.
# Submit with EXP_TAG, TASK, DUR_TARGET, and optional DUR_MAX (must be >= DUR_TARGET).
# MASK_MODE: prob (default, e57) | entropy | prob_entropy (combined rate+entropy Lagrangian).
# DUR_MODE: fixed (default, e57 goal_duration_reg) | lagrangian (mask-style dual-ascent
#   Lagrangian on |E[dur]-target|, folded out into its own independently-scaled loss key).
# RECIPE: vark_masked (default, e57-style masked variable-K) |
#   mask_fixedk (masking package WITHOUT var-K: fixed manager_sample_freq=MGR_FREQ,
#     no duration head/prior -- isolates masking vs var-K, e.g. hopper locality ctrl) |
#   plain_vark ("e124 + var-K": full goal replacement, no mask, no cond, struct
#     defaults to 0 -- isolates the var-K machinery from the masking package).
# SEED: replicate index (default 0).
# Example:
#   sbatch -J e139_cheetah --export=ALL,EXP_TAG=e139,TASK=dmc_cheetah_run,DUR_TARGET=4.0 run_v3_prior_vargoal_small.sbatch
EXP_TAG="${EXP_TAG:?set EXP_TAG}"
TASK="${TASK:-dmc_cartpole_swingup}"
DUR_TARGET="${DUR_TARGET:-4.0}"
DUR_MIN="${DUR_MIN:-1}"
DUR_MAX="${DUR_MAX:-16}"
if awk "BEGIN {exit !($DUR_TARGET > $DUR_MAX)}"; then
  DUR_MAX="$DUR_TARGET"
fi

RECIPE="${RECIPE:-vark_masked}"
MGR_FREQ="${MGR_FREQ:-8}"

RUN_SUFFIX="${EXP_TAG}_${TASK}_vargoal_masked_durreg_soft${DUR_TARGET%%.*}"
if [ "$RECIPE" != "vark_masked" ]; then
  RUN_SUFFIX="${EXP_TAG}_${TASK}_${RECIPE}"
fi
# Requeue-safe RUN_DIR: keyed to SLURM_JOB_ID (stable across requeues) so a
# requeued job resumes from checkpoint instead of restarting in a fresh mktemp dir.
RUN_DIR="${RUN_DIR:-/work/DoyaU/vasilache/work/${RUN_SUFFIX}_j${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"

export MUJOCO_GL=egl

MASK_TOPK="${MASK_TOPK:-0}"
MASK_MODE="${MASK_MODE:-prob}"
MASK_MAX="${MASK_MAX:-5.0}"
MASK_TARGET="${MASK_TARGET:-0.3}"
MGR_ACHIEVE="${MGR_ACHIEVE:-False}"
DUR_REG="${DUR_REG:-0.01}"
DUR_MODE="${DUR_MODE:-fixed}"
DUR_LAGRANGE_MAX="${DUR_LAGRANGE_MAX:-5.0}"
SEED="${SEED:-0}"

if [ "$DUR_MODE" = "lagrangian" ]; then
  DUR_FLAGS=(
    --agent.goal_duration_reg 0.0
    --agent.goal_duration_adapt False
    --agent.goal_duration_lagrange True
    --agent.goal_duration_lagrange_max "$DUR_LAGRANGE_MAX"
  )
else
  DUR_FLAGS=(
    --agent.goal_duration_reg "$DUR_REG"
    --agent.goal_duration_adapt False
    --agent.goal_duration_lagrange False
  )
fi

MASK_FLAGS=(
  --agent.mask_topk "$MASK_TOPK"
  --agent.mask_sparsity_mode "$MASK_MODE"
  --agent.mask_sparsity_max "$MASK_MAX"
  --agent.mask_sparsity_target "$MASK_TARGET"
  --agent.mgr_cond_goalcode True
  --agent.mgr_cond_achieve "$MGR_ACHIEVE"
)
VARK_FLAGS=(
  --agent.variable_goal_length True
  --agent.goal_duration_min "$DUR_MIN"
  --agent.goal_duration_max "$DUR_MAX"
  --agent.goal_duration_target "$DUR_TARGET"
  "${DUR_FLAGS[@]}"
)

case "$RECIPE" in
  vark_masked)
    CONFIGS="size6m masked_goals variable_goals"
    STRUCT_W="${STRUCT_W:-200.0}"
    RECIPE_FLAGS=( "${MASK_FLAGS[@]}" "${VARK_FLAGS[@]}" )
    ;;
  mask_fixedk)
    CONFIGS="size6m masked_goals"
    STRUCT_W="${STRUCT_W:-200.0}"
    RECIPE_FLAGS=(
      "${MASK_FLAGS[@]}"
      --agent.manager_sample_freq "$MGR_FREQ"
      --agent.imag_length 32
    )
    ;;
  plain_vark)
    CONFIGS="size6m variable_goals"
    STRUCT_W="${STRUCT_W:-0.0}"
    RECIPE_FLAGS=( "${VARK_FLAGS[@]}" )
    ;;
  *)
    echo "unknown RECIPE=$RECIPE" >&2; exit 1
    ;;
esac

echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"
echo "EXP_TAG=$EXP_TAG TASK=$TASK RECIPE=$RECIPE DUR_TARGET=$DUR_TARGET DUR_MODE=$DUR_MODE DUR_REG=$DUR_REG DUR_MAX=$DUR_MAX MASK_MODE=$MASK_MODE MASK_TARGET=$MASK_TARGET STRUCT_W=$STRUCT_W MGR_FREQ=$MGR_FREQ SEED=$SEED" >> "$RUN_DIR/job.env"
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
