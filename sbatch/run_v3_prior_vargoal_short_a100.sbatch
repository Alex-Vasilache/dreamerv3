#!/bin/bash -l
#SBATCH -J prior_vargoal_short_a100
#SBATCH -p short-a100
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300

# Same recipe as run_v3_prior_vargoal_big_a100.sbatch, but on the new low-priority
# `short-a100` partition (2h limit, 1h non-preemptible, preempted jobs requeue --
# see reference_short_a100_partition memory / SCDA announcement 2026-07-10).
# RUN_DIR MUST be passed in fixed at submission time (not mktemp'd here) so a
# requeue after preemption resumes from checkpoint in the same logdir instead of
# starting a fresh run.
# Example:
#   RUN_DIR=/work/DoyaU/vasilache/work/e160_... (mktemp'd by the caller)
#   sbatch -J e160_cartpole_BIG --export=ALL,EXP_TAG=e160,TASK=dmc_cartpole_swingup,RUN_DIR="$RUN_DIR",STRUCT_W=0.0 sbatch/run_v3_prior_vargoal_short_a100.sbatch
set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
unset DREAMERV3_CONV_IMPL   # native cuDNN conv (A100 sm_80 ~2x faster)

EXP_TAG="${EXP_TAG:?set EXP_TAG}"
TASK="${TASK:-dmc_cheetah_run}"
DUR_TARGET="${DUR_TARGET:-4.0}"
DUR_MIN="${DUR_MIN:-1}"
DUR_MAX="${DUR_MAX:-16}"
if awk "BEGIN {exit !($DUR_TARGET > $DUR_MAX)}"; then
  DUR_MAX="$DUR_TARGET"
fi

MASK_TOPK="${MASK_TOPK:-0}"
STRUCT_W="${STRUCT_W:-200.0}"
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

RUN_SUFFIX="${EXP_TAG}_${TASK}_vargoal_masked_durreg_soft${DUR_TARGET%%.*}_BIG_shortq"
# Fallback keyed to SLURM_JOB_ID (stable across requeues) so even without a
# caller-supplied RUN_DIR a requeue resumes from checkpoint, never a fresh dir.
RUN_DIR="${RUN_DIR:-/work/DoyaU/vasilache/work/${RUN_SUFFIX}_j${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"
export MUJOCO_GL=egl

cat > "$RUN_DIR/job.env" <<EOF
RUN_DIR=$RUN_DIR
EXP_TAG=$EXP_TAG
CONFIGS=director_match masked_goals variable_goals
TASK=$TASK
SIZE=BIG IMAGE=64x64 IMAG=16 BATCH=16x64 TRAIN_RATIO=64 CONV=native
DUR_TARGET=$DUR_TARGET DUR_MODE=$DUR_MODE DUR_REG=$DUR_REG DUR_MAX=$DUR_MAX MASK_TARGET=$MASK_TARGET MASK_MODE=$MASK_MODE STRUCT_W=$STRUCT_W SEED=$SEED
NOTE=Group D (combined) recipe, struct_weight=0 ablation, run on short-a100 (requeue-safe fixed RUN_DIR)
EOF
RUN_STEPS="${RUN_STEPS:-4000000}"

AGENT_FLAGS=(
  --agent.goal_autoencoder_beta 0.25
  --agent.goal_rec_loss_agg sum
  --agent.goal_opt.lr 4e-5
  --agent.goal_opt.wd 0.0
  --agent.goal_opt.eps 1e-20
  --agent.manager_actent_impl mult
  --agent.manager_actent_target 0.5
  --agent.mask_topk "$MASK_TOPK"
  --agent.goal_struct_weight "$STRUCT_W"
  --agent.mask_sparsity_mode "$MASK_MODE"
  --agent.mask_sparsity_max "$MASK_MAX"
  --agent.mask_sparsity_target "$MASK_TARGET"
  --agent.mgr_cond_goalcode True
  --agent.mgr_cond_achieve "$MGR_ACHIEVE"
  --agent.variable_goal_length True
  --agent.goal_duration_min "$DUR_MIN"
  --agent.goal_duration_max "$DUR_MAX"
  --agent.goal_duration_target "$DUR_TARGET"
  "${DUR_FLAGS[@]}"
  --agent.imag_length 16
  --agent.report False
)

# Self-requeue near the 2h wall limit: preemption auto-requeues (--requeue) but
# wall-time TIMEOUT is terminal — this is why e160/e161 stalled at ~0.6M steps.
# USR1 arrives 300s early; requeue keeps the job id and RUN_DIR -> checkpoint resume.
on_usr1() {
  echo "[self-requeue] USR1 near walltime; requeueing job $SLURM_JOB_ID to resume in $RUN_DIR"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap on_usr1 USR1

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --seed "$SEED" \
  --configs director_match masked_goals variable_goals \
  --task "$TASK" \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 64 64 \
  --batch_size 16 \
  --batch_length 64 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.mgr_expl_weight 0.1 \
  --agent.mgr_retnorm.impl meanstd \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 8 \
  --run.steps "$RUN_STEPS" \
  --run.train_ratio 64 \
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
