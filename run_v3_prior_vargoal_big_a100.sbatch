#!/bin/bash -l
#SBATCH -J prior_vargoal_big_a100
#SBATCH -p gpu-a100
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600

# BIG-scale port of the e57 prior recipe (masked + variable-K + soft fixed duration
# prior, dur_target=4 / mask_target=0.3 -> cartpole best-15 839, max 854) to
# director_match 64x64 on a single A100-80GB (real default batch b16x64, imag16,
# train_ratio 64 -- matches run_v3_hrl_auto_big_a100.sbatch's proven-to-fit scale;
# imag_length forced to 16 since variable_goals' default 32 won't fit BIG). Uses
# explicit prior flags (this script), not hrl_auto's adaptive Lagrange targets.
# Submit with EXP_TAG, TASK, and optional DUR_TARGET/MASK_TARGET.
# Example:
#   sbatch -J e141_cartpole_BIG --export=ALL,EXP_TAG=e141,TASK=dmc_cartpole_swingup run_v3_prior_vargoal_big_a100.sbatch
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
MASK_MODE="${MASK_MODE:-prob}"
MASK_MAX="${MASK_MAX:-5.0}"
MASK_TARGET="${MASK_TARGET:-0.3}"
MGR_ACHIEVE="${MGR_ACHIEVE:-False}"
DUR_REG="${DUR_REG:-0.01}"
DUR_MODE="${DUR_MODE:-fixed}"
DUR_LAGRANGE_MAX="${DUR_LAGRANGE_MAX:-5.0}"
SEED="${SEED:-0}"
RECIPE="${RECIPE:-vark_masked}"   # vark_masked | mask_fixedk | plain_vark (see small script)
MGR_FREQ="${MGR_FREQ:-8}"
MGR_REWARD_AGG="${MGR_REWARD_AGG:-mean}"   # mean (default) | sum (couples mgr return to hold length)
MGR_EXPL_W="${MGR_EXPL_W:-0.1}"            # manager exploration-reward weight (default 0.1)

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
    CONFIGS="director_match masked_goals variable_goals"
    STRUCT_W="${STRUCT_W:-200.0}"
    RECIPE_FLAGS=( "${MASK_FLAGS[@]}" "${VARK_FLAGS[@]}" )
    ;;
  mask_fixedk)
    CONFIGS="director_match masked_goals"
    STRUCT_W="${STRUCT_W:-200.0}"
    RECIPE_FLAGS=( "${MASK_FLAGS[@]}" --agent.manager_sample_freq "$MGR_FREQ" )
    ;;
  plain_vark)
    CONFIGS="director_match variable_goals"
    STRUCT_W="${STRUCT_W:-0.0}"
    RECIPE_FLAGS=( "${VARK_FLAGS[@]}" )
    ;;
  *)
    echo "unknown RECIPE=$RECIPE" >&2; exit 1
    ;;
esac

RUN_SUFFIX="${EXP_TAG}_${TASK}_vargoal_masked_durreg_soft${DUR_TARGET%%.*}_BIG"
if [ "$RECIPE" != "vark_masked" ]; then
  RUN_SUFFIX="${EXP_TAG}_${TASK}_${RECIPE}_BIG"
fi
# Requeue-safe RUN_DIR: requeues keep the same SLURM_JOB_ID, so the fallback is
# deterministic and a preempted/requeued job resumes from checkpoint in the same
# dir instead of restarting fresh in a new mktemp dir (e177 lost 19h to this).
RUN_DIR="${RUN_DIR:-/work/DoyaU/vasilache/work/${RUN_SUFFIX}_j${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"
export MUJOCO_GL=egl

cat > "$RUN_DIR/job.env" <<EOF
RUN_DIR=$RUN_DIR
EXP_TAG=$EXP_TAG
CONFIGS=$CONFIGS
TASK=$TASK
SIZE=BIG IMAGE=64x64 IMAG=16 BATCH=16x64 TRAIN_RATIO=64 CONV=native
RECIPE=$RECIPE DUR_TARGET=$DUR_TARGET DUR_MODE=$DUR_MODE DUR_REG=$DUR_REG DUR_MAX=$DUR_MAX MASK_TARGET=$MASK_TARGET MASK_MODE=$MASK_MODE STRUCT_W=$STRUCT_W MGR_FREQ=$MGR_FREQ MGR_REWARD_AGG=$MGR_REWARD_AGG MGR_EXPL_W=$MGR_EXPL_W SEED=$SEED
NOTE=prior-vargoal template (RECIPE-switched) at BIG director_match scale
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
  --agent.goal_struct_weight "$STRUCT_W"
  "${RECIPE_FLAGS[@]}"
  --agent.imag_length 16
  --agent.report False
)

# Self-requeue near walltime (USR1 arrives 600s before the limit): requeue keeps
# the job id -> same RUN_DIR -> checkpoint resume. Wall-time TIMEOUT is terminal
# otherwise (only preemption auto-requeues).
on_usr1() {
  echo "[self-requeue] USR1 near walltime; requeueing job $SLURM_JOB_ID to resume in $RUN_DIR"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap on_usr1 USR1

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --seed "$SEED" \
  --configs $CONFIGS \
  --task "$TASK" \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 64 64 \
  --batch_size 16 \
  --batch_length 64 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.mgr_expl_weight "$MGR_EXPL_W" \
  --agent.mgr_reward_agg "$MGR_REWARD_AGG" \
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
# Backgrounded + wait so the USR1 trap can run while training is in flight
# (bash defers traps until a foreground command returns).
wait "$TRAIN_PID"

exit $?
