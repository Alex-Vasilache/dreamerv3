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

# BIG-scale port of the prior recipe (variable-K + soft fixed duration prior,
# dur_target=4) to
# director_match 64x64 on a single A100-80GB (real default batch b16x64, imag16,
# train_ratio 64 -- matches run_v3_hrl_auto_big_a100.sbatch's proven-to-fit scale;
# imag_length forced to 16 since variable_goals' default 32 won't fit BIG). Uses
# explicit prior flags (this script), not hrl_auto's adaptive Lagrange targets.
# Submit with EXP_TAG, TASK, and optional DUR_TARGET.
# Example:
#   sbatch -J e141_cartpole_BIG --export=ALL,EXP_TAG=e141,TASK=dmc_cartpole_swingup sbatch/run_v3_prior_vargoal_big_a100.sbatch
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

DUR_REG="${DUR_REG:-0.01}"
DUR_MODE="${DUR_MODE:-fixed}"
DUR_LAGRANGE_MAX="${DUR_LAGRANGE_MAX:-5.0}"
# Duration-head adaptive entropy regularizer target (embodied.jax.AutoAdapt,
# 'mgr_dur_actent' in agent.py). <0 = config default (inherits
# manager_actent_target, currently 0.5) -- unchanged behavior. Set near 0.0 to
# effectively disable the controller as an ablation: the target is then
# already satisfied by any nonnegative entropy, so the multiplier relaxes to
# its floor and never pushes back against duration-policy collapse.
DUR_ACTENT_TARGET="${DUR_ACTENT_TARGET:--1.0}"
# Bypass the duration head entirely: every decision holds for exactly this
# many steps (agent.py _duration_steps, "control: variable training graph at
# a fixed K"). 0 = off (duration head samples normally). Only meaningful
# under variable_goal_length; wired into VARK_FLAGS below.
DUR_FIXED="${DUR_FIXED:-0}"
SEED="${SEED:-0}"
WORKER_TIMED_GOALS="${WORKER_TIMED_GOALS:-False}"   # HiTS-style worker countdown conditioning
STRUCT_ADAPT="${STRUCT_ADAPT:-False}"   # AutoAdapt multiplier toward struct-loss ceiling (goal_struct_adapt)
STRUCT_ADAPT_TARGET="${STRUCT_ADAPT_TARGET:-0.005}"  # struct-loss setpoint (dual-ascent unless one_sided)
# Log goal/struct_corr[_hard] every train step even with the struct loss off
# (goal_struct_weight=0, STRUCT_ADAPT=False) -- e.g. for pure-Director
# baselines where we want the geometry-preservation trajectory across
# training without optimizing against it. See experiments/goal_code_struct_corr/.
GOAL_STRUCT_DIAG="${GOAL_STRUCT_DIAG:-False}"
# Condition the manager on the previous goal code so it can deliberately reuse
# blocks (required for learnable implicit sparsity). Valid on plain Director too.
MGR_COND_GOALCODE="${MGR_COND_GOALCODE:-False}"
# Direct (REINFORCE-free) implicit sparsity (07-16, F18 follow-up #2) -- see
# small script / agent.py goal_soft_reuse_adapt / EXPERIMENTS.md §7. Mutually
# exclusive with GOAL_DELTA_MODE. Implies mgr_cond_goalcode.
GOAL_SOFT_REUSE_ADAPT="${GOAL_SOFT_REUSE_ADAPT:-False}"
GOAL_SOFT_REUSE_TARGET="${GOAL_SOFT_REUSE_TARGET:-0.7}"
GOAL_SOFT_REUSE_TARGET_INIT="${GOAL_SOFT_REUSE_TARGET_INIT:-$GOAL_SOFT_REUSE_TARGET}"
GOAL_SOFT_REUSE_TARGET_VEL="${GOAL_SOFT_REUSE_TARGET_VEL:-0.01}"
RECIPE="${RECIPE:-plain_vark}"   # plain_vark | director
MGR_FREQ="${MGR_FREQ:-8}"
MGR_REWARD_AGG="${MGR_REWARD_AGG:-mean}"   # mean (default) | sum (couples mgr return to hold length)
MGR_EXPL_W="${MGR_EXPL_W:-0.1}"            # manager exploration-reward weight (default 0.1)
# Hindsight-relabel a horizon-truncated hold's duration class (see
# hrl/tensors.py relabel_truncated_last_duration); ablate by setting False.
# Only consulted when TRUNC_POLICY=keep.
GOAL_DURATION_RELABEL_TRUNCATED="${GOAL_DURATION_RELABEL_TRUNCATED:-True}"
# How the manager POLICY term treats a hold cut short by the imagination
# horizon: keep | drop_duration | drop_decision (see configs.yaml). The
# critic is never gated -- its truncated target is a valid variable-n TD
# target either way.
TRUNC_POLICY="${TRUNC_POLICY:-keep}"
# NOTE: imag_length was previously hardcoded to 16 here ("won't fit BIG" at
# larger values) -- now overridable. 32 roughly doubles imagination rollout
# activation memory; watch for OOM on gpu-a100 if raising it.
IMAG_LENGTH="${IMAG_LENGTH:-16}"

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
  --agent.goal_duration_fixed "$DUR_FIXED"
  "${DUR_FLAGS[@]}"
)

case "$RECIPE" in
  plain_vark)
    CONFIGS="director_match variable_goals"
    STRUCT_W="${STRUCT_W:-0.0}"
    RECIPE_FLAGS=( "${VARK_FLAGS[@]}" )
    ;;
  director)
    # Pure Director baseline: fixed-K goals, no variable-K.
    CONFIGS="director_match"
    STRUCT_W="${STRUCT_W:-0.0}"
    RECIPE_FLAGS=( --agent.manager_sample_freq "$MGR_FREQ" )
    ;;
  *)
    echo "unknown RECIPE=$RECIPE" >&2; exit 1
    ;;
esac

RUN_SUFFIX="${EXP_TAG}_${TASK}_${RECIPE}_BIG"
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
SIZE=BIG IMAGE=64x64 IMAG=$IMAG_LENGTH BATCH=16x64 TRAIN_RATIO=64 CONV=native
RECIPE=$RECIPE DUR_TARGET=$DUR_TARGET DUR_MODE=$DUR_MODE DUR_REG=$DUR_REG DUR_MAX=$DUR_MAX STRUCT_W=$STRUCT_W STRUCT_ADAPT=$STRUCT_ADAPT STRUCT_ADAPT_TARGET=$STRUCT_ADAPT_TARGET GOAL_STRUCT_DIAG=$GOAL_STRUCT_DIAG WORKER_TIMED_GOALS=$WORKER_TIMED_GOALS DUR_ACTENT_TARGET=$DUR_ACTENT_TARGET DUR_FIXED=$DUR_FIXED MGR_FREQ=$MGR_FREQ MGR_REWARD_AGG=$MGR_REWARD_AGG GOAL_DURATION_RELABEL_TRUNCATED=$GOAL_DURATION_RELABEL_TRUNCATED TRUNC_POLICY=$TRUNC_POLICY MGR_EXPL_W=$MGR_EXPL_W MGR_COND_GOALCODE=$MGR_COND_GOALCODE GOAL_SOFT_REUSE_ADAPT=$GOAL_SOFT_REUSE_ADAPT GOAL_SOFT_REUSE_TARGET=$GOAL_SOFT_REUSE_TARGET GOAL_SOFT_REUSE_TARGET_INIT=$GOAL_SOFT_REUSE_TARGET_INIT GOAL_SOFT_REUSE_TARGET_VEL=$GOAL_SOFT_REUSE_TARGET_VEL SEED=$SEED
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
  --agent.goal_struct_adapt "$STRUCT_ADAPT"
  --agent.goal_struct_adapt_target "$STRUCT_ADAPT_TARGET"
  --agent.goal_struct_diag "$GOAL_STRUCT_DIAG"
  --agent.mgr_cond_goalcode "$MGR_COND_GOALCODE"
  --agent.goal_soft_reuse_adapt "$GOAL_SOFT_REUSE_ADAPT"
  --agent.goal_soft_reuse_target "$GOAL_SOFT_REUSE_TARGET"
  --agent.goal_soft_reuse_target_init "$GOAL_SOFT_REUSE_TARGET_INIT"
  --agent.goal_soft_reuse_target_vel "$GOAL_SOFT_REUSE_TARGET_VEL"
  --agent.worker_timed_goals "$WORKER_TIMED_GOALS"
  --agent.manager_actent_duration_target "$DUR_ACTENT_TARGET"
  "${RECIPE_FLAGS[@]}"
  --agent.imag_length "$IMAG_LENGTH"
  --agent.goal_duration_relabel_truncated "$GOAL_DURATION_RELABEL_TRUNCATED"
  --agent.goal_duration_truncated_policy "$TRUNC_POLICY"
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
