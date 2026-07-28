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

MASK_TOPK="${MASK_TOPK:-0}"
MASK_MODE="${MASK_MODE:-prob}"
MASK_MAX="${MASK_MAX:-5.0}"
MASK_TARGET="${MASK_TARGET:-0.3}"
# Open-loop anneal of the sparsity target itself (agent.Ratchet): ramps from
# MASK_TARGET_INIT toward MASK_TARGET by at most MASK_TARGET_VEL per training
# call. Default init == target -> no-op (byte-for-byte prior behavior).
MASK_TARGET_INIT="${MASK_TARGET_INIT:-$MASK_TARGET}"
MASK_TARGET_VEL="${MASK_TARGET_VEL:-0.01}"
MGR_ACHIEVE="${MGR_ACHIEVE:-False}"
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
# Implicit-sparsity controller (mask-free): drive measured implicit block sparsity
# toward IMPL_TARGET via a per-decision REINFORCE change-cost. See small script.
IMPL_MODE="${IMPL_MODE:-none}"
IMPL_TARGET="${IMPL_TARGET:-0.5}"
IMPL_TARGET_INIT="${IMPL_TARGET_INIT:-$IMPL_TARGET}"
IMPL_TARGET_VEL="${IMPL_TARGET_VEL:-0.001}"
IMPL_ONE_SIDED="${IMPL_ONE_SIDED:-False}"
# Condition the manager on the previous goal code so it can deliberately reuse
# blocks (required for learnable implicit sparsity). Valid on plain Director too.
MGR_COND_GOALCODE="${MGR_COND_GOALCODE:-False}"
# Delta goal generation (07-16, F18 follow-up): additive sigmoid-vote combination
# with the previous goal code's soft distribution -- see small script / agent.py
# _delta_combine_skill / EXPERIMENTS.md §7. Implies mgr_cond_goalcode.
GOAL_DELTA_MODE="${GOAL_DELTA_MODE:-False}"
GOAL_DELTA_CLIP="${GOAL_DELTA_CLIP:-1.0}"
# Differentiable continuous-goal-space reuse loss (07-16 follow-up) -- see
# small script / agent._decode_goal_no_decoder_grad / EXPERIMENTS.md §7.
GOAL_REUSE_WEIGHT="${GOAL_REUSE_WEIGHT:-0.0}"
GOAL_REUSE_ADAPT="${GOAL_REUSE_ADAPT:-False}"
GOAL_REUSE_TARGET="${GOAL_REUSE_TARGET:-0.9}"
GOAL_REUSE_TARGET_INIT="${GOAL_REUSE_TARGET_INIT:-$GOAL_REUSE_TARGET}"
GOAL_REUSE_TARGET_VEL="${GOAL_REUSE_TARGET_VEL:-0.01}"
# Direct (REINFORCE-free) implicit sparsity (07-16, F18 follow-up #2) -- see
# small script / agent.py goal_soft_reuse_adapt / EXPERIMENTS.md §7. Mutually
# exclusive with GOAL_DELTA_MODE. Implies mgr_cond_goalcode.
GOAL_SOFT_REUSE_ADAPT="${GOAL_SOFT_REUSE_ADAPT:-False}"
GOAL_SOFT_REUSE_TARGET="${GOAL_SOFT_REUSE_TARGET:-0.7}"
GOAL_SOFT_REUSE_TARGET_INIT="${GOAL_SOFT_REUSE_TARGET_INIT:-$GOAL_SOFT_REUSE_TARGET}"
GOAL_SOFT_REUSE_TARGET_VEL="${GOAL_SOFT_REUSE_TARGET_VEL:-0.01}"
RECIPE="${RECIPE:-vark_masked}"   # vark_masked | mask_fixedk | plain_vark (see small script)
MGR_FREQ="${MGR_FREQ:-8}"
MGR_REWARD_AGG="${MGR_REWARD_AGG:-mean}"   # mean (default) | sum (couples mgr return to hold length)
MGR_EXPL_W="${MGR_EXPL_W:-0.1}"            # manager exploration-reward weight (default 0.1)
# Director-style block-pooled manager credit under variable-K (default False =
# full-resolution). Pools reward (via MGR_REWARD_AGG)/continuation over each
# realized hold segment instead of training the critic on every raw step. See
# EXPERIMENTS.md "Variable-K block-pooled reward" / agent.py
# variable_block_director_tensors + relabel_truncated_last_duration (hindsight-
# relabels a hold's duration class when the imagination horizon cuts it short,
# so REINFORCE credits the duration actually realized, not the one sampled).
VARIABLE_GOAL_BLOCK_REW="${VARIABLE_GOAL_BLOCK_REW:-False}"
# A/B switch for the 2026-07-27 padded-axis normalization fix ALONE: True (default)
# normalizes packed manager losses by the real decision count; False restores the
# old ~K-fold down-weighting while keeping the other block-pooled fixes on.
MGR_DEC_RESCALE="${MGR_DEC_RESCALE:-True}"
GOAL_DURATION_RELABEL_TRUNCATED="${GOAL_DURATION_RELABEL_TRUNCATED:-True}"  # hindsight-relabel a horizon-truncated hold's duration class (see agent.py relabel_truncated_last_duration); ablate by setting False
# NOTE: imag_length was previously hardcoded to 16 here ("won't fit BIG" at
# larger values) -- now overridable. 32 roughly doubles imagination rollout
# activation memory; watch for OOM on gpu-a100 if raising it.
IMAG_LENGTH="${IMAG_LENGTH:-16}"

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
  --agent.mask_sparsity_target_init "$MASK_TARGET_INIT"
  --agent.mask_sparsity_target_vel "$MASK_TARGET_VEL"
  --agent.mgr_cond_goalcode True
  --agent.mgr_cond_achieve "$MGR_ACHIEVE"
)
VARK_FLAGS=(
  --agent.variable_goal_length True
  --agent.goal_duration_min "$DUR_MIN"
  --agent.goal_duration_max "$DUR_MAX"
  --agent.goal_duration_target "$DUR_TARGET"
  --agent.variable_goal_block_rew "$VARIABLE_GOAL_BLOCK_REW"
  --agent.goal_duration_fixed "$DUR_FIXED"
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
  director)
    # Pure Director baseline: fixed-K plain goals (no masked goals, no variable-K).
    CONFIGS="director_match"
    STRUCT_W="${STRUCT_W:-0.0}"
    RECIPE_FLAGS=( --agent.manager_sample_freq "$MGR_FREQ" )
    ;;
  joint_fixedk)
    # Single-head ("joint") masked manager on the fixed-K Director base. Minimal:
    # no mgr_cond, no topk. MASK_MODE=none -> fully free; MASK_MODE=prob +
    # MASK_TARGET_INIT/VEL -> ratchet P(abstain) 0->0.7. STRUCT via STRUCT_ADAPT.
    # imag_length is forced to 16 in the common flags below (BIG memory).
    CONFIGS="director_match masked_goals"
    STRUCT_W="${STRUCT_W:-0.0}"
    RECIPE_FLAGS=(
      --agent.mask_joint_edit True
      --agent.manager_sample_freq "$MGR_FREQ"
      --agent.mask_topk 0
      --agent.mask_sparsity_mode "$MASK_MODE"
      --agent.mask_sparsity_max "$MASK_MAX"
      --agent.mask_sparsity_target "$MASK_TARGET"
      --agent.mask_sparsity_target_init "$MASK_TARGET_INIT"
      --agent.mask_sparsity_target_vel "$MASK_TARGET_VEL"
    )
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
SIZE=BIG IMAGE=64x64 IMAG=$IMAG_LENGTH BATCH=16x64 TRAIN_RATIO=64 CONV=native
RECIPE=$RECIPE DUR_TARGET=$DUR_TARGET DUR_MODE=$DUR_MODE DUR_REG=$DUR_REG DUR_MAX=$DUR_MAX MASK_TARGET=$MASK_TARGET MASK_TARGET_INIT=$MASK_TARGET_INIT MASK_TARGET_VEL=$MASK_TARGET_VEL MASK_MODE=$MASK_MODE STRUCT_W=$STRUCT_W STRUCT_ADAPT=$STRUCT_ADAPT STRUCT_ADAPT_TARGET=$STRUCT_ADAPT_TARGET WORKER_TIMED_GOALS=$WORKER_TIMED_GOALS DUR_ACTENT_TARGET=$DUR_ACTENT_TARGET DUR_FIXED=$DUR_FIXED MGR_FREQ=$MGR_FREQ MGR_REWARD_AGG=$MGR_REWARD_AGG VARIABLE_GOAL_BLOCK_REW=$VARIABLE_GOAL_BLOCK_REW MGR_DEC_RESCALE=$MGR_DEC_RESCALE GOAL_DURATION_RELABEL_TRUNCATED=$GOAL_DURATION_RELABEL_TRUNCATED MGR_EXPL_W=$MGR_EXPL_W IMPL_MODE=$IMPL_MODE IMPL_TARGET=$IMPL_TARGET IMPL_TARGET_INIT=$IMPL_TARGET_INIT IMPL_TARGET_VEL=$IMPL_TARGET_VEL IMPL_ONE_SIDED=$IMPL_ONE_SIDED MGR_COND_GOALCODE=$MGR_COND_GOALCODE GOAL_DELTA_MODE=$GOAL_DELTA_MODE GOAL_DELTA_CLIP=$GOAL_DELTA_CLIP GOAL_REUSE_WEIGHT=$GOAL_REUSE_WEIGHT GOAL_REUSE_ADAPT=$GOAL_REUSE_ADAPT GOAL_REUSE_TARGET=$GOAL_REUSE_TARGET GOAL_REUSE_TARGET_INIT=$GOAL_REUSE_TARGET_INIT GOAL_REUSE_TARGET_VEL=$GOAL_REUSE_TARGET_VEL GOAL_SOFT_REUSE_ADAPT=$GOAL_SOFT_REUSE_ADAPT GOAL_SOFT_REUSE_TARGET=$GOAL_SOFT_REUSE_TARGET GOAL_SOFT_REUSE_TARGET_INIT=$GOAL_SOFT_REUSE_TARGET_INIT GOAL_SOFT_REUSE_TARGET_VEL=$GOAL_SOFT_REUSE_TARGET_VEL SEED=$SEED
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
  --agent.impl_sparsity_mode "$IMPL_MODE"
  --agent.impl_sparsity_target "$IMPL_TARGET"
  --agent.impl_sparsity_target_init "$IMPL_TARGET_INIT"
  --agent.impl_sparsity_target_vel "$IMPL_TARGET_VEL"
  --agent.impl_sparsity_one_sided "$IMPL_ONE_SIDED"
  --agent.mgr_cond_goalcode "$MGR_COND_GOALCODE"
  --agent.goal_delta_mode "$GOAL_DELTA_MODE"
  --agent.goal_delta_clip "$GOAL_DELTA_CLIP"
  --agent.goal_reuse_weight "$GOAL_REUSE_WEIGHT"
  --agent.goal_reuse_adapt "$GOAL_REUSE_ADAPT"
  --agent.goal_reuse_target "$GOAL_REUSE_TARGET"
  --agent.goal_reuse_target_init "$GOAL_REUSE_TARGET_INIT"
  --agent.goal_reuse_target_vel "$GOAL_REUSE_TARGET_VEL"
  --agent.goal_soft_reuse_adapt "$GOAL_SOFT_REUSE_ADAPT"
  --agent.goal_soft_reuse_target "$GOAL_SOFT_REUSE_TARGET"
  --agent.goal_soft_reuse_target_init "$GOAL_SOFT_REUSE_TARGET_INIT"
  --agent.goal_soft_reuse_target_vel "$GOAL_SOFT_REUSE_TARGET_VEL"
  --agent.worker_timed_goals "$WORKER_TIMED_GOALS"
  --agent.manager_actent_duration_target "$DUR_ACTENT_TARGET"
  "${RECIPE_FLAGS[@]}"
  --agent.imag_length "$IMAG_LENGTH"
  --agent.goal_duration_relabel_truncated "$GOAL_DURATION_RELABEL_TRUNCATED"
  --agent.mgr_decision_mean_rescale "$MGR_DEC_RESCALE"
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
