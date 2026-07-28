#!/bin/bash -l
#SBATCH -J prior_vargoal_big_multiv100
#SBATCH -p gpu-v100
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=192G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600

# V100 twin of run_v3_prior_vargoal_big_a100.sbatch: identical RECIPE-switched
# mechanism flags (director | plain_vark |
# but sharded across 4x V100 (16GB each) via data parallelism
# instead of a single A100-80GB -- one V100 can't hold the real default batch
# b16x64 (~59GB on A100); NGPU=4 splits it to batch16/NGPU=4 per device
# (~12GB/device, fit validated by e119-e124). Requires the Route-A-fixed clone
# env for native cuDNN conv to work on V100 sm_70 (see EXPERIMENTS.md §7 /
# CLAUDE.md "native cuDNN conv" note) -- the regular dreamerv3_env's native
# conv is broken on V100 by an unrelated cuDNN/CUDA version drift.
# Submit with EXP_TAG, TASK, and the same RECIPE/*_TARGET/*_ADAPT flags as the
# A100 template. Example (restart a pending A100 job on V100 instead):
#   sbatch -J e294_cartpole_BIG --export=ALL,EXP_TAG=e294,TASK=dmc_cartpole_swingup,RECIPE=director,... sbatch/run_v3_prior_vargoal_big_multiv100_routeA.sbatch
set -uo pipefail

CLONE=/work/DoyaU/vasilache/work/dreamerv3_env_routeA

# --- activate the Route-A clone env (native conv fixed on V100), mirror e124 ---
module purge
_PY311_LIB=/apps/free88/python/3.11.4/lib
if ! module is-avail python/3.11.4 2>/dev/null; then
  module use /apps/.modulefiles88 2>/dev/null || true
  module use /apps/.modulefiles95 2>/dev/null || true
fi
module load python/3.11.4 2>/dev/null || export LD_LIBRARY_PATH="${_PY311_LIB}:${LD_LIBRARY_PATH:-}"
source "$CLONE/bin/activate"
python -c 'import sys' >/dev/null 2>&1 || export LD_LIBRARY_PATH="${_PY311_LIB}:${LD_LIBRARY_PATH:-}"
unset DREAMERV3_CONV_IMPL   # native conv (Route-A clone; ~2x faster than reference)
export MUJOCO_GL=egl
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false"
# This script activates the clone venv directly (does not source
# source_dreamerv3_env.sh, which would `module purge`) -- read the WANDB key
# from that canonical env file at runtime instead of duplicating the secret.
_ENVFILE=/apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export WANDB_API_KEY="$(sed -n 's/^export WANDB_API_KEY=//p' "$_ENVFILE" | head -1)"
export WANDB_PROJECT=dreamerv3
export WANDB_ENTITY=vasilache
[ -n "${WANDB_API_KEY:-}" ] || echo "[WARN] WANDB_API_KEY not found in $_ENVFILE — wandb will drop to jsonl-only"

NGPU="${NGPU:-4}"
DEVLIST=$(seq -s ' ' 0 $((NGPU-1)))

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
SEED="${SEED:-0}"
WORKER_TIMED_GOALS="${WORKER_TIMED_GOALS:-False}"   # HiTS-style worker countdown conditioning
STRUCT_ADAPT="${STRUCT_ADAPT:-False}"   # AutoAdapt multiplier toward struct-loss ceiling (goal_struct_adapt)
STRUCT_ADAPT_TARGET="${STRUCT_ADAPT_TARGET:-0.005}"  # struct-loss setpoint (dual-ascent unless one_sided)
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

RUN_SUFFIX="${EXP_TAG}_${TASK}_${RECIPE}_BIG_multiv100"
# Requeue-safe RUN_DIR: requeues keep the same SLURM_JOB_ID, so the fallback is
# deterministic and a preempted/requeued job resumes from checkpoint in the same
# dir instead of restarting fresh (e177 lost 19h to this).
RUN_DIR="${RUN_DIR:-/work/DoyaU/vasilache/work/${RUN_SUFFIX}_j${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"

cat > "$RUN_DIR/job.env" <<EOF
RUN_DIR=$RUN_DIR
EXP_TAG=$EXP_TAG
CONFIGS=$CONFIGS
TASK=$TASK
SIZE=BIG IMAGE=64x64 IMAG=16 BATCH=16x64 TRAIN_RATIO=64 CONV=native ENV=routeA_clone NGPU=$NGPU
RECIPE=$RECIPE DUR_TARGET=$DUR_TARGET DUR_MODE=$DUR_MODE DUR_REG=$DUR_REG DUR_MAX=$DUR_MAX STRUCT_W=$STRUCT_W STRUCT_ADAPT=$STRUCT_ADAPT STRUCT_ADAPT_TARGET=$STRUCT_ADAPT_TARGET WORKER_TIMED_GOALS=$WORKER_TIMED_GOALS MGR_FREQ=$MGR_FREQ MGR_REWARD_AGG=$MGR_REWARD_AGG MGR_EXPL_W=$MGR_EXPL_W MGR_COND_GOALCODE=$MGR_COND_GOALCODE GOAL_SOFT_REUSE_ADAPT=$GOAL_SOFT_REUSE_ADAPT GOAL_SOFT_REUSE_TARGET=$GOAL_SOFT_REUSE_TARGET GOAL_SOFT_REUSE_TARGET_INIT=$GOAL_SOFT_REUSE_TARGET_INIT GOAL_SOFT_REUSE_TARGET_VEL=$GOAL_SOFT_REUSE_TARGET_VEL SEED=$SEED
NOTE=prior-vargoal template (RECIPE-switched) at BIG director_match scale, sharded across NGPU V100s (Route-A clone)
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
  --agent.mgr_cond_goalcode "$MGR_COND_GOALCODE"
  --agent.goal_soft_reuse_adapt "$GOAL_SOFT_REUSE_ADAPT"
  --agent.goal_soft_reuse_target "$GOAL_SOFT_REUSE_TARGET"
  --agent.goal_soft_reuse_target_init "$GOAL_SOFT_REUSE_TARGET_INIT"
  --agent.goal_soft_reuse_target_vel "$GOAL_SOFT_REUSE_TARGET_VEL"
  --agent.worker_timed_goals "$WORKER_TIMED_GOALS"
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
  --jax.train_devices $DEVLIST \
  --jax.policy_devices 0 \
  "${AGENT_FLAGS[@]}" &
TRAIN_PID=$!
# Backgrounded + wait so the USR1 trap can run while training is in flight
# (bash defers traps until a foreground command returns).
wait "$TRAIN_PID"

exit $?
