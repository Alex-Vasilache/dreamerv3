#!/bin/bash -l
#SBATCH -J bench
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600

# 2026-09-04 benchmark launcher: dreamerv3 / director / som_lip.
#
# DELIBERATELY MINIMAL. The older launchers (run_v3_director_baseline_multigpu,
# run_v3_goal_ae_ablation_big_a100) build a ~20-entry AGENT_FLAGS array, and a
# CLI flag beats a config block -- which is how the 2026-09-01 defaults never
# reached a run (commit 9ae4ee0) and how `director_stable` could not turn on
# the goal-AE weight decay it was written to test. This script passes NOTHING
# that a config block owns: no lr, no wd, no beta, no actent target, no
# retnorm, no imag_length, no train_ratio, no replay size. `--configs $CONFIG`
# is the single source of truth for the agent.
#
# Usage (see sbatch/submit_benchmark.sh, which generates the whole matrix):
#   sbatch -p gpu-a100 --gres=gpu:a100:1 -c 16 --mem=64G \
#     --export=ALL,CONFIG=director,TASK=dmc_cheetah_run,SEED=0,EXP_TAG=b001 \
#     sbatch/run_benchmark.sbatch

set -uo pipefail

REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3
WD=/work/DoyaU/vasilache/work
mkdir -p "$WD/slurm_logs"

# Array mode: one line per task in PARAMS, "CONFIG TASK SEED EXP_TAG".
# Single-job mode: CONFIG/TASK/SEED/EXP_TAG come straight from --export.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  PARAMS="${PARAMS:?array mode needs PARAMS=<file>}"
  mapfile -t ROWS < "$PARAMS"
  read -r CONFIG TASK SEED EXP_TAG <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
  echo "[array] task=$SLURM_ARRAY_TASK_ID -> $CONFIG $TASK seed=$SEED tag=$EXP_TAG"
fi

CONFIG="${CONFIG:?set CONFIG=dreamerv3|director|som_lip}"
TASK="${TASK:?set TASK}"
SEED="${SEED:-0}"
EXP_TAG="${EXP_TAG:-bench}"
# NOTE: source_dreamerv3_env.sh exports WANDB_PROJECT=dreamerv3, so this value
# is stashed under a private name and re-applied AFTER the source below.
# Resolving into WANDB_PROJECT here would be silently overwritten -- the same
# generic-environment-name collision as WORK=/work in the 2026-08-15 watchdog.
BENCH_WANDB_PROJECT="${WANDB_PROJECT:-dreamerv3-bench-2026-09}"
RUN_ENVS="${RUN_ENVS:-8}"

# Fixed RUN_DIR keyed on the job id so a requeue RESUMES instead of restarting.
# SLURM_JOB_ID is stable across `scontrol requeue` of the same job.
JOBKEY="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && JOBKEY="${JOBKEY}_${SLURM_ARRAY_TASK_ID}"
RUN_DIR="${RUN_DIR:-$WD/${EXP_TAG}_${TASK}_${CONFIG}_s${SEED}_j${JOBKEY}}"
mkdir -p "$RUN_DIR/logdir"

# Env must be sourced (not the venv activated directly): source_dreamerv3_env.sh
# is what exports WANDB_API_KEY. Activating the venv alone silently drops it and
# the run logs to jsonl only -- that is what happened to e119-e122.
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
# Conv implementation, per GPU generation.
#   A100 (sm_80): native cuDNN, ~2x faster.
#   V100/P100:    native conv dies here with "<unknown cudnn status: 5003>" in
#                 this env (cuDNN/CUDA drifted past the pinned <=12.2 that the
#                 Route A clone restores), so keep the reference impl. It used
#                 to OOM at the old 107M `director_match` size; at size6m the
#                 model is ~18x smaller and it fits.
# Override with CONV_IMPL=native|reference.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
if [ -n "${CONV_IMPL:-}" ]; then
  :
elif printf '%s' "$GPU_NAME" | grep -qi 'A100'; then
  CONV_IMPL=native
else
  CONV_IMPL=reference
fi
if [ "$CONV_IMPL" = native ]; then
  unset DREAMERV3_CONV_IMPL
else
  export DREAMERV3_CONV_IMPL=reference
fi
echo "[conv] gpu='$GPU_NAME' impl=$CONV_IMPL"
export MUJOCO_GL=egl
# Re-apply after the source, which sets its own WANDB_PROJECT.
WANDB_PROJECT="$BENCH_WANDB_PROJECT"
export WANDB_PROJECT
export WANDB_RUN_GROUP="${CONFIG}_${TASK}"

NGPU=$(python3 - <<'EOF'
import os
v = os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS') or ''
n = len([x for x in v.split(',') if x.strip()])
print(n if n else int(os.environ.get('SLURM_GPUS_ON_NODE', '1')))
EOF
)
DEVLIST=$(python3 -c "print(' '.join(str(i) for i in range($NGPU)))")

{
  echo "EXP_TAG=$EXP_TAG"; echo "CONFIG=$CONFIG"; echo "TASK=$TASK"
  echo "SEED=$SEED"; echo "RUN_DIR=$RUN_DIR"; echo "JOB=$SLURM_JOB_ID"
  echo "NODE=$(hostname)"; echo "NGPU=$NGPU"; echo "WANDB_PROJECT=$WANDB_PROJECT"
  echo "GPU=$GPU_NAME"; echo "CONV_IMPL=$CONV_IMPL"
} | tee "$RUN_DIR/job.env"

on_usr1() {
  echo "[self-requeue] USR1 near walltime; requeueing $SLURM_JOB_ID to resume in $RUN_DIR"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap on_usr1 USR1

# DMC tasks need the image/proprio switches; pinpad is image-only and rejects
# the dmc.* flags, so they are only added for dmc_* tasks.
ENV_FLAGS=()
case "$TASK" in
  dmc_*) ENV_FLAGS=(--env.dmc.image True --env.dmc.proprio True --env.dmc.size 64 64) ;;
esac

run_training() {
  python -u "$REPO/dreamerv3/main.py" \
    --logdir "$RUN_DIR/logdir" \
    --seed "$SEED" \
    --configs "$CONFIG" \
    --task "$TASK" \
    "${ENV_FLAGS[@]}" \
    --online_learning False \
    --online_actor_cpu False \
    --run.envs "$RUN_ENVS" \
    --run.log_video False \
    --agent.report False \
    --logger.outputs jsonl,wandb \
    --logger.wandb_mode online \
    --logger.wandb_entity vasilache \
    --logger.wandb_project "$WANDB_PROJECT" \
    --jax.platform cuda \
    --jax.train_devices ${DEVLIST} \
    --jax.policy_devices 0 \
    --jax.prealloc "${PREALLOC:-True}" &
  TRAIN_PID=$!
  # Backgrounded + wait so the USR1 trap can fire while training is in flight
  # (bash defers traps until a foreground command returns).
  wait "$TRAIN_PID"
  return $?
}

run_training
RC=$?
echo "[done] rc=$RC run_dir=$RUN_DIR"
exit $RC
