#!/bin/bash -l
#SBATCH -J benchpack
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600

# Run SEVERAL training runs concurrently on ONE GPU.
#
# WHY. At size6m a run uses ~2.4 GB of an 80 GB A100 and nvidia-smi reports
# compute_avg 0.97 -- but that metric only says a kernel was resident, not that
# the SMs were busy. A 3.94M-parameter model at batch 16x64 does not fill an
# A100. SLURM cannot help here: this cluster exposes `GresTypes = gpu,mic` with
# no MPS or shard type, so it hands out whole GPUs. Packing inside one job is
# the only route.
#
# Each process gets its own RUN_DIR, its own wandb run, and its own replay. They
# share the device. `--jax.prealloc False` is mandatory: the default grabs ~75%
# of the card per process and the second one would OOM immediately.
#
# Usage:
#   sbatch -p gpu-a100 --gres=gpu:a100:1 -c 32 --mem=160G \
#     --export=ALL,PARAMS=<file>,ROWS=0,1,2,3 \
#     sbatch/run_benchmark_packed.sbatch
#
# ROWS is a comma-separated list of 0-based line numbers in PARAMS.

set -uo pipefail

REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3
WD=/work/DoyaU/vasilache/work
mkdir -p "$WD/slurm_logs"

PARAMS="${PARAMS:?set PARAMS=<file>}"
ROWS="${ROWS:?set ROWS=0,1,2,...}"
BENCH_WANDB_PROJECT="${WANDB_PROJECT:-dreamerv3-bench-2026-09}"
RUN_ENVS="${RUN_ENVS:-4}"

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl
WANDB_PROJECT="$BENCH_WANDB_PROJECT"; export WANDB_PROJECT

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
if [ -n "${CONV_IMPL:-}" ]; then :
elif printf '%s' "$GPU_NAME" | grep -qi 'A100'; then CONV_IMPL=native
else CONV_IMPL=reference; fi
if [ "$CONV_IMPL" = native ]; then unset DREAMERV3_CONV_IMPL
else export DREAMERV3_CONV_IMPL=reference; fi
echo "[packed] gpu='$GPU_NAME' conv=$CONV_IMPL rows=$ROWS"

mapfile -t ALLROWS < "$PARAMS"
IFS=',' read -ra IDXS <<< "$ROWS"

PIDS=(); DIRS=()
on_usr1() {
  echo "[packed] USR1: requeueing $SLURM_JOB_ID"
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap on_usr1 USR1

for idx in "${IDXS[@]}"; do
  read -r CONFIG TASK SEED EXP_TAG EXTRA_BLOCKS <<< "${ALLROWS[$idx]}"
  CONFIG="$CONFIG${EXTRA_BLOCKS:+ $EXTRA_BLOCKS}"
  CONFIG_TAG="$(printf %s "$CONFIG" | tr ' ' '-')"
  JOBKEY="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
  [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && JOBKEY="${JOBKEY}_${SLURM_ARRAY_TASK_ID}"
  RUN_DIR="$WD/${EXP_TAG}_${TASK}_${CONFIG_TAG}_s${SEED}_j${JOBKEY}"
  mkdir -p "$RUN_DIR/logdir"
  DIRS+=("$RUN_DIR")
  {
    echo "EXP_TAG=$EXP_TAG"; echo "CONFIG=$CONFIG"; echo "TASK=$TASK"
    echo "SEED=$SEED"; echo "RUN_DIR=$RUN_DIR"; echo "JOB=$SLURM_JOB_ID"
    echo "NODE=$(hostname)"; echo "GPU=$GPU_NAME"; echo "CONV_IMPL=$CONV_IMPL"
    echo "PACKED=${#IDXS[@]}"; echo "WANDB_PROJECT=$WANDB_PROJECT"
  } > "$RUN_DIR/job.env"

  ENV_FLAGS=()
  case "$TASK" in
    dmc_*) ENV_FLAGS=(--env.dmc.image True --env.dmc.proprio True --env.dmc.size 64 64) ;;
  esac

  WANDB_RUN_GROUP="${CONFIG_TAG}_${TASK}" \
  python -u "$REPO/dreamerv3/main.py" \
    --logdir "$RUN_DIR/logdir" \
    --seed "$SEED" \
    --configs $CONFIG \
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
    --jax.train_devices 0 \
    --jax.policy_devices 0 \
    --jax.prealloc False \
    > "$RUN_DIR/stdout.log" 2> "$RUN_DIR/stderr.log" &
  PIDS+=($!)
  echo "[packed] started row=$idx pid=${PIDS[-1]} $CONFIG $TASK s$SEED -> $RUN_DIR"
  sleep 20   # stagger compilation so they do not all JIT at once
done

RC=0
for p in "${PIDS[@]}"; do
  wait "$p" || RC=$?
done
echo "[packed] all done rc=$RC"
for d in "${DIRS[@]}"; do echo "[done] rc=$RC run_dir=$d"; done
exit $RC
