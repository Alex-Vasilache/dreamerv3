#!/bin/bash -l
#SBATCH -J e131_big_v100
#SBATCH -p gpu-v100
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=192G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Auto-tuning HRL BIG cell sharded across 4x V100 (16GB each) via data parallelism:
# `hrl_auto` config at director_match 64x64, real default batch (b16x64, imag16,
# train_ratio 64), batch16/NGPU per device. Requires the Route-A-fixed clone env
# (native cuDNN conv on V100 sm_70). Fit validated by e119/e122 (~11-13 GB/device).
# TASK default hopper_hop. imag_length forced to 16 (hrl_auto's 32 won't fit BIG).
set -uo pipefail

CLONE=/work/DoyaU/vasilache/work/dreamerv3_env_routeA
module purge
_PY311_LIB=/apps/free88/python/3.11.4/lib
if ! module is-avail python/3.11.4 2>/dev/null; then
  module use /apps/.modulefiles88 2>/dev/null || true
  module use /apps/.modulefiles95 2>/dev/null || true
fi
module load python/3.11.4 2>/dev/null || export LD_LIBRARY_PATH="${_PY311_LIB}:${LD_LIBRARY_PATH:-}"
source "$CLONE/bin/activate"
python -c 'import sys' >/dev/null 2>&1 || export LD_LIBRARY_PATH="${_PY311_LIB}:${LD_LIBRARY_PATH:-}"
unset DREAMERV3_CONV_IMPL   # native conv (Route-A clone)
export MUJOCO_GL=egl
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false"
# WANDB key from the canonical env file at runtime (do not embed the secret here).
_ENVFILE=/apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export WANDB_API_KEY="$(sed -n 's/^export WANDB_API_KEY=//p' "$_ENVFILE" | head -1)"
export WANDB_PROJECT=dreamerv3
export WANDB_ENTITY=vasilache
[ -n "${WANDB_API_KEY:-}" ] || echo "[WARN] WANDB_API_KEY not found in $_ENVFILE — wandb -> jsonl only"

NGPU="${NGPU:-4}"
EXP_TAG="${EXP_TAG:-e131}"
TASK="${TASK:-dmc_hopper_hop}"
MASK_MODE="${MASK_MODE:-prob}"
DEVLIST=$(seq -s ' ' 0 $((NGPU-1)))
RUN_SUFFIX="${EXP_TAG}_${TASK}_hrl_auto_BIG_multiv100"
RUN_DIR="${RUN_DIR:-$(mktemp -d /work/DoyaU/vasilache/work/${RUN_SUFFIX}_$(date +%Y%m%d_%H%M%S)_XXXXXX)}"
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"
echo "[$EXP_TAG] NODE=$(hostname) TASK=$TASK NGPU=$NGPU DEVLIST='$DEVLIST' RUN_DIR=$RUN_DIR"

cat > "$RUN_DIR/job.env" <<EOF
RUN_DIR=$RUN_DIR
EXP_TAG=$EXP_TAG
CONFIGS=director_match hrl_auto
TASK=$TASK
SIZE=BIG IMAGE=64x64 IMAG=16 BATCH=16x64 TRAIN_RATIO=64 NGPU=$NGPU CONV=native ENV=routeA_clone
MASK_MODE=$MASK_MODE
NOTE=auto-tuning HRL BIG (all-adaptive; cond off, mean agg, no costs, no K prior)
EOF

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs director_match hrl_auto \
  --task "$TASK" \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 64 64 \
  --batch_size 16 \
  --batch_length 64 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.imag_length 16 \
  --agent.mask_sparsity_mode "$MASK_MODE" \
  --agent.report False \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 8 \
  --run.steps 4000000 \
  --run.train_ratio 64 \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  --jax.train_devices $DEVLIST \
  --jax.policy_devices 0
rc=$?
echo "[$EXP_TAG] EXIT=$rc TASK=$TASK NGPU=$NGPU  (rc!=0 + RESOURCE_EXHAUSTED => OOM at $NGPU GPUs)"
exit $rc
