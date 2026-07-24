#!/bin/bash -l
#SBATCH -J e124_director
#SBATCH -p gpu-v100
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=192G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# e124/e125: PURE Director-style HRL baseline (defaults only) sharded across 4x
# V100 (16GB each) via data parallelism. Same intent as e123 (A100 cheetah) but
# for the tasks the base must also match. Uses ONLY Director-faithful defaults:
#   use_masked_goals=False (OVERWRITE ENTIRE GOALS), variable_goal_length=False
#   (FIXED goal length K=manager_sample_freq=8), use_hrl=True.
# NO masked-goals / variable-K / struct / per-block-credit / switch-edit costs.
# director_match 64x64 + real default batch (b16x64=1024 frames, imag16, tr64).
#
# Requires the Route-A-fixed clone env (native cuDNN conv works on V100 sm_70).
# One 16GB V100 can't hold b16x64 (needs ~59GB on A100); split across NGPU
# devices (batch16/NGPU per device). Fit validated by e119-e122 (~12GB/device).
#
#   TASK    (default dmc_hopper_hop; set dmc_acrobot_swingup etc.)
#   EXP_TAG (default e124; use e125 for acrobot — give each task its own number)
#   NGPU    (default 4, one node max for single-process JAX)
#   STEPS   (default 4000000 full run; lower for a fit-test)
set -uo pipefail

CLONE=/work/DoyaU/vasilache/work/dreamerv3_env_routeA

# --- activate the Route-A clone env (native conv fixed), mirror module setup ---
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
# WANDB creds: this script activates the clone venv directly (does not source
# source_dreamerv3_env.sh, which would `module purge`). Read the API key from that
# canonical env file at runtime instead of embedding the secret here, so the key
# is not duplicated into another shared, inspectable file.
_ENVFILE=/apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export WANDB_API_KEY="$(sed -n 's/^export WANDB_API_KEY=//p' "$_ENVFILE" | head -1)"
export WANDB_PROJECT=dreamerv3
export WANDB_ENTITY=vasilache
[ -n "${WANDB_API_KEY:-}" ] || echo "[WARN] WANDB_API_KEY not found in $_ENVFILE — wandb will drop to jsonl-only"

NGPU="${NGPU:-4}"
STEPS="${STEPS:-4000000}"
EXP_TAG="${EXP_TAG:-e124}"
TASK="${TASK:-dmc_hopper_hop}"
DEVLIST=$(seq -s ' ' 0 $((NGPU-1)))
RUN_SUFFIX="${EXP_TAG}_${TASK}_director_baseline_multiv100"
RUN_DIR="${RUN_DIR:-$(mktemp -d /work/DoyaU/vasilache/work/${RUN_SUFFIX}_$(date +%Y%m%d_%H%M%S)_XXXXXX)}"
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"
echo "[$EXP_TAG] NODE=$(hostname) TASK=$TASK NGPU=$NGPU DEVLIST='$DEVLIST' STEPS=$STEPS CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?} RUN_DIR=$RUN_DIR"

cat > "$RUN_DIR/job.env" <<EOF
RUN_DIR=$RUN_DIR
EXP_TAG=$EXP_TAG
CONFIGS=director_match
TASK=$TASK
IMAGE_SIZE=64x64
BATCH=16 LEN=64 IMAG=16 TRAIN_RATIO=64 NGPU=$NGPU
CONV=native ENV=routeA_clone
MASKED_GOALS=False VARIABLE_GOAL_LENGTH=False MANAGER_SAMPLE_FREQ=8
NOTE=pure Director baseline (defaults only, multi-V100 data-parallel)
EOF

# All flags below == config defaults; explicit for reproducibility.
AGENT_FLAGS=(
  --agent.use_hrl True
  --agent.use_masked_goals False
  --agent.variable_goal_length False
  --agent.manager_sample_freq 8
  --agent.goal_autoencoder_beta 0.25
  --agent.goal_rec_loss_agg sum
  --agent.goal_opt.lr 4e-5
  --agent.goal_opt.wd 0.0
  --agent.goal_opt.eps 1e-20
  --agent.manager_actent_impl mult
  --agent.manager_actent_target 0.5
  --agent.mgr_reward_agg mean
  --agent.imag_length 16
  --agent.report False
)

# wandb for the full (>=1M) run; short fit-test stays jsonl/scope.
LOGOUT="jsonl,scope"; [ "$STEPS" -ge 1000000 ] && LOGOUT="jsonl,scope,wandb"

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs director_match \
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
  --run.steps "$STEPS" \
  --run.train_ratio 64 \
  --logger.outputs "$LOGOUT" \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  --jax.train_devices $DEVLIST \
  --jax.policy_devices 0 \
  "${AGENT_FLAGS[@]}"
rc=$?
echo "[$EXP_TAG] EXIT=$rc TASK=$TASK NGPU=$NGPU STEPS=$STEPS  (rc!=0 + RESOURCE_EXHAUSTED => OOM at $NGPU GPUs)"
exit $rc
