#!/bin/bash -l
# Numeric-parity harness for the 2026-07-28 agent.py modularization.
#
# Runs the surviving production code paths on CPU jax at a fixed seed and
# dumps each leg's metrics.jsonl. Run it once against the pre-refactor tree and
# once against the refactored tree, then diff the loss columns: the refactor is
# a pure restructuring, so every retained path must reproduce bit-for-bit.
#
# CPU-only on purpose -- `intel` has no GrpTRES cap for us, so this never
# competes with the live A100 batch for GPU quota.
#
# Usage:
#   REPO=<tree> OUT=<dir> [BLOCK_REW_FLAG=1] sbatch sbatch/run_parity_smoke.sbatch
#     REPO           repo root to test (default: this script's repo)
#     OUT            where the per-leg logdirs go (required)
#     BLOCK_REW_FLAG 1 = pass --agent.variable_goal_block_rew True explicitly
#                    (needed for the PRE-refactor tree, where block pooling is
#                    still opt-in). Omit for the refactored tree, where it is
#                    unconditional and the flag no longer exists.
#SBATCH -J parity_smoke
#SBATCH -p intel
#SBATCH -t 0-02:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
# Determinism: single-threaded XLA CPU, so reductions keep a fixed order across
# runs. Without this the thread-count-dependent reduction tree makes the loss
# columns differ in the last few ulps between otherwise identical runs.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:?set OUT to a parity output dir}"
MAIN="$REPO/dreamerv3/main.py"
mkdir -p "$OUT"
echo "REPO=$REPO"
echo "OUT=$OUT"

BLOCK_REW=()
if [ "${BLOCK_REW_FLAG:-0}" = "1" ]; then
  BLOCK_REW=( --agent.variable_goal_block_rew True )
fi

COMMON=(
  --task dmc_cartpole_swingup
  --env.dmc.image True --env.dmc.proprio True --env.dmc.size 32 32
  --agent.enc.simple.outer True --agent.dec.simple.outer True
  --seed 0
  --run.steps 800
  --agent.imag_length 16
  --logger.outputs jsonl
  --jax.platform cpu
)

# LEG A -- fixed-K Director. The retained non-variable-K baseline (RECIPE=director).
echo "=================== LEG A: fixed-K Director ==================="
python -u "$MAIN" --logdir "$OUT/fixedk" --configs size6m debug "${COMMON[@]}" \
  --agent.use_hrl True \
  --agent.variable_goal_length False \
  --agent.manager_sample_freq 8
echo "LEG A OK"

# LEG B -- var-K, block-pooled, fixed duration prior. The e360-e362/e365
# "pinned" production cell (DUR_MODE=fixed, DUR_FIXED=8).
echo "=================== LEG B: var-K block-pooled, pinned duration ==================="
python -u "$MAIN" --logdir "$OUT/vark_pinned" --configs size6m debug "${COMMON[@]}" \
  --agent.use_hrl True \
  --agent.variable_goal_length True "${BLOCK_REW[@]}" \
  --agent.goal_duration_min 1 --agent.goal_duration_max 16 \
  --agent.goal_duration_target 8.0 --agent.goal_duration_fixed 8 \
  --agent.goal_duration_reg 0.01 --agent.goal_duration_lagrange False \
  --agent.mgr_reward_agg mean
echo "LEG B OK"

# LEG C -- var-K, block-pooled, Lagrangian duration prior, sum aggregation,
# relabel OFF. The e363/e364 production research cell, and the path that
# exercises horizon-truncated holds (goal_duration_max == imag_length).
echo "=================== LEG C: var-K block-pooled, Lagrangian + sum ==================="
python -u "$MAIN" --logdir "$OUT/vark_lagrange" --configs size6m debug "${COMMON[@]}" \
  --agent.use_hrl True \
  --agent.variable_goal_length True "${BLOCK_REW[@]}" \
  --agent.goal_duration_min 1 --agent.goal_duration_max 16 \
  --agent.goal_duration_target 8.0 \
  --agent.goal_duration_reg 0.0 --agent.goal_duration_lagrange True \
  --agent.goal_duration_lagrange_max 5.0 \
  --agent.mgr_reward_agg sum \
  --agent.goal_duration_relabel_truncated False
echo "LEG C OK"

# LEG D -- var-K block-pooled with the retained implicit-masking mechanism
# (goal_soft_reuse_adapt + goal_struct_adapt) explicitly ratcheted, so the
# reuse/struct controllers are exercised rather than sitting at their defaults.
echo "=================== LEG D: soft-reuse + struct-adapt ==================="
python -u "$MAIN" --logdir "$OUT/softreuse" --configs size6m debug "${COMMON[@]}" \
  --agent.use_hrl True \
  --agent.variable_goal_length True "${BLOCK_REW[@]}" \
  --agent.goal_duration_min 1 --agent.goal_duration_max 16 \
  --agent.goal_duration_target 8.0 \
  --agent.goal_soft_reuse_adapt True \
  --agent.goal_soft_reuse_target 0.5 \
  --agent.goal_soft_reuse_target_init 0.0 \
  --agent.goal_soft_reuse_target_vel 1.0e-3 \
  --agent.goal_struct_adapt True --agent.goal_struct_adapt_target 0.01
echo "LEG D OK"

echo "ALL LEGS OK"
exit 0
