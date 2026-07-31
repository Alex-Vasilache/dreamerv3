#!/bin/bash -l
#SBATCH -J diag_structcorr
#SBATCH -p gpu-v100
#SBATCH -t 0-01:00:00
#SBATCH -c 10
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH --array=0-2
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%A_%a.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%A_%a.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

# Hardcoded, not derived from ${BASH_SOURCE[0]}: sbatch copies the submitted
# script into /var/spool/slurmd/... before running it, so self-discovery via
# BASH_SOURCE resolves to that ephemeral spool path, not this file's location.
SCRIPT_DIR=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3/experiments/goal_code_struct_corr
# Default output straight into the experiment folder's results/ dir (already
# committed there for the baselines below); override OUTDIR for a fresh run.
OUTDIR="${OUTDIR:-$SCRIPT_DIR/results}"
mkdir -p "$OUTDIR"

BUCKET=/bucket/DoyaU/vasilache/bucket/results/dreamerv3
NAMES=(e124_hopper e123_cheetah e180_acrobot)
declare -A RUNS=(
  [e124_hopper]="$BUCKET/e124_dmc_hopper_hop_director_baseline_multiv100_20260706_094230_C3AT29"
  [e123_cheetah]="$BUCKET/e123_dmc_cheetah_run_director_baseline_20260706_155117_Vk7oa4"
  [e180_acrobot]="$BUCKET/e180_dmc_acrobot_swingup_director_baseline_j4660140"
)

# One array task per env, run concurrently across separate GPU allocations;
# each task additionally drives n_envs=8 parallel workers of its own env for
# the rollout, so the two levels of parallelism multiply the wall-clock
# speedup vs. one sequential single-env loop.
name="${NAMES[$SLURM_ARRAY_TASK_ID]}"
echo "=================== $name (array task $SLURM_ARRAY_TASK_ID) ==================="
python -u "$SCRIPT_DIR/diag_goal_struct_corr.py" \
  --run_dir "${RUNS[$name]}" \
  --n_batches 10 --n_envs 8 \
  --out "$OUTDIR/${name}.npz"
echo "=================== $name DONE ==================="
