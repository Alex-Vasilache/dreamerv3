#!/bin/bash -l
#SBATCH -J diag_structcorr
#SBATCH -p gpu-v100
#SBATCH -t 0-00:15:00
#SBATCH -c 10
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%A_%a.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%A_%a.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

# Hardcoded, not derived from ${BASH_SOURCE[0]}: sbatch copies the submitted
# script into /var/spool/slurmd/... before running it, so self-discovery via
# BASH_SOURCE resolves to that ephemeral spool path, not this file's location.
SCRIPT_DIR=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3/experiments/goal_code_struct_corr
OUTDIR="${OUTDIR:-$SCRIPT_DIR/results}"
mkdir -p "$OUTDIR"

# ${MANIFEST} is written by submit_diag_goal_struct_corr.sh: one row per
# (task, seed) baseline that has actually finished training (see
# ../baselines_common.py), array-indexed 0-based by SLURM_ARRAY_TASK_ID.
# Columns: task  short_task  seed  run_dir  tag
MANIFEST="${MANIFEST:?set by submit_diag_goal_struct_corr.sh}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
IFS=$'\t' read -r TASK SHORT_TASK SEED RUN_DIR TAG <<< "$LINE"
OUT_TAG="${TAG}_${SHORT_TASK}_seed${SEED}"

echo "=================== $OUT_TAG (array task $SLURM_ARRAY_TASK_ID) ==================="
echo "task=$TASK seed=$SEED run_dir=$RUN_DIR"
# n_envs=8 parallel workers + n_batches=10 x n_states~1024 keeps each task at
# ~2-3 min wall time; the array runs all tasks concurrently across separate
# GPU allocations, so total wall time stays ~2-3 min regardless of how many
# (task, seed) pairs are in the manifest.
python -u "$SCRIPT_DIR/diag_goal_struct_corr.py" \
  --run_dir "$RUN_DIR" --seed 0 \
  --n_batches 10 --n_envs 8 \
  --out "$OUTDIR/${OUT_TAG}.npz"
echo "=================== $OUT_TAG DONE ==================="
