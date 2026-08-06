#!/bin/bash -l
#SBATCH -J diag_structcorr_snap
#SBATCH -p gpu-v100
#SBATCH -t 0-00:15:00
#SBATCH -c 10
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Single-checkpoint variant of run_diag_goal_struct_corr.sbatch: measures
# goal-code/goal-space correlation against one specific mid-training
# checkpoint SNAPSHOT (not the run's live ckpt/latest), submitted by
# watch_and_diag_running.py right after it copies that snapshot aside.
set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

SCRIPT_DIR=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3/experiments/goal_code_struct_corr
RUN_DIR="${RUN_DIR:?set by watch_and_diag_running.py}"
CKPT_PATH="${CKPT_PATH:?set by watch_and_diag_running.py}"
OUT_PATH="${OUT_PATH:?set by watch_and_diag_running.py}"

echo "run_dir=$RUN_DIR ckpt_path=$CKPT_PATH out_path=$OUT_PATH train_step=${TRAIN_STEP:-unknown}"
python -u "$SCRIPT_DIR/diag_goal_struct_corr.py" \
  --run_dir "$RUN_DIR" --ckpt_path "$CKPT_PATH" --seed 0 \
  --n_batches 10 --n_envs 8 \
  ${TRAIN_STEP:+--train_step "$TRAIN_STEP"} \
  --out "$OUT_PATH"
# Only reached on success (set -e above) -- the ~1GB checkpoint snapshot has
# served its purpose (measurement saved to $OUT_PATH) and would otherwise
# accumulate indefinitely across milestones/seeds.
rm -rf "$CKPT_PATH"
echo "removed snapshot $CKPT_PATH"
echo "DONE"
