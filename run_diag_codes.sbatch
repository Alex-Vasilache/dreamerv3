#!/bin/bash -l
#SBATCH -J diag_codes
#SBATCH -p gpu-v100
#SBATCH -t 0-00:30:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

RUN_DIR="${RUN_DIR:-/work/DoyaU/vasilache/work/v3_cartpole_masked_goals_6m_20260608_110826_RMqdiq}"
BATCHES="${BATCHES:-8}"
OUT="${OUT:-/work/DoyaU/vasilache/work/slurm_logs/diag_codes_${SLURM_JOB_ID}.npz}"

cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3

python -u diag_codes.py \
  --run_dir "$RUN_DIR" \
  --batches "$BATCHES" \
  --out "$OUT"

exit $?
