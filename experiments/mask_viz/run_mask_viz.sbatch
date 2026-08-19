#!/bin/bash -l
#SBATCH -J mask_viz
#SBATCH -p gpu-v100
#SBATCH -t 01:30:00
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
set -uo pipefail
REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3_somvae
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export DREAMERV3_CONV_IMPL=reference   # V100
export MUJOCO_GL=egl
OUTDIR="${OUTDIR:-$REPO/experiments/mask_viz/out}"
mkdir -p "$OUTDIR"
for run in $RUNS; do
  name=$(basename "${run%/}"); tag="${name%%_*}"
  echo "=============== $name ==============="
  python -u "$REPO/experiments/mask_viz/render_mask_viz.py" \
    --run_dir "$run" --out "$OUTDIR/$tag" --steps "${STEPS:-520}" \
    || echo "FAILED $name"
done
