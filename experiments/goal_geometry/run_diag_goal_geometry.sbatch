#!/bin/bash -l
#SBATCH -J diag_goal_geometry
#SBATCH -p short-a100
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Goal-code geometry for one or more runs. Needs a GPU node for the live
# rollout (MUJOCO_GL=egl) even though the measurement itself is small.
#
#   RUNS="/work/.../e510_... /work/.../e502_..." OUTDIR=results \
#     sbatch experiments/goal_geometry/run_diag_goal_geometry.sbatch
#
# CKPT_SUFFIX picks a milestone instead of the live checkpoint, which is what
# to use while a run is still training -- logdir/ckpt is being rewritten under
# you, the milestones are not.
#   CKPT_SUFFIX=logdir/ckpt_milestones/000002000000
set -uo pipefail

REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3_somvae
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
# Native cuDNN conv is ~2x faster but is broken on V100 sm_70 in this env (it
# needs the Route-A clone). This measurement is a forward pass over a few
# thousand states, so the conv path costs nothing either way -- keep the
# reference impl when running on a V100, which is where there is free capacity
# while the A100 lanes are saturated by the batch.
#   CONV=native  on A100        CONV=reference  on V100/P100
CONV="${CONV:-native}"
[ "$CONV" = "native" ] && unset DREAMERV3_CONV_IMPL || export DREAMERV3_CONV_IMPL=reference
export MUJOCO_GL=egl

RUNS="${RUNS:?set RUNS to one or more run dirs}"
OUTDIR="${OUTDIR:-$REPO/experiments/goal_geometry/results}"
CKPT_SUFFIX="${CKPT_SUFFIX:-}"
N_BATCHES="${N_BATCHES:-4}"
N_ENVS="${N_ENVS:-8}"
mkdir -p "$OUTDIR"

fail=0
for run in $RUNS; do
  name=$(basename "${run%/}")
  args=(--run_dir "$run" --out "$OUTDIR/${name}.npz"
        --n_batches "$N_BATCHES" --n_envs "$N_ENVS")
  if [ -n "$CKPT_SUFFIX" ]; then
    if [ -d "$run/$CKPT_SUFFIX" ]; then
      args+=(--ckpt_path "$run/$CKPT_SUFFIX")
    else
      echo "SKIP $name: no $CKPT_SUFFIX"; continue
    fi
  fi
  echo "=============== $name ==============="
  python -u "$REPO/experiments/goal_geometry/diag_goal_geometry.py" "${args[@]}" \
    || { echo "FAILED $name"; fail=1; }
done
echo "=============== GEOMETRY: $([ $fail -eq 0 ] && echo PASS || echo FAIL) ==============="
exit $fail
