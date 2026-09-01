#!/bin/bash -l
#SBATCH -J novelty_viz_cpu
#SBATCH -p intel
#SBATCH -t 03:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Behaviour panels for the count-novelty runs, rendered on CPU.
#
# No GPU is needed for this. It is a single-env rollout of a few hundred steps,
# not training -- the only GPU-shaped work is the per-step policy forward pass,
# which JAX runs on CPU fine. Meanwhile both the v100 and p100 GrpTRES quotas
# are fully consumed by e644-e647 and every A100 in the cluster is allocated, so
# a GPU render sits in the queue behind other people's jobs indefinitely.
#
# Rendering backend by task:
#   pinpad_*  pure numpy gridworld (embodied/envs/pinpad.py), no MuJoCo at all
#   dmc_*     MuJoCo, so MUJOCO_GL=osmesa for software rendering instead of egl
#
#   RUN=<run_dir> CKPT=<milestone> TAG=<name> sbatch run_novelty_viz_cpu.sbatch
set -uo pipefail
REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export DREAMERV3_CONV_IMPL=reference     # no cuDNN on CPU
export MUJOCO_GL=osmesa                  # software GL; egl needs a GPU
export JAX_PLATFORMS=cpu
export XLA_FLAGS="${XLA_FLAGS:-} --xla_force_host_platform_device_count=1"

OUTDIR="${OUTDIR:-$REPO/experiments/mask_viz/novelty}"
mkdir -p "$OUTDIR"

RUN="${RUN:?set RUN=<run_dir>}"
TAG="${TAG:?set TAG=<name>}"
echo "=============== $TAG (CPU) ==============="
echo "run  = $RUN"
echo "ckpt = ${CKPT:-<live ckpt/latest>}"

python -u "$REPO/experiments/mask_viz/render_mask_viz.py" \
  --run_dir "$RUN" \
  ${CKPT:+--ckpt_path "$CKPT"} \
  --platform cpu \
  --out "$OUTDIR/$TAG" \
  --steps "${STEPS:-520}"
rc=$?
echo "=============== $TAG EXIT=$rc ==============="
exit $rc
