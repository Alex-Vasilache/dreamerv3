#!/bin/bash -l
#SBATCH -J novelty_viz
#SBATCH -p short-a100
#SBATCH -t 01:30:00
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Behaviour panels for the count-novelty runs (e636-e643), same three-row format
# as experiments/manager_head/policy: observation / running goal code Z_t with
# changed blocks tinted / the active goal decoded back to pixels.
#
# Two differences from run_mask_viz.sbatch:
#
#   MILESTONE, not the live checkpoint. These runs are still training, so
#   `ckpt/latest` is being rewritten underneath us; loading it races with the
#   trainer. `ckpt_milestones/<step>/` is immutable once written.
#
#   short-a100, not gpu-v100. Both the v100 and p100 GrpTRES gres/gpu=8 quotas
#   are fully consumed by e644-e647 (4 GPUs each), so a 1-GPU job there is
#   quota-blocked even though the hardware is idle. short-a100 is a separate
#   32-GPU association.
#
#   RUN=<run_dir> CKPT=<milestone dir> TAG=<name> sbatch run_novelty_viz.sbatch
set -uo pipefail
REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3_somvae
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
# A100 sm_80 handles native conv; the V100 script forces reference for a reason
# that does not apply here, and reference is ~2x slower.
unset DREAMERV3_CONV_IMPL
export MUJOCO_GL=egl

OUTDIR="${OUTDIR:-$REPO/experiments/mask_viz/novelty}"
mkdir -p "$OUTDIR"

RUN="${RUN:?set RUN=<run_dir>}"
TAG="${TAG:?set TAG=<name>}"
echo "=============== $TAG ==============="
echo "run  = $RUN"
echo "ckpt = ${CKPT:-<live ckpt/latest>}"

python -u "$REPO/experiments/mask_viz/render_mask_viz.py" \
  --run_dir "$RUN" \
  ${CKPT:+--ckpt_path "$CKPT"} \
  --out "$OUTDIR/$TAG" \
  --steps "${STEPS:-520}"
rc=$?
echo "=============== $TAG EXIT=$rc ==============="
exit $rc
