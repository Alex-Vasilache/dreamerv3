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

# Overridable, not hardcoded to the main checkout: sbatch copies the script
# into a spool dir before running it, so self-locating via ${BASH_SOURCE[0]}
# does NOT work here (resolves to /var/spool/..., confirmed the hard way). A
# plain `REPO=/path/to/main/checkout` silently runs the MAIN CHECKOUT's
# diag_goal_geometry.py even when this script is submitted from a worktree
# -- pass REPO=<worktree path> at submission time when testing worktree
# edits, or FORCE=1 will happily overwrite the main checkout's results/*.npz
# with output from code the worktree never intended to run.
REPO="${REPO:-/apps/unit/DoyaU/vasilache/apps/code/dreamerv3_somvae}"
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

# Measuring an untrained model says nothing -- its codebook is still at
# initialization, every state quantizes to one entry, and the geometry is that
# of random weights. A run directory can exist with a checkpoint and no
# training at all (a job that was submitted, created its dir, and was cancelled
# before its first log), so require some actual progress before spending a slot
# on it.
MIN_STEPS="${MIN_STEPS:-100000}"

fail=0
for run in $RUNS; do
  name=$(basename "${run%/}")
  steps=$(python3 - "$run" <<'PY'
import json, os, sys
p = os.path.join(sys.argv[1], 'logdir', 'metrics.jsonl')
best = 0
if os.path.exists(p):
    with open(p, 'rb') as f:
        f.seek(0, 2); f.seek(max(0, f.tell() - 200_000))
        for line in f.read().decode('utf8', 'ignore').splitlines()[1:]:
            try:
                best = max(best, int(json.loads(line).get('step', 0)))
            except Exception:
                pass
print(best)
PY
)
  if [ "${steps:-0}" -lt "$MIN_STEPS" ]; then
    echo "SKIP $name: only ${steps:-0} steps trained (< $MIN_STEPS)"
    continue
  fi
  # Idempotent: an existing result is only redone on FORCE=1. This is what lets
  # the final pass be pointed at all forty run directories and measure only
  # what is missing, instead of tracking which round measured what.
  if [ -f "$OUTDIR/${name}.npz" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "HAVE $name: $OUTDIR/${name}.npz (FORCE=1 to redo)"
    continue
  fi
  args=(--run_dir "$run" --out "$OUTDIR/${name}.npz"
        --n_batches "$N_BATCHES" --n_envs "$N_ENVS")
  # FAST=1: only the total-index sweep, one batch of states. About a fifth of
  # the wall time; the output carries index_total/* and meta only, so write it
  # to its own OUTDIR rather than over a full result.
  [ "${FAST:-0}" = "1" ] && args+=(--fast)
  # PAIRS=1: only the pairwise correlations and their pair pools (the
  # goal-code-vs-goal-space figures). One encode, no decode sweeps.
  [ "${PAIRS:-0}" = "1" ] && args+=(--pairs-only)
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
