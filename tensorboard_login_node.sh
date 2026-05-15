#!/usr/bin/env bash
# TensorBoard on the Saion *login* node only (not inside sbatch/salloc/srun steps).
# Re-execs under a login shell (``bash -l``, same idea as ``#!/bin/bash -l`` in sbatch)
# so ``module load python/3.11.4`` from ``source_dreamerv3_env.sh`` matches training jobs.
#
#   cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3
#   LOGDIR=/work/DoyaU/vasilache/work/<run_id>/ PORT=6010 ./tensorboard_login_node.sh
#   LOGDIR=... PORT=6010 BACKGROUND=1 ./tensorboard_login_node.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "tensorboard_login_node.sh: run on the login node outside Slurm (SLURM_JOB_ID=${SLURM_JOB_ID})." >&2
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
: "${LOGDIR:?Set LOGDIR to your run directory (parent of logdir/ or logdir itself; see Slurm .out)}"
export LOGDIR
export PORT="${PORT:-6006}"
export BACKGROUND="${BACKGROUND:-0}"

exec bash -l -c "cd \"$HERE\" && exec ./tensorboard_login.sh"
