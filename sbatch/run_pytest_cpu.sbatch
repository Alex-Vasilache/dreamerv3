#!/bin/bash -l
# Run the JAX test suites on a CPU-only node.
#
# Why not a GPU partition: our account's GPU quota is shared, so while a full
# 8-job A100 batch is in flight a `-p gpu-v100`/`gpu-a100` test job pends on
# AssocGrpGRES/AssocGrpCpuLimit indefinitely. The `intel` partition has no
# GrpTRES limit for us and its nodes are el8, so `dreamerv3_env` (Python 3.11)
# imports fine there -- unlike the login node (glibc too old).
#
# Usage:
#   sbatch sbatch/run_pytest_cpu.sbatch                      # default targets
#   TESTS="embodied/tests/test_block_pooled_equivalence.py" \
#     sbatch sbatch/run_pytest_cpu.sbatch
# Or interactively (fast, no queue wait when intel is free):
#   srun -p intel -c 8 --mem=24G -t 0:20:00 bash -lc '...'
#SBATCH -J pytest_cpu
#SBATCH -p intel
#SBATCH -t 0-00:30:00
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export JAX_PLATFORMS=cpu
# Tests that construct a DMC env need a GL backend even with image=False,
# because dm_control builds a render context up front. `egl` needs a GPU, so
# submit those to -p gpu-v100; on `intel` they skip themselves.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# REPO defaults to the main tree, so a copy of this script sitting in a
# worktree still tests the main tree unless asked otherwise. Override it to
# test the tree the script lives in:
#   REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3_somvae sbatch ...
REPO="${REPO:-/apps/unit/DoyaU/vasilache/apps/code/dreamerv3}"
cd "$REPO"
# Lustre's coarse mtime resolution can leave pytest's assertion-rewritten
# __pycache__ entries stale, so a run silently executes the PREVIOUS version of
# a test file (symptom: pytest prints `>   ???` for the failing line and a line
# number that does not match the source). Clear it every run.
find "$REPO" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$REPO/.pytest_cache" 2>/dev/null || true

TESTS="${TESTS:-embodied/tests/test_block_pooled_equivalence.py embodied/tests/test_variable_goals.py}"
python -m pytest $TESTS -v -rA
exit $?
