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
cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3
TESTS="${TESTS:-embodied/tests/test_block_pooled_equivalence.py embodied/tests/test_variable_goals.py}"
python -m pytest $TESTS -v -rA
exit $?
