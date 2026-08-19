#!/bin/bash
# Run pytest on the CPU-only `intel` partition.
#
# The login node cannot run dreamerv3_env at all (glibc too old -- Python 3.11
# segfaults at import), and the GPU partitions are usually saturated by real
# training. `intel` is el8, has no GPU quota to spend, and is free.
#
#   sbatch --export=ALL,TESTS="embodied/tests/test_mgr_poisson.py" \
#          sbatch/run_tests_intel.sbatch
#
#SBATCH -J dv3_tests
#SBATCH -p intel
#SBATCH -t 0-02:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x_%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x_%j.err
set -uo pipefail

REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3_somvae
TESTS="${TESTS:-embodied/tests/}"
PYTEST_ARGS="${PYTEST_ARGS:--x -q}"

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

# No GPU on this partition; keep JAX on CPU and quiet about it.
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export MUJOCO_GL=egl

cd "$REPO" || exit 1
echo "host=$(hostname)  tests=$TESTS"
python -c "import jax; print('jax', jax.__version__, jax.devices())"
python -m pytest $TESTS $PYTEST_ARGS
rc=$?
echo "pytest exit $rc"
exit $rc
