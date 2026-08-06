#!/bin/bash -l
# Run pytest under `dreamerv3_env` on a compute node. The login node cannot run
# it (glibc too old for Python 3.11), so every test run needs an allocation.
#
# Defaults to a CPU-only slice of `gpu-v100` because the `intel` partition has
# been failing to boot nodes ("Nodes saion-intelNN are still not ready").
# Set TEST_PART=intel (and TEST_GRES=) to go back to intel once it recovers --
# intel costs no GPU quota, which matters while a full GPU batch is live.
#
# Usage:  tools/run_tests_intel.sh embodied/tests/test_goal_lipschitz.py -x -q
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PART="${TEST_PART:-gpu-v100}"
GRES="${TEST_GRES-gpu:v100:1}"
ARGS=(-p "$PART" -c "${TEST_CPUS:-8}" --mem="${TEST_MEM:-24G}" -t "${TEST_TIME:-0:30:00}")
[ -n "$GRES" ] && ARGS+=(--gres="$GRES")
# JAX_PLATFORMS=cpu keeps the tests deterministic and off the GPU even when one
# is allocated; unset it (TEST_GPU=1) to exercise the real accelerator path.
PLATFORMS="cpu"; [ "${TEST_GPU:-0}" = "1" ] && PLATFORMS="cuda"
PYTEST_ARGS="$(printf '%q ' "$@")"
srun "${ARGS[@]}" bash -lc "source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh >/dev/null 2>&1
            cd '$REPO'
            export JAX_PLATFORMS='$PLATFORMS'
            export PYTHONPATH='$REPO':\${PYTHONPATH:-}
            python -m pytest $PYTEST_ARGS"
