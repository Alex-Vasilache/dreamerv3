#!/bin/bash -l
#SBATCH -J routeA2_v100
#SBATCH -p gpu-v100
#SBATCH -t 0-01:00:00
#SBATCH -c 16
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Route A v2: clone already has jax 0.4.33 + cuDNN 9.1.0.70, but native conv still
# EXECUTION-FAILED on V100 because the CUDA runtime drifted to 12.9 (dreamer pins
# nvcc<=12.2). Restore the whole CUDA runtime stack to 12.2 so cuDNN 9.1 + runtime
# + jaxlib 0.4.33 are consistent, then retest native conv on V100.
set -uo pipefail
CLONE=/work/DoyaU/vasilache/work/dreamerv3_env_routeA

module purge
_PY311_LIB=/apps/free88/python/3.11.4/lib
if ! module is-avail python/3.11.4 2>/dev/null; then
  module use /apps/.modulefiles88 2>/dev/null || true; module use /apps/.modulefiles95 2>/dev/null || true
fi
module load python/3.11.4 2>/dev/null || export LD_LIBRARY_PATH="${_PY311_LIB}:${LD_LIBRARY_PATH:-}"
source "$CLONE/bin/activate"
python -c 'import sys' >/dev/null 2>&1 || export LD_LIBRARY_PATH="${_PY311_LIB}:${LD_LIBRARY_PATH:-}"

echo "[routeA2] pin CUDA runtime stack -> 12.2 (keep cuDNN 9.1.0.70)"
python -m pip install --no-input \
  "nvidia-cuda-runtime-cu12>=12.2,<12.3" \
  "nvidia-cuda-nvcc-cu12>=12.2,<12.3" \
  "nvidia-cuda-cupti-cu12>=12.2,<12.3" \
  "nvidia-cublas-cu12>=12.2,<12.3" \
  "nvidia-cusparse-cu12>=12.1,<12.2" \
  "nvidia-cusolver-cu12>=11.5,<11.6" \
  "nvidia-cufft-cu12>=11.0,<11.1" \
  "nvidia-curand-cu12>=10.3,<10.4" \
  "nvidia-nvjitlink-cu12>=12.2,<12.3" \
  "nvidia-cudnn-cu12==9.1.0.70"
PIP_RC=$?
echo "[routeA2] pip rc=$PIP_RC"
echo "[routeA2] AFTER:"; python -m pip freeze 2>/dev/null | grep -iE "^jax|jaxlib|jax-cuda12|nvidia-" || true
[ $PIP_RC -ne 0 ] && { echo "[routeA2] PIP INSTALL FAILED"; exit $PIP_RC; }

unset DREAMERV3_CONV_IMPL
export MUJOCO_GL=egl
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false"
RUN_DIR=$(mktemp -d /work/DoyaU/vasilache/work/routeA2_smoke_XXXXXX); cd "$RUN_DIR"
echo "[routeA2] RUN_DIR=$RUN_DIR"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs director_match masked_goals variable_goals \
  --task dmc_cheetah_run \
  --env.dmc.image True --env.dmc.proprio True --env.dmc.size 64 64 \
  --batch_size 4 --batch_length 32 \
  --agent.enc.simple.outer True --agent.dec.simple.outer True \
  --agent.mgr_expl_weight 0.1 --agent.mgr_retnorm.impl meanstd \
  --agent.variable_goal_length True --agent.imag_length 32 \
  --agent.goal_duration_max 32 --agent.report False \
  --online_learning False --online_actor_cpu False \
  --run.envs 8 --run.steps 8000 --run.train_ratio 8 --run.log_every 30 \
  --logger.outputs jsonl,scope \
  --jax.platform cuda --jax.prealloc False
rc=$?
echo "[routeA2] SMOKE EXIT=$rc (0 + fps rows => NATIVE CONV FIXED ON V100)"
exit $rc
