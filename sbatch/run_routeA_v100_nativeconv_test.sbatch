#!/bin/bash -l
#SBATCH -J routeA_v100
#SBATCH -p gpu-v100
#SBATCH -t 0-01:00:00
#SBATCH -c 16
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Route A: in a CLONED env, restore the stack to requirements.txt intent
# (jax[cuda12]==0.4.33, all four jax components aligned) and pin cuDNN DOWN to
# 9.1.0.70 (the plugin's tested floor) instead of the drifted 9.22, then test
# whether native cuDNN conv works on V100 (sm_70) at director_match 64x64.
# The working /apps env is untouched.
set -uo pipefail

CLONE=/work/DoyaU/vasilache/work/dreamerv3_env_routeA

# --- module + LD setup (mirror source_dreamerv3_env.sh, but activate the CLONE) ---
module purge
_PY311_LIB=/apps/free88/python/3.11.4/lib
if ! module is-avail python/3.11.4 2>/dev/null; then
  module use /apps/.modulefiles88 2>/dev/null || true
  module use /apps/.modulefiles95 2>/dev/null || true
fi
module load python/3.11.4 2>/dev/null || export LD_LIBRARY_PATH="${_PY311_LIB}:${LD_LIBRARY_PATH:-}"

# Fix the venv path baked into the cloned activate script, then activate the clone.
sed -i "s#/apps/unit/DoyaU/vasilache/apps/dreamerv3_env#${CLONE}#g" "$CLONE/bin/activate"
source "$CLONE/bin/activate"
python -c 'import sys' >/dev/null 2>&1 || export LD_LIBRARY_PATH="${_PY311_LIB}:${LD_LIBRARY_PATH:-}"

echo "[routeA] which python: $(which python)"
echo "[routeA] prefix: $(python -c 'import sys;print(sys.prefix)')"

# --- Route A install: align jax to 0.4.33 + pin cuDNN down to 9.1.0.70 ---
echo "[routeA] BEFORE:"; python -m pip freeze 2>/dev/null | grep -iE "^jax|jaxlib|jax-cuda12|nvidia-cudnn|nvidia-cuda-nvcc|nvidia-cuda-runtime" || true
python -m pip install --no-input \
  "jax==0.4.33" "jaxlib==0.4.33" "jax-cuda12-plugin==0.4.33" "jax-cuda12-pjrt==0.4.33" \
  "nvidia-cudnn-cu12==9.1.0.70"
PIP_RC=$?
echo "[routeA] pip rc=$PIP_RC"
echo "[routeA] AFTER:"; python -m pip freeze 2>/dev/null | grep -iE "^jax|jaxlib|jax-cuda12|nvidia-cudnn|nvidia-cuda-nvcc|nvidia-cuda-runtime" || true
[ $PIP_RC -ne 0 ] && { echo "[routeA] PIP INSTALL FAILED — aborting"; exit $PIP_RC; }

# --- native-conv V100 smoke at director_match 64x64 (the config that 5003'd before) ---
unset DREAMERV3_CONV_IMPL   # NATIVE conv — the whole point of the test
export MUJOCO_GL=egl
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false"
RUN_DIR=$(mktemp -d /work/DoyaU/vasilache/work/routeA_smoke_XXXXXX); cd "$RUN_DIR"
echo "[routeA] RUN_DIR=$RUN_DIR CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"

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
echo "[routeA] SMOKE EXIT=$rc  (0 + fps rows => NATIVE CONV FIXED ON V100 via Route A)"
exit $rc
