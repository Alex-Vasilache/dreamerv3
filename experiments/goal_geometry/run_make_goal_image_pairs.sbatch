#!/bin/bash -l
#SBATCH -J goal_image_pairs
#SBATCH -p gpu-v100
#SBATCH -t 00:45:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Renders sample goal-code -> goal-image pairs at controlled similarity bins
# (make_goal_image_pairs.py). Needs a GPU node for the live rollout
# (MUJOCO_GL=egl) and the JAX forward passes.
#
#   RUN_DIR=/work/.../e506_..._j4676887 OUT=results_img_pairs/e506_director \
#     sbatch experiments/goal_geometry/run_make_goal_image_pairs.sbatch
set -uo pipefail

REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3_somvae/.claude/worktrees/goal-image-pairs
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
# Reference conv: this is a small forward-pass measurement, not worth the
# Route-A native-conv dependency, and V100 native conv needs that clone (see
# run_diag_goal_geometry.sbatch's CONV comment).
export DREAMERV3_CONV_IMPL=reference
export MUJOCO_GL=egl

RUN_DIR="${RUN_DIR:?set RUN_DIR to a run directory}"
OUT="${OUT:?set OUT to an output prefix (writes <OUT>_hard.png / _soft.png / _README.txt)}"
CKPT_SUFFIX="${CKPT_SUFFIX:-}"
N_STATES="${N_STATES:-512}"
N_ENVS="${N_ENVS:-8}"
N_PER_BIN="${N_PER_BIN:-4}"
SEED="${SEED:-0}"

mkdir -p "$(dirname "$OUT")"
args=(--run_dir "$RUN_DIR" --out "$OUT" --n_states "$N_STATES" \
      --n_envs "$N_ENVS" --n_per_bin "$N_PER_BIN" --seed "$SEED")
if [ -n "$CKPT_SUFFIX" ]; then
  if [ -d "$RUN_DIR/$CKPT_SUFFIX" ]; then
    args+=(--ckpt_path "$RUN_DIR/$CKPT_SUFFIX")
  else
    echo "SKIP: no $CKPT_SUFFIX under $RUN_DIR"; exit 1
  fi
fi

python -u "$REPO/experiments/goal_geometry/make_goal_image_pairs.py" "${args[@]}"
exit $?
