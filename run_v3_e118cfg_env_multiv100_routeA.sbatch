#!/bin/bash -l
#SBATCH -J e118cfg_mv100
#SBATCH -p gpu-v100
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=192G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Generic e118-config run on 4x V100 (data-parallel) via the Route-A-fixed clone.
# Same config as e118 (director_match 64x64, batch16/len64/imag16, train_ratio 64,
# native conv) for an arbitrary DMC TASK. Fits ~11-13GB/device on 16GB V100s.
# Pass TASK, EXP_TAG, RUN_DIR (fixed for chained resumes), STEPS.
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
unset DREAMERV3_CONV_IMPL   # native conv (Route-A clone)
export MUJOCO_GL=egl
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false"
# WANDB creds: this script activates the Route-A clone venv directly instead of
# sourcing source_dreamerv3_env.sh, so it must set these itself or wandb init fails
# with "No API key configured" (silently drops to jsonl-only). Keep in sync with
# source_dreamerv3_env.sh.
export WANDB_API_KEY=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
export WANDB_PROJECT=dreamerv3
export WANDB_ENTITY=vasilache

TASK="${TASK:?set TASK, e.g. dmc_acrobot_swingup}"
EXP_TAG="${EXP_TAG:?set EXP_TAG, e.g. e119}"
NGPU="${NGPU:-4}"
STEPS="${STEPS:-4000000}"
DEVLIST=$(seq -s ' ' 0 $((NGPU-1)))
ENVNAME="${TASK#dmc_}"
RUN_SUFFIX="${EXP_TAG}_${ENVNAME}_mv100_b16"
RUN_DIR="${RUN_DIR:-$(mktemp -d /work/DoyaU/vasilache/work/${RUN_SUFFIX}_$(date +%Y%m%d_%H%M%S)_XXXXXX)}"
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"
echo "[e118cfg] TASK=$TASK EXP_TAG=$EXP_TAG NODE=$(hostname) NGPU=$NGPU DEVLIST='$DEVLIST' STEPS=$STEPS RUN_DIR=$RUN_DIR"

AGENT_FLAGS=(
  --agent.goal_autoencoder_beta 0.25
  --agent.goal_rec_loss_agg sum
  --agent.goal_opt.lr 4e-5
  --agent.goal_opt.wd 0.0
  --agent.goal_opt.eps 1e-20
  --agent.manager_actent_impl mult
  --agent.manager_actent_target 0.5
  --agent.mask_topk 0
  --agent.goal_struct_weight 200.0
  --agent.mask_sparsity_mode prob
  --agent.mask_sparsity_max 5.0
  --agent.mask_sparsity_target 0.3
  --agent.mgr_reward_agg sum
  --agent.goal_switch_cost 0.02
  --agent.goal_edit_cost 0.0
  --agent.goal_edit_cost_ach 0.0
  --agent.mask_sparsemax False
  --agent.mask_sparsity_prob_ach_weight 0.0
  --agent.mask_perblock_credit True
  --agent.perblock_edit_cost 0.3
  --agent.mgr_cond_goalcode True
  --agent.mgr_cond_achieve False
  --agent.variable_goal_length True
  --agent.goal_duration_min 1
  --agent.goal_duration_max 16
  --agent.goal_duration_reg 0.0
  --agent.goal_duration_target 4.0
  --agent.goal_duration_adapt False
  --agent.imag_length 16
  --agent.report False
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs director_match masked_goals variable_goals \
  --task "$TASK" \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 64 64 \
  --batch_size 16 \
  --batch_length 64 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.mgr_expl_weight 0.1 \
  --agent.mgr_retnorm.impl meanstd \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 8 \
  --run.steps "$STEPS" \
  --run.train_ratio 64 \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  --jax.train_devices $DEVLIST \
  --jax.policy_devices 0 \
  "${AGENT_FLAGS[@]}"

exit $?
