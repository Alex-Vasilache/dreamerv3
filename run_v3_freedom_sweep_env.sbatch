#!/bin/bash -l
#SBATCH -p gpu-v100
#SBATCH -t 2-00:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Generalized manager-freedom sweep: same knobs as run_v3_cartpole_freedom_sweep.sbatch
# but the DMC task is parameterized so the e88-derived recipe can be ported across
# environments. Set TASK (full dmc_<domain>_<task> string) and ENV_NAME (short tag
# used in the run-dir / job suffix). Knobs default to the e57 champion recipe.
#   TASK       dmc_cartpole_swingup   full task string passed to --task
#   ENV_NAME   cartpole               short tag for RUN_SUFFIX
#   MGR_REWARD_AGG mean|sum   sum = K-neutral SMDP option return (length freedom)
#   DUR_REG    0.0|0.01       soft duration prior weight (0 = prior off)
#   DUR_TARGET 4.0            prior target K (only used when DUR_REG>0)
#   MASK_MODE  prob|none      none = no sparsity target (emergent sparsity)
#   SWITCH_COST 0.0           fixed per-decision cost (length lever)
#   PERBLOCK_CREDIT False      Tier 2 per-block Q_g credit
#   PERBLOCK_EDIT_COST 0.0     Tier 2 per-block edit cost (the sparsity dial)
# Submit e.g.:
#   sbatch -J e94_acrobot --export=ALL,EXP_TAG=e94,ENV_NAME=acrobot,\
#     TASK=dmc_acrobot_swingup,DUR_REG=0.01,MASK_MODE=none,MGR_REWARD_AGG=sum,\
#     DUR_MAX=32,SWITCH_COST=0.02,PERBLOCK_CREDIT=True,PERBLOCK_EDIT_COST=0.3 \
#     run_v3_freedom_sweep_env.sbatch
set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

EXP_TAG="${EXP_TAG:?set EXP_TAG=e94...}"
TASK="${TASK:-dmc_cartpole_swingup}"
ENV_NAME="${ENV_NAME:-cartpole}"
MGR_REWARD_AGG="${MGR_REWARD_AGG:-mean}"
DUR_REG="${DUR_REG:-0.0}"
DUR_TARGET="${DUR_TARGET:-4.0}"
DUR_MIN="${DUR_MIN:-1}"
DUR_MAX="${DUR_MAX:-16}"
MASK_MODE="${MASK_MODE:-prob}"
MASK_TARGET="${MASK_TARGET:-0.3}"
MASK_MAX="${MASK_MAX:-5.0}"
MASK_TOPK="${MASK_TOPK:-0}"
SWITCH_COST="${SWITCH_COST:-0.0}"
EDIT_COST="${EDIT_COST:-0.0}"
EDIT_COST_ACH="${EDIT_COST_ACH:-0.0}"
MASK_SPARSEMAX="${MASK_SPARSEMAX:-False}"
PROB_ACH_WEIGHT="${PROB_ACH_WEIGHT:-0.0}"
PERBLOCK_CREDIT="${PERBLOCK_CREDIT:-False}"
PERBLOCK_EDIT_COST="${PERBLOCK_EDIT_COST:-0.0}"
STRUCT_W="${STRUCT_W:-200.0}"
ACTENT_TARGET="${ACTENT_TARGET:-0.5}"
RUN_STEPS="${RUN_STEPS:-4000000}"

RUN_SUFFIX="${EXP_TAG}_${ENV_NAME}_freedom"
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir "$RUN_SUFFIX")}"
mkdir -p "$RUN_DIR"

export MUJOCO_GL=egl

echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"
echo "EXP_TAG=$EXP_TAG TASK=$TASK ENV_NAME=$ENV_NAME MGR_REWARD_AGG=$MGR_REWARD_AGG DUR_REG=$DUR_REG DUR_TARGET=$DUR_TARGET DUR_MAX=$DUR_MAX MASK_MODE=$MASK_MODE SWITCH_COST=$SWITCH_COST PERBLOCK_CREDIT=$PERBLOCK_CREDIT PERBLOCK_EDIT_COST=$PERBLOCK_EDIT_COST" >> "$RUN_DIR/job.env"

cd "$RUN_DIR"
AGENT_FLAGS=(
  --agent.goal_autoencoder_beta 0.25
  --agent.goal_rec_loss_agg sum
  --agent.goal_opt.lr 4e-5
  --agent.goal_opt.wd 0.0
  --agent.goal_opt.eps 1e-20
  --agent.manager_actent_impl mult
  --agent.manager_actent_target "$ACTENT_TARGET"
  --agent.mask_topk "$MASK_TOPK"
  --agent.goal_struct_weight "$STRUCT_W"
  --agent.mask_sparsity_mode "$MASK_MODE"
  --agent.mask_sparsity_max "$MASK_MAX"
  --agent.mask_sparsity_target "$MASK_TARGET"
  --agent.mgr_reward_agg "$MGR_REWARD_AGG"
  --agent.goal_switch_cost "$SWITCH_COST"
  --agent.goal_edit_cost "$EDIT_COST"
  --agent.goal_edit_cost_ach "$EDIT_COST_ACH"
  --agent.mask_sparsemax "$MASK_SPARSEMAX"
  --agent.mask_sparsity_prob_ach_weight "$PROB_ACH_WEIGHT"
  --agent.mask_perblock_credit "$PERBLOCK_CREDIT"
  --agent.perblock_edit_cost "$PERBLOCK_EDIT_COST"
  --agent.mgr_cond_goalcode True
  --agent.mgr_cond_achieve False
  --agent.variable_goal_length True
  --agent.goal_duration_min "$DUR_MIN"
  --agent.goal_duration_max "$DUR_MAX"
  --agent.goal_duration_reg "$DUR_REG"
  --agent.goal_duration_target "$DUR_TARGET"
  --agent.goal_duration_adapt False
  --agent.report False
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size6m masked_goals variable_goals \
  --task "$TASK" \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 32 32 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.mgr_expl_weight 0.1 \
  --agent.mgr_retnorm.impl meanstd \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 16 \
  --run.steps "$RUN_STEPS" \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}"

exit $?
