#!/bin/bash -l
#SBATCH -p gpu-v100
#SBATCH -t 2-00:00:00
# 8 CPU/job: 8 sweep jobs x 8 = 64 <= v100 assoc cap (cpu=72). Cartpole is light
# (e57 ran on 4); 8 gives headroom for 16 envs without starving the assoc.
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

# Manager-freedom sweep (e66+): replace the forced sparsity target (0.3) and soft
# duration prior (->8) with *priced activity* so the manager freely chooses edit
# fraction and hold-length. Knobs (all default to the e57 champion recipe):
#   MGR_REWARD_AGG  mean|sum   sum = K-neutral SMDP option return (length freedom)
#   DUR_REG         0.0|0.01   soft duration prior weight (0 = prior off)
#   DUR_TARGET      4.0        prior target K (only used when DUR_REG>0)
#   MASK_MODE       prob|none  none = no sparsity target (sparsity freedom)
#   SWITCH_COST     0.0        fixed per-decision cost (length lever on mean agg)
#   EDIT_COST       0.0        fixed per-edited-block cost (sparsity lever)
#   EDIT_COST_ACH     0.0        B.3 achievability-gated edit cost (reward path)
#   MASK_SPARSEMAX    False      C.5 sparsemax edit gate
#   PROB_ACH_WEIGHT   0.0        Tier 1 analytic prob_ach weight (use MASK_MODE=prob_ach)
#   PERBLOCK_CREDIT   False      Tier 2 per-block Q_g credit
#   PERBLOCK_EDIT_COST 0.0       Tier 2 tie-break edit cost
# Submit with EXP_TAG and the knobs you want, e.g.:
#   sbatch -J e66_cartpole --export=ALL,EXP_TAG=e66,MGR_REWARD_AGG=sum,DUR_REG=0.0 \
#     sbatch/run_v3_cartpole_freedom_sweep.sbatch
EXP_TAG="${EXP_TAG:?set EXP_TAG=e66...}"
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
EDIT_COST_ACH="${EDIT_COST_ACH:-0.0}"   # B.3 achievability-gated edit cost
MASK_SPARSEMAX="${MASK_SPARSEMAX:-False}"  # C.5 sparsemax edit gate
PROB_ACH_WEIGHT="${PROB_ACH_WEIGHT:-0.0}"  # Tier 1: set >0 with MASK_MODE=prob_ach
PERBLOCK_CREDIT="${PERBLOCK_CREDIT:-False}"  # Tier 2 per-block Q_g credit
PERBLOCK_EDIT_COST="${PERBLOCK_EDIT_COST:-0.0}"  # Tier 2 tie-break edit cost
STRUCT_W="${STRUCT_W:-200.0}"
ACTENT_TARGET="${ACTENT_TARGET:-0.5}"
RUN_STEPS="${RUN_STEPS:-4000000}"

RUN_SUFFIX="${EXP_TAG}_cartpole_freedom"
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir "$RUN_SUFFIX")}"
mkdir -p "$RUN_DIR"

export MUJOCO_GL=egl

echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"
echo "EXP_TAG=$EXP_TAG MGR_REWARD_AGG=$MGR_REWARD_AGG DUR_REG=$DUR_REG DUR_TARGET=$DUR_TARGET DUR_MAX=$DUR_MAX MASK_MODE=$MASK_MODE SWITCH_COST=$SWITCH_COST EDIT_COST=$EDIT_COST EDIT_COST_ACH=$EDIT_COST_ACH MASK_SPARSEMAX=$MASK_SPARSEMAX PROB_ACH_WEIGHT=$PROB_ACH_WEIGHT PERBLOCK_CREDIT=$PERBLOCK_CREDIT PERBLOCK_EDIT_COST=$PERBLOCK_EDIT_COST" >> "$RUN_DIR/job.env"

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
  --task dmc_cartpole_swingup \
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
