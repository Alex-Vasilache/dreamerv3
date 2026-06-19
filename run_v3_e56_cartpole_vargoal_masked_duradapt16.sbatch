#!/bin/bash -l
#SBATCH -J e56_cartpole
#SBATCH -p gpu-v100
#SBATCH -t 0-24:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir e56_cartpole_vargoal_masked_duradapt16)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

export MUJOCO_GL=egl

# e56 = masked goals (e25 recipe), variable duration, ADAPTIVE prior toward E[duration]=16.
# Pairs with e54 (masked, no prior) and e53 (plain, adapt@16). Uses capped AutoAdapt
# duration prior (e48 recipe) with target 16 for longer goal holds under partial edits.
MASK_TOPK="${MASK_TOPK:-0}"
STRUCT_W="${STRUCT_W:-200.0}"
MASK_MODE="${MASK_MODE:-prob}"
MASK_MAX="${MASK_MAX:-5.0}"
MASK_TARGET="${MASK_TARGET:-0.3}"
MGR_ACHIEVE="${MGR_ACHIEVE:-False}"
DUR_MIN="${DUR_MIN:-1}"
DUR_MAX="${DUR_MAX:-16}"
DUR_TARGET="${DUR_TARGET:-16.0}"
ADAPT_MAX="${ADAPT_MAX:-0.05}"
ADAPT_SETPOINT="${ADAPT_SETPOINT:-0.5}"
RUN_STEPS="${RUN_STEPS:-4000000}"

cd "$RUN_DIR"
AGENT_FLAGS=(
  --agent.goal_autoencoder_beta 0.25
  --agent.goal_rec_loss_agg sum
  --agent.goal_opt.lr 4e-5
  --agent.goal_opt.wd 0.0
  --agent.goal_opt.eps 1e-20
  --agent.manager_actent_impl mult
  --agent.manager_actent_target 0.5
  --agent.mask_topk "$MASK_TOPK"
  --agent.goal_struct_weight "$STRUCT_W"
  --agent.mask_sparsity_mode "$MASK_MODE"
  --agent.mask_sparsity_max "$MASK_MAX"
  --agent.mask_sparsity_target "$MASK_TARGET"
  --agent.mgr_cond_goalcode True
  --agent.mgr_cond_achieve "$MGR_ACHIEVE"
  --agent.variable_goal_length True
  --agent.goal_duration_min "$DUR_MIN"
  --agent.goal_duration_max "$DUR_MAX"
  --agent.goal_duration_reg 0.0
  --agent.goal_duration_target "$DUR_TARGET"
  --agent.goal_duration_adapt True
  --agent.goal_duration_adapt_setpoint "$ADAPT_SETPOINT"
  --agent.goal_duration_adapt_max "$ADAPT_MAX"
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
