#!/bin/bash -l
#SBATCH -J e55_walker
#SBATCH -p gpu-a100
#SBATCH -t 0-24:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir e55_walker_vargoal_masked_notarget)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

export MUJOCO_GL=egl

# e55 = same masked + variable-duration / no-prior recipe as e54, on dmc_walker_walk
# (64x64, locomotion — better HRL testbed than cartpole for temporal abstraction).
# a100 slot; fewer parallel envs than cartpole due to image size.
MASK_TOPK="${MASK_TOPK:-0}"
STRUCT_W="${STRUCT_W:-200.0}"
MASK_MODE="${MASK_MODE:-prob}"
MASK_MAX="${MASK_MAX:-5.0}"
MASK_TARGET="${MASK_TARGET:-0.3}"
MGR_ACHIEVE="${MGR_ACHIEVE:-False}"
DUR_MIN="${DUR_MIN:-1}"
DUR_MAX="${DUR_MAX:-16}"
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
  --agent.goal_duration_adapt False
  --agent.report False
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size6m masked_goals variable_goals \
  --task dmc_walker_walk \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 64 64 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.mgr_expl_weight 0.1 \
  --agent.mgr_retnorm.impl meanstd \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 4 \
  --run.steps "$RUN_STEPS" \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}"

exit $?
