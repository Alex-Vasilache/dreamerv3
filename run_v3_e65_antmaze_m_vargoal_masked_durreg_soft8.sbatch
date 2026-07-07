#!/bin/bash -l
#SBATCH -J e65_antmaze
#SBATCH -p gpu-v100
#SBATCH -t 2-00:00:00
#SBATCH -c 32
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=128G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

# e65 — Director Antmaze-M (loconav_ant_maze_m): masked + variable-K + soft fixed
# duration prior (reg=0.01 toward target 8). This is the e58 recipe ported to the
# navigation task. Obs = camera-4 egocentric image (64x64) + ant proprio, matching
# Director (re-enabled in loconav.py; dreamerv3 port had regressed to proprio-only).
#
# Walltime is 48h (gpu-v100 assoc MaxWall cap); RUN_STEPS targets 10M, so this will
# NOT finish in one window. Resume to continue: re-run with the SAME --logdir
#   RUN_DIR=<this run dir> sbatch run_v3_e65_antmaze_m_vargoal_masked_durreg_soft8.sbatch
DUR_MIN="${DUR_MIN:-1}"
DUR_MAX="${DUR_MAX:-16}"
DUR_REG="${DUR_REG:-0.01}"
DUR_TARGET="${DUR_TARGET:-8.0}"
MASK_TOPK="${MASK_TOPK:-0}"
STRUCT_W="${STRUCT_W:-200.0}"
MASK_MODE="${MASK_MODE:-prob}"
MASK_MAX="${MASK_MAX:-5.0}"
MASK_TARGET="${MASK_TARGET:-0.3}"
MGR_ACHIEVE="${MGR_ACHIEVE:-False}"
RUN_STEPS="${RUN_STEPS:-10000000}"

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir e65_antmaze_m_vargoal_masked_durreg_soft8)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"
echo "DUR_TARGET=$DUR_TARGET DUR_REG=$DUR_REG DUR_MAX=$DUR_MAX RUN_STEPS=$RUN_STEPS" >> "$RUN_DIR/job.env"

export MUJOCO_GL=egl

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
  --agent.goal_duration_reg "$DUR_REG"
  --agent.goal_duration_target "$DUR_TARGET"
  --agent.goal_duration_adapt False
  --agent.report False
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size6m loconav masked_goals variable_goals \
  --env.loconav.image True \
  --env.loconav.size 64 64 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.mgr_expl_weight 0.1 \
  --agent.mgr_retnorm.impl meanstd \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs "${RUN_ENVS:-16}" \
  --run.steps "$RUN_STEPS" \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}"

exit $?
