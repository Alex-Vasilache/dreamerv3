#!/bin/bash -l
#SBATCH -J estruct
#SBATCH -p gpu-v100
#SBATCH -t 0-24:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

# Generic struct-correlation-ablation launcher. Base recipe = e25 (size6m +
# masked_goals + variable_goals, the variable-K masked baseline). All struct knobs
# come from env so one template serves the e33-e40 set; the per-run resolved
# config is captured in $RUN_DIR/logdir/config.yaml for archival/reproducibility.
RUN_PREFIX="${RUN_PREFIX:-estruct}"
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir "$RUN_PREFIX")}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

export MUJOCO_GL=egl

# --- struct-correlation knobs (the variables under test) ---
STRUCT_W="${STRUCT_W:-200.0}"          # fixed weight (ignored when adapt=True)
STRUCT_TARGET="${STRUCT_TARGET:-deter}"  # deter | feat
STRUCT_LOSS="${STRUCT_LOSS:-mse}"        # mse | margin
STRUCT_MARGIN="${STRUCT_MARGIN:-0.1}"
STRUCT_ADAPT="${STRUCT_ADAPT:-False}"
STRUCT_ADAPT_TARGET="${STRUCT_ADAPT_TARGET:-0.01}"
STRUCT_ADAPT_INIT="${STRUCT_ADAPT_INIT:-200.0}"

# --- fixed e25 recipe knobs ---
MASK_TOPK="${MASK_TOPK:-0}"
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
  --agent.goal_struct_target "$STRUCT_TARGET"
  --agent.goal_struct_loss "$STRUCT_LOSS"
  --agent.goal_struct_margin "$STRUCT_MARGIN"
  --agent.goal_struct_adapt "$STRUCT_ADAPT"
  --agent.goal_struct_adapt_target "$STRUCT_ADAPT_TARGET"
  --agent.goal_struct_adapt_init "$STRUCT_ADAPT_INIT"
  --agent.mask_sparsity_mode "$MASK_MODE"
  --agent.mask_sparsity_max "$MASK_MAX"
  --agent.mask_sparsity_target "$MASK_TARGET"
  --agent.mgr_cond_goalcode True
  --agent.mgr_cond_achieve "$MGR_ACHIEVE"
  --agent.variable_goal_length True
  --agent.goal_duration_min "$DUR_MIN"
  --agent.goal_duration_max "$DUR_MAX"
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
