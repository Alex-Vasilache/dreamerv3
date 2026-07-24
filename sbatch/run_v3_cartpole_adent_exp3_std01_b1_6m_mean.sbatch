#!/bin/bash -l
#SBATCH -J v3_cartpole_adent_exp3_std01_b1_6m_mean
#SBATCH -p gpu-v100
#SBATCH -t 0-24:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir v3_cartpole_adent_exp3_std01_b1_6m_mean)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

export MUJOCO_GL=egl

cd "$RUN_DIR"
# Exp3: Goal AE beta 1.0 | goal_opt lr 1e-4, wd 1e-2, eps 1e-6 (bumped optimizer) |
# manager adv scaling std (meanstd) + 0.1 expl, adaptive actent 0.5 | size 6m |
# goal rec loss aggregation mean (per-dim, rescaled by goal_rec_dim_ref=1024;
# at 6m deter=1024 this equals the sum, so the scale is unchanged here).
AGENT_FLAGS=(
  --agent.goal_autoencoder_beta 1.0
  --agent.goal_rec_loss_agg mean
  --agent.goal_rec_dim_ref 1024
  --agent.goal_opt.lr 1e-4
  --agent.goal_opt.wd 1e-2
  --agent.goal_opt.eps 1e-6
  --agent.manager_actent_impl mult
  --agent.manager_actent_target 0.5
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size6m \
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
  --run.steps 4000000 \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}"

exit $?
