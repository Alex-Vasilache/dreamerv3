#!/bin/bash -l
#SBATCH -J eval_e25_maskviz
#SBATCH -p gpu-v100
#SBATCH -t 0-01:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir eval_e25_maskviz)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

# Stable copy of e25's latest checkpoint (e25 is still training and rotates ckpts).
CKPT=/work/DoyaU/vasilache/work/e25_maskviz_eval_ckpt/ckpt/20260616T082901F230080

# eval_only: load the e25 (masked + variable-goal-length) agent and roll out a few
# cartpole episodes with the deterministic eval policy. The new online 4-row mask_viz
# panel is emitted every step and logged at episode end as
# epstats/policy_log/mask_viz_image (a gif in WandB). Config flags mirror the e25
# training recipe EXACTLY so the checkpoint loads into a matching architecture.
python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size6m masked_goals variable_goals \
  --script eval_only \
  --run.from_checkpoint "$CKPT" \
  --run.steps 4200 \
  --run.envs 1 \
  --run.log_every 5 \
  --task dmc_cartpole_swingup \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 32 32 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.mgr_expl_weight 0.1 \
  --agent.mgr_retnorm.impl meanstd \
  --agent.goal_autoencoder_beta 0.25 \
  --agent.goal_rec_loss_agg sum \
  --agent.goal_opt.lr 4e-5 \
  --agent.goal_opt.wd 0.0 \
  --agent.goal_opt.eps 1e-20 \
  --agent.manager_actent_impl mult \
  --agent.manager_actent_target 0.5 \
  --agent.mask_topk 0 \
  --agent.goal_struct_weight 200.0 \
  --agent.mask_sparsity_mode prob \
  --agent.mask_sparsity_max 5.0 \
  --agent.mask_sparsity_target 0.3 \
  --agent.mgr_cond_goalcode True \
  --agent.mgr_cond_achieve False \
  --agent.variable_goal_length True \
  --agent.goal_duration_min 1 \
  --agent.goal_duration_max 16 \
  --online_learning False \
  --online_actor_cpu False \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False

echo "EVAL DONE"
exit $?
