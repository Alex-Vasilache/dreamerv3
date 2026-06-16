#!/bin/bash -l
#SBATCH -J eval_exp1_maskviz
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

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir eval_exp1_maskviz)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

# exp1 = PLAIN HRL goals (NO masking, NO variable goal length) -- the pre-masking
# baseline. use_masked_goals is False, so the 3-row mask_viz panel is NOT emitted
# (there is no running goal code / edit mask to overlay). This produces the 2-row
# epstats/policy_log/image_with_goal (obs | decoded goal) for comparison against the
# goal rows of e17 / e25. Config flags mirror exp1's training recipe exactly.
CKPT=/work/DoyaU/vasilache/work/exp1_eval_ckpt/ckpt/20260605T025523F248743

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size6m \
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
  --agent.report False \
  --online_learning False \
  --online_actor_cpu False \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False

echo "EVAL DONE"
exit $?
