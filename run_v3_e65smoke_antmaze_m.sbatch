#!/bin/bash -l
#SBATCH -J e65smoke_antmaze
#SBATCH -p gpu-v100
#SBATCH -t 0-00:40:00
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

# Smoke test for the e65 Antmaze-M run: validates that the (patched) loconav env
# imports, produces the camera-4 egocentric image + proprio obs, and that the
# masked + variable-K HRL agent trains a handful of steps. Tiny nets + CPU via
# `debug`; ~a few minutes. NOT an experiment — no wandb, throwaway logdir.
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir e65smoke_antmaze_m)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

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
  --agent.mask_topk 0
  --agent.goal_struct_weight 200.0
  --agent.mask_sparsity_mode prob
  --agent.mask_sparsity_max 5.0
  --agent.mask_sparsity_target 0.3
  --agent.mgr_cond_goalcode True
  --agent.mgr_cond_achieve False
  --agent.variable_goal_length True
  --agent.goal_duration_min 1
  --agent.goal_duration_max 16
  --agent.goal_duration_reg 0.01
  --agent.goal_duration_target 8.0
  --agent.goal_duration_adapt False
  --agent.report False
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size6m loconav masked_goals variable_goals debug \
  --env.loconav.image True \
  --env.loconav.size 64 64 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.mgr_expl_weight 0.1 \
  --agent.mgr_retnorm.impl meanstd \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 4 \
  --run.steps 2000 \
  --logger.outputs jsonl \
  "${AGENT_FLAGS[@]}"

echo "SMOKE EXIT=$?"
exit $?
