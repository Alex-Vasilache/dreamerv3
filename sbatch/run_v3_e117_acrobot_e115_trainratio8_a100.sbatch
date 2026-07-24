#!/bin/bash -l
#SBATCH -J e117_acrobot
#SBATCH -p gpu-a100
#SBATCH -t 2-00:00:00
#SBATCH -c 16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# e117: e116 recipe (e88 manager + director_match + 64x64, native conv,
# train_ratio 8) on acrobot_swingup instead of cartpole.
#   director_match batch_steps = batch_size 4 * batch_length 32 = 128
#   train_ratio 8 -> gradient step every 16 env steps (== TF Director train_every 16).
# Prior acrobot attempt e110 (e88 verbatim, size6m 32x32) FAILED to learn (19/1/132).
# This isolates whether Director-scale capacity + 64x64 resolution rescues acrobot,
# a harder swingup than cartpole. imag_length stays 32 (variable_goals) so a full
# manager decision (dur_max 32) fits in one imagined rollout for per-block credit.
set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

# --- speed lever A (ENABLED): native cuDNN conv instead of the einsum reference
# impl. source_dreamerv3_env.sh forces DREAMERV3_CONV_IMPL=reference (a Saion
# workaround); we unset it so nets.py uses jax.lax.conv_general_dilated. Smoke-
# tested on gpu-a100 (el8) 2026-07-03 with this exact director_match+64x64 model:
# no segfault, and ~1.98x faster (fps/train ~792 native vs ~401 reference at
# train_ratio 8). The XLA cuDNN autotuner prints one-time "slow_operation_alarm"
# warnings during first compile -- expected, not an error.
unset DREAMERV3_CONV_IMPL

EXP_TAG="${EXP_TAG:-e117}"
RUN_SUFFIX="${EXP_TAG}_acrobot_e115_trainratio8"
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir "$RUN_SUFFIX")}"
mkdir -p "$RUN_DIR"

export MUJOCO_GL=egl

cat > "$RUN_DIR/job.env" <<EOF
RUN_DIR=$RUN_DIR
EXP_TAG=$EXP_TAG
CONFIGS=director_match masked_goals variable_goals
TASK=dmc_acrobot_swingup
IMAGE_SIZE=64x64
TRAIN_RATIO=8
MGR_REWARD_AGG=sum DUR_REG=0.0 DUR_MAX=32 SWITCH_COST=0.02
MASK_MODE=prob PERBLOCK_CREDIT=True PERBLOCK_EDIT_COST=0.3
EOF

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
  --agent.mgr_reward_agg sum
  --agent.goal_switch_cost 0.02
  --agent.goal_edit_cost 0.0
  --agent.goal_edit_cost_ach 0.0
  --agent.mask_sparsemax False
  --agent.mask_sparsity_prob_ach_weight 0.0
  --agent.mask_perblock_credit True
  --agent.perblock_edit_cost 0.3
  --agent.mgr_cond_goalcode True
  --agent.mgr_cond_achieve False
  --agent.variable_goal_length True
  --agent.goal_duration_min 1
  --agent.goal_duration_max 32
  --agent.goal_duration_reg 0.0
  --agent.goal_duration_target 4.0
  --agent.goal_duration_adapt False
  --agent.report False
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs director_match masked_goals variable_goals \
  --task dmc_acrobot_swingup \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 64 64 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.mgr_expl_weight 0.1 \
  --agent.mgr_retnorm.impl meanstd \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 8 \
  --run.steps 4000000 \
  --run.train_ratio 8 \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}"

exit $?
