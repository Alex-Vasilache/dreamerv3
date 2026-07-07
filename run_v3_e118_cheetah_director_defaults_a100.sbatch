#!/bin/bash -l
#SBATCH -J e118_cheetah
#SBATCH -p gpu-a100
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# e118: cheetah_run at the TRUE DreamerV3/Director default batch on A100-80GB.
# Same e88/e116 manager recipe + director_match 64x64 + native conv, but:
#   batch_size 16, batch_length 64 (=1024 frames/grad-step, the real default),
#   imag_length 16, train_ratio 64 (== Director "train every 16 env steps").
# Validated fit: ~59/80 GB, ~1127 fps/train on A100-80GB (native conv).
# train_ratio 64 @ 4M steps is ~62h > 48h wall, so this script RESUMES from a
# FIXED RUN_DIR and is CHAINED (afterany) to continue past the wall limit.
set -uo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
unset DREAMERV3_CONV_IMPL   # native cuDNN conv (A100 sm_80 — ~2x faster)

EXP_TAG="${EXP_TAG:-e118}"
RUN_SUFFIX="${EXP_TAG}_cheetah_director_defaults"
# RUN_DIR must be passed in for chained resumes; only mint one on the first job.
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir "$RUN_SUFFIX")}"
mkdir -p "$RUN_DIR"
export MUJOCO_GL=egl

cat > "$RUN_DIR/job.env" <<EOF
RUN_DIR=$RUN_DIR
EXP_TAG=$EXP_TAG
CONFIGS=director_match masked_goals variable_goals
TASK=dmc_cheetah_run
IMAGE_SIZE=64x64
BATCH=16 LEN=64 IMAG=16 TRAIN_RATIO=64
CONV=native
MGR_REWARD_AGG=sum DUR_REG=0.0 DUR_MAX=16 SWITCH_COST=0.02
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
  --agent.goal_duration_max 16
  --agent.goal_duration_reg 0.0
  --agent.goal_duration_target 4.0
  --agent.goal_duration_adapt False
  --agent.imag_length 16
  --agent.report False
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs director_match masked_goals variable_goals \
  --task dmc_cheetah_run \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 64 64 \
  --batch_size 16 \
  --batch_length 64 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.mgr_expl_weight 0.1 \
  --agent.mgr_retnorm.impl meanstd \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 8 \
  --run.steps 4000000 \
  --run.train_ratio 64 \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}"

exit $?
