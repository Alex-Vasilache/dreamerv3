#!/bin/bash -l
#SBATCH -J e123_director
#SBATCH -p gpu-a100
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600

# e123: PURE Director-style HRL baseline in our DreamerV3 implementation.
#
# Purpose: confirm the BASE hierarchy (manager/worker/goal-VAE) reproduces
# Director's published DMC numbers BEFORE stacking any of our research changes.
# Unlike e118 (which loaded `masked_goals variable_goals` + mask-sparsity +
# per-block credit + goal_struct + variable-K + switch/edit costs), this run uses
# ONLY the Director-faithful defaults:
#   * use_hrl=True            (default) -> Director manager/worker/goal-VAE
#   * use_masked_goals=False  (default) -> OVERWRITE ENTIRE GOALS (whole skill
#                                          code replaced each manager decision)
#   * variable_goal_length=False (default) -> FIXED goal length K=manager_sample_freq=8
# plus `director_match` (RSSM deter/hidden 1024, classes 32, CNN depth 64,
# 4-layer 512-wide MLPs) + 64x64 images for architecture parity with TF Director,
# and the real DreamerV3/Director default batch (16x64=1024 frames/grad-step,
# imag 16, train_ratio 64 == Director "train every 16 env steps").
#
# Everything in AGENT_FLAGS below is EQUAL TO THE CONFIG DEFAULT and is passed
# only for explicit reproducibility (nothing here diverges from `defaults`).
#
# TASK is overridable to sweep the four tasks the base must match:
#   dmc_cheetah_run (default, Director ~500), dmc_hopper_hop, dmc_acrobot_swingup,
#   dmc_cartpole_swingup. Give each task a NEW experiment number when launched.
#
# Fit: b16x64 director_match 64x64 needs A100-80GB (~59/80 GB on e118); does not
# fit on V100-16GB. train_ratio 64 @ 4M is ~62h > 48h wall -> pass a FIXED RUN_DIR
# and CHAIN (afterany) to resume past the wall limit.
set -uo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
unset DREAMERV3_CONV_IMPL   # native cuDNN conv (A100 sm_80 — ~2x faster)

EXP_TAG="${EXP_TAG:-e123}"
TASK="${TASK:-dmc_cheetah_run}"
RUN_SUFFIX="${EXP_TAG}_${TASK}_director_baseline"
# Requeue-safe RUN_DIR: keyed to SLURM_JOB_ID (stable across requeues) so a
# preempted/requeued job resumes from checkpoint; pass RUN_DIR for chained resumes.
RUN_DIR="${RUN_DIR:-/work/DoyaU/vasilache/work/${RUN_SUFFIX}_j${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"
export MUJOCO_GL=egl

cat > "$RUN_DIR/job.env" <<EOF
RUN_DIR=$RUN_DIR
EXP_TAG=$EXP_TAG
CONFIGS=director_match
TASK=$TASK
IMAGE_SIZE=64x64
BATCH=16 LEN=64 IMAG=16 TRAIN_RATIO=64
CONV=native
MASKED_GOALS=False VARIABLE_GOAL_LENGTH=False MANAGER_SAMPLE_FREQ=8
NOTE=pure Director baseline (defaults only, no masked/variable-K/struct/costs)
EOF

cd "$RUN_DIR"

# Self-requeue near walltime (USR1 600s early); requeue keeps job id -> same
# RUN_DIR -> checkpoint resume.
on_usr1() {
  echo "[self-requeue] USR1 near walltime; requeueing job $SLURM_JOB_ID to resume in $RUN_DIR"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap on_usr1 USR1

# All flags below == config defaults; explicit for reproducibility.
AGENT_FLAGS=(
  --agent.use_hrl True
  --agent.use_masked_goals False
  --agent.variable_goal_length False
  --agent.manager_sample_freq 8
  --agent.goal_autoencoder_beta 0.25
  --agent.goal_rec_loss_agg sum
  --agent.goal_opt.lr 4e-5
  --agent.goal_opt.wd 0.0
  --agent.goal_opt.eps 1e-20
  --agent.manager_actent_impl mult
  --agent.manager_actent_target 0.5
  --agent.mgr_reward_agg mean
  --agent.imag_length 16
  --agent.report False
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs director_match \
  --task "$TASK" \
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
  "${AGENT_FLAGS[@]}" &
TRAIN_PID=$!
# Backgrounded + wait so the USR1 trap can run while training is in flight.
wait "$TRAIN_PID"

exit $?
