#!/bin/bash -l
#SBATCH -J antmaze_m_director_clean
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

# Pure Director baseline on loconav_ant_maze_m, with the two auxiliary
# mechanisms that default to True since 2026-07-22 explicitly DISABLED.
#
# This supersedes run_v3_antmaze_m_director_baseline_big_a100.sbatch, which
# passed no struct/reuse flag and therefore inherited goal_struct_adapt=True.
# In e410-e414 that term reached a multiplier of 843-2000 and accounted for
# 7-56% of the goal-autoencoder loss (verified against the logged
# loss/goal_autoencoder), which disqualifies those runs as a control for any
# work on goal-code geometry -- goal_struct_adapt optimizes exactly that
# geometry. e390-e409 (DMC) set both flags False and are unaffected.
#
# Pure Director baseline (fixed-K, no masked-goals/variable-K) ported to
# loconav_ant_maze_m, at the same director_match BIG/A100 scale (batch16x64,
# imag16, native conv) as the e390-e409 DMC pure-Director baselines. Goal:
# an antmaze-M Director run to sanity-check the HRL implementation on a
# navigation task with sparse reward -- the only prior antmaze attempt (e65,
# masked+variable-K recipe) never encountered reward and was abandoned
# (EXPERIMENTS.md e65 row). This uses ONLY Director-faithful defaults
# (agent.use_hrl=True is already the global default; RECIPE is fixed-K here,
# no masked_goals/variable_goals configs).
#
# loconav's own config block (dreamerv3/configs.yaml) sets
# run.train_ratio=256 (not overridden here -- higher train ratio suits the
# sparser reward / harder exploration of navigation vs. DMC locomotion) and
# env.loconav.repeat=1.
#
# Requeue-safe: job-id RUN_DIR + USR1 self-requeue at T-600s, so a 10M-step
# run spanning multiple 48h segments resumes from checkpoint automatically.
#
# Usage:
#   sbatch -J e475_antmaze_clean_s0 --export=ALL,EXP_TAG=e475,SEED=0 \
#     sbatch/run_v3_antmaze_m_director_clean_big_a100.sbatch
set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
unset DREAMERV3_CONV_IMPL   # native cuDNN conv (A100 sm_80 ~2x faster)

EXP_TAG="${EXP_TAG:?set EXP_TAG}"
SEED="${SEED:-0}"
MGR_FREQ="${MGR_FREQ:-8}"
MGR_REWARD_AGG="${MGR_REWARD_AGG:-mean}"
MGR_EXPL_W="${MGR_EXPL_W:-0.1}"
IMAG_LENGTH="${IMAG_LENGTH:-16}"
RUN_ENVS="${RUN_ENVS:-16}"
RUN_STEPS="${RUN_STEPS:-10000000}"

RUN_SUFFIX="${EXP_TAG}_antmaze_m_director_clean_BIG"
# Requeue-safe RUN_DIR: requeues keep the same SLURM_JOB_ID, so a preempted/
# requeued job resumes from checkpoint in the same dir instead of restarting
# fresh (see run_v3_prior_vargoal_big_a100.sbatch e177 note).
RUN_DIR="${RUN_DIR:-/work/DoyaU/vasilache/work/${RUN_SUFFIX}_j${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"
export MUJOCO_GL=egl

cat > "$RUN_DIR/job.env" <<EOF
RUN_DIR=$RUN_DIR
EXP_TAG=$EXP_TAG
CONFIGS=director_match loconav
TASK=loconav_ant_maze_m
SIZE=BIG IMAGE=64x64 IMAG=$IMAG_LENGTH BATCH=16x64 CONV=native
RECIPE=director STRUCT_ADAPT=False GOAL_SOFT_REUSE_ADAPT=False MGR_FREQ=$MGR_FREQ MGR_REWARD_AGG=$MGR_REWARD_AGG MGR_EXPL_W=$MGR_EXPL_W
RUN_ENVS=$RUN_ENVS RUN_STEPS=$RUN_STEPS SEED=$SEED
NOTE=pure Director baseline on antmaze-M with struct_adapt and soft_reuse_adapt OFF (supersedes e410-e414)
EOF

AGENT_FLAGS=(
  # The point of this script: both default to True in configs.yaml and must be
  # off for a Director control. goal_struct_weight is already 0.0 by default,
  # but goal_struct_adapt=True alone is enough to add the term to the loss.
  --agent.goal_struct_weight 0.0
  --agent.goal_struct_adapt False
  --agent.goal_soft_reuse_adapt False
  --agent.mgr_cond_goalcode False
  --agent.goal_autoencoder_beta 0.25
  --agent.goal_rec_loss_agg sum
  --agent.goal_opt.lr 4e-5
  --agent.goal_opt.wd 0.0
  --agent.goal_opt.eps 1e-20
  --agent.manager_actent_impl mult
  --agent.manager_actent_target 0.5
  --agent.manager_sample_freq "$MGR_FREQ"
  --agent.imag_length "$IMAG_LENGTH"
  --agent.report False
)

# Self-requeue near walltime (USR1 arrives 600s before the limit): requeue keeps
# the job id -> same RUN_DIR -> checkpoint resume. Wall-time TIMEOUT is terminal
# otherwise (only preemption auto-requeues).
on_usr1() {
  echo "[self-requeue] USR1 near walltime; requeueing job $SLURM_JOB_ID to resume in $RUN_DIR"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap on_usr1 USR1

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --seed "$SEED" \
  --configs director_match loconav \
  --env.loconav.image True \
  --env.loconav.size 64 64 \
  --batch_size 16 \
  --batch_length 64 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.mgr_expl_weight "$MGR_EXPL_W" \
  --agent.mgr_reward_agg "$MGR_REWARD_AGG" \
  --agent.mgr_retnorm.impl meanstd \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs "$RUN_ENVS" \
  --run.steps "$RUN_STEPS" \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}" &
TRAIN_PID=$!
wait "$TRAIN_PID"

exit $?
