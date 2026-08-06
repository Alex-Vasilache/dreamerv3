#!/bin/bash -l
#SBATCH -J goal_ae_ablation
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

# Goal-autoencoder ablation (motivation.tex, Lipschitz-constrained SOM-VAE).
#
# Identical in every respect to the pure-Director baselines e390-e409 -- same
# BIG/director_match scale, batch 16x64, imag16, train_ratio 64, 8 envs, 4M
# steps, native conv, MGR_FREQ=8, STRUCT_ADAPT=False, GOAL_SOFT_REUSE_ADAPT=False
# -- EXCEPT for the goal autoencoder, selected by ARM:
#
#   director    Director categorical + straight-through head (reproduces e390-e409)
#   vq          blockwise vector quantizer, no SOM, no Lipschitz  (extra control)
#   som         + circular neighborhood loss + second reconstruction
#   lipvq       + row-wise weight normalization + Lipschitz penalty
#   som_lipvq   both
#   som_lipvq_mgr  both, plus a manager whose logits are distances to the
#              codebook, so the ring is usable by the policy and not only by
#              the decoder (hrl/goal_ae.py ManagerRingHead)
#
# The `vq` arm is not one of motivation.tex's four but is worth running: it
# isolates "quantization instead of categorical sampling" so a som/lipvq result
# can be attributed to the SOM / Lipschitz component rather than to the
# bottleneck swap.
#
# IMPORTANT -- this script runs main.py from the dreamerv3_somvae WORKTREE, not
# the primary code/dreamerv3 tree (which stays pinned on `main` for the live
# antmaze baselines e410-e414). SLURM freezes this script at submit time but NOT
# the Python source, which is re-read on every requeue: do not switch branches
# or land unrelated commits in dreamerv3_somvae while these runs are in flight.
#
# Usage (see sbatch/submit_goal_ae_ablation.sh for the staged matrix launcher):
#   sbatch -J e415_hopper_som --export=ALL,EXP_TAG=e415,TASK=dmc_hopper_hop,ARM=som,SEED=0 \
#     sbatch/run_v3_goal_ae_ablation_big_a100.sbatch
set -euo pipefail

REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3_somvae

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
unset DREAMERV3_CONV_IMPL   # native cuDNN conv (A100 sm_80 ~2x faster)

EXP_TAG="${EXP_TAG:?set EXP_TAG}"
TASK="${TASK:-dmc_hopper_hop}"
ARM="${ARM:?set ARM (director|vq|som|lipvq|som_lipvq)}"
SEED="${SEED:-0}"
MGR_FREQ="${MGR_FREQ:-8}"
MGR_REWARD_AGG="${MGR_REWARD_AGG:-mean}"
MGR_EXPL_W="${MGR_EXPL_W:-0.1}"
IMAG_LENGTH="${IMAG_LENGTH:-16}"
RUN_ENVS="${RUN_ENVS:-8}"
RUN_STEPS="${RUN_STEPS:-4000000}"
# Track goal-code/goal-space geometry across training. This is the property the
# whole architecture targets, so it is ON here (unlike the baselines, where only
# seed 4 logged it) -- costs an O((batch_size*batch_length)^2) similarity matrix
# per train step and does NOT enter any loss.
GOAL_STRUCT_DIAG="${GOAL_STRUCT_DIAG:-True}"

case "$ARM" in
  director)  ARM_CONFIG="" ;;
  vq)        ARM_CONFIG="goal_vq" ;;
  som)       ARM_CONFIG="goal_som" ;;
  lipvq)     ARM_CONFIG="goal_lipvq" ;;
  som_lipvq) ARM_CONFIG="goal_som_lipvq" ;;
  som_lipvq_mgr) ARM_CONFIG="goal_som_lipvq_mgr" ;;
  *) echo "unknown ARM=$ARM (director|vq|som|lipvq|som_lipvq|som_lipvq_mgr)" >&2; exit 1 ;;
esac
CONFIGS="director_match${ARM_CONFIG:+ $ARM_CONFIG}"

RUN_SUFFIX="${EXP_TAG}_${TASK}_${ARM}_s${SEED}_BIG"
# Requeue-safe RUN_DIR: a requeue keeps the same SLURM_JOB_ID, so the job
# resumes from checkpoint in the same dir instead of restarting fresh.
RUN_DIR="${RUN_DIR:-/work/DoyaU/vasilache/work/${RUN_SUFFIX}_j${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"
export MUJOCO_GL=egl

cat > "$RUN_DIR/job.env" <<EOF
RUN_DIR=$RUN_DIR
EXP_TAG=$EXP_TAG
CONFIGS=$CONFIGS
TASK=$TASK
ARM=$ARM SEED=$SEED
SIZE=BIG IMAGE=64x64 IMAG=$IMAG_LENGTH BATCH=16x64 TRAIN_RATIO=64 CONV=native
RECIPE=director MGR_FREQ=$MGR_FREQ MGR_REWARD_AGG=$MGR_REWARD_AGG MGR_EXPL_W=$MGR_EXPL_W
STRUCT_W=0.0 STRUCT_ADAPT=False GOAL_SOFT_REUSE_ADAPT=False GOAL_STRUCT_DIAG=$GOAL_STRUCT_DIAG
CODE=$REPO GIT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
NOTE=goal-AE ablation vs e390-e409 pure-Director baselines (identical except the goal AE)
EOF

AGENT_FLAGS=(
  --agent.goal_autoencoder_beta 0.25
  --agent.goal_rec_loss_agg sum
  --agent.goal_opt.lr 4e-5
  --agent.goal_opt.wd 0.0
  --agent.goal_opt.eps 1e-20
  --agent.manager_actent_impl mult
  --agent.manager_actent_target 0.5
  # Explicitly off: the e390-e409 baselines set these False, and the struct
  # term in particular optimizes the very geometry this architecture is meant
  # to provide -- leaving it on would confound the comparison. (The antmaze
  # e410-e414 runs did NOT set them and so carry both; see EXPERIMENTS.md.)
  --agent.goal_struct_weight 0.0
  --agent.goal_struct_adapt False
  --agent.goal_soft_reuse_adapt False
  --agent.mgr_cond_goalcode False
  --agent.goal_struct_diag "$GOAL_STRUCT_DIAG"
  --agent.worker_timed_goals False
  --agent.variable_goal_length False
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

python -u "$REPO/dreamerv3/main.py" \
  --logdir "$RUN_DIR/logdir" \
  --seed "$SEED" \
  --configs $CONFIGS \
  --task "$TASK" \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 64 64 \
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
  --run.train_ratio 64 \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}" &
TRAIN_PID=$!
# Backgrounded + wait so the USR1 trap can run while training is in flight
# (bash defers traps until a foreground command returns).
wait "$TRAIN_PID"

exit $?
