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
#   som_lipvq_smooth  both, plus ring-smoothed REINFORCE credit
#              (hrl/heads.py ring_smooth_event). Differs from som_lipvq by the
#              single number goal_vq.mgr_smooth, so it isolates credit
#              assignment without changing the policy class.
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
# Track goal-code/goal-space geometry across training. Costs an
# O((batch_size*batch_length)^2 = 1024^2) similarity matrix per train step --
# i.e. 64 of them per env step at train_ratio 64 -- and does NOT enter any
# loss. Now OFF by default: it was the single biggest avoidable cost in the
# 4M-step budget (see below), and nothing in the current programme reads
# goal/struct_corr*. Set GOAL_STRUCT_DIAG=True to get it back.
GOAL_STRUCT_DIAG="${GOAL_STRUCT_DIAG:-False}"
# Speed knobs. None of these change what is computed by training -- they only
# remove diagnostics/IO -- so runs with different values stay comparable.
#   PREALLOC   True = grab the GPU allocation once up front instead of growing
#              it per step (memory use is ~21% of an 80GB A100, so this is free)
#   LOG_OUTPUT scope wrote ~400MB of episode video per run that nothing reads
#   LOG_VIDEO  False stops stacking every frame of worker 0's episodes in RAM
PREALLOC="${PREALLOC:-True}"
LOG_OUTPUTS="${LOG_OUTPUTS:-jsonl,wandb}"
LOG_VIDEO="${LOG_VIDEO:-False}"
#   RUN_DEBUG  Driver placement. True = envs stepped in-process, False = one
#              process per env. Measured 2026-08-09 on this exact recipe:
#              in-process 40.0 env fps / GPU 0.90, subprocess 36.4 / 0.79 --
#              piping 64x64 observations costs more than stepping DMC inline.
RUN_DEBUG="${RUN_DEBUG:-True}"
# Replay capacity. The upstream DreamerV3 default is 5e6, which EXCEEDS the 4e6
# steps these runs take, so `Replay._insert` never reaches capacity and nothing
# is ever evicted: `replay/items` grows linearly to the step count (measured in
# e403). Sampling is `fracs.uniform: 1.0` and only ~12.5% of a batch comes from
# the online queue, so ~87.5% of every training batch is drawn uniformly over
# the WHOLE run history, mean sample age ~2M steps. TF Director uses 1e6.
# Written as an integer, not 1e6: PyYAML parses unquoted `1e6` as a STRING.
REPLAY_SIZE="${REPLAY_SIZE:-1000000}"
# Checkpoint interval in seconds. 900 is the default and is right for a 48h
# job that never restarts; on `short-a100`, where the association caps a job at
# 2h and the run requeues itself ~16 times over its life, every requeue throws
# away the progress since the last checkpoint, so a shorter interval pays for
# itself (a 1GB checkpoint write costs seconds; 15 minutes of lost training
# costs minutes). Bookkeeping only -- it changes nothing about what is
# computed, so runs with different values stay comparable.
SAVE_EVERY="${SAVE_EVERY:-900}"
# Persistent XLA compilation cache. Also only bookkeeping, and the same reason:
# a 2h slice spends 10-15 minutes compiling the BIG graph and autotuning cuDNN
# from cold, which is ~10% of the slice. The cache is keyed on the HLO, the
# backend and the jax version, so arms with different configs cannot collide,
# and concurrent jobs are what the cache is designed for. Set JAX_CACHE_DIR=
# (empty) to switch it off.
JAX_CACHE_DIR="${JAX_CACHE_DIR:-/work/DoyaU/vasilache/work/jax_cache}"
if [ -n "$JAX_CACHE_DIR" ]; then
  mkdir -p "$JAX_CACHE_DIR"
  export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR"
fi

case "$ARM" in
  director)  ARM_CONFIG="" ;;
  vq)        ARM_CONFIG="goal_vq" ;;
  som)       ARM_CONFIG="goal_som" ;;
  lipvq)     ARM_CONFIG="goal_lipvq" ;;
  som_lipvq) ARM_CONFIG="goal_som_lipvq" ;;
  som_lipvq_mgr) ARM_CONFIG="goal_som_lipvq_mgr" ;;
  som_lipvq_smooth) ARM_CONFIG="goal_som_lipvq_smooth" ;;
  som_line)      ARM_CONFIG="goal_som_line" ;;
  som_lipvq_line) ARM_CONFIG="goal_som_lipvq_line" ;;
  som_lipvq_line_smooth) ARM_CONFIG="goal_som_lipvq_line_smooth" ;;
  som_lipvq_line_mgr) ARM_CONFIG="goal_som_lipvq_line_mgr" ;;
  som_orig_line) ARM_CONFIG="goal_som_orig_line" ;;
  som_orig_lipvq_line) ARM_CONFIG="goal_som_orig_lipvq_line" ;;
  som_orig_line_adapt) ARM_CONFIG="goal_som_orig_line_adapt" ;;
  # Liu et al. 2022 verbatim: scalar per-layer bound, `prod` penalty, relu.
  som_orig_lipvq_line_paper) ARM_CONFIG="goal_som_orig_lipvq_line_paper" ;;
  # The e510-e549 family: Liu et al.'s penalty FORM (scalar per-layer bound,
  # `prod`) without their activation change, so each arm is a single-factor
  # change from its neighbours. See configs.yaml.
  lipvq_prod)               ARM_CONFIG="goal_lipvq_prod" ;;
  som_lipvq_line_prod)      ARM_CONFIG="goal_som_lipvq_line_prod" ;;
  som_orig_lipvq_line_prod) ARM_CONFIG="goal_som_orig_lipvq_line_prod" ;;
  *) echo "unknown ARM=$ARM (director|vq|som|lipvq|som_lipvq|som_lipvq_mgr|som_lipvq_smooth|som_line|som_lipvq_line|som_lipvq_line_smooth|som_lipvq_line_mgr|som_orig_line|som_orig_lipvq_line|som_orig_line_adapt|som_orig_lipvq_line_paper|lipvq_prod|som_lipvq_line_prod|som_orig_lipvq_line_prod)" >&2; exit 1 ;;
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
REPLAY_SIZE=$REPLAY_SIZE
COMMIT_SCALE=${COMMIT_SCALE:-default} SOM_SCALE=${SOM_SCALE:-default}
RECIPE=director MGR_FREQ=$MGR_FREQ MGR_REWARD_AGG=$MGR_REWARD_AGG MGR_EXPL_W=$MGR_EXPL_W
STRUCT_W=0.0 STRUCT_ADAPT=False GOAL_SOFT_REUSE_ADAPT=False GOAL_STRUCT_DIAG=$GOAL_STRUCT_DIAG
PREALLOC=$PREALLOC LOG_OUTPUTS=$LOG_OUTPUTS LOG_VIDEO=$LOG_VIDEO RUN_DEBUG=$RUN_DEBUG
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

# Optional goal-AE weight overrides, appended only when set, so an arm can be
# swept without a new config block. COMMIT_SCALE is the alpha of
# motivation.tex's commitment term: at the SOM-VAE default of 1.0 that term
# pulls the encoder toward the codebook harder than the reconstruction pushes
# it to stay informative (measured gradient ratio 0.87 at initialization), and
# the codebook collapses. 0.25 is the VQ-VAE commitment weight and gives 3.47.
if [ -n "${COMMIT_SCALE:-}" ]; then
  AGENT_FLAGS+=(--agent.goal_vq.commit_scale "$COMMIT_SCALE")
fi
if [ -n "${SOM_SCALE:-}" ]; then
  AGENT_FLAGS+=(--agent.goal_vq.som_scale "$SOM_SCALE")
fi

# Self-requeue near walltime (USR1 arrives 600s before the limit): requeue keeps
# the job id -> same RUN_DIR -> checkpoint resume. Wall-time TIMEOUT is terminal
# otherwise (only preemption auto-requeues).
on_usr1() {
  echo "[self-requeue] USR1 near walltime; requeueing job $SLURM_JOB_ID to resume in $RUN_DIR"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap on_usr1 USR1

run_training() {
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
    --run.save_every "$SAVE_EVERY" \
    --run.train_ratio 64 \
    --run.log_video "$LOG_VIDEO" \
    --run.debug "$RUN_DEBUG" \
    --replay.size "$REPLAY_SIZE" \
    --logger.outputs "$LOG_OUTPUTS" \
    --logger.wandb_mode online \
    --logger.wandb_entity vasilache \
    --jax.platform cuda \
    --jax.prealloc "$PREALLOC" \
    "${AGENT_FLAGS[@]}" &
  TRAIN_PID=$!
  # Backgrounded + wait so the USR1 trap can run while training is in flight
  # (bash defers traps until a foreground command returns).
  wait "$TRAIN_PID"
}

# In-job retry (added 2026-08-10 for the unattended e510-e557 batch). A crash
# that is not a signal -- a transient CUDA/NCCL error, a filesystem hiccup, an
# OOM from a co-tenant -- otherwise ends the job and leaves its GPU slot to the
# next queued run, so the failed experiment silently drops out of the matrix
# until someone looks. Retrying in place resumes from the checkpoint in the
# same RUN_DIR and costs only the restart.
#
# Deliberately NOT retried:
#   rc >= 128  the process was signalled. That is SLURM preempting or killing
#              the job (short-a100 is preemptible); --requeue handles it, and
#              retrying here just races the teardown.
#   attempt 4+ a fault that survives three restarts is a real bug, not a
#              transient, and looping on it would burn the slot for days.
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
attempt=1
rc=0
while :; do
  rc=0
  run_training || rc=$?
  [ "$rc" -eq 0 ] && break
  if [ "$rc" -ge 128 ]; then
    echo "[retry] attempt $attempt ended on signal (rc=$rc); leaving it to SLURM"
    break
  fi
  if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    echo "[retry] attempt $attempt failed (rc=$rc); giving up after $MAX_ATTEMPTS attempts"
    break
  fi
  echo "[retry] attempt $attempt failed (rc=$rc); resuming from checkpoint in 60s"
  attempt=$((attempt + 1))
  sleep 60
done

exit $rc
