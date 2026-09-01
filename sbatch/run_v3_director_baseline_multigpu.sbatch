#!/bin/bash -l
#SBATCH -J director_multigpu
#SBATCH -p gpu-v100
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=192G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600

# Director baselines on V100/P100, for the 5-seeds-per-environment tracker in
# DIRECTOR_BASELINES.md.
#
# Derived from run_v3_goal_ae_ablation_big_a100.sbatch and IDENTICAL to it in
# every training respect, so seeds produced here are directly comparable with
# the existing A100 Director seeds (e502-e509, e554-e557, e566-e569, e624-e625,
# e632-e633). Only the device placement differs.
#
# WHY FOUR GPUS. Batch 16x64 needs ~59GB on an A100 and does not fit one 16GB
# V100/P100. Split across 4 devices it is ~12GB each (fit validated by
# e119-e122, see run_v3_e124_director_baseline_multiv100_routeA.sbatch line 21).
# NGPU=2 would need ~24GB per device and OOM. So one job takes a whole node's
# worth of our quota: 8 GPUs per partition means 2 concurrent jobs.
#
# WHY NOT THE e124 SCRIPT. That one builds RUN_DIR with mktemp plus a timestamp,
# so a requeue starts from zero rather than resuming -- fatal for a 4M run that
# needs 2 requeues to clear 48h walls -- and it takes no SEED. This one inherits
# the job-id RUN_DIR and USR1 self-requeue from the A100 script.
#
#   sbatch --export=ALL,EXP_TAG=e644,TASK=pinpad_five,ARM=director,SEED=2 \
#          sbatch/run_v3_director_baseline_multigpu.sbatch
#   # P100: add -p gpu-p100 --gres=gpu:p100:4 --nodelist=saion-gpu[11-14]

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

# Environment depends on the GPU generation.
#
# The A100 path takes the stock env and native cuDNN conv (~2x). That does NOT
# work on V100/P100: with the current env the conv algorithm picker fails
# outright --
#   INTERNAL: All algorithms tried for ... custom-call(__cudnn$convForward) failed
#   Profiling failure on cuDNN engine ...: CUDNN_STATUS_INTERNAL_ERROR
# which killed jobs 4700184/4700185 (V100) and 4700186 (P100) within a minute.
# The fix is the Route-A clone (jax 0.4.33 + cuDNN 9.1.0.70 + CUDA 12.2, pinned
# against the drift that broke native conv on sm_70) plus a non-strict algorithm
# picker so a failed profile falls back instead of aborting.
#
# The clone is activated directly and therefore never sources
# source_dreamerv3_env.sh, which is where WANDB_API_KEY lives -- without the
# explicit export below the run silently degrades to jsonl-only.
case "${SLURM_JOB_PARTITION:-gpu-a100}" in
  gpu-v100|gpu-p100)
    CLONE=/work/DoyaU/vasilache/work/dreamerv3_env_routeA
    module purge
    _PY311_LIB=/apps/free88/python/3.11.4/lib
    if ! module is-avail python/3.11.4 2>/dev/null; then
      module use /apps/.modulefiles88 2>/dev/null || true
      module use /apps/.modulefiles95 2>/dev/null || true
    fi
    module load python/3.11.4 2>/dev/null || \
      export LD_LIBRARY_PATH="${_PY311_LIB}:${LD_LIBRARY_PATH:-}"
    source "$CLONE/bin/activate"
    python -c 'import sys' >/dev/null 2>&1 || \
      export LD_LIBRARY_PATH="${_PY311_LIB}:${LD_LIBRARY_PATH:-}"
    export MUJOCO_GL=egl
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false"
    _ENVFILE=/apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
    export WANDB_API_KEY="$(sed -n 's/^export WANDB_API_KEY=//p' "$_ENVFILE" | head -1)"
    export WANDB_PROJECT=dreamerv3
    export WANDB_ENTITY=vasilache
    [ -n "${WANDB_API_KEY:-}" ] || \
      echo "[WARN] WANDB_API_KEY not found in $_ENVFILE -- wandb will drop to jsonl-only"
    # Native conv on both. The reference implementation is not a safe fallback
    # here: forcing it on P100 (job 4700192) OOM'd at 13.4GB per device, because
    # reference conv costs more memory than cuDNN, and 4x16GB only fits BIG at
    # the ~12GB/device the native path uses. So the choice on a 16GB card is
    # native conv or nothing, and the Route-A clone plus the non-strict
    # algorithm picker is what makes native conv work off sm_80.
    unset DREAMERV3_CONV_IMPL
    ;;
  *)
    source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
    unset DREAMERV3_CONV_IMPL   # native cuDNN conv (A100 sm_80 ~2x faster)
    ;;
esac

# Data-parallel device list. NGPU must match --gres; 4 is the only value that
# fits BIG on a 16GB card (see the header).
NGPU="${NGPU:-4}"
DEVLIST=$(seq -s ' ' 0 $((NGPU-1)))
echo "[devices] partition=${SLURM_JOB_PARTITION:-?} NGPU=$NGPU DEVLIST='$DEVLIST'" \
     "conv=${DREAMERV3_CONV_IMPL:-native} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

EXP_TAG="${EXP_TAG:?set EXP_TAG}"
TASK="${TASK:-dmc_hopper_hop}"
ARM="${ARM:?set ARM (director|vq|som|lipvq|som_lipvq)}"
SEED="${SEED:-0}"
MGR_FREQ="${MGR_FREQ:-8}"
MGR_REWARD_AGG="${MGR_REWARD_AGG:-mean}"
MGR_EXPL_W="${MGR_EXPL_W:-0.1}"
IMAG_LENGTH="${IMAG_LENGTH:-16}"
RUN_ENVS="${RUN_ENVS:-8}"
# A CLI flag beats a config block, so this has to be a variable: leaving it
# hardcoded at 64 would silently override loconav's train_ratio 256.
TRAIN_RATIO="${TRAIN_RATIO:-64}"
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
# Goal-AE optimizer weight decay. Has to be a variable, not the hardcoded 0.0 it
# used to be: AGENT_FLAGS puts it on the command line, and a CLI flag beats a
# config block (same trap as TRAIN_RATIO and MGR_ACTENT_TARGET above), so
# EXTRA_CONFIGS=director_stable could not turn weight decay on for the goal AE
# while this was pinned. Default 0.0 keeps every earlier submission identical.
GOAL_OPT_WD="${GOAL_OPT_WD:-0.0}"
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
  # SOM-line + the unimodal Poisson manager policy (Zhu et al. 2024). The only
  # difference from som_line is the manager head's parameterization.
  som_line_poisson) ARM_CONFIG="goal_som_line mgr_poisson" ;;
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
  # SOM-line+LiP with the unimodal Poisson manager. Differs from
  # som_lipvq_line_prod by the manager head alone, so it is a direct
  # comparison against e534-e573.
  som_lipvq_line_prod_poisson) ARM_CONFIG="goal_som_lipvq_line_prod mgr_poisson" ;;
  # Same, with the discretized-Gaussian manager instead of the Poisson one.
  som_lipvq_line_prod_gaussian) ARM_CONFIG="goal_som_lipvq_line_prod mgr_gaussian" ;;
  # SOM-line + LiP with RELU (so prod_l softplus(c_l) is an exact Lipschitz
  # bound) + the discretized-Gaussian manager. Straight-through stays ON --
  # goal_som_lipvq_line_prod is the ste:'on' member of the family.
  som_lipvq_line_relu_gaussian) ARM_CONFIG="goal_som_lipvq_line_prod lip_relu mgr_gaussian" ;;
  # Same, plus the 10% eps-greedy index jump in the environment rollout.
  som_lipvq_line_relu_gaussian_eps) ARM_CONFIG="goal_som_lipvq_line_prod lip_relu mgr_gaussian mgr_explore" ;;
  # Student-t (nu=2.5) manager instead of the Gaussian: polynomial tails so the
  # far half of the ordered codebook is both reachable AND still carries
  # direction, which lets the manager WALK the mean there instead of needing
  # the eps-greedy jump to find it. Differs from som_lipvq_line_relu_gaussian
  # by the manager head alone. nu=1 (a true Cauchy) is NOT usable -- its
  # tie-state entropy floor is 0.595, above manager_actent_target. See the
  # mgr_studentt config block.
  som_lipvq_line_relu_studentt)
    ARM_CONFIG="goal_som_lipvq_line_prod lip_relu mgr_studentt"
    # nu=1's tie-state entropy floor is 0.601, so this arm CANNOT be regulated
    # at the default 0.5 -- the adapter rails as in e479. The mgr_studentt
    # config block sets 0.7, but AGENT_FLAGS below passes the target on the CLI
    # and a CLI flag beats a config block (same trap as TRAIN_RATIO above), so
    # the arm has to state it here too or the block is silently overridden.
    ARM_ACTENT_TARGET=0.7 ;;
  # SOM-line + LiP + ReLU with the plain CATEGORICAL manager. This combination
  # had never been run: ReLU only ever shipped bundled with a different manager
  # head (mgr_gaussian in e592-e619, mgr_studentt in e628-e635), so activation
  # and head always moved together. It is both a single-factor ReLU-vs-silu
  # contrast against e534-e573 and the control for som_lipvq_line_relu_rao.
  som_lipvq_line_relu)      ARM_CONFIG="goal_som_lipvq_line_prod lip_relu" ;;
  # The same, plus Rao's quadratic entropy on the skill head held at normalized
  # 0.5 while manager_actent_target stays at 0.5. Differs from
  # som_lipvq_line_relu by that one regularizer, so the pair isolates it.
  som_lipvq_line_relu_rao)  ARM_CONFIG="goal_som_lipvq_line_prod lip_relu mgr_rao" ;;
  # SOM-line + LiP + ReLU + categorical manager, with count-based novelty as a
  # THIRD manager reward and critic alongside the reconstruction-error bonus.
  # Differs from som_lipvq_line_relu by that channel alone.
  som_lipvq_line_relu_count) ARM_CONFIG="goal_som_lipvq_line_prod lip_relu mgr_novel" ;;
  # The same, with the frontier (reachability) gate on the novelty bonus.
  som_lipvq_line_relu_countf) ARM_CONFIG="goal_som_lipvq_line_prod lip_relu mgr_novel_frontier" ;;
  som_orig_lipvq_line_prod) ARM_CONFIG="goal_som_orig_lipvq_line_prod" ;;
  *) echo "unknown ARM=$ARM (director|vq|som|lipvq|som_lipvq|som_lipvq_mgr|som_lipvq_smooth|som_line|som_line_poisson|som_lipvq_line|som_lipvq_line_smooth|som_lipvq_line_mgr|som_orig_line|som_orig_lipvq_line|som_orig_line_adapt|som_orig_lipvq_line_paper|lipvq_prod|som_lipvq_line_prod|som_lipvq_line_prod_poisson|som_lipvq_line_prod_gaussian|som_lipvq_line_relu_gaussian|som_lipvq_line_relu_gaussian_eps|som_lipvq_line_relu_studentt|som_lipvq_line_relu|som_lipvq_line_relu_rao|som_lipvq_line_relu_count|som_lipvq_line_relu_countf|som_orig_lipvq_line_prod)" >&2; exit 1 ;;
esac
# EXTRA_CONFIGS is appended last so it wins over the arm block. Needed for
# loconav/antmaze, which carries its own env repeat and train ratio; without it
# an antmaze task would silently run with the DMC settings.
CONFIGS="director_match${ARM_CONFIG:+ $ARM_CONFIG}${EXTRA_CONFIGS:+ $EXTRA_CONFIGS}"
# Manager entropy setpoint. Defaults to whatever the ARM needs (0.5 for every
# arm whose head can reach it), overridable from the environment. It has to be
# a variable for the same reason TRAIN_RATIO does: AGENT_FLAGS puts it on the
# command line, which beats any config block that tries to set it.
MGR_ACTENT_TARGET="${MGR_ACTENT_TARGET:-${ARM_ACTENT_TARGET:-0.5}}"

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
SIZE=BIG IMAGE=64x64 IMAG=$IMAG_LENGTH BATCH=16x64 TRAIN_RATIO=$TRAIN_RATIO CONV=native
REPLAY_SIZE=$REPLAY_SIZE
COMMIT_SCALE=${COMMIT_SCALE:-default} SOM_SCALE=${SOM_SCALE:-default}
RECIPE=director MGR_FREQ=$MGR_FREQ MGR_REWARD_AGG=$MGR_REWARD_AGG MGR_EXPL_W=$MGR_EXPL_W
MGR_ACTENT_TARGET=$MGR_ACTENT_TARGET
STRUCT_W=0.0 STRUCT_ADAPT=False GOAL_SOFT_REUSE_ADAPT=False GOAL_STRUCT_DIAG=$GOAL_STRUCT_DIAG
EXTRA_CONFIGS=${EXTRA_CONFIGS:-} GOAL_OPT_WD=$GOAL_OPT_WD
PREALLOC=$PREALLOC LOG_OUTPUTS=$LOG_OUTPUTS LOG_VIDEO=$LOG_VIDEO RUN_DEBUG=$RUN_DEBUG
CODE=$REPO GIT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
NOTE=goal-AE ablation vs e390-e409 pure-Director baselines (identical except the goal AE)
EOF

AGENT_FLAGS=(
  --agent.goal_autoencoder_beta 0.25
  --agent.goal_rec_loss_agg sum
  --agent.goal_opt.lr 4e-5
  --agent.goal_opt.wd "$GOAL_OPT_WD"
  --agent.goal_opt.eps 1e-20
  --agent.manager_actent_impl mult
  --agent.manager_actent_target "$MGR_ACTENT_TARGET"
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
# Rao setpoint override for the *_rao arms. Only appended when set, so the
# mgr_rao config block's 0.5 stands by default -- appending it unconditionally
# would put it on the command line, where it would silently beat the block for
# every arm including the ones that do not enable manager_rao at all.
if [ -n "${MGR_RAO_TARGET:-}" ]; then
  AGENT_FLAGS+=(--agent.manager_rao_target "$MGR_RAO_TARGET")
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
    --run.train_ratio "$TRAIN_RATIO" \
    --run.log_video "$LOG_VIDEO" \
    --run.debug "$RUN_DEBUG" \
    --replay.size "$REPLAY_SIZE" \
    --logger.outputs "$LOG_OUTPUTS" \
    --logger.wandb_mode online \
    --logger.wandb_entity vasilache \
    --jax.platform cuda \
  --jax.train_devices ${DEVLIST} \
  --jax.policy_devices 0 \
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
