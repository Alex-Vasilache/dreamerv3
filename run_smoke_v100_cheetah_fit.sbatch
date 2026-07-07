#!/bin/bash -l
#SBATCH -J smoke_v100fit
#SBATCH -p gpu-v100
#SBATCH -t 0-00:35:00
#SBATCH -c 16
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Fit-test: can the EXACT e117 model (director_match ~77M params, 64x64) run on a
# 16GB V100 for cheetah_run, using only batch_size / imag_length as levers?
#   BATCH   (batch_size,  default 4)  -- halving ~halves world-model activations
#   IMAG    (imag_length, default 32) -- halving ~halves imagination activations
#   DURMAX  (goal_duration_max, default = IMAG) -- keep a full manager decision
#           inside one imagined rollout so per-block credit stays valid.
# Image is FIXED 64x64 (same model). Reference conv is KEPT (native cuDNN conv
# fails on V100 cc7.0 with "cudnn status 5003" / giant workspace OOM).
set -uo pipefail   # no -e: capture OOM exit and still print summary

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
# CONV=native  -> unset DREAMERV3_CONV_IMPL (memory-efficient cuDNN conv). On V100
#   this needs DTYPE=float16 (V100 sm_70 has fp16 tensor cores but NO bf16 conv
#   engine -> native bf16 conv errors with "cudnn status 5003"). fp16 conv is
#   supported on sm_70.
# CONV=reference -> keep the einsum fallback (works with bf16 but materializes
#   huge patch tensors -> OOMs at 64x64 even at batch 1).
CONV="${CONV:-reference}"
DTYPE="${DTYPE:-bfloat16}"
if [ "$CONV" = "native" ]; then unset DREAMERV3_CONV_IMPL; fi
# EXTRA_XLA: append experimental XLA flags (e.g. disable the cuDNN conv+bias fusion
# that has no working algorithm on V100 sm_70, forcing plain __cudnn$convForward).
export XLA_FLAGS="${XLA_FLAGS:-} ${EXTRA_XLA:-}"
echo "[fit] CONV=$CONV DREAMERV3_CONV_IMPL=${DREAMERV3_CONV_IMPL:-<unset>} DTYPE=$DTYPE"
echo "[fit] XLA_FLAGS=$XLA_FLAGS"

BATCH="${BATCH:-4}"
LEN="${LEN:-32}"
IMAG="${IMAG:-32}"
DURMAX="${DURMAX:-$IMAG}"
TAG="${TAG:-${CONV}_${DTYPE}_b${BATCH}_T${LEN}_L${IMAG}}"

RUN_SUFFIX="smoke_v100fit_cheetah_${TAG}"
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir "$RUN_SUFFIX")}"
mkdir -p "$RUN_DIR"
export MUJOCO_GL=egl
cd "$RUN_DIR"
echo "[fit] NODE=$(hostname) BATCH=$BATCH IMAG=$IMAG DURMAX=$DURMAX RUN_DIR=$RUN_DIR"

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
  --agent.goal_duration_max "$DURMAX"
  --agent.goal_duration_reg 0.0
  --agent.goal_duration_target 4.0
  --agent.goal_duration_adapt False
  --agent.imag_length "$IMAG"
  --agent.report False
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs director_match masked_goals variable_goals \
  --task dmc_cheetah_run \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 64 64 \
  --batch_size "$BATCH" \
  --batch_length "$LEN" \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.mgr_expl_weight 0.1 \
  --agent.mgr_retnorm.impl meanstd \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 8 \
  --run.steps 30000 \
  --run.train_ratio 8 \
  --run.log_every 30 \
  --logger.outputs jsonl,scope \
  --jax.platform cuda \
  --jax.prealloc False \
  --jax.compute_dtype "$DTYPE" \
  "${AGENT_FLAGS[@]}"
rc=$?
echo "[fit] EXIT=$rc BATCH=$BATCH IMAG=$IMAG DURMAX=$DURMAX TAG=$TAG  (rc!=0 + RESOURCE_EXHAUSTED => OOM)"
exit $rc
