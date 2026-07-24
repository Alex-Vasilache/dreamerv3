#!/bin/bash -l
#SBATCH -J smoke_v100mgpu
#SBATCH -p gpu-v100
#SBATCH -t 0-00:35:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=128G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Fit-test lever C: run the EXACT e117 model (director_match, 64x64, batch 4,
# imag 32) on 4x V100 (16GB each) on ONE node, data-parallel. train_mesh '-1,1,1'
# shards the batch on the 'd' axis across all train_devices, so each GPU holds the
# full ~77M-param model but only batch_size/NGPU worth of activations. With
# batch_size 4 over 4 GPUs -> ~batch 1 per device, which should fit in 16GB.
# Reference conv KEPT (native cuDNN conv is broken on V100 cc7.0).
set -uo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
echo "[mgpu] DREAMERV3_CONV_IMPL=${DREAMERV3_CONV_IMPL:-<unset>} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"

BATCH="${BATCH:-4}"
IMAG="${IMAG:-32}"
NGPU="${NGPU:-4}"
TAG="${TAG:-mgpu${NGPU}_b${BATCH}_L${IMAG}}"

RUN_SUFFIX="smoke_v100mgpu_cheetah_${TAG}"
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir "$RUN_SUFFIX")}"
mkdir -p "$RUN_DIR"
export MUJOCO_GL=egl
cd "$RUN_DIR"
# Build the train_devices list 0..NGPU-1 as space-separated flag args.
DEVLIST=$(seq -s ' ' 0 $((NGPU-1)))
echo "[mgpu] NODE=$(hostname) NGPU=$NGPU DEVLIST='$DEVLIST' BATCH=$BATCH IMAG=$IMAG RUN_DIR=$RUN_DIR"

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
  --agent.goal_duration_max "$IMAG"
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
  --jax.train_devices $DEVLIST \
  --jax.policy_devices 0 \
  "${AGENT_FLAGS[@]}"
rc=$?
echo "[mgpu] EXIT=$rc NGPU=$NGPU BATCH=$BATCH IMAG=$IMAG TAG=$TAG"
exit $rc
