#!/bin/bash -l
#SBATCH -J smoke_a100fit
#SBATCH -p gpu-a100
#SBATCH -t 0-00:40:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Fit-test on A100 (native bf16 conv works here, unlike V100): can ONE A100 hold
# the EXACT e117 model (director_match, 64x64) at the target batch/len/imag?
#   BATCH (default 16), LEN (batch_length, default 64), IMAG (default 16)
# Reports GPU total memory so we know 40 vs 80 GB.
set -uo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
unset DREAMERV3_CONV_IMPL   # native cuDNN conv (works on A100 sm_80, memory-efficient)
echo "[a100fit] GPU:"; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

BATCH="${BATCH:-16}"
LEN="${LEN:-64}"
IMAG="${IMAG:-16}"
DURMAX="${DURMAX:-$IMAG}"
TAG="${TAG:-a100_b${BATCH}_T${LEN}_L${IMAG}}"

RUN_SUFFIX="smoke_a100fit_cheetah_${TAG}"
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir "$RUN_SUFFIX")}"
mkdir -p "$RUN_DIR"
export MUJOCO_GL=egl
cd "$RUN_DIR"
echo "[a100fit] NODE=$(hostname) BATCH=$BATCH LEN=$LEN IMAG=$IMAG DURMAX=$DURMAX RUN_DIR=$RUN_DIR"

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
  "${AGENT_FLAGS[@]}"
rc=$?
echo "[a100fit] EXIT=$rc BATCH=$BATCH LEN=$LEN IMAG=$IMAG TAG=$TAG"
exit $rc
