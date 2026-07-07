#!/bin/bash -l
#SBATCH -J smoke_nativeconv
#SBATCH -p gpu-a100
#SBATCH -t 0-00:40:00
#SBATCH -c 16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Smoke test: does NATIVE cuDNN conv work on the A100 (el8) for the e115 model?
# source_dreamerv3_env.sh forces DREAMERV3_CONV_IMPL=reference (einsum fallback,
# slow). Here we UNSET it so nets.py uses jax.lax.conv_general_dilated. Same
# director_match + 64x64 conv shapes as e115/e116, just a short run to check:
#   (1) does it import/run without segfault, and
#   (2) is train FPS higher than the reference impl?
# CONV_MODE=reference sbatch <this>   # to run the reference baseline for comparison
set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

CONV_MODE="${CONV_MODE:-native}"
if [ "$CONV_MODE" = "native" ]; then
  unset DREAMERV3_CONV_IMPL
  echo "[smoke] CONV: native jax.lax.conv_general_dilated (DREAMERV3_CONV_IMPL unset)"
else
  export DREAMERV3_CONV_IMPL=reference
  echo "[smoke] CONV: reference einsum fallback (DREAMERV3_CONV_IMPL=reference)"
fi

RUN_SUFFIX="smoke_${CONV_MODE}conv"
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir "$RUN_SUFFIX")}"
mkdir -p "$RUN_DIR"
export MUJOCO_GL=egl
cd "$RUN_DIR"

# Same manager recipe as e116 but short + no wandb (jsonl/scope only) so it's a
# pure throughput/does-it-crash probe.
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
  --task dmc_cartpole_swingup \
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
  --run.steps 30000 \
  --run.train_ratio 8 \
  --run.log_every 30 \
  --logger.outputs jsonl,scope \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}"

echo "[smoke] exit=$? conv=$CONV_MODE run_dir=$RUN_DIR"
