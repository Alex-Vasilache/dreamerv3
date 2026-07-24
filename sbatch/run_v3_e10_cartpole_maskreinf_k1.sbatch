#!/bin/bash -l
#SBATCH -J e10_cartpole
#SBATCH -p gpu-v100
#SBATCH -t 0-24:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir e10_cartpole_maskreinf_k1)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

export MUJOCO_GL=egl

# e10 = e7 setup (soft sparsity, struct=200, K=1) but with mask_sparsity_mode=
# reinforce (C): the realized edit fraction is subtracted from the manager
# extrinsic reward (coeff = adaptive Lagrange scale), so the mask logp is
# reinforced toward sparser commands -- the unbiased estimator for the discrete
# mask. Reward-scale penalty, so the ceiling is small (mask_sparsity_max=1.0,
# init=0.1) to stay comparable to mgr_extr_rew (~0.6).
MASK_TOPK="${MASK_TOPK:-0}"
STRUCT_W="${STRUCT_W:-200.0}"
MGR_FREQ="${MGR_FREQ:-1}"
MASK_MODE="${MASK_MODE:-reinforce}"
MASK_MAX="${MASK_MAX:-1.0}"
MASK_INIT="${MASK_INIT:-0.1}"

cd "$RUN_DIR"
AGENT_FLAGS=(
  --agent.goal_autoencoder_beta 0.25
  --agent.goal_rec_loss_agg sum
  --agent.goal_opt.lr 4e-5
  --agent.goal_opt.wd 0.0
  --agent.goal_opt.eps 1e-20
  --agent.manager_actent_impl mult
  --agent.manager_actent_target 0.5
  --agent.mask_topk "$MASK_TOPK"
  --agent.goal_struct_weight "$STRUCT_W"
  --agent.manager_sample_freq "$MGR_FREQ"
  --agent.mask_sparsity_mode "$MASK_MODE"
  --agent.mask_sparsity_max "$MASK_MAX"
  --agent.mask_sparsity_init "$MASK_INIT"
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size6m masked_goals \
  --task dmc_cartpole_swingup \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 32 32 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.mgr_expl_weight 0.1 \
  --agent.mgr_retnorm.impl meanstd \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 16 \
  --run.steps 4000000 \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}"

exit $?
