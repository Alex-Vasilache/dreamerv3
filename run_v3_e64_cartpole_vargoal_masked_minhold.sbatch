#!/bin/bash -l
#SBATCH -J e64_cartpole
#SBATCH -p gpu-v100
#SBATCH -t 2-00:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir e64_cartpole_vargoal_masked_minhold)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

export MUJOCO_GL=egl

# e64 = e54 (masked goals, variable duration, NO duration prior, B2 prob sparsity 0.3,
# struct=200, mgr_cond_goalcode) + MINIMUM-HOLD FLOOR: goal_duration_min=4 (range [4,16]
# instead of [1,16]). Architectural constraint on the action space, NOT a reward prior —
# the duration head can't choose degenerate short churn (<4) that caused e25/e54 to drift
# to ~2.3. The duration head sampling already clamps via p = goal_duration_min + idx, so
# no code change. Fresh scratch.
DUR_MIN="${DUR_MIN:-4}"
DUR_MAX="${DUR_MAX:-16}"
STRUCT_W="${STRUCT_W:-200.0}"
RUN_STEPS="${RUN_STEPS:-4000000}"

cd "$RUN_DIR"
AGENT_FLAGS=(
  --agent.goal_autoencoder_beta 0.25
  --agent.goal_rec_loss_agg sum
  --agent.goal_opt.lr 4e-5
  --agent.goal_opt.wd 0.0
  --agent.goal_opt.eps 1e-20
  --agent.manager_actent_impl mult
  --agent.manager_actent_target 0.5
  --agent.mask_topk 0
  --agent.goal_struct_weight "$STRUCT_W"
  --agent.mask_sparsity_mode prob
  --agent.mask_sparsity_max 5.0
  --agent.mask_sparsity_target 0.3
  --agent.mgr_cond_goalcode True
  --agent.mgr_cond_achieve False
  --agent.variable_goal_length True
  --agent.goal_duration_min "$DUR_MIN"
  --agent.goal_duration_max "$DUR_MAX"
  --agent.goal_duration_reg 0.0
  --agent.goal_duration_adapt False
  --agent.report False
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size6m masked_goals variable_goals \
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
  --run.steps "$RUN_STEPS" \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}"

exit $?
