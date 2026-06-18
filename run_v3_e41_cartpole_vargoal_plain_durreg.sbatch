#!/bin/bash -l
#SBATCH -J e41_cartpole
#SBATCH -p gpu-a100
#SBATCH -t 0-24:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir e41_cartpole_vargoal_plain_durreg)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

export MUJOCO_GL=egl

# e41 = e24 recipe (PLAIN goals + variable goal length) + a soft DURATION PRIOR
# pulling the manager's expected goal duration toward ~8. Motivation: the variable
# temporal goal (variable K) degraded score vs fixed K=8; this isolates whether the
# problem is durations drifting away from the good K=8 regime. "Fixed goal mask" =
# plain goals (no masking, whole code replaced each switch, no spatial variability);
# only the duration varies, and it is regularized toward goal_duration_target=8 via
# goal_duration_reg * (E[duration] - 8)^2 added to the manager policy loss at each
# decision step (direct pull on the duration logits, independent of REINFORCE).
DUR_MIN="${DUR_MIN:-1}"
DUR_MAX="${DUR_MAX:-16}"
STRUCT_W="${STRUCT_W:-200.0}"
DUR_REG="${DUR_REG:-0.1}"
DUR_TARGET="${DUR_TARGET:-8.0}"
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
  --agent.goal_struct_weight "$STRUCT_W"
  --agent.variable_goal_length True
  --agent.goal_duration_min "$DUR_MIN"
  --agent.goal_duration_max "$DUR_MAX"
  --agent.goal_duration_reg "$DUR_REG"
  --agent.goal_duration_target "$DUR_TARGET"
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size6m variable_goals \
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
