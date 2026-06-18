#!/bin/bash -l
#SBATCH -J e42_cartpole
#SBATCH -p gpu-v100
#SBATCH -t 0-24:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir e42_cartpole_fixedk8_plain_struct)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

export MUJOCO_GL=egl

# V100: full report() JIT precompile segfaults (PTX 12.9 vs driver CUDA 12.8).
# Disable report panels; scalars still log via jsonl/scores/wandb.
V100_AGENT_FLAGS=(
  --agent.report False
)

# e42 = FIXED K=8 plain-goals control matched to e41's struct/AE recipe.
# Isolates the variable-K training graph from duration sampling: same goal_struct_weight,
# goal-AE knobs, and cartpole env as e41, but variable_goal_length=False and
# manager_sample_freq=8 (Director fixed window). imag_length defaults to 16 (exp1 parity);
# set IMAG_LENGTH=32 for e41-length imagination horizon parity.
# Optional: resume from e41 peak (~1.33M) via RUN_FROM_CKPT (warm-start world model only;
# manager duration head is absent in fixed-K mode).
MGR_FREQ="${MGR_FREQ:-8}"
STRUCT_W="${STRUCT_W:-200.0}"
IMAG_LENGTH="${IMAG_LENGTH:-16}"
RUN_STEPS="${RUN_STEPS:-4000000}"
RUN_FROM_CKPT="${RUN_FROM_CKPT:-}"

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
  --agent.variable_goal_length False
  --agent.manager_sample_freq "$MGR_FREQ"
  --agent.imag_length "$IMAG_LENGTH"
  "${V100_AGENT_FLAGS[@]}"
)

EXTRA_FLAGS=()
if [[ -n "$RUN_FROM_CKPT" ]]; then
  EXTRA_FLAGS+=(--run.from_checkpoint "$RUN_FROM_CKPT")
fi

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs size6m \
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
  "${AGENT_FLAGS[@]}" \
  "${EXTRA_FLAGS[@]}"

exit $?
