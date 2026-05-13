#!/bin/bash -l
#SBATCH -J dreamerv3_smoke
#SBATCH -p gpu
#SBATCH -t 0-0:45:00
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#
# Short offline train to validate the stack after code changes (DreamerV3
# defaults, or DreamerV4-style toggles). Same pattern as run_dreamerv3_gpu.sbatch
# and run_dreamerv3_online.sbatch: source_dreamerv3_env.sh + RUN_DIR on /work.
#
# Submit: sbatch run_dreamerv3_smoke_train.sbatch
# Optional: RUN_DIR=/work/.../my_smoke sbatch run_dreamerv3_smoke_train.sbatch
#
# Toggle blocks below: leave AGENT_FLAGS empty for baseline; uncomment for V4 flags.

set -euo pipefail

mkdir -p /work/DoyaU/vasilache/work/slurm_logs

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir dreamerv3_smoke)}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" > "$RUN_DIR/job.env"

export MUJOCO_GL=egl

cd "$RUN_DIR"

# DreamerV4-style agent flags (elements.Config dotted keys). Empty = baseline.
AGENT_FLAGS=(
  # --agent.use_single_rollout True
  # --agent.use_rms_loss_norm True
  # --agent.imag_loss.use_pmpo_actor True
)

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs defaults debug \
  --task dmc_cartpole_swingup \
  --env.dmc.image False \
  --run.envs 4 \
  --run.steps 3000 \
  --run.log_every 50 \
  --run.report_every 200 \
  --run.save_every 500 \
  --logger.outputs jsonl \
  --jax.platform cuda \
  --jax.prealloc False \
  "${AGENT_FLAGS[@]}"

exit $?
