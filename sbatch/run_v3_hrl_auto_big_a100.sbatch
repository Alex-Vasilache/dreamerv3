#!/bin/bash -l
#SBATCH -J e130_big_a100
#SBATCH -p gpu-a100
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Auto-tuning HRL BIG cell on A100-80GB: `hrl_auto` config at director_match 64x64,
# real default batch (b16x64, imag16, train_ratio 64). Single A100 device (b16x64 at
# this scale needs ~59/80 GB — does not fit a 16GB V100 single-device). Native conv.
# TASK default cheetah_run. imag_length forced to 16 (hrl_auto's 32 won't fit BIG).
set -uo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
unset DREAMERV3_CONV_IMPL   # native cuDNN conv (A100 sm_80 ~2x faster)

EXP_TAG="${EXP_TAG:-e130}"
TASK="${TASK:-dmc_cheetah_run}"
MASK_MODE="${MASK_MODE:-prob}"
RUN_SUFFIX="${EXP_TAG}_${TASK}_hrl_auto_BIG"
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir "$RUN_SUFFIX")}"
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"
export MUJOCO_GL=egl

cat > "$RUN_DIR/job.env" <<EOF
RUN_DIR=$RUN_DIR
EXP_TAG=$EXP_TAG
CONFIGS=director_match hrl_auto
TASK=$TASK
SIZE=BIG IMAGE=64x64 IMAG=16 BATCH=16x64 TRAIN_RATIO=64 CONV=native
MASK_MODE=$MASK_MODE
NOTE=auto-tuning HRL BIG (all-adaptive; cond off, mean agg, no costs, no K prior)
EOF

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs director_match hrl_auto \
  --task "$TASK" \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size 64 64 \
  --batch_size 16 \
  --batch_length 64 \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.imag_length 16 \
  --agent.mask_sparsity_mode "$MASK_MODE" \
  --agent.report False \
  --online_learning False \
  --online_actor_cpu False \
  --run.envs 8 \
  --run.steps 4000000 \
  --run.train_ratio 64 \
  --logger.outputs jsonl,scope,wandb \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False
exit $?
