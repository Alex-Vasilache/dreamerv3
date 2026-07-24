#!/bin/bash -l
#SBATCH -J hrl_auto
#SBATCH -p gpu-v100
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Auto-tuning HRL panel cell: masked goals + variable-K with EVERY regularizer an
# adaptive Lagrange multiplier on a dimensionless target (config block `hrl_auto`).
# One config, meant to transfer across the env x size panel with no per-cell tuning.
#
#   SIZE  = 6m  (size6m, 32x32, 1 V100 — fast, this is the priority tier)
#         | BIG (director_match, 64x64 — needs A100/multi-V100; see the _big launcher)
#   TASK  = dmc_cartpole_swingup | dmc_cheetah_run | dmc_acrobot_swingup | dmc_hopper_hop | ...
#   EXP_TAG, STEPS (default 4e6), MASK_MODE (prob default; 'entropy' for entropy-native)
set -uo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

SIZE="${SIZE:-6m}"
TASK="${TASK:-dmc_cartpole_swingup}"
EXP_TAG="${EXP_TAG:-eXX}"
STEPS="${STEPS:-4000000}"
MASK_MODE="${MASK_MODE:-prob}"
export MUJOCO_GL=egl

if [ "$SIZE" = "6m" ]; then
  SIZE_CFG=size6m; IMG=32; IMAG=32
else
  SIZE_CFG=director_match; IMG=64; IMAG=16   # BIG: shrink imag to fit; use A100/multi-V100
fi

RUN_SUFFIX="${EXP_TAG}_${TASK}_hrl_auto_${SIZE}"
RUN_DIR="${RUN_DIR:-$(dreamerv3_mktemp_run_dir "$RUN_SUFFIX")}"
mkdir -p "$RUN_DIR"; cd "$RUN_DIR"

cat > "$RUN_DIR/job.env" <<EOF
RUN_DIR=$RUN_DIR
EXP_TAG=$EXP_TAG
CONFIGS=$SIZE_CFG hrl_auto
TASK=$TASK
SIZE=$SIZE IMAGE=${IMG}x${IMG} IMAG=$IMAG
MASK_MODE=$MASK_MODE STEPS=$STEPS
NOTE=auto-tuning HRL (all-adaptive, dimensionless targets; cond off, mean agg, no costs, no K prior)
EOF

# Short smoke: STEPS small (<5000) -> jsonl/scope only, report off.
LOGOUT="jsonl,scope,wandb"; [ "$STEPS" -lt 5000 ] && LOGOUT="jsonl,scope"

python -u /apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py \
  --logdir "$RUN_DIR/logdir" \
  --configs "$SIZE_CFG" hrl_auto \
  --task "$TASK" \
  --env.dmc.image True \
  --env.dmc.proprio True \
  --env.dmc.size "$IMG" "$IMG" \
  --agent.enc.simple.outer True \
  --agent.dec.simple.outer True \
  --agent.imag_length "$IMAG" \
  --agent.mask_sparsity_mode "$MASK_MODE" \
  --agent.report False \
  --online_learning False \
  --run.envs 8 \
  --run.steps "$STEPS" \
  --logger.outputs "$LOGOUT" \
  --logger.wandb_mode online \
  --logger.wandb_entity vasilache \
  --jax.platform cuda \
  --jax.prealloc False
exit $?
