#!/bin/bash -l
#SBATCH -J smoke_goalhold
#SBATCH -p gpu-v100
#SBATCH -t 0-00:45:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

ROOT=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3
MAIN="$ROOT/dreamerv3/main.py"
BASE="${BASE:-$(dreamerv3_mktemp_run_dir smoke_goalhold)}"
mkdir -p "$BASE"
echo "BASE=$BASE"

echo "=================== LEG 1: goal_deter hold invariant (GPU) ==================="
python -u "$ROOT/smoke_goal_hold.py"
echo "LEG 1 OK"

echo "=================== LEG 2: short train + mask_viz (GPU) ==================="
python -u "$MAIN" --logdir "$BASE/maskviz" \
  --configs size6m debug masked_goals \
  --task dmc_cartpole_swingup \
  --env.dmc.image True --env.dmc.proprio True --env.dmc.size 32 32 \
  --agent.enc.simple.outer True --agent.dec.simple.outer True \
  --run.steps 1100 --run.envs 1 \
  --agent.imag_length 16 \
  --agent.variable_goal_length True \
  --agent.goal_duration_min 1 --agent.goal_duration_max 8 \
  --agent.mask_sparsity_mode prob --agent.mask_sparsity_target 0.3 \
  --agent.mgr_cond_goalcode True \
  --logger.outputs jsonl,scope \
  --logger.filter 'score|length|fps|epstats|train/loss/' \
  --jax.platform cuda \
  --jax.prealloc False
echo "LEG 1 train finished; checking epstats mask_viz in scope"
ls "$BASE/maskviz/scope/" 2>/dev/null | grep -i mask_viz || true
grep -ao "policy_mask_viz[a-z_]*" "$BASE/maskviz/metrics.jsonl" 2>/dev/null | sort -u || true
echo "LEG 2 OK"

echo "SMOKE GOAL_HOLD DONE"
exit 0
