#!/bin/bash -l
#SBATCH -J smoke_maskviz_cpu
#SBATCH -p intel
#SBATCH -t 0-01:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

MAIN=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py
BASE="${BASE:-$(dreamerv3_mktemp_run_dir smoke_maskviz_cpu)}"
mkdir -p "$BASE"
echo "BASE=$BASE"

# CPU-only variant (intel partition, no GPU) of run_smoke_maskviz.sbatch so it does
# not compete with the e26-e32 GPU runs for the GPU quota. Validates the sticky
# last_edit_mask change: masked goals + variable duration through policy() over a
# full cartpole episode, so the carry round-trips last_edit_mask and the 4-row
# mask_viz panel is emitted without shape/tree errors.
python -u "$MAIN" --logdir "$BASE/maskviz" \
  --configs size6m debug masked_goals \
  --task dmc_cartpole_swingup \
  --env.dmc.image True --env.dmc.proprio True --env.dmc.size 32 32 \
  --agent.enc.simple.outer True --agent.dec.simple.outer True \
  --run.steps 1100 \
  --agent.imag_length 16 \
  --agent.variable_goal_length True \
  --agent.goal_duration_min 1 --agent.goal_duration_max 6 \
  --agent.mask_sparsity_mode prob --agent.mask_sparsity_target 0.3 \
  --agent.mgr_cond_goalcode True \
  --logger.outputs jsonl \
  --jax.platform cpu

echo "RUN OK; scanning metrics for mask_viz key"
grep -ao "policy_mask_viz[a-z_]*" "$BASE/maskviz/metrics.jsonl" | sort -u || true
echo "SMOKE MASKVIZ DONE"
exit 0
