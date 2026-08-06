#!/bin/bash
# Discovers all currently-finished pure-Director baseline seeds (4 tasks x
# however many of seeds 0-4 have completed training, see baselines_common.py)
# and submits one SLURM array task per (task, seed) pair. Re-run this any
# time -- as seeds 2-4 (e398-e409) finish, they get picked up automatically
# and the array just grows, no manual edits.
set -euo pipefail

SCRIPT_DIR=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3/experiments/goal_code_struct_corr
EXPERIMENTS_DIR=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3/experiments
MANIFEST="$SCRIPT_DIR/manifest.tsv"

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh >/dev/null 2>&1 || true

python3 - "$EXPERIMENTS_DIR" "$MANIFEST" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
from baselines_common import discover_baselines, TASKS

short = {
    'dmc_cartpole_swingup': 'cartpole',
    'dmc_hopper_hop': 'hopper',
    'dmc_acrobot_swingup': 'acrobot',
    'dmc_cheetah_run': 'cheetah',
}
found = discover_baselines()
n = 0
with open(sys.argv[2], 'w') as f:
  for task in TASKS:
    for seed, run_dir, tag in found[task]:
      f.write(f'{task}\t{short[task]}\t{seed}\t{run_dir}\t{tag}\n')
      n += 1
print(f'wrote {n} manifest rows to {sys.argv[2]}')
if n == 0:
  raise SystemExit('no complete baseline checkpoints found -- nothing to submit')
EOF

N=$(wc -l < "$MANIFEST")
echo "submitting array of $N tasks (one per task x seed)"
sbatch --array=0-$((N - 1)) --export=ALL,MANIFEST="$MANIFEST" \
    "$SCRIPT_DIR/run_diag_goal_struct_corr.sbatch"
