#!/bin/bash -l
#SBATCH -J smoke_durmods
#SBATCH -p gpu-v100
#SBATCH -t 0-00:25:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl
MAIN=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py
BASE="$(dreamerv3_mktemp_run_dir smoke_durmods)"
echo "BASE=$BASE"

COMMON=(
  --task dmc_cartpole_swingup
  --env.dmc.image True --env.dmc.proprio True --env.dmc.size 32 32
  --agent.enc.simple.outer True --agent.dec.simple.outer True
  --run.steps 1000 --agent.imag_length 16 --agent.report False
  --jax.platform cuda --jax.prealloc False
  --logger.outputs jsonl
)

# Leg A: variable-K plain — validates the new duration std/min/max/hist metrics.
python -u "$MAIN" --logdir "$BASE/var_plain" --configs size6m debug "${COMMON[@]}" \
  --agent.variable_goal_length True \
  --agent.goal_duration_min 1 --agent.goal_duration_max 8 \
  --agent.goal_duration_reg 0.01

# Leg B: adaptive duration prior — validates the AutoAdapt wiring + capped scale.
python -u "$MAIN" --logdir "$BASE/var_adapt" --configs size6m debug "${COMMON[@]}" \
  --agent.variable_goal_length True \
  --agent.goal_duration_min 1 --agent.goal_duration_max 8 \
  --agent.goal_duration_adapt True --agent.goal_duration_adapt_max 0.05

# Leg C: fixed-duration-through-variable-graph — validates goal_duration_fixed.
python -u "$MAIN" --logdir "$BASE/var_fixed4" --configs size6m debug "${COMMON[@]}" \
  --agent.variable_goal_length True \
  --agent.goal_duration_min 1 --agent.goal_duration_max 8 \
  --agent.goal_duration_fixed 4

echo "=== metric keys present (var_plain) ==="
python - "$BASE/var_plain/metrics.jsonl" <<'PY'
import json, sys
keys=set()
for line in open(sys.argv[1]):
    line=line.strip()
    if line:
        try: keys|=set(json.loads(line).keys())
        except: pass
want=['train/goal/mgr_duration_std','train/goal/mgr_duration_min','train/goal/mgr_duration_max',
      'train/goal/mgr_duration_hist_p1_4','train/goal/mgr_duration_hist_p5_8']
for k in want:
    print(('FOUND ' if k in keys else 'MISSING ')+k)
PY
echo "=== adapter scale (var_adapt) ==="
python - "$BASE/var_adapt/metrics.jsonl" <<'PY'
import json, sys
last={}
for line in open(sys.argv[1]):
    line=line.strip()
    if line:
        try: last.update(json.loads(line))
        except: pass
for k,v in last.items():
    if 'duration_adapt' in k: print(k,'=',v)
PY
echo "ALL LEGS DONE"
exit 0
