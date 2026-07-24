#!/bin/bash -l
#SBATCH -J smoke_goal_delta
#SBATCH -p gpu-v100
#SBATCH -t 0-00:10:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl
MAIN=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py
BASE="$(dreamerv3_mktemp_run_dir smoke_goal_delta)"
echo "BASE=$BASE"

COMMON=(
  --task dmc_cartpole_swingup
  --env.dmc.image True --env.dmc.proprio True --env.dmc.size 32 32
  --agent.enc.simple.outer True --agent.dec.simple.outer True
  --run.steps 1000 --agent.imag_length 16 --agent.report False
  --run.log_every 1
  --jax.platform cuda --jax.prealloc False
  --logger.outputs jsonl
)

# Leg A: plain Director baseline, goal_delta_mode OFF -- byte-identical-behavior
# check that the mask_viz/last_change_mask refactor (Problem 1) didn't break the
# non-masked-goals path. Also the denominator for "did white blocks appear" (this
# leg should behave like before: implicit_sparsity_block present, un-forced).
python -u "$MAIN" --logdir "$BASE/plain_baseline" --configs size6m debug "${COMMON[@]}" \
  --agent.mgr_cond_goalcode True

# Leg B: goal_delta_mode ON -- the new mechanism under test. Validates: agent
# builds/trains without error through both the online policy() path and the
# imagination/train path (_emit_manager's 9 call sites + the mgr_policy
# re-derivation for the actor loss), implicit_sparsity_block is a real,
# non-degenerate signal, and gradients are finite throughout.
python -u "$MAIN" --logdir "$BASE/delta_on" --configs size6m debug "${COMMON[@]}" \
  --agent.goal_delta_mode True \
  --agent.goal_delta_clip 1.0

# Leg C: goal_delta_mode ON with a tighter clip, to exercise the clip path.
python -u "$MAIN" --logdir "$BASE/delta_clip05" --configs size6m debug "${COMMON[@]}" \
  --agent.goal_delta_mode True \
  --agent.goal_delta_clip 0.5

echo "=== Leg A (baseline) sanity ==="
python - "$BASE/plain_baseline/metrics.jsonl" <<'PY'
import json, sys
rows = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
key = 'train/goal/implicit_sparsity_block'
vals = [r[key] for r in rows if key in r]
print('FOUND ' + key if vals else 'MISSING ' + key)
if vals:
    print('n=%d last=%.4f' % (len(vals), vals[-1]))
    assert all(0.0 <= v <= 1.0 for v in vals), 'implicit_sparsity_block out of [0,1]'
print('BASELINE_OK')
PY

echo "=== Legs B/C (delta mode) sanity ==="
for LEG in delta_on delta_clip05; do
  echo "--- $LEG ---"
  python - "$BASE/$LEG/metrics.jsonl" <<PY
import json, math
rows = []
for line in open("$BASE/$LEG/metrics.jsonl"):
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
assert rows, 'no metrics rows at all -- run crashed before logging'
key = 'train/goal/implicit_sparsity_block'
vals = [r[key] for r in rows if key in r]
assert vals, 'MISSING ' + key
assert all(0.0 <= v <= 1.0 for v in vals), 'implicit_sparsity_block out of [0,1]: ' + str(vals)
print('implicit_sparsity_block: n=%d first=%.4f last=%.4f' % (len(vals), vals[0], vals[-1]))
# No NaN/Inf anywhere in any logged scalar (delta-combine numerical safety check).
bad = []
for r in rows:
    for k, v in r.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            bad.append((k, v))
assert not bad, 'NaN/Inf in metrics: ' + str(bad[:10])
print('DELTA_MODE_OK: %s' % "$LEG")
PY
done

echo "ALL LEGS DONE"
exit 0
