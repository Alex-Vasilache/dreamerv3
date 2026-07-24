#!/bin/bash -l
#SBATCH -J smoke_goal_reuse
#SBATCH -p gpu-v100
#SBATCH -t 0-00:15:00
#SBATCH -c 4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -euo pipefail
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl
MAIN=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3/dreamerv3/main.py
BASE="$(dreamerv3_mktemp_run_dir smoke_goal_reuse)"
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

# Leg A: baseline, goal_reuse off entirely -- byte-identical-path check.
python -u "$MAIN" --logdir "$BASE/baseline" --configs size6m debug "${COMMON[@]}" \
  --agent.mgr_cond_goalcode True

# Leg B: goal_reuse_weight ON (fixed-weight ablation mode), plain Director.
python -u "$MAIN" --logdir "$BASE/reuse_fixed" --configs size6m debug "${COMMON[@]}" \
  --agent.goal_reuse_weight 1.0

# Leg C: goal_reuse_adapt ON (the recommended adaptive Lagrangian mode),
# target similarity 0.9, plain Director.
python -u "$MAIN" --logdir "$BASE/reuse_adapt" --configs size6m debug "${COMMON[@]}" \
  --agent.goal_reuse_adapt True \
  --agent.goal_reuse_target 0.9

# Leg D: goal_reuse_adapt + goal_delta_mode together -- compatibility check
# (different levels: delta mode is a sampling mechanism, goal_reuse is a loss).
python -u "$MAIN" --logdir "$BASE/reuse_adapt_and_delta" --configs size6m debug "${COMMON[@]}" \
  --agent.goal_reuse_adapt True \
  --agent.goal_reuse_target 0.9 \
  --agent.goal_delta_mode True

echo "=== Leg A (baseline) sanity ==="
python - "$BASE/baseline/metrics.jsonl" <<PY
import json
rows = []
for line in open("$BASE/baseline/metrics.jsonl"):
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
assert rows, 'no metrics rows -- run crashed before logging'
assert not any('goal_reuse' in k for r in rows for k in r), 'goal_reuse key leaked into baseline (should be off)'
goal_grad = [r['train/opt/goal_grad_norm'] for r in rows if 'train/opt/goal_grad_norm' in r]
print('BASELINE_OK: n_rows=%d, no goal_reuse keys present' % len(rows))
if goal_grad:
    print('  opt/goal_grad_norm (baseline, no goal_reuse): last=%.6f' % goal_grad[-1])
PY

echo "=== Legs B/C/D (goal_reuse) sanity ==="
for LEG in reuse_fixed reuse_adapt reuse_adapt_and_delta; do
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
sim_key = 'train/goal/reuse_sim_mean'
loss_key = 'train/loss/goal_reuse'
sims = [r[sim_key] for r in rows if sim_key in r]
losses = [r[loss_key] for r in rows if loss_key in r]
assert sims, 'MISSING ' + sim_key
assert losses, 'MISSING ' + loss_key
assert all(-1.0001 <= v <= 1.0001 for v in sims), 'reuse_sim_mean out of [-1,1]: ' + str(sims)
print('reuse_sim_mean: n=%d first=%.4f last=%.4f' % (len(sims), sims[0], sims[-1]))
print('loss/goal_reuse: n=%d first=%.4f last=%.4f' % (len(losses), losses[0], losses[-1]))
scale_key = 'train/goal/reuse_adapt_scale_mean'
scales = [r[scale_key] for r in rows if scale_key in r]
if "$LEG" != "reuse_fixed":
    assert scales, 'MISSING ' + scale_key + ' (adapt mode should log adapter scale)'
    assert all(1e-5 <= v <= 100.0 for v in scales), 'adapter scale out of [min,max]: ' + str(scales)
    print('reuse_adapt_scale_mean: n=%d first=%.4f last=%.4f' % (len(scales), scales[0], scales[-1]))
goal_grad = [r['train/opt/goal_grad_norm'] for r in rows if 'train/opt/goal_grad_norm' in r]
if goal_grad:
    print('  opt/goal_grad_norm (with goal_reuse on): last=%.6f (compare to baseline printed above -- ' \
          'should be similar order of magnitude, not systematically larger, if the decoder-gradient ' \
          'block is working)' % goal_grad[-1])
bad = []
for r in rows:
    for k, v in r.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            bad.append((k, v))
assert not bad, 'NaN/Inf in metrics: ' + str(bad[:10])
print('GOAL_REUSE_OK: $LEG')
PY
done

echo "ALL LEGS DONE"
exit 0
