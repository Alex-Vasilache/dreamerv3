#!/bin/bash -l
#SBATCH -J smoke_soft_reuse
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
BASE="$(dreamerv3_mktemp_run_dir smoke_soft_reuse)"
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

# Leg A: plain Director baseline, goal_soft_reuse_adapt OFF -- regression check
# that broadening the goal_delta_mode-only skill_probs plumbing (carry init,
# _emit_manager, _mgr_input) to also fire for goal_soft_reuse_adapt didn't
# change anything on the existing (off) path.
python -u "$MAIN" --logdir "$BASE/plain_baseline" --configs size6m debug "${COMMON[@]}" \
  --agent.mgr_cond_goalcode True

# Leg B: goal_soft_reuse_adapt ON, fixed target (init == target, no ratchet).
# Validates: agent builds/trains without error through both the online policy()
# path (skill_probs carried + fed as soft conditioning) and the imagination/train
# path (new preedit_eff['skill_probs'] branch at all 3 sites, the direct overlap
# loss), goal/soft_reuse_overlap_mean is a real signal bounded in [0,1] (Cauchy-
# Schwarz on two probability vectors), and gradients are finite throughout.
python -u "$MAIN" --logdir "$BASE/soft_reuse_direct" --configs size6m debug "${COMMON[@]}" \
  --agent.mgr_cond_goalcode True \
  --agent.goal_soft_reuse_adapt True \
  --agent.goal_soft_reuse_target 0.7

# Leg C: goal_soft_reuse_adapt ON with the target ratcheted (fast vel so it
# visibly moves within a 1000-step smoke run), mirroring how
# run_smoke_goal_reuse_ratchet.sbatch validated the analogous goal_reuse ramp.
python -u "$MAIN" --logdir "$BASE/soft_reuse_ratchet" --configs size6m debug "${COMMON[@]}" \
  --agent.mgr_cond_goalcode True \
  --agent.goal_soft_reuse_adapt True \
  --agent.goal_soft_reuse_target 0.7 \
  --agent.goal_soft_reuse_target_init 0.0 \
  --agent.goal_soft_reuse_target_vel 0.05

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
assert rows, 'no metrics rows at all -- run crashed before logging'
key = 'train/goal/implicit_sparsity_block'
vals = [r[key] for r in rows if key in r]
print('FOUND ' + key if vals else 'MISSING ' + key)
if vals:
    print('n=%d last=%.4f' % (len(vals), vals[-1]))
    assert all(0.0 <= v <= 1.0 for v in vals), 'implicit_sparsity_block out of [0,1]'
# Must NOT have any soft_reuse keys on this leg (mechanism fully off).
bad_keys = [k for r in rows for k in r if 'soft_reuse' in k]
assert not bad_keys, 'soft_reuse metrics leaked into the OFF leg: ' + str(set(bad_keys))
print('BASELINE_OK')
PY

echo "=== Legs B/C (soft reuse) sanity ==="
for LEG in soft_reuse_direct soft_reuse_ratchet; do
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
key = 'train/goal/soft_reuse_overlap_mean'
vals = [r[key] for r in rows if key in r]
assert vals, 'MISSING ' + key
assert all(-1e-4 <= v <= 1.0 + 1e-4 for v in vals), 'overlap out of [0,1]: ' + str(vals)
print('soft_reuse_overlap_mean: n=%d first=%.4f last=%.4f' % (len(vals), vals[0], vals[-1]))
tkey = 'train/goal/soft_reuse_target_now'
tvals = [r[tkey] for r in rows if tkey in r]
assert tvals, 'MISSING ' + tkey
print('soft_reuse_target_now: first=%.4f last=%.4f' % (tvals[0], tvals[-1]))
skey = 'train/goal/soft_reuse_scale_mean'
svals = [r[skey] for r in rows if skey in r]
assert svals, 'MISSING ' + skey
print('soft_reuse_scale_mean: first=%.4f last=%.4f' % (svals[0], svals[-1]))
# No NaN/Inf anywhere in any logged scalar.
bad = []
for r in rows:
    for k, v in r.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            bad.append((k, v))
assert not bad, 'NaN/Inf in metrics: ' + str(bad[:10])
print('SOFT_REUSE_OK: %s' % "$LEG")
PY
done

echo "=== Leg C ratchet-specific check ==="
python - <<PY
import json
rows = []
for line in open("$BASE/soft_reuse_ratchet/metrics.jsonl"):
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
key = 'train/goal/soft_reuse_target_now'
vals = [(r['step'], r[key]) for r in rows if key in r]
steps, targets = zip(*vals)
assert targets[0] < 0.3, 'ratchet did not start near 0.0: ' + str(targets[0])
assert targets[-1] > targets[0], 'ratchet did not move upward'
assert targets[-1] <= 0.7 + 1e-6, 'ratchet overshot final target 0.7: ' + str(targets[-1])
assert all(b >= a - 1e-6 for a, b in zip(targets, targets[1:])), 'ratchet moved downward at some point'
print('RATCHET_OK')
PY

echo "ALL LEGS DONE"
exit 0
