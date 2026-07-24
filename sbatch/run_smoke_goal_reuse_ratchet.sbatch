#!/bin/bash -l
#SBATCH -J smoke_reuse_ratchet
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
BASE="$(dreamerv3_mktemp_run_dir smoke_reuse_ratchet)"
echo "BASE=$BASE"

# Fast vel (0.05/call, vs the real launch's ~4e-6/1.5e-5) so the ratchet
# visibly moves within a 1000-step smoke run -- validates AutoAdapt's
# per-call target= override + the Ratchet schedule wiring, mirroring how
# run_smoke_mask_ratchet.sbatch validated the earlier mask-target ratchet.
python -u "$MAIN" --logdir "$BASE/ratchet" --configs size6m debug \
  --task dmc_cartpole_swingup \
  --env.dmc.image True --env.dmc.proprio True --env.dmc.size 32 32 \
  --agent.enc.simple.outer True --agent.dec.simple.outer True \
  --run.steps 1000 --agent.imag_length 16 --agent.report False \
  --run.log_every 1 \
  --jax.platform cuda --jax.prealloc False \
  --logger.outputs jsonl \
  --agent.goal_reuse_adapt True \
  --agent.goal_reuse_target 0.95 \
  --agent.goal_reuse_target_init 0.0 \
  --agent.goal_reuse_target_vel 0.05

echo "=== ratchet trajectory ==="
python - "$BASE/ratchet/metrics.jsonl" <<PY
import json
rows = []
for line in open("$BASE/ratchet/metrics.jsonl"):
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
key = 'train/goal/reuse_target_now'
vals = [(r['step'], r[key]) for r in rows if key in r]
print('FOUND ' + key if vals else 'MISSING ' + key)
if not vals:
    raise SystemExit(1)
steps, targets = zip(*vals)
print('n=%d first=%.4f last=%.4f' % (len(targets), targets[0], targets[-1]))
assert targets[0] < 0.3, 'ratchet did not start near 0.0: ' + str(targets[0])
assert targets[-1] > targets[0], 'ratchet did not move upward'
assert targets[-1] <= 0.95 + 1e-6, 'ratchet overshot final target 0.95: ' + str(targets[-1])
# Monotonically non-decreasing (open-loop ramp, no feedback).
assert all(b >= a - 1e-6 for a, b in zip(targets, targets[1:])), 'ratchet moved downward at some point'
print('RATCHET_OK')
PY

echo "ALL LEGS DONE"
exit 0
