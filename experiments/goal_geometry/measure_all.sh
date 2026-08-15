#!/bin/bash
# Submit the geometry measurement for every e5xx run that does not have one.
#
# The per-run job is idempotent (it skips runs it has already measured and runs
# that have not trained), so this can be fired at any time and will only do the
# outstanding work. It goes to the V100 lane, which costs no A100 quota and is
# ample: the measurement is a forward pass and takes ~2.5 min a run.
#
#   experiments/goal_geometry/measure_all.sh          # submit
#   DRY_RUN=1 experiments/goal_geometry/measure_all.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# See watchdog_e510_e557.sh: the login profile exports WORK=/work.
WD_RUNS_DIR="${WD_RUNS_DIR:-/work/DoyaU/vasilache/work}"
RESULTS="${RESULTS:-$REPO/experiments/goal_geometry/results}"
MIN_STEPS="${MIN_STEPS:-3900000}"   # only measure finished runs by default
PER_JOB="${PER_JOB:-10}"            # run dirs per submitted job

todo=()
for d in "$WD_RUNS_DIR"/e5*_BIG_j*/; do
  [ -d "$d" ] || continue
  name=$(basename "${d%/}")
  [ -f "$RESULTS/${name}.npz" ] && continue
  steps=$(python3 - "$d" <<'PY'
import json, os, sys
p = os.path.join(sys.argv[1], 'logdir', 'metrics.jsonl')
best = 0
if os.path.exists(p):
    with open(p, 'rb') as f:
        f.seek(0, 2); f.seek(max(0, f.tell() - 200_000))
        for line in f.read().decode('utf8', 'ignore').splitlines()[1:]:
            try:
                best = max(best, int(json.loads(line).get('step', 0)))
            except Exception:
                pass
print(best)
PY
)
  [ "${steps:-0}" -ge "$MIN_STEPS" ] && todo+=("$d")
done

echo "${#todo[@]} run(s) to measure (>= $MIN_STEPS steps, no result yet)"
[ "${#todo[@]}" -eq 0 ] && exit 0

i=0
while [ $i -lt ${#todo[@]} ]; do
  chunk="${todo[@]:$i:$PER_JOB}"
  n=$(echo "$chunk" | wc -w)
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] $n runs: $(for r in $chunk; do basename "${r%/}" | cut -d_ -f1; done | tr '\n' ' ')"
  else
    RUNS="$chunk" N_BATCHES=4 N_ENVS=4 CONV=reference MIN_STEPS="$MIN_STEPS" \
      sbatch -J geom_all -p gpu-v100 --gres=gpu:v100:1 -t 02:00:00 \
        "$REPO/experiments/goal_geometry/run_diag_goal_geometry.sbatch"
  fi
  i=$((i + PER_JOB))
done
