#!/bin/bash
# e574-e581: SOM-line (no LiP) on cheetah_run and hopper_hop.
#
#   e574-e577   som_line   dmc_cheetah_run   seeds 0-3   4M
#   e578-e581   som_line   dmc_hopper_hop    seeds 0-3   4M
#
# Extends the SOM-only arm, which so far exists only on cartpole_swingup
# (e510-e513) and hopper_stand (e514-e517), onto the two further tasks
# SOM-line+LiP already covers. Baselines are in place for both: e554-e557 for
# cheetah and e566-e569 for hopper_hop.
#
# Still missing for full parity with SOM-line+LiP: cartpole_swingup_sparse.
#
# Queued on gpu-a100 behind everything already there, with --nice 9000 against
# the 6000/7000/8000 of the e558-e573 groups. The lane gives 8 concurrent GPUs
# and e566-e573 hold all of them, so these start as that batch drains.
#
# Rows are appended to the e558-e573 TSV on purpose: the watchdog re-reads that
# file every pass, so these get the same supervision with no second watchdog.
#
#   DRY_RUN=1 sbatch/submit_e574_e581_som_line_cheetah_hopperhop.sh
#   sbatch/submit_e574_e581_som_line_cheetah_hopperhop.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/sbatch/run_v3_goal_ae_ablation_big_a100.sbatch"
LOG="$REPO/job_logs/e558_e573_hopperhop_sparse_cheetah.tsv"

PARTITION="${PARTITION:-gpu-a100}"
WALL="${WALL:-2-00:00:00}"
SAVE_EVERY="${SAVE_EVERY:-900}"
GRES="${GRES:-gpu:a100:1}"
STEPS="${STEPS:-4000000}"
SEEDS="${SEEDS:-0 1 2 3}"
NICE="${NICE:-9000}"
DRY_RUN="${DRY_RUN:-0}"
WORK="${WORK:-/work/DoyaU/vasilache/work}"
ARM=som_line

find_run_dir() {  # tag task arm seed
  python3 - "$WORK" "$1" "$2" "$3" "$4" <<'PY'
import glob, json, os, sys
work, tag, task, arm, seed = sys.argv[1:6]
best, bestdir = -1, ''
for d in glob.glob(f'{work}/{tag}_{task}_{arm}_s{seed}_BIG_j*'):
    p = os.path.join(d, 'logdir', 'metrics.jsonl')
    if not os.path.exists(p):
        continue
    step = -1
    with open(p, 'rb') as f:
        f.seek(0, 2); f.seek(max(0, f.tell() - 200_000))
        for line in f.read().decode('utf8', 'ignore').splitlines()[1:]:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if 'step' in row:
                step = max(step, int(row['step']))
    if step > best:
        best, bestdir = step, d
print(bestdir if best > 0 else '')
PY
}

submit() {  # exp task label seed
  local exp="$1" task="$2" label="$3" seed="$4"
  local tag="e${exp}" name="e${1}_${3}_${ARM}_s${4}"
  local ev="ALL,EXP_TAG=$tag,TASK=$task,ARM=$ARM,SEED=$seed"
  ev="$ev,RUN_STEPS=$STEPS,SAVE_EVERY=$SAVE_EVERY"
  local pinned
  pinned="$(find_run_dir "$tag" "$task" "$ARM" "$seed")"
  [ -n "$pinned" ] && ev="$ev,RUN_DIR=$pinned"
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] nice=$NICE $name  $task${pinned:+   # resumes $pinned}"
    return
  fi
  local out jobid
  out="$(sbatch -J "$name" -p "$PARTITION" -t "$WALL" --gres="$GRES" \
        --nice="$NICE" --requeue --export="$ev" "$SCRIPT")"
  jobid="${out##* }"
  echo "$out  ($name)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "$jobid" "$tag" "$task" "$ARM" "$seed" "$name" \
    "$PARTITION" "$STEPS" >> "$LOG"
  sleep 2
}

for seed in $SEEDS; do
  submit $((574 + seed)) dmc_cheetah_run cheetah "$seed"
done
for seed in $SEEDS; do
  submit $((578 + seed)) dmc_hopper_hop hopperhop "$seed"
done

echo
echo "log: $LOG  (the e558-e573 watchdog picks these up on its next pass)"
