#!/bin/bash
# e558-e573: SOM-line+LiP on three further tasks, with a matched Director
# baseline on hopper hop. Launched 2026-08-17 to run unattended.
#
#   group  exps        arm                  task                      steps
#   1      e558-e561   som_lipvq_line_prod  dmc_cartpole_swingup_sparse  4M
#          e562-e565   som_lipvq_line_prod  dmc_cheetah_run              4M
#   2      e566-e569   director             dmc_hopper_hop               4M
#   3      e570-e573   som_lipvq_line_prod  dmc_hopper_hop               4M
#
# Group 1's baseline already exists (e550-e557, the Director extras on those
# two tasks). Group 2 is the baseline group 3 needs, and it runs FRESH to 4M
# rather than resuming e495-e499: those were archived with SKIP_REPLAY=1, so
# their replay buffers are gone and only the 2.3M checkpoints survive. A
# warm start from there would refill replay from scratch mid-run, which is a
# different experiment from group 3's fresh 4M runs and would not be a valid
# baseline for them.
#
# ORDERING. gpu-a100 gives this account 8 concurrent GPUs and is not
# preemptible. Group 1 is exactly 8 runs, so it fills the lane; groups 2 and 3
# are 4 each and fill it together as group 1 drains. Order is enforced with
# --nice rather than --dependency: a dependency would idle four GPUs waiting
# for group 2 to finish before group 3 could start, where nice gets the same
# priority order with none of the waste. Nothing is submitted to short-a100 --
# measured at a ~17% duty cycle on 2026-08-10, all 32 cluster A100s allocated.
#
# All three groups queue behind the e550-e557 extras still running, and behind
# any rescue the e510-e557 watchdog needs to make (nice 6000+ vs its 5000).
#
#   DRY_RUN=1 sbatch/submit_e558_e573_hopperhop_sparse_cheetah.sh
#   sbatch/submit_e558_e573_hopperhop_sparse_cheetah.sh
#   GROUP=1 ...      # one group only
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/sbatch/run_v3_goal_ae_ablation_big_a100.sbatch"
LOGDIR="$REPO/job_logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/e558_e573_hopperhop_sparse_cheetah.tsv"

PARTITION="${PARTITION:-gpu-a100}"
WALL="${WALL:-2-00:00:00}"
SAVE_EVERY="${SAVE_EVERY:-900}"
GRES="${GRES:-gpu:a100:1}"
STEPS="${STEPS:-4000000}"
SEEDS="${SEEDS:-0 1 2 3}"
DRY_RUN="${DRY_RUN:-0}"
GROUP="${GROUP:-all}"
WORK="${WORK:-/work/DoyaU/vasilache/work}"

# Furthest-along existing run dir for an experiment, or empty. Same rule the
# watchdog uses, so a relaunch and a rescue land on the same directory.
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

submit() {  # exp task arm seed label nice
  local exp="$1" task="$2" arm="$3" seed="$4" label="$5" nice="$6"
  local tag="e${exp}"
  local name="${tag}_${label}_${arm}_s${seed}"
  local exportvars="ALL,EXP_TAG=$tag,TASK=$task,ARM=$arm,SEED=$seed"
  exportvars="$exportvars,RUN_STEPS=$STEPS,SAVE_EVERY=$SAVE_EVERY"
  local pinned
  pinned="$(find_run_dir "$tag" "$task" "$arm" "$seed")"
  [ -n "$pinned" ] && exportvars="$exportvars,RUN_DIR=$pinned"
  local cmd=(sbatch -J "$name" -p "$PARTITION" -t "$WALL" --gres="$GRES"
       --nice="$nice" --requeue --export="$exportvars" "$SCRIPT")
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] nice=$nice ${name}  ${task}${pinned:+   # resumes $pinned}"
    return
  fi
  local out jobid
  out="$("${cmd[@]}")"
  jobid="${out##* }"
  echo "$out  ($name, nice=$nice)${pinned:+  resuming $(basename "$pinned")}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "$jobid" "$tag" "$task" "$arm" "$seed" "$name" \
    "$PARTITION" "$STEPS" >> "$LOG"
  sleep 2
}

# group 1: som_lipvq_line_prod on the two tasks e550-e557 already baseline
if [ "$GROUP" = "all" ] || [ "$GROUP" = "1" ]; then
  for seed in $SEEDS; do
    submit $((558 + seed)) dmc_cartpole_swingup_sparse som_lipvq_line_prod \
      "$seed" cpsparse 6000
    submit $((562 + seed)) dmc_cheetah_run som_lipvq_line_prod \
      "$seed" cheetah 6000
  done
fi

# group 2: the hopper hop Director baseline group 3 is measured against
if [ "$GROUP" = "all" ] || [ "$GROUP" = "2" ]; then
  for seed in $SEEDS; do
    submit $((566 + seed)) dmc_hopper_hop director "$seed" hopperhop 7000
  done
fi

# group 3: som_lipvq_line_prod on hopper hop
if [ "$GROUP" = "all" ] || [ "$GROUP" = "3" ]; then
  for seed in $SEEDS; do
    submit $((570 + seed)) dmc_hopper_hop som_lipvq_line_prod \
      "$seed" hopperhop 8000
  done
fi

echo
echo "log: $LOG"
