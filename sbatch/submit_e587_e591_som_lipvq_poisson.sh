#!/bin/bash
# e587-e591: SOM-line+LiP + the unimodal Poisson manager, one seed each.
#
#   e587  dmc_cartpole_swingup          e590  dmc_cheetah_run
#   e588  dmc_hopper_stand              e591  dmc_hopper_hop
#   e589  dmc_cartpole_swingup_sparse
#
# Differs from the e534-e573 SOM-line+LiP runs by the MANAGER HEAD ALONE, so it
# is the direct comparison those runs support: same goal autoencoder (SOM on a
# line, Lipschitz-regularized, silu trunk -- verified against the checkpoints,
# not just the config), categorical head swapped for the Poisson one.
#
# Replaces e582-e586 (SOM-line + Poisson, no LiP) in the A100 lane; those were
# paused at ~1.9M/4M with checkpoint and replay intact and resume via
# sbatch/resume_e582_e586_som_line_poisson.sh.
#
# ONE SEED PER TASK, as with e582-e586: a screen across the five benchmarks,
# not a powered comparison. Nothing here can resolve hopper_hop.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/sbatch/run_v3_goal_ae_ablation_big_a100.sbatch"
LOG="$REPO/job_logs/e558_e573_hopperhop_sparse_cheetah.tsv"

PARTITION="${PARTITION:-gpu-a100}"
WALL="${WALL:-2-00:00:00}"
SAVE_EVERY="${SAVE_EVERY:-900}"
GRES="${GRES:-gpu:a100:1}"
STEPS="${STEPS:-4000000}"
SEED="${SEED:-0}"
NICE="${NICE:-5000}"
DRY_RUN="${DRY_RUN:-0}"
WORK="${WORK:-/work/DoyaU/vasilache/work}"
ARM=som_lipvq_line_prod_poisson

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

submit() {  # exp task label
  local exp="$1" task="$2" label="$3"
  local tag="e${exp}" name="e${1}_${3}_${ARM}_s${SEED}"
  local ev="ALL,EXP_TAG=$tag,TASK=$task,ARM=$ARM,SEED=$SEED"
  ev="$ev,RUN_STEPS=$STEPS,SAVE_EVERY=$SAVE_EVERY"
  local pinned
  pinned="$(find_run_dir "$tag" "$task" "$ARM" "$SEED")"
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
    "$(date -Is)" "$jobid" "$tag" "$task" "$ARM" "$SEED" "$name" \
    "$PARTITION" "$STEPS" >> "$LOG"
  sleep 2
}

submit 587 dmc_cartpole_swingup        cpswingup
submit 588 dmc_hopper_stand            hopperstand
submit 589 dmc_cartpole_swingup_sparse cpsparse
submit 590 dmc_cheetah_run             cheetah
submit 591 dmc_hopper_hop              hopperhop

echo
echo "log: $LOG  (the e558-e573 watchdog picks these up on its next pass)"
