#!/bin/bash
# e582-e586: SOM-line + the unimodal Poisson manager policy, one seed each.
#
#   e582  som_line_poisson  dmc_cartpole_swingup         seed 0   4M
#   e583  som_line_poisson  dmc_hopper_stand             seed 0   4M
#   e584  som_line_poisson  dmc_cartpole_swingup_sparse  seed 0   4M
#   e585  som_line_poisson  dmc_cheetah_run              seed 0   4M
#   e586  som_line_poisson  dmc_hopper_hop               seed 0   4M
#
# The manager's per-block head stops emitting 8 free logits and emits a Poisson
# rate + a temperature instead (Zhu et al. 2024, Eqs. 8-10), so its class
# distribution is unimodal and REINFORCE credit spreads to neighbouring classes.
# On a SOM line those neighbours really are neighbours in goal space, which is
# the whole hypothesis. Differs from the plain som_line arm by the manager head
# alone -- same goal AE, same everything else.
#
# ONE SEED PER TASK, deliberately: this is a first look across the five
# benchmarks, not a powered comparison. A single seed cannot resolve anything
# on hopper_hop (retired earlier for exactly that reason) and only barely on
# the others; read it as "does this train at all, and does it look broken",
# then decide where to spend 4 seeds.
#
# Baselines already in hand for all five: som_line has cartpole_swingup
# (e510-e513) and hopper_stand (e514-e517); Director has all five.
# som_line still has no cartpole_swingup_sparse / cheetah / hopper_hop, so
# e584-e586 compare against Director and SOM+LiP rather than against som_line.
#
# Rows are appended to the e558-e573 TSV on purpose: the watchdog re-reads that
# file every pass, so these get crash/stall supervision with no second watchdog.
# It resubmits via ARM=, and som_line_poisson is wired into the launch script's
# case, so a resubmit reproduces the same config.
#
#   DRY_RUN=1 sbatch/submit_e582_e586_som_line_poisson.sh
#   sbatch/submit_e582_e586_som_line_poisson.sh
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
ARM=som_line_poisson

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

submit 582 dmc_cartpole_swingup        cpswingup
submit 583 dmc_hopper_stand            hopperstand
submit 584 dmc_cartpole_swingup_sparse cpsparse
submit 585 dmc_cheetah_run             cheetah
submit 586 dmc_hopper_hop              hopperhop

echo
echo "log: $LOG  (the e558-e573 watchdog picks these up on its next pass)"
