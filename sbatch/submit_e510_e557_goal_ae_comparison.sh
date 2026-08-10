#!/bin/bash
# e510-e557: the goal-autoencoder comparison, four seeds per cell, launched
# 2026-08-10 to run unattended for a week.
#
# Five goal-AE arms x two tasks x four seeds (e510-e549), plus eight
# lower-priority pure-Director baselines on two further tasks (e550-e557).
# Everything else -- BIG/director_match scale, 64x64, batch 16x64, imag 16,
# train_ratio 64, 8 envs, replay 1e6, native conv, 4M steps, struct/soft-reuse
# off -- is identical to the e502-e509 Director baselines, which are the
# reference distribution for the first two tasks and are already running.
#
#   arm                            exps        differs from its neighbour by
#   som_line                       e510-e517   (SOM on a line, STE on)
#   som_orig_line                  e518-e525   ... straight-through OFF
#   lipvq_prod                     e526-e533   (Lipschitz only, no SOM)
#   som_lipvq_line_prod            e534-e541   = som_line + Lipschitz
#   som_orig_lipvq_line_prod       e542-e549   = som_orig_line + Lipschitz
#
#   director (cartpole_swingup_sparse) e550-e553
#   director (cheetah_run)             e554-e557
#
# Exp numbers are grouped by arm so the log reads by arm, but jobs are
# SUBMITTED seed-major: the first ten jobs are all five arms on both tasks at
# seed 0, the next ten are seed 1, and so on. With ~16-20 concurrent slots that
# means the first wave to finish is a COMPLETE comparison at n=2 rather than
# two finished arms and three that never started -- which is what matters if
# the cluster turns out to be slower than planned.
#
# Partition: both A100 lanes. `gpu-a100` is capped at 8 GPUs for this account
# and is not preemptible; `short-a100` allows 32, is PriorityTier=1, and its
# jobs get requeued when a higher-tier job needs the node. Requeue keeps the
# job id, RUN_DIR is keyed to the job id, and the run resumes from its
# checkpoint, so preemption costs a restart and nothing else.
#
# The e550-e557 baselines are submitted with --nice so they always queue behind
# the five arms, which are the priority.
#
# Usage:
#   DRY_RUN=1 sbatch/submit_e510_e557_goal_ae_comparison.sh   # preview
#   sbatch/submit_e510_e557_goal_ae_comparison.sh             # launch all 48
#   SUBSET=arms sbatch/submit_e510_e557_goal_ae_comparison.sh # e510-e549 only
#   SUBSET=director ...                                       # e550-e557 only
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/sbatch/run_v3_goal_ae_ablation_big_a100.sbatch"
LOGDIR="$REPO/job_logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/e510_e557_goal_ae_comparison.tsv"

# Two lanes, and they must be requested separately: a comma-separated
# partition list is rejected outright for this account ("Multiple partition job
# request not supported when a partition is set in the association").
#
#   short-a100  PriorityTier=1, preemptible-with-requeue. The association
#               allows 32 GPUs but only 256 CPUs, so at 16 CPUs a job it is
#               really 16 concurrent, and MaxWall is 2h -- a 4M-step run
#               requeues itself ~16 times to get through. Empty and idle right
#               now, so jobs sent here start immediately.
#   gpu-a100    PriorityTier=10, not preemptible, 48h jobs that never restart,
#               but 128 CPUs = 8 concurrent, and all 8 are occupied by
#               e502-e509 until ~2026-08-11: anything sent here waits ~10h.
#
# Seeds 0-1 and part of seed 2 therefore go to short-a100, where they start
# now, and the tail of the order goes to gpu-a100, where it is pulled forward
# to whenever the baselines finish rather than waiting behind the whole
# short-a100 backlog. Both lanes draw on the same 32 physical A100s, so the
# combined throughput is ~24 runs at a time.
# REVISED 2026-08-10 12:50 after measuring what short-a100 actually delivers.
# e510 there ran 28:43 before being preempted, of which ~10 min was compiling
# the BIG graph, and then sat queued for another 80 minutes: a duty cycle
# around 17%, because every one of the cluster's 32 A100s is allocated and a
# PriorityTier=1 job only gets what nobody else wants. So the batch now goes
# entirely into `gpu-a100`, the 8 GPUs this account is actually guaranteed, and
# short-a100 is used only if a later check-in finds it idle.
PARTITION="${PARTITION:-gpu-a100}"
PART_WALL="${PART_WALL:-2-00:00:00}"
PART_SAVE_EVERY="${PART_SAVE_EVERY:-900}"
ALT_PARTITION="${ALT_PARTITION:-short-a100}"
ALT_WALL="${ALT_WALL:-02:00:00}"
ALT_SAVE_EVERY="${ALT_SAVE_EVERY:-300}"
ALT_TAIL="${ALT_TAIL:-0}"        # last N arm jobs submitted to ALT_PARTITION
GRES="${GRES:-gpu:a100:1}"
DRY_RUN="${DRY_RUN:-0}"
SEEDS="${SEEDS:-0 1 2 3}"
# Separate horizons. Eight guaranteed GPUs put the 40 arm runs at five rounds:
# at 4M (~27h a run) that is 135h and leaves the Director extras no room at
# all, at 3M (~20.4h) it is 102h and everything fits with a day to spare. The
# arms are a comparison against each other and against e502-e509, which log
# continuously and snapshot every 500k, so 3M costs nothing but the last
# quarter of a curve; "did not finish" would cost the whole experiment.
# The Director extras stay at 4M because their job is to sit alongside the
# e390-e409 baselines in the paper's motivation figures, which are read at 4M.
ARM_STEPS="${ARM_STEPS:-3000000}"
DIR_STEPS="${DIR_STEPS:-4000000}"
# Resume in place: if this experiment already has a run directory with
# progress in it, pin RUN_DIR to the furthest one instead of starting over.
PIN_DIRS="${PIN_DIRS:-1}"
WORK="${WORK:-/work/DoyaU/vasilache/work}"
SUBSET="${SUBSET:-all}"
NICE="${NICE:-5000}"

ARMS=(som_line som_orig_line lipvq_prod som_lipvq_line_prod som_orig_lipvq_line_prod)
TASKS=(dmc_cartpole_swingup dmc_hopper_stand)
LABELS=(cartpole hopper)

DIR_TASKS=(dmc_cartpole_swingup_sparse dmc_cheetah_run)
DIR_LABELS=(cpsparse cheetah)

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

submit() {  # partition exp task arm seed label steps extra_sbatch_args...
  local part="$1" exp="$2" task="$3" arm="$4" seed="$5" label="$6" steps="$7"
  shift 7
  local tag="e${exp}"
  local name="${tag}_${label}_${arm}_s${seed}"
  local wall="$PART_WALL" save="$PART_SAVE_EVERY"
  # The script's own header asks for USR1 600s before the limit, which is 8% of
  # a 2h slice spent not training. 240s is ample for `scontrol requeue; exit`
  # and buys back ~6 minutes on every one of the ~16 slices a run needs.
  local sig=(--signal=B:USR1@240)
  if [ "$part" = "$ALT_PARTITION" ]; then
    wall="$ALT_WALL"; save="$ALT_SAVE_EVERY"; sig=()
  fi
  local exportvars="ALL,EXP_TAG=$tag,TASK=$task,ARM=$arm,SEED=$seed"
  exportvars="$exportvars,RUN_STEPS=$steps,SAVE_EVERY=$save"
  local pinned=''
  if [ "$PIN_DIRS" = "1" ]; then
    pinned="$(find_run_dir "$tag" "$task" "$arm" "$seed")"
    [ -n "$pinned" ] && exportvars="$exportvars,RUN_DIR=$pinned"
  fi
  local cmd=(sbatch -J "$name" -p "$part" -t "$wall" "${sig[@]}" --gres="$GRES" "$@"
       --export="$exportvars" "$SCRIPT")
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] ${cmd[*]}${pinned:+   # resumes $pinned}"
    return
  fi
  local out jobid
  out="$("${cmd[@]}")"
  jobid="${out##* }"
  echo "$out  ($name)${pinned:+  resuming $(basename "$pinned")}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "$jobid" "$tag" "$task" "$arm" "$seed" "$name" "$part" \
    "$steps" >> "$LOG"
  sleep 2
}

n_arms=$(( $(echo $SEEDS | wc -w) * ${#ARMS[@]} * ${#TASKS[@]} ))
i=0

if [ "$SUBSET" = "all" ] || [ "$SUBSET" = "arms" ]; then
  for seed in $SEEDS; do
    for ai in "${!ARMS[@]}"; do
      for ti in "${!TASKS[@]}"; do
        i=$((i + 1))
        part="$PARTITION"
        [ "$i" -gt "$((n_arms - ALT_TAIL))" ] && part="$ALT_PARTITION"
        submit "$part" $((510 + ai * 8 + ti * 4 + seed)) \
          "${TASKS[$ti]}" "${ARMS[$ai]}" "$seed" "${LABELS[$ti]}" "$ARM_STEPS"
      done
    done
  done
fi

# The Director extras are the user's lowest priority ("can run once the others
# finish"), so they go in the same lane as the bulk of the arms with a nice
# value that keeps them behind every one of them. --nice only orders jobs
# within a partition, which is why they are not sent to the other lane.
if [ "$SUBSET" = "all" ] || [ "$SUBSET" = "director" ]; then
  for seed in $SEEDS; do
    for ti in "${!DIR_TASKS[@]}"; do
      submit "$PARTITION" $((550 + ti * 4 + seed)) \
        "${DIR_TASKS[$ti]}" director "$seed" "${DIR_LABELS[$ti]}" "$DIR_STEPS" \
        --nice="$NICE"
    done
  done
fi

[ "$DRY_RUN" = "1" ] || echo "logged to $LOG"
