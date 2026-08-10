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

PARTITION="${PARTITION:-gpu-a100,short-a100}"
GRES="${GRES:-gpu:a100:1}"
DRY_RUN="${DRY_RUN:-0}"
SEEDS="${SEEDS:-0 1 2 3}"
RUN_STEPS="${RUN_STEPS:-4000000}"
SUBSET="${SUBSET:-all}"
NICE="${NICE:-5000}"

ARMS=(som_line som_orig_line lipvq_prod som_lipvq_line_prod som_orig_lipvq_line_prod)
TASKS=(dmc_cartpole_swingup dmc_hopper_stand)
LABELS=(cartpole hopper)

DIR_TASKS=(dmc_cartpole_swingup_sparse dmc_cheetah_run)
DIR_LABELS=(cpsparse cheetah)

submit() {  # exp task arm seed label extra_sbatch_args...
  local exp="$1" task="$2" arm="$3" seed="$4" label="$5"; shift 5
  local tag="e${exp}"
  local name="${tag}_${label}_${arm}_s${seed}"
  local cmd=(sbatch -J "$name" -p "$PARTITION" --gres="$GRES" "$@"
       --export="ALL,EXP_TAG=$tag,TASK=$task,ARM=$arm,SEED=$seed,RUN_STEPS=$RUN_STEPS"
       "$SCRIPT")
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] ${cmd[*]}"
    return
  fi
  local out jobid
  out="$("${cmd[@]}")"
  jobid="${out##* }"
  echo "$out  ($name)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "$jobid" "$tag" "$task" "$arm" "$seed" "$name" >> "$LOG"
  sleep 2
}

if [ "$SUBSET" = "all" ] || [ "$SUBSET" = "arms" ]; then
  for seed in $SEEDS; do
    for ai in "${!ARMS[@]}"; do
      for ti in "${!TASKS[@]}"; do
        submit $((510 + ai * 8 + ti * 4 + seed)) \
          "${TASKS[$ti]}" "${ARMS[$ai]}" "$seed" "${LABELS[$ti]}"
      done
    done
  done
fi

if [ "$SUBSET" = "all" ] || [ "$SUBSET" = "director" ]; then
  for seed in $SEEDS; do
    for ti in "${!DIR_TASKS[@]}"; do
      submit $((550 + ti * 4 + seed)) \
        "${DIR_TASKS[$ti]}" director "$seed" "${DIR_LABELS[$ti]}" --nice="$NICE"
    done
  done
fi

[ "$DRY_RUN" = "1" ] || echo "logged to $LOG"
