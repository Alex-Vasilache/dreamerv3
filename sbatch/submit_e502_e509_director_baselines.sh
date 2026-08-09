#!/bin/bash
# e502-e509: our Director implementation on two tasks it should be able to
# solve, 4 seeds each, BIG/A100. 8 jobs = the whole gpu-a100 GPU quota.
#
#   e502-e505  dmc_cartpole_swingup  seeds 0-3
#   e506-e509  dmc_hopper_stand      seeds 0-3
#
# Why these two tasks: `hopper_hop` was retired as a comparison substrate on
# 2026-08-09 -- its outcome distribution is bimodal and 5 seeds have ~1% power
# (EXPERIMENTS.md, "Why the restart"). `cartpole_swingup` is one of Director's
# configured DMC tasks; `hopper_stand` keeps the hopper body without hop's
# variance.
#
# ARM=director is the pure-Director arm (no goal AE), i.e. the same recipe as
# the e390-e409 baselines, now with the 2026-08-09 run-cost defaults
# (goal_struct_diag off, prealloc on, no scope output, no episode video) and
# 500k-step milestone checkpoints.
#
# Usage:
#   DRY_RUN=1 sbatch/submit_e502_e509_director_baselines.sh   # preview
#   sbatch/submit_e502_e509_director_baselines.sh             # launch
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/sbatch/run_v3_goal_ae_ablation_big_a100.sbatch"
LOGDIR="$REPO/job_logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/e502_e509_director_baselines.tsv"

PARTITION="${PARTITION:-gpu-a100}"
GRES="${GRES:-gpu:a100:1}"
DRY_RUN="${DRY_RUN:-0}"
SEEDS="${SEEDS:-0 1 2 3}"
BASE=502

short_env() { echo "${1#dmc_}" | cut -d_ -f1; }

n=0
for env in dmc_cartpole_swingup dmc_hopper_stand; do
  for seed in $SEEDS; do
    exp=$((BASE + n)); n=$((n + 1))
    tag="e${exp}"
    name="${tag}_$(short_env "$env")_director_s${seed}"
    cmd=(sbatch -J "$name" -p "$PARTITION" --gres="$GRES"
         --export="ALL,EXP_TAG=$tag,TASK=$env,ARM=director,SEED=$seed${RUN_STEPS:+,RUN_STEPS=$RUN_STEPS}"
         "$SCRIPT")
    if [ "$DRY_RUN" = "1" ]; then
      echo "[dry-run] ${cmd[*]}"
      continue
    fi
    out="$("${cmd[@]}")"
    jobid="${out##* }"
    echo "$out  ($name)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date -Is)" "$jobid" "$tag" "$env" "director" "$seed" "$name" >> "$LOG"
  done
done

[ "$DRY_RUN" = "1" ] || echo "logged to $LOG"
