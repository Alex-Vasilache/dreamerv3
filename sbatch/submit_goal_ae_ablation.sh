#!/bin/bash
# Staged matrix launcher for the goal-autoencoder ablation.
#
# Three waves, so the first results arrive before the full 60-job matrix is
# committed to the queue:
#
#   WAVE=1   hopper, seed 0, 3 arms                        ->  3 jobs
#   WAVE=2   cartpole + acrobot + cheetah, seed 0, 3 arms  ->  9 jobs
#   WAVE=3   all 4 envs, seeds 1-4, 3 arms                 -> 48 jobs
#
# Experiment numbers are allocated from START_EXP in a fixed order so they stay
# global and monotonic (CLAUDE.md); each wave's base is pinned below rather than
# computed, so re-running a wave reuses the same numbers instead of drifting.
#
# Usage:
#   DRY_RUN=1 WAVE=1 sbatch/submit_goal_ae_ablation.sh     # preview
#   WAVE=1 sbatch/submit_goal_ae_ablation.sh               # launch
#   WAVE=1 ARMS="som" sbatch/submit_goal_ae_ablation.sh    # subset
#
# Env knobs: PARTITION (default gpu-a100), ARMS, ENVS, SEEDS, RUN_STEPS.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/sbatch/run_v3_goal_ae_ablation_big_a100.sbatch"
LOGDIR="$REPO/job_logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/goal_ae_ablation.tsv"

WAVE="${WAVE:-1}"
PARTITION="${PARTITION:-gpu-a100}"
GRES="${GRES:-gpu:a100:1}"
DRY_RUN="${DRY_RUN:-0}"

# Fixed allocation. 3 arms x (1 env x 1 seed | 3 envs x 1 seed | 4 envs x 4 seeds).
ALL_ARMS="som lipvq som_lipvq"
case "$WAVE" in
  1) BASE=415; ENVS="${ENVS:-dmc_hopper_hop}"; SEEDS="${SEEDS:-0}" ;;
  2) BASE=418; ENVS="${ENVS:-dmc_cartpole_swingup dmc_acrobot_swingup dmc_cheetah_run}"
     SEEDS="${SEEDS:-0}" ;;
  3) BASE=427
     ENVS="${ENVS:-dmc_hopper_hop dmc_cartpole_swingup dmc_acrobot_swingup dmc_cheetah_run}"
     SEEDS="${SEEDS:-1 2 3 4}" ;;
  *) echo "WAVE must be 1, 2 or 3" >&2; exit 1 ;;
esac
ARMS="${ARMS:-$ALL_ARMS}"

short_env() { echo "${1#dmc_}" | cut -d_ -f1; }

# Walk the FULL wave grid so numbering does not depend on the ARMS/ENVS/SEEDS
# subset actually being submitted; skip the entries not requested.
n=0
for seed in $SEEDS; do
  for env in $ENVS; do
    for arm in $ALL_ARMS; do
      exp=$((BASE + n)); n=$((n + 1))
      case " $ARMS " in *" $arm "*) ;; *) continue ;; esac
      tag="e${exp}"
      name="${tag}_$(short_env "$env")_${arm}_s${seed}"
      cmd=(sbatch -J "$name" -p "$PARTITION" --gres="$GRES"
           --export="ALL,EXP_TAG=$tag,TASK=$env,ARM=$arm,SEED=$seed${RUN_STEPS:+,RUN_STEPS=$RUN_STEPS}"
           "$SCRIPT")
      if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] ${cmd[*]}"
        continue
      fi
      out="$(${cmd[@]})"
      jobid="${out##* }"
      echo "$out  ($name)"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Is)" "$jobid" "$tag" "$env" "$arm" "$seed" "$name" >> "$LOG"
    done
  done
done

[ "$DRY_RUN" = "1" ] || echo "logged to $LOG"
