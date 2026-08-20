#!/bin/bash
# Resume e582-e586 (SOM-line + Poisson manager), paused 2026-08-20 at ~1.9M/4M
# steps to free the A100 lane for e587-e591.
#
# They were cancelled, not archived, so both the checkpoint and the 4-7GB
# replay buffer are still on /work and each run picks up where it stopped.
# Archiving with SKIP_REPLAY=1 would keep the checkpoint but delete the replay,
# which makes a run un-resumable -- do not do that to these until they finish.
#
# Their rows were removed from the e558-e573 watchdog TSV at pause time (or it
# would have resubmitted them within one pass); this script puts them back.
#
#   DRY_RUN=1 sbatch/resume_e582_e586_som_line_poisson.sh
#   sbatch/resume_e582_e586_som_line_poisson.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/sbatch/run_v3_goal_ae_ablation_big_a100.sbatch"
PAUSED="$REPO/job_logs/e582_e586_paused.tsv"
LOG="$REPO/job_logs/e558_e573_hopperhop_sparse_cheetah.tsv"

PARTITION="${PARTITION:-gpu-a100}"
WALL="${WALL:-2-00:00:00}"
SAVE_EVERY="${SAVE_EVERY:-900}"
GRES="${GRES:-gpu:a100:1}"
STEPS="${STEPS:-4000000}"
NICE="${NICE:-5000}"
DRY_RUN="${DRY_RUN:-0}"
ARM=som_line_poisson

[ -f "$PAUSED" ] || { echo "no $PAUSED -- nothing recorded as paused" >&2; exit 1; }

while IFS=$'\t' read -r ts jobid tag task arm seed name part steps dir; do
  [ -z "${tag:-}" ] && continue
  if [ ! -d "$dir" ]; then
    echo "MISSING run dir for $tag: $dir" >&2; continue
  fi
  if [ ! -e "$dir/logdir/ckpt/latest" ]; then
    echo "NO CHECKPOINT for $tag in $dir -- would restart from scratch" >&2
    continue
  fi
  ev="ALL,EXP_TAG=$tag,TASK=$task,ARM=$ARM,SEED=$seed"
  ev="$ev,RUN_STEPS=$STEPS,SAVE_EVERY=$SAVE_EVERY,RUN_DIR=$dir"
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] $name resumes $dir"
    continue
  fi
  out="$(sbatch -J "$name" -p "$PARTITION" -t "$WALL" --gres="$GRES" \
        --nice="$NICE" --requeue --export="$ev" "$SCRIPT")"
  echo "$out  ($name resumes $dir)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "${out##* }" "$tag" "$task" "$ARM" "$seed" "$name" \
    "$PARTITION" "$STEPS" >> "$LOG"
  sleep 2
done < "$PAUSED"

echo
echo "watchdog TSV: $LOG (rows re-added, supervision resumes next pass)"
