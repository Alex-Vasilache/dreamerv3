#!/bin/bash
# Archive + clean the cancelled 2026-07-09/10 batches (e133-e143 leftovers, e144-e159,
# smoke dirs) per the CLAUDE.md /work hygiene policy: copy to bucket with SKIP_REPLAY=1,
# verify rsync exit 0, then delete the run dir and its SLURM logs.
set -uo pipefail
COPY=/work/DoyaU/vasilache/work/copy_dreamerv3_runs_to_bucket.sh
WORK=/work/DoyaU/vasilache/work
LOGS=$WORK/slurm_logs
OUT=$WORK/archive_cancelled_20260710.log
: > "$OUT"

# Everything matching these prefixes is from stopped/cancelled runs (verified: no
# running jobs remain from e133-e159 as of 2026-07-10; e160/e161 run in their own
# dirs which do NOT match these patterns).
shopt -s nullglob
for d in $WORK/e13[3-9]_* $WORK/e14[0-9]_* $WORK/e15[0-9]_* $WORK/smoke144_* $WORK/smokeD_* $WORK/smokeK_* $WORK/smokeP_*; do
  base=$(basename "$d")
  echo "== $base" >> "$OUT"
  if SKIP_REPLAY=1 "$COPY" "$base" >> "$OUT" 2>&1; then
    rm -rf "$d"
    # SLURM logs are named by JOB NAME, which starts with the same experiment tag
    # (e.g. e144_mask_entropy_cartpole_small-4657451.out for dir e144_dmc_...).
    tag="${base%%_*}"
    rm -f "$LOGS/${tag}_"*.out "$LOGS/${tag}_"*.err 2>/dev/null
    echo "   archived + deleted (logs: ${tag}_*)" >> "$OUT"
  else
    echo "   COPY FAILED - kept in /work" >> "$OUT"
  fi
done
# smoke job logs use ad-hoc job names; sweep them once at the end
rm -f "$LOGS"/smoke*.out "$LOGS"/smoke*.err 2>/dev/null
echo "DONE $(date)" >> "$OUT"
