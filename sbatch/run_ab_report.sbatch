#!/bin/bash -l
# Scheduled re-analysis of the e372-e379 refactor A/B.
#
# Submit with a deferred start, e.g.
#   sbatch --begin=2026-07-29T06:00:00 sbatch/run_ab_report.sbatch
#
# Runs on `intel` (no GPU quota cost, no competition with the A/B runs) and
# writes a timestamped report to $OUT_DIR. Deliberately a SLURM job rather than
# a cloud routine: the whole analysis reads run directories under /work and
# queries squeue/sacct, none of which exist outside the cluster.
#SBATCH -J ab_report
#SBATCH -p intel
#SBATCH -t 0-00:20:00
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

set -uo pipefail   # NOT -e: a partial/failed arm must still produce a report

REPO="${REPO:-/apps/unit/DoyaU/vasilache/apps/code/dreamerv3_refactor}"
OUT_DIR="${OUT_DIR:-/work/DoyaU/vasilache/work/ab_reports}"
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%dT%H%M%S)
OUT="$OUT_DIR/ab_report_${STAMP}.txt"

{
  echo "=============================================================="
  echo " Refactor A/B report -- $(date)"
  echo " e372-e377 pre/post arms, e378-e379 same-code controls"
  echo "=============================================================="
  echo
  echo "--- job states ---"
  for j in 4671612 4671613 4671614 4671615 4671616 4671617 4671631 4671632; do
    printf '%-9s %s\n' "$j" \
      "$(sacct -j "$j" --format=JobName%28,State,Elapsed,ExitCode -n 2>/dev/null | head -1)"
  done
  echo
  echo "--- any arm that died ---"
  for j in 4671612 4671613 4671614 4671615 4671616 4671617 4671631 4671632; do
    st=$(sacct -j "$j" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in
      RUNNING|PENDING|COMPLETED|REQUEUED|RESIZING|SUSPENDED) ;;
      *) echo "  !! $j is $st"
         f=$(ls /work/DoyaU/vasilache/work/slurm_logs/*-"$j".err 2>/dev/null | head -1)
         [ -n "$f" ] && tail -15 "$f" | sed 's/^/     /' ;;
    esac
  done
  echo
  echo "--- requeue contamination check ---"
  # The five pre-side arms read the LIVE tree, which was merged to the
  # refactored code at 2026-07-29T07:51:44+09:00. Their running processes hold
  # the old code in memory, but a REQUEUE re-execs and would silently pick up
  # the new code, invalidating that arm as a "pre" measurement. SLURM keeps one
  # accounting row per run attempt, so >1 row means the job restarted.
  MERGE_EPOCH=$(date -d '2026-07-29T07:51:44+09:00' +%s)
  for j in 4671612 4671614 4671616 4671631 4671632; do
    n=$(sacct -j "$j" --duplicates --format=JobID -n 2>/dev/null | grep -cE "^ *$j( |$)")
    start=$(sacct -j "$j" --format=Start -n 2>/dev/null | tail -1 | tr -d ' ')
    se=$(date -d "$start" +%s 2>/dev/null || echo 0)
    if [ "${n:-1}" -gt 1 ] || { [ "$se" -gt 0 ] && [ "$se" -gt "$MERGE_EPOCH" ]; }; then
      echo "  !! $j RESTARTED after the merge -- it is no longer a 'pre' arm."
      echo "     Discard it; its results now come from the refactored code."
    else
      echo "  ok $j (single run attempt, started $start)"
    fi
  done
  echo
  echo "--- comparison (matched step windows, vs same-code control) ---"
  source /apps/unit/DoyaU/vasilache/apps/source_rl_env.sh >/dev/null 2>&1 || true
  python3 "$REPO/tools/compare_refactor_ab.py" 2>&1
  echo
  echo "--- reminder of the read-out rule ---"
  echo "A gap is evidence of a behavior change only if it clearly exceeds the"
  echo "self-gap (same code, rerun). Watch the DIRECTION of any soft-reuse /"
  echo "implicit-sparsity deviation across tasks: a real code difference biases"
  echo "consistently, noise flips sign."
} > "$OUT" 2>&1

cp "$OUT" "$OUT_DIR/ab_report_latest.txt"
echo "report written: $OUT"
cat "$OUT"
exit 0
