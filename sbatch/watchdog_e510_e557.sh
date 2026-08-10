#!/bin/bash
# Unattended supervisor for the e510-e557 batch (2026-08-10 .. 2026-08-17).
#
# One pass over every experiment listed in the launch TSV, deciding one of:
#
#   DONE      the run reached its step target -- never touched again
#   ALIVE     a job with that name is pending or running, and its metrics file
#             is still advancing -- left alone
#   STALLED   running, but metrics.jsonl has not been written for STALL_MIN
#             minutes (it is written every 120s, and the run checkpoints every
#             900s). Requeued: same job id, same RUN_DIR, resumes from the
#             checkpoint.
#   RESUBMIT  no job in the queue and the step target not reached -- the run
#             died. Resubmitted with RUN_DIR pinned to the existing directory,
#             so it continues from the checkpoint instead of starting over.
#
# The whole state is derived from squeue and the filesystem on every pass, so
# the script is idempotent and can be killed and restarted at any time.
#
# Guardrails:
#   * only experiments named in the TSV are ever touched
#   * nothing is ever cancelled or deleted
#   * MAX_RESUBMITS per experiment, after which it is only reported
#   * no resubmission after DEADLINE -- past that a fresh run cannot finish
#     before the user is back, and a half-run experiment is worse than a clean
#     "did not finish" in the log
#
# Usage:  watchdog_e510_e557.sh          one pass, prints a report
#         LOOP=1 watchdog_e510_e557.sh   loop forever every INTERVAL seconds
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/sbatch/run_v3_goal_ae_ablation_big_a100.sbatch"
TSV="${TSV:-$REPO/job_logs/e510_e557_goal_ae_comparison.tsv}"
STATE="${STATE:-$REPO/job_logs/e510_e557_watchdog_state.tsv}"
WORK="${WORK:-/work/DoyaU/vasilache/work}"

PARTITION="${PARTITION:-gpu-a100,short-a100}"
GRES="${GRES:-gpu:a100:1}"
RUN_STEPS="${RUN_STEPS:-4000000}"
DONE_FRAC="${DONE_FRAC:-0.999}"
STALL_MIN="${STALL_MIN:-90}"
MIN_RUN_MIN="${MIN_RUN_MIN:-30}"     # grace period before the stall check bites
MAX_RESUBMITS="${MAX_RESUBMITS:-5}"
DEADLINE="${DEADLINE:-2026-08-16T12:00:00}"
INTERVAL="${INTERVAL:-1800}"
LOOP="${LOOP:-0}"
DRY_RUN="${DRY_RUN:-0}"

deadline_epoch=$(date -d "$DEADLINE" +%s)
touch "$STATE"

log() { echo "[$(date -Is)] $*"; }

# Highest env step recorded in any run dir belonging to this experiment, and
# the directory it came from. Prints "<step> <dir> <mtime_epoch>"; step -1 when
# the experiment has no run dir at all yet.
probe() {  # tag task arm seed
  local tag="$1" task="$2" arm="$3" seed="$4"
  python3 - "$WORK" "$tag" "$task" "$arm" "$seed" <<'PY'
import glob, json, os, sys
work, tag, task, arm, seed = sys.argv[1:6]
best, bestdir, bestmtime = -1, '', 0
pattern = f'{work}/{tag}_{task}_{arm}_s{seed}_BIG_j*'
for d in glob.glob(pattern):
    path = os.path.join(d, 'logdir', 'metrics.jsonl')
    if not os.path.exists(path):
        continue
    step, mtime = -1, os.path.getmtime(path)
    # Read the tail only: these files reach tens of MB over a 4M-step run.
    with open(path, 'rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 200_000))
        for line in f.read().decode('utf8', 'ignore').splitlines()[1:]:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if 'step' in row:
                step = max(step, int(row['step']))
    if step > best:
        best, bestdir, bestmtime = step, d, mtime
if not bestdir:  # a dir may exist with no metrics yet
    dirs = sorted(glob.glob(pattern))
    if dirs:
        bestdir = dirs[-1]
        bestmtime = os.path.getmtime(bestdir)
print(best, bestdir, int(bestmtime))
PY
}

count_of() { awk -F'\t' -v n="$1" '$2==n {c++} END {print c+0}' "$STATE"; }

pass() {
  local now target ndone nalive nresub nstall nfail
  now=$(date +%s)
  target=$(python3 -c "print(int($RUN_STEPS * $DONE_FRAC))")
  ndone=0; nalive=0; nresub=0; nstall=0; nfail=0

  # One squeue call for the whole pass: name -> state. %S is the start time as
  # a parseable timestamp, which is what the stall check needs -- a job that
  # only just started has an old metrics file from its previous life and must
  # not be requeued for it.
  declare -A qstate qjob qstart
  while IFS='|' read -r jid jname jst jstart; do
    [ -z "${jname:-}" ] && continue
    qstate["$jname"]="$jst"; qjob["$jname"]="$jid"
    qstart["$jname"]=$(date -d "$jstart" +%s 2>/dev/null || echo 0)
  done < <(squeue -u "$(whoami)" -h -o "%i|%j|%T|%S" 2>/dev/null)

  while IFS=$'\t' read -r _ts _jobid tag task arm seed name; do
    [ -z "${name:-}" ] && continue
    read -r step dir mtime < <(probe "$tag" "$task" "$arm" "$seed")

    if [ "$step" -ge "$target" ] 2>/dev/null; then
      ndone=$((ndone + 1)); continue
    fi

    local st="${qstate[$name]:-}"
    if [ -n "$st" ]; then
      # In the queue. Only a RUNNING job with a stale metrics file is suspect.
      if [ "$st" = "RUNNING" ] && [ -n "$dir" ] && [ "$mtime" -gt 0 ]; then
        local age_min=$(( (now - mtime) / 60 ))
        local start="${qstart[$name]:-0}"
        local run_min=$(( start > 0 ? (now - start) / 60 : 0 ))
        if [ "$age_min" -ge "$STALL_MIN" ] && [ "$run_min" -ge "$MIN_RUN_MIN" ]; then
          nstall=$((nstall + 1))
          log "STALLED  $name job ${qjob[$name]} step=$step metrics ${age_min}m old -> requeue"
          [ "$DRY_RUN" = "1" ] || scontrol requeue "${qjob[$name]}" 2>&1 | sed 's/^/         /'
          printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$name" requeue "${qjob[$name]}" >> "$STATE"
          continue
        fi
      fi
      nalive=$((nalive + 1)); continue
    fi

    # Not queued and not finished: the run died.
    local n; n=$(count_of "$name")
    if [ "$n" -ge "$MAX_RESUBMITS" ]; then
      nfail=$((nfail + 1))
      log "GIVEUP   $name step=$step after $n resubmits -- needs a human"
      continue
    fi
    if [ "$now" -ge "$deadline_epoch" ]; then
      nfail=$((nfail + 1))
      log "PASTDUE  $name step=$step died after the $DEADLINE resubmit deadline"
      continue
    fi

    nresub=$((nresub + 1))
    local exportvars="ALL,EXP_TAG=$tag,TASK=$task,ARM=$arm,SEED=$seed,RUN_STEPS=$RUN_STEPS"
    [ -n "$dir" ] && exportvars="$exportvars,RUN_DIR=$dir"
    log "RESUBMIT $name step=$step resume=${dir:-<fresh>} (attempt $((n + 1)))"
    if [ "$DRY_RUN" != "1" ]; then
      local out
      out=$(sbatch -J "$name" -p "$PARTITION" --gres="$GRES" \
              --export="$exportvars" "$SCRIPT" 2>&1)
      log "         $out"
      printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$name" resubmit "${out##* }" >> "$STATE"
    fi
  done < "$TSV"

  log "PASS done=$ndone alive=$nalive requeued=$nstall resubmitted=$nresub stuck=$nfail"
}

if [ "$LOOP" = "1" ]; then
  while :; do
    pass
    sleep "$INTERVAL"
  done
else
  pass
fi
