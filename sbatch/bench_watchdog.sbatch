#!/bin/bash -l
#SBATCH -J bench_watchdog
#SBATCH -p intel
#SBATCH -t 3-00:00:00
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300

# Watchdog for the 2026-09 benchmark matrix.
#
# WHAT IT DETECTS. A run that is not finished and has no live job behind it.
# The first version keyed off "the log says rc!=0", which misses the most
# common way a run actually dies: anything that kills the process before it
# writes its exit line -- scancel, an OOM kill, a node going away. e795 sat
# dead for 25 hours that way. Staleness of logdir/metrics.jsonl is the honest
# signal, because a live run rewrites it every `run.log_every` (120 s).
#
# HOW IT AVOIDS DOUBLE-SUBMITTING. Three independent checks, all cheap:
#   1. the job id recorded in the run's own job.env is still in the queue;
#   2. the job id THIS watchdog last submitted for that run is still in the
#      queue -- job.env is stale until the resubmitted job actually starts, and
#      that window is exactly where a naive check would fire twice;
#   3. metrics.jsonl was touched within BW_STALE_MIN.
# Any one of them means "leave it alone".
#
# GUARDRAILS, from the 2026-08-15 watchdog that resubmitted 32 finished runs:
#   * every variable is BW_-prefixed (that incident was a collision with the
#     login profile's WORK=/work);
#   * a pass aborts without acting if the run-directory count falls below half
#     the most it has ever seen -- a missing mount looks exactly like "they all
#     died", and only the scale distinguishes the two;
#   * a per-run retry cap, so a deterministically-crashing config cannot loop.

set -uo pipefail

BW_REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3
BW_RUNS_DIR=/work/DoyaU/vasilache/work
BW_STATE="$BW_RUNS_DIR/bench_watchdog_state.tsv"
BW_DROPLIST="$BW_RUNS_DIR/bench_droplist.txt"
BW_INTERVAL="${BW_INTERVAL:-900}"
BW_TARGET_STEPS="${BW_TARGET_STEPS:-1090000}"
BW_MAX_RETRIES="${BW_MAX_RETRIES:-4}"
BW_STALE_MIN="${BW_STALE_MIN:-45}"
BW_DEADLINE="${BW_DEADLINE:-2026-10-01T00:00:00}"
# Space reclaim: a FINISHED run's logdir/replay is dead weight -- multi-GB of
# raw experience needed only to resume training. Deleting it is safe once the
# run is at target and, unlike archiving to /bucket, works from a compute node
# (/bucket is read-only there). Checkpoints and ckpt_milestones are untouched.
BW_REAP_REPLAY="${BW_REAP_REPLAY:-1}"
BW_PROJECT="${BW_PROJECT:-dreamerv3-bench-2026-09}"

# The walltime is 3 days but this has to survive weeks, so hand off to a
# successor shortly before being killed rather than relying on a human.
bw_relaunch() {
  if [ "$(date +%s)" -lt "$(date -d "$BW_DEADLINE" +%s)" ]; then
    local nxt
    nxt=$(sbatch --parsable "$BW_REPO/sbatch/bench_watchdog.sbatch" 2>&1)
    echo "[watchdog] walltime near; successor submitted: $nxt"
  else
    echo "[watchdog] walltime near and deadline passed; not relaunching"
  fi
  exit 0
}
trap bw_relaunch USR1

touch "$BW_STATE"
echo "[watchdog] start $(date -Is) interval=${BW_INTERVAL}s stale=${BW_STALE_MIN}min deadline=$BW_DEADLINE"

bw_get() { awk -F'\t' -v k="$1" -v f="$2" '$1==k{v=$f} END{print v}' "$BW_STATE"; }
bw_put() {  # key retries lastjob
  grep -v -P "^\Q$1\E\t" "$BW_STATE" > "$BW_STATE.tmp" 2>/dev/null || : > "$BW_STATE.tmp"
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date -Is)" >> "$BW_STATE.tmp"
  mv "$BW_STATE.tmp" "$BW_STATE"
}
bw_in_queue() { [ -n "${1:-}" ] && [ "$1" != "-" ] && squeue -j "$1" -h -o "%i" 2>/dev/null | grep -q . ; }

while [ "$(date +%s)" -lt "$(date -d "$BW_DEADLINE" +%s)" ]; do
  # Which dirs are ours: the ones whose job.env carries the benchmark's wandb
  # project. A job-id prefix glob (j4706*/j4707*) was wrong twice -- it missed
  # every later submission, so the director_og and size50m arrays ran with no
  # watchdog at all. Job ids only go up; a marker does not age.
  BW_DIRS=()
  for d in "$BW_RUNS_DIR"/e[0-9]*_j*; do
    [ -d "$d" ] || continue
    grep -qxF "WANDB_PROJECT=$BW_PROJECT" "$d/job.env" 2>/dev/null || continue
    BW_DIRS+=("$d")
  done
  BW_NDIRS=${#BW_DIRS[@]}
  BW_HWM=$(bw_get __hwm__ 2); BW_HWM=${BW_HWM:-0}
  if [ "$BW_NDIRS" -gt "$BW_HWM" ]; then bw_put __hwm__ "$BW_NDIRS" -; BW_HWM=$BW_NDIRS; fi
  if [ "$BW_NDIRS" -lt $(( BW_HWM / 2 )) ]; then
    echo "[watchdog] ABORT PASS: $BW_NDIRS dirs against high-water mark $BW_HWM. Not acting."
    sleep "$BW_INTERVAL"; continue
  fi

  BW_ACTED=0; BW_DONE=0; BW_LIVE=0
  for BW_DIR in "${BW_DIRS[@]}"; do
    [ -d "$BW_DIR" ] || continue
    BW_NAME=$(basename "$BW_DIR")
    BW_M="$BW_DIR/logdir/metrics.jsonl"

    # Target is per run, not global: the size6m arms stop at 1.1M but the
    # director_og and size50m arms run 4M. A single hardcoded 1.09M would call
    # a 4M run finished at 27% and -- worse -- reap its replay, which is the one
    # thing that makes it resumable. Read what the run itself was configured
    # with; the 0.99 margin is because the driver checks between chunks and
    # stops a few thousand steps short.
    BW_CFGSTEPS=$(awk '/^run:/{r=1;next} /^[a-z]/{r=0} r&&/^  steps:/{print $2;exit}' \
      "$BW_DIR/logdir/config.yaml" 2>/dev/null)
    BW_TGT=$(awk -v s="${BW_CFGSTEPS:-0}" -v d="$BW_TARGET_STEPS" \
      'BEGIN{printf "%d", (s+0 > 0) ? s*0.99 : d}')

    BW_STEP=$(python3 - "$BW_M" <<'PY' 2>/dev/null
import json, sys
step = 0
try:
    for line in open(sys.argv[1]):
        if line.strip():
            try: step = json.loads(line).get('step', step) or step
            except Exception: pass
except Exception:
    pass
print(int(step))
PY
)
    if [ "${BW_STEP:-0}" -ge "$BW_TGT" ]; then
      BW_DONE=$((BW_DONE+1))
      if [ "$BW_REAP_REPLAY" = 1 ] && [ -d "$BW_DIR/logdir/replay" ]; then
        BW_SZ=$(du -sm "$BW_DIR/logdir/replay" 2>/dev/null | cut -f1)
        rm -rf "$BW_DIR/logdir/replay" && \
          echo "[watchdog] reaped ${BW_SZ}MB of replay from finished $BW_NAME"
      fi
      continue
    fi

    if [ -f "$BW_M" ]; then
      BW_AGE=$(( ( $(date +%s) - $(stat -c %Y "$BW_M") ) / 60 ))
    else
      BW_AGE=$(( ( $(date +%s) - $(stat -c %Y "$BW_DIR") ) / 60 ))
    fi
    if [ "$BW_AGE" -lt "$BW_STALE_MIN" ]; then BW_LIVE=$((BW_LIVE+1)); continue; fi

    BW_ENVJOB=$(awk -F= '$1=="JOB"{print $2}' "$BW_DIR/job.env" 2>/dev/null)
    if bw_in_queue "$BW_ENVJOB"; then BW_LIVE=$((BW_LIVE+1)); continue; fi
    BW_OURJOB=$(bw_get "$BW_NAME" 3)
    if bw_in_queue "$BW_OURJOB"; then BW_LIVE=$((BW_LIVE+1)); continue; fi

    # Explicit drop list: runs deliberately abandoned. Without this a cancelled
    # run looks exactly like a crashed one 45 minutes later, and the watchdog
    # dutifully brings it back.
    if [ -f "$BW_DROPLIST" ] && grep -qxF "$BW_NAME" "$BW_DROPLIST"; then continue; fi

    BW_N=$(bw_get "$BW_NAME" 2); BW_N=${BW_N:-0}
    if [ "$BW_N" -ge "$BW_MAX_RETRIES" ]; then
      echo "[watchdog] $BW_NAME: $BW_N retries already, giving up (step=$BW_STEP idle=${BW_AGE}min)"
      continue
    fi

    BW_CFG=$(awk -F= '$1=="CONFIG"{print $2}' "$BW_DIR/job.env" 2>/dev/null)
    BW_TASK=$(awk -F= '$1=="TASK"{print $2}' "$BW_DIR/job.env" 2>/dev/null)
    BW_SEED=$(awk -F= '$1=="SEED"{print $2}' "$BW_DIR/job.env" 2>/dev/null)
    BW_TAG=$(awk -F= '$1=="EXP_TAG"{print $2}' "$BW_DIR/job.env" 2>/dev/null)
    [ -n "$BW_CFG" ] && [ -n "$BW_TASK" ] || { echo "[watchdog] $BW_NAME: unreadable job.env, skipping"; continue; }

    # One partition only: a multi-partition request is rejected outright when
    # the association is per-partition.
    # size50m needs 24 GB of GPU memory; a 16 GB V100/P100 cannot hold it, so
    # that arm is A100-only. The small arms may go anywhere.
    case "$BW_CFG" in
      *size50m*) BW_CANDS="gpu-a100"; BW_MEM=96G ;;
      *)         BW_CANDS="gpu-a100 gpu-v100 gpu-p100"; BW_MEM=48G ;;
    esac
    BW_PART=""
    for cand in $BW_CANDS; do
      [ "$(squeue -u "$(whoami)" -h -p "$cand" -t RUNNING,PENDING -o '%i' | wc -l)" -lt 8 ] && { BW_PART=$cand; break; }
    done
    [ -n "$BW_PART" ] || { echo "[watchdog] $BW_NAME: all partitions full, retry next pass"; continue; }
    BW_EXTRA=""; [ "$BW_PART" = gpu-p100 ] && BW_EXTRA="--nodelist=saion-gpu[11-14]"

    # Values travel through the ENVIRONMENT with --export=ALL: the --export list
    # is comma-separated, so a layered CONFIG ("director mgr_expl_perc01") would
    # end it early and silently drop TASK/SEED/RUN_DIR.
    # shellcheck disable=SC2086
    BW_JID=$(CONFIG="$BW_CFG" TASK="$BW_TASK" SEED="$BW_SEED" EXP_TAG="$BW_TAG" \
      RUN_DIR="$BW_DIR" WANDB_PROJECT="$BW_PROJECT" \
      sbatch --parsable -J bench_retry -p "$BW_PART" $BW_EXTRA \
      --gres=gpu:1 -c 8 --mem="$BW_MEM" -t 2-00:00:00 --export=ALL \
      "$BW_REPO/sbatch/run_benchmark.sbatch" 2>&1)
    if [[ "$BW_JID" =~ ^[0-9]+$ ]]; then
      bw_put "$BW_NAME" $((BW_N + 1)) "$BW_JID"
      echo "[watchdog] RESUBMIT $BW_NAME step=$BW_STEP idle=${BW_AGE}min retry=$((BW_N+1)) -> job $BW_JID on $BW_PART"
      BW_ACTED=$((BW_ACTED + 1))
    else
      echo "[watchdog] $BW_NAME: sbatch failed: $BW_JID"
    fi
  done

  echo "[watchdog] $(date -Is) dirs=$BW_NDIRS done=$BW_DONE live=$BW_LIVE resubmitted=$BW_ACTED"
  sleep "$BW_INTERVAL"
done
echo "[watchdog] deadline reached, exiting"
