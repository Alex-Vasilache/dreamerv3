#!/bin/bash -l
#SBATCH -J bench_watchdog
#SBATCH -p intel
#SBATCH -t 3-00:00:00
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Watchdog for the 2026-09-04 benchmark matrix.
#
# SLURM's --requeue covers preemption and node failure but NOT a non-zero exit
# from the application, so a task that dies the way som_lip did on V100 (a NaN
# assertion two minutes in) stays dead. This resubmits those, and only those.
#
# GUARDRAILS, learned from the 2026-08-15 watchdog that resubmitted all 32
# finished runs at once:
#
#  * No generic variable names. That incident was caused by the login profile
#    exporting WORK=/work while the script assumed its own value, so every glob
#    silently pointed at the wrong tree and "no run directory" read as "the run
#    died". Everything here is prefixed BW_.
#  * A pass ABORTS without acting if the number of run directories drops to
#    less than half the most it has ever seen. A missing mount, a wrong root,
#    or an unreadable filesystem all look exactly like "everything died", and
#    the difference is the scale. The comparison is against a HIGH-WATER MARK
#    rather than the expected total, because run directories only appear as
#    tasks start: keying it to 54 would leave the watchdog inert for as long as
#    the A100 queue was backed up, which is exactly when it is needed.
#  * A run is only resubmitted if its log says rc!=0 AND its final step is
#    short of the target AND it is not currently queued or running.
#  * Each task is resubmitted at most BW_MAX_RETRIES times, tracked in a state
#    file, so a deterministically-crashing config cannot loop forever.

set -uo pipefail

BW_REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3
BW_RUNS_DIR=/work/DoyaU/vasilache/work
BW_LOGS="$BW_RUNS_DIR/slurm_logs"
BW_STATE="$BW_RUNS_DIR/bench_watchdog_state.tsv"
BW_INTERVAL="${BW_INTERVAL:-900}"
BW_TARGET_STEPS="${BW_TARGET_STEPS:-1090000}"
BW_MAX_RETRIES="${BW_MAX_RETRIES:-3}"
BW_DEADLINE="${BW_DEADLINE:-2026-09-12T00:00:00}"
BW_EXPECTED="${BW_EXPECTED:-54}"

touch "$BW_STATE"
echo "[watchdog] start $(date -Is) interval=${BW_INTERVAL}s deadline=$BW_DEADLINE"

bw_retries() { awk -F'\t' -v k="$1" '$1==k{n=$2} END{print n+0}' "$BW_STATE"; }
bw_bump() {
  local k=$1 n
  n=$(( $(bw_retries "$k") + 1 ))
  grep -v -P "^\Q$k\E\t" "$BW_STATE" > "$BW_STATE.tmp" 2>/dev/null || : > "$BW_STATE.tmp"
  printf '%s\t%s\t%s\n' "$k" "$n" "$(date -Is)" >> "$BW_STATE.tmp"
  mv "$BW_STATE.tmp" "$BW_STATE"
}

while [ "$(date +%s)" -lt "$(date -d "$BW_DEADLINE" +%s)" ]; do
  BW_NDIRS=$(ls -d "$BW_RUNS_DIR"/e7*_j4706[0-9][0-9][0-9]_* 2>/dev/null | wc -l)
  BW_HWM=$(awk -F'\t' '$1=="__hwm__"{n=$2} END{print n+0}' "$BW_STATE")
  if [ "$BW_NDIRS" -gt "$BW_HWM" ]; then
    grep -v -P '^__hwm__\t' "$BW_STATE" > "$BW_STATE.tmp" 2>/dev/null || : > "$BW_STATE.tmp"
    printf '__hwm__\t%s\t%s\n' "$BW_NDIRS" "$(date -Is)" >> "$BW_STATE.tmp"
    mv "$BW_STATE.tmp" "$BW_STATE"
    BW_HWM=$BW_NDIRS
  fi
  if [ "$BW_NDIRS" -lt $((BW_HWM / 2)) ]; then
    echo "[watchdog] ABORT PASS: $BW_NDIRS run dirs against a high-water mark of $BW_HWM. Not acting."
    sleep "$BW_INTERVAL"; continue
  fi

  BW_ACTED=0
  for BW_OUT in "$BW_LOGS"/bench_*.out; do
    [ -f "$BW_OUT" ] || continue
    BW_RC=$(grep -aoE '^\[done\] rc=[0-9]+' "$BW_OUT" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
    [ -n "$BW_RC" ] && [ "$BW_RC" != 0 ] || continue
    BW_DIR=$(grep -aoE 'run_dir=\S+' "$BW_OUT" 2>/dev/null | tail -1 | cut -d= -f2)
    [ -n "$BW_DIR" ] && [ -d "$BW_DIR" ] || continue
    BW_NAME=$(basename "$BW_DIR")

    # Finished? Then the non-zero rc was on a later, already-complete run.
    BW_STEP=$(python3 - "$BW_DIR" <<'PY' 2>/dev/null
import json, os, sys
p = os.path.join(sys.argv[1], 'logdir', 'metrics.jsonl')
step = 0
try:
    with open(p, 'rb') as f:
        end = f.seek(0, 2); f.seek(max(0, end - 8192))
        for line in f.read().decode(errors='ignore').split('\n'):
            if line.strip():
                try: step = max(step, json.loads(line).get('step', 0) or 0)
                except Exception: pass
except Exception:
    pass
print(int(step))
PY
)
    [ "${BW_STEP:-0}" -ge "$BW_TARGET_STEPS" ] && continue

    # Still queued or running under this name? Leave it alone.
    squeue -u "$(whoami)" -h -o "%Z" 2>/dev/null | grep -qF "$BW_DIR" && continue
    grep -qF "$BW_NAME" <(squeue -u "$(whoami)" -h -o "%j" 2>/dev/null) && continue

    BW_N=$(bw_retries "$BW_NAME")
    if [ "$BW_N" -ge "$BW_MAX_RETRIES" ]; then
      echo "[watchdog] $BW_NAME: $BW_N retries already, giving up (rc=$BW_RC step=$BW_STEP)"
      continue
    fi

    BW_ENV="$BW_DIR/job.env"
    BW_CFG=$(awk -F= '$1=="CONFIG"{print $2}' "$BW_ENV" 2>/dev/null)
    BW_TASK=$(awk -F= '$1=="TASK"{print $2}' "$BW_ENV" 2>/dev/null)
    BW_SEED=$(awk -F= '$1=="SEED"{print $2}' "$BW_ENV" 2>/dev/null)
    BW_TAG=$(awk -F= '$1=="EXP_TAG"{print $2}' "$BW_ENV" 2>/dev/null)
    [ -n "$BW_CFG" ] && [ -n "$BW_TASK" ] || { echo "[watchdog] $BW_NAME: unreadable job.env, skipping"; continue; }

    # Resubmit on A100 with the same RUN_DIR so it RESUMES from the checkpoint.
    # ONE partition only: our association is per-partition, and a multi-partition
    # request is rejected outright with "Multiple partition job request not
    # supported when a partition is set in the association". Pick whichever of
    # our three has a free slot, else fall back to the preemptible short-a100
    # (runs are resumable, so preemption costs one checkpoint interval).
    BW_PART=""
    for cand in gpu-a100 gpu-v100 gpu-p100; do
      BW_USED=$(squeue -u "$(whoami)" -h -p "$cand" -t RUNNING,PENDING -o "%D" | wc -l)
      if [ "$BW_USED" -lt 8 ]; then BW_PART=$cand; break; fi
    done
    [ -n "$BW_PART" ] || BW_PART=short-a100
    BW_EXTRA=""
    [ "$BW_PART" = gpu-p100 ] && BW_EXTRA="--nodelist=saion-gpu[11-14]"
    # shellcheck disable=SC2086
    # Values go through the ENVIRONMENT with --export=ALL, not through the
    # --export list. A layered arm's CONFIG contains a space
    # ("director mgr_expl_perc01") and sbatch's --export parser treats that as
    # the end of the list, silently dropping everything after it.
    BW_JID=$(CONFIG="$BW_CFG" TASK="$BW_TASK" SEED="$BW_SEED" EXP_TAG="$BW_TAG" \
      RUN_DIR="$BW_DIR" WANDB_PROJECT=dreamerv3-bench-2026-09 \
      sbatch --parsable -J "bench_retry" -p "$BW_PART" $BW_EXTRA \
      --gres=gpu:1 -c 8 --mem=48G -t 2-00:00:00 --export=ALL \
      "$BW_REPO/sbatch/run_benchmark.sbatch" 2>&1)
    if [[ "$BW_JID" =~ ^[0-9]+$ ]]; then
      bw_bump "$BW_NAME"
      echo "[watchdog] resubmitted $BW_NAME (rc=$BW_RC step=$BW_STEP retry=$((BW_N + 1))) as job $BW_JID"
      BW_ACTED=$((BW_ACTED + 1))
    else
      echo "[watchdog] $BW_NAME: sbatch failed: $BW_JID"
    fi
  done

  echo "[watchdog] $(date -Is) pass done, dirs=$BW_NDIRS resubmitted=$BW_ACTED"
  sleep "$BW_INTERVAL"
done
echo "[watchdog] deadline reached, exiting"
