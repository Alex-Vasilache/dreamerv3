#!/bin/bash
# Keep e592-e611 running in strict seed-major order, against preemption.
#
# --nice orders the PENDING queue, which is not enough. When a running job is
# preempted it leaves the queue, is briefly held before becoming eligible again,
# and the backfill scheduler gives its slot to whatever is next in line. That is
# how e595 (seed 0) lost its GPU to e600 (seed 2) on 2026-08-20 -- nice cannot
# protect a job that is momentarily not in the queue.
#
# The fix is to keep every later-seed job HELD, so there is nothing for the
# scheduler to backfill with, and release them one at a time as slots free. A
# held job cannot jump the queue no matter what the scheduler does.
#
# The lane caps at 8 concurrent jobs on BOTH limits at once:
#   gpu-a100  cpu=128, gres/gpu=8   and each job takes -c 16, so 8 x 16 = 128.
# There is no slack, so a job that leaves must wait for one to finish.
#
#   nohup sbatch/seed_major_releaser.sh &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANE="${LANE:-gpu-a100}"
WANT="${WANT:-8}"                 # concurrent jobs the lane supports
POLL="${POLL:-300}"

me="$(whoami)"

while :; do
  # Everything of ours in the lane that is NOT user-held is either running or
  # eligible to run; that is what counts against the cap.
  active=$(squeue -u "$me" -h -p "$LANE" -o "%i %T %r" \
           | grep -v 'JobHeldUser' | wc -l)
  held=$(squeue -u "$me" -h -p "$LANE" -o "%i %r" | grep -c 'JobHeldUser' || true)

  if [ "$held" -eq 0 ]; then
    echo "[$(date -Is)] nothing held; releaser done"
    exit 0
  fi

  if [ "$active" -lt "$WANT" ]; then
    # Release in EXPERIMENT-NUMBER order, which is seed-major by construction
    # (e592-e595 seed 0, e596-e599 seed 1, ...).
    n=$((WANT - active))
    nxt=$(squeue -u "$me" -h -p "$LANE" -o "%i|%j|%r" \
          | grep 'JobHeldUser' \
          | awk -F'|' '{split($2,a,"_"); print substr(a[1],2)"|"$1}' \
          | sort -n | head -"$n" | cut -d'|' -f2)
    for j in $nxt; do
      name=$(squeue -j "$j" -h -o "%j" 2>/dev/null)
      echo "[$(date -Is)] active=$active/$WANT -> releasing $j ($name)"
      scontrol release "$j"
      sleep 3
    done
  else
    echo "[$(date -Is)] active=$active/$WANT held=$held, nothing to do"
  fi
  sleep "$POLL"
done
