#!/bin/bash
# Let preempted (requeued) array tasks restart before never-started ones.
# SLURM gave fresh tasks the free slots while requeued ones waited; this holds
# every pending task that has no run dir yet whenever a task WITH a run dir
# (i.e. already started, then preempted) is pending, and releases them after.
# Usage: nohup requeue_first.sh <array_job_id> <run-dir substring> &
ARR=$1; MATCH=$2; W=/work/DoyaU/vasilache/work
while :; do
  if ! squeue -u "$USER" -h -o '%i' > /dev/null 2>&1; then sleep 120; continue; fi
  pend=$(squeue -u "$USER" -h -r -t PENDING -o '%i %R' | awk -v a="${ARR}_" 'index($1,a)==1')
  [ -z "$(squeue -u "$USER" -h -r -o '%i' | grep "^${ARR}_")" ] && { echo "$(date '+%F %T') requeue_first: array done"; exit 0; }
  started=(); fresh=()
  while read -r id reason; do
    [ -n "$id" ] || continue
    t=${id#${ARR}_}
    if ls -d $W/*_j${ARR}_${t} 2>/dev/null | grep -q -- "$MATCH"; then started+=("$id"); else fresh+=("$id:$reason"); fi
  done <<< "$pend"
  if [ ${#started[@]} -gt 0 ]; then
    for f in "${fresh[@]}"; do id=${f%%:*}; [[ $f == *JobHeldUser* ]] || scontrol hold "$id" 2>/dev/null; done
    [ ${#fresh[@]} -gt 0 ] && echo "$(date '+%F %T') requeue_first: ${#started[@]} requeued waiting (${started[*]}), fresh tasks held"
  else
    for f in "${fresh[@]}"; do [[ $f == *JobHeldUser* ]] && scontrol release "${f%%:*}" 2>/dev/null; done
  fi
  sleep 120
done
