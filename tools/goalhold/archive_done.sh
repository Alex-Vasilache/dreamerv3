#!/bin/bash
# Archive goal-hold array runs whose SLURM task is in a terminal state. Copy (no replay), verify, then delete.
A=4812978; W=/work/DoyaU/vasilache/work; B=/bucket/DoyaU/vasilache/bucket/results/dreamerv3; L=$W/slurm_logs
for d in $W/h0[34]*_j${A}_*; do
  [ -d "$d" ] || continue
  run=$(basename $d); task=${run##*_}; job=$(grep '^JOB=' $d/job.env | cut -d= -f2)
  st=$(sacct -j ${A}_${task} -X -n -o State | head -1 | tr -d ' +')
  case "$st" in COMPLETED|CANCELLED|FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) ;; *) continue ;; esac
  squeue -h -j ${A}_${task} 2>/dev/null | grep -q . && continue
  echo "== $run ($st, job $job)"
  SKIP_REPLAY=1 $W/copy_dreamerv3_runs_to_bucket.sh $run > /dev/null 2>&1; rc=$?
  mkdir -p $B/$run/slurm_logs; cp -n $L/goalhold-${job}.out $L/goalhold-${job}.err $B/$run/slurm_logs/ 2>/dev/null
  ok=1
  [ $rc -eq 0 ] || ok=0
  [ -s $B/$run/logdir/metrics.jsonl ] && cmp -s $d/logdir/scores.jsonl $B/$run/logdir/scores.jsonl || ok=0
  [ -s $B/$run/slurm_logs/goalhold-${job}.out ] || ok=0
  if [ "$st" = COMPLETED ]; then [ -d $B/$run/logdir/ckpt_milestones/000004000000 ] || ok=0; fi
  if [ $ok -eq 1 ]; then rm -rf "$d"; rm -f $L/goalhold-${job}.out $L/goalhold-${job}.err; echo "   archived + deleted"; else echo "   VERIFY FAILED (rc=$rc), left in place"; fi
done
