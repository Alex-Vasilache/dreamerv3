#!/bin/bash
# Keep up to 8 of my jobs on gpu-a100: whenever one finishes, move the next
# queued V100/P100 job there (tools/collapse_study/migrate_to_a100.sh).
# Usage: nohup auto_migrate.sh <jobid> [...] &   (in priority order)
cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3
for jid in "$@"; do
  [ -n "$(squeue -h -j "$jid" 2>/dev/null)" ] || continue   # already finished
  while [ "$(squeue -u "$USER" -h -p gpu-a100 -t RUNNING,PENDING | wc -l)" -ge 8 ]; do sleep 120; done
  [ -n "$(squeue -h -j "$jid" 2>/dev/null)" ] || continue
  echo "$(date '+%F %T') $(tools/collapse_study/migrate_to_a100.sh "$jid")"
done
echo "$(date '+%F %T') auto_migrate done"
