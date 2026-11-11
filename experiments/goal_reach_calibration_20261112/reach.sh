#!/bin/bash
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl
cd /work/DoyaU/vasilache/work/meetings/tce_investigation
B=/bucket/DoyaU/vasilache/bucket/results/dreamerv3
while read name run step; do
  grep -q "^saved" reach/$name.log 2>/dev/null && continue
  python -u reach_probe.py $(basename $B/${run}_*) $step reach/$name > reach/$name.log 2>&1; echo "$name exit $?"
done
