#!/bin/bash
# ckptsweep.sh TAG FLOOR N STRIDE RUN...: evalpol over the last N milestones (every STRIDE-th) of each run.
TAG=$1; FLOOR=$2; N=$3; STRIDE=$4; shift 4
PY=/home/g/george-vasilache/tmp/claude-11568/-apps-unit-DoyaU-vasilache-apps-code-robot/ecfef5f0-0409-4746-b1d6-0fa2c30562a7/scratchpad/pv/bin/python
for run in "$@"; do
  for s in $(ls $run/logdir/ckpt_milestones | sort | awk -v n=$N -v k=$STRIDE '{a[NR]=$1} END {for (i=NR; i>0 && c<n; i-=k) {print a[i]+0; c++}}'); do
    CKPT=$s $PY evalpol.py $TAG ${CODE:-/work/DoyaU/vasilache/work/sysid/code_r6t4} $FLOOR $run | tail -1
  done
done
