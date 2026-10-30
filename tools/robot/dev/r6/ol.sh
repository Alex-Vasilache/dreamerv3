#!/bin/bash
#SBATCH -p intel -c 32 --mem=32G -t 0-1 -o /work/DoyaU/vasilache/work/sysid/r6/logs/%x_%j.out
# ol.sh TAG FIT [overrides...]: open loop 0.3 s on the command + balancing + held-out sets
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh >/dev/null 2>&1
export CODE=/apps/unit/DoyaU/vasilache/apps/code/robot/dreamerv3
S=/work/DoyaU/vasilache/work/sysid; R=$S/recordings/sysid
TAG=$1; FIT=$2; shift 2
python -u $S/r5/ol.py $FIT ${WIN:-0.3} $S/r6/ol_$TAG.pkl \
  $SETS \
  -- "$@" 2>&1 | grep -v -i "warn\|unstable"
