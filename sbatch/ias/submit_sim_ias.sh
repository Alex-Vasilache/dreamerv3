#!/bin/bash
# Launch a sim round on the IAS cluster as a throttled Slurm array. Run on the
# IAS login node from a code snapshot:
#   PARAMS=<file> CONC=7 QOS=normal [FIRST=0 LAST=n-1] sbatch/ias/submit_sim_ias.sh
# CONC caps concurrent runs so live demos keep GPUs (limit 10 per user).
set -euo pipefail
cd "$(dirname "$0")/../.."
CODE=$PWD; PARAMS=$(realpath "${PARAMS:?}"); QOS=${QOS:-normal}; CONC=${CONC:-7}
n=$(grep -c -v -e '^#' -e '^\s*$' "$PARAMS")
FIRST=${FIRST:-0}; LAST=${LAST:-$((n - 1))}
LOG=${RUNS:-$HOME/robot_runs}/slurm_logs; mkdir -p "$LOG"
cd "$LOG"
CODE="$CODE" PARAMS="$PARAMS" sbatch --parsable -J "$(basename "${PARAMS%.txt}")" \
  --qos="$QOS" ${TIME:+-t $TIME} --array="${FIRST}-${LAST}%${CONC}" --export=ALL ${DEP:+--dependency=$DEP} \
  "$CODE/sbatch/ias/run_sim_ias.sbatch"
