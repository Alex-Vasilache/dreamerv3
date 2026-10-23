#!/bin/bash -l
#SBATCH -J sysid_fit
#SBATCH -p intel
#SBATCH -t 10:00:00
#SBATCH -c 40
#SBATCH --mem=32G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Fit the MuJoCo robot to sysid_probe.py recordings (tools/robot/dev/sysid_fit.py)
# on a CPU node: a global search over motor and floors together, then
# Nelder-Mead from its best. Code and recordings are synced from the Mac into
# $SYSID (tools/robot/dev/sysid_fit.py's docstring); each window's rollout is
# one worker task, so -c sets the speed.
#
#   SYSID=/work/DoyaU/vasilache/work/sysid NAME=fit2 \
#   ARGS="--joint --method de --iters 60" sbatch sbatch/run_sysid_fit.sbatch

set -euo pipefail
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
SYSID=${SYSID:-/work/DoyaU/vasilache/work/sysid}
NAME=${NAME:-fit}
cd "$SYSID/code"
mkdir -p "$SYSID/fits"
python -u tools/robot/dev/sysid_fit.py "$SYSID"/data/* \
  --workers "${SLURM_CPUS_PER_TASK:-40}" \
  --out "$SYSID/fits/$NAME.json" ${ARGS:-} 2>&1 \
  | grep -v --line-buffered 'simulation is unstable\|^$' \
  | tee "$SYSID/fits/$NAME.log"
