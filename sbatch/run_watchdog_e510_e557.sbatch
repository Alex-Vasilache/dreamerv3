#!/bin/bash -l
#SBATCH -J watchdog_e510_e557
#SBATCH -p intel
#SBATCH -t 7-12:00:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

# Keeps the e510-e557 batch alive while nobody is watching it: every 30 minutes
# it resumes runs that died and requeues runs that stopped writing metrics. See
# sbatch/watchdog_e510_e557.sh for what it will and will not do.
#
# It lives on `intel` because that partition costs no GPU quota, has no wall
# limit, and is el8 -- and because a supervisor that competes with the runs it
# supervises for a GPU is not a supervisor. The loop derives all of its state
# from squeue and the filesystem on every pass, so a requeue of this job (intel
# is preemptible) costs nothing but the gap.
#
#   sbatch sbatch/run_watchdog_e510_e557.sbatch
#   scancel <jobid>     # to stop supervision
set -uo pipefail

REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3_somvae

echo "[watchdog] starting at $(date -Is) as job $SLURM_JOB_ID"
echo "[watchdog] repo=$REPO interval=${INTERVAL:-1800}s deadline=${DEADLINE:-2026-08-16T12:00:00}"

LOOP=1 INTERVAL="${INTERVAL:-1800}" exec "$REPO/sbatch/watchdog_e510_e557.sh"
