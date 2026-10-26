#!/bin/bash -l
#SBATCH -J ucb_kde_fit
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# ucb_kde_fit.py: MODE=collect TASK=<task> collects seeds 0-1 of V+SOM at 0.5M,
# 2M and final; MODE=fit runs the grid over everything collected.
set -uo pipefail
REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export DREAMERV3_CONV_IMPL=reference
export MUJOCO_GL=egl
OUT=/work/DoyaU/vasilache/work/paper_geom/ucb_kde_fit
mkdir -p $OUT
B=/bucket/DoyaU/vasilache/bucket/results/dreamerv3
cd $REPO
if [ "${MODE:?}" = fit ]; then
  python -u tools/collapse_study/ucb_kde_fit.py fit $OUT; exit $?
fi
if [ "$MODE" = metric ]; then
  fail=0
  for f in $OUT/*_${TASK:?}_*_deter.npy; do
    b=$(basename $f _deter.npy); m=$OUT/${b}_metric.npz
    [ -f $m ] && continue
    ck=${b##*_}; run=$(ls -d $B/${b%%_*}_*); [ "$ck" = final ] || true
    python -u tools/collapse_study/ucb_kde_fit.py metric $run $ck $m || fail=1
  done
  exit $fail
fi
fail=0
for run in $(ls -d $B/e20[0-5]?_${TASK:?}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-pubmatch_s[01]_*); do
  for ck in 000000500000 000002000000 final; do
    f=$OUT/$(basename $run | cut -c1-5)_${TASK}_$ck.npz
    [ -f $f ] || python -u tools/collapse_study/ucb_kde_fit.py collect $run $ck $f || fail=1
  done
done
exit $fail
