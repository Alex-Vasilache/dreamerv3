#!/bin/bash -l
#SBATCH -J ucb_table_fit
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# ucb_table_fit.py on V+SOM+even (e2006-e2011 seed 0, e2018-e2023 seed 1) at 2M:
# all collects, then the fit, in one job.
set -uo pipefail
REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export DREAMERV3_CONV_IMPL=reference
export MUJOCO_GL=egl
OUT=/work/DoyaU/vasilache/work/paper_geom/ucb_table_fit
mkdir -p $OUT
B=/bucket/DoyaU/vasilache/bucket/results/dreamerv3
cd $REPO
for run in $(ls -d $B/e20[0-5]?_*_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-pubmatch_s[01]_*); do
  f=$OUT/$(basename $run | cut -d_ -f1)_2M.npz
  [ -f $f ] || python -u tools/collapse_study/ucb_table_fit.py collect $run 000002000000 $f
done
python -u tools/collapse_study/ucb_table_fit.py fit $OUT
