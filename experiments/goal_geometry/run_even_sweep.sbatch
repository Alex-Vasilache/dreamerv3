#!/bin/bash -l
#SBATCH -J even_sweep
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Offline evenness-weight sweep (even_sweep.py) for one task: collect deter
# states with the OG+SOM seed-0 run's policy, then fit fresh goal AEs per weight.
#   TASK=dmc_cartpole_swingup sbatch run_even_sweep.sbatch
set -uo pipefail
REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export DREAMERV3_CONV_IMPL=reference
export MUJOCO_GL=egl
OUT=/work/DoyaU/vasilache/work/paper_geom/even_sweep
mkdir -p $OUT
B=/bucket/DoyaU/vasilache/bucket/results/dreamerv3
RUN=$(ls -d $B/e1[34]??_${TASK:?}_director-director_og-goal_som_line-pubmatch_s0_* | head -1)
cd "$REPO/experiments/goal_geometry"
[ -f $OUT/${TASK}_states.npz ] || python -u even_sweep.py collect --run_dir "$RUN" --out $OUT/${TASK}_states.npz || exit 1
python -u even_sweep.py fit --run_dir "$RUN" --states $OUT/${TASK}_states.npz --out $OUT/${TASK}_fit.npz ${WEIGHTS:+--weights $WEIGHTS}
