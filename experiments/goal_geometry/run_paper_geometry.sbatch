#!/bin/bash -l
#SBATCH -J paper_geom
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Goal-code geometry for the paper on the matched-protocol runs (2026-10-15):
# code_sim (final + 1M/2M/3M/4M milestones) and idx_dist, per run, read
# straight from the bucket. RUNS is "arm:run_dir arm:run_dir ...".
#
#   RUNS="now:/bucket/.../e1233_... som:/bucket/.../e1333_..." \
#     sbatch experiments/goal_geometry/run_paper_geometry.sbatch
#
# Output: $OUT/code_sim/<arm>/<run>.npz, $OUT/code_sim_ckpt/<arm>/<step>/<run>.npz,
# $OUT/idx_dist/<arm>/<run>.npz. Existing files are skipped.
set -uo pipefail
REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export DREAMERV3_CONV_IMPL=reference   # forward pass only; V100 lane
export MUJOCO_GL=egl
OUT="${OUT:-/work/DoyaU/vasilache/work/paper_geom}"
G=$REPO/experiments/goal_geometry
cd "$G"

fail=0
for item in ${RUNS:?set RUNS}; do
  arm=${item%%:*}; run=${item#*:}; name=$(basename "$run")
  f=$OUT/code_sim/$arm/$name
  [ -f "$f.npz" ] || python -u code_sim_to_state_sim.py --run_dir "$run" --out "$f" \
      --n_codes 1000 --n_partners 100 || fail=1
  for s in 000001000000 000002000000 000003000000 000004000000; do
    f=$OUT/code_sim_ckpt/$arm/$s/$name
    [ -f "$f.npz" ] || python -u code_sim_to_state_sim.py --run_dir "$run" --out "$f" \
        --n_codes 1000 --n_partners 100 --ckpt "$run/logdir/ckpt_milestones/$s" || fail=1
  done
  f=$OUT/idx_dist/$arm/$name
  [ -f "$f.npz" ] || python -u code_sim_index_distance.py --run_dir "$run" --out "$f" || fail=1
done
exit $fail
