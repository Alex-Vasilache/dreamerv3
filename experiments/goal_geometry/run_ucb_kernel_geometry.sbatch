#!/bin/bash -l
#SBATCH -J ucb_kgeom
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Per-block class geometry for the UCB count kernel (ucb_kernel_geometry.py) on
# the matched-protocol OG / OG+SOM / OG+SOM+LiP runs, final checkpoint.
# Output: $OUT/<arm>/<run>.npz; existing files are skipped.
set -uo pipefail
REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export DREAMERV3_CONV_IMPL=reference
export MUJOCO_GL=egl
OUT="${OUT:-/work/DoyaU/vasilache/work/paper_geom/ucb_kernel}"
B=/bucket/DoyaU/vasilache/bucket/results/dreamerv3
cd "$REPO/experiments/goal_geometry"
fail=0
for t in ${TASKS:-pinpad_four pinpad_five pinpad_six dmc_cartpole_swingup dmc_cheetah_run dmc_hopper_hop}; do
  for spec in "og:e12[3-8]?_${t}_director-director_og-pubmatch_s*" \
              "som:e1[34]??_${t}_director-director_og-goal_som_line-pubmatch_s*" \
              "somlip:e1[34]??_${t}_director-director_og-goal_som_lipvq_line_prod-lip_relu-goal_vq_f32-pubmatch_s*"; do
    arm=${spec%%:*}
    for run in $B/${spec#*:}; do
      f=$OUT/$arm/$(basename "$run")
      [ -f "$f.npz" ] || python -u ucb_kernel_geometry.py --run_dir "$run" --out "$f" || fail=1
    done
  done
done
exit $fail
