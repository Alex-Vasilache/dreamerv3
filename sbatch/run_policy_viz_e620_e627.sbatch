#!/bin/bash -l
#SBATCH -J policy_viz
#SBATCH -p gpu-v100
#SBATCH -t 0-04:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Policy mask-viz panels for the e620-e627 exploration benchmarks, in the same
# form as experiments/manager_head/policy/e592.gif: per frame, the observation,
# the running goal code with the blocks the manager changed tinted, and the
# active goal decoded back to pixels.
#
# ON gpu-v100 DELIBERATELY. These are read-only rollouts and the a100 partition
# is full of live training; v100 is idle and a 520-step rollout does not need
# the speed. Do NOT `unset DREAMERV3_CONV_IMPL` here -- native cuDNN conv is
# broken on this V100 image (see the Route-A note), and the default `reference`
# impl the env script sets is what works.
#
# Reads ckpt_milestones/, never ckpt/latest: four of these runs are training
# right now and `latest` can be mid-write, while a milestone is immutable. All
# four antmaze runs share 1.5M and all four pinpad runs share 4.0M, so each
# arm-vs-arm pair is compared at a matched step.
#
#   sbatch --export=ALL,FAMILY=antmaze sbatch/run_policy_viz_e620_e627.sbatch
#   sbatch --export=ALL,FAMILY=pinpad  sbatch/run_policy_viz_e620_e627.sbatch
set -uo pipefail

REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3
W=/work/DoyaU/vasilache/work
OUT="$REPO/experiments/manager_head/policy"

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

FAMILY="${FAMILY:?set FAMILY=antmaze|pinpad}"
STEPS="${STEPS:-520}"

# Each row is "<output basename> <run dir>". The basename carries experiment,
# task, arm and seed so the files are distinguishable without opening them --
# `e622.mp4` said nothing about what it showed.
case "$FAMILY" in
  antmaze)
    STEP="${STEP:-000001500000}"; TAG=1.5M
    RUNS=(
      "e620_antmazeXL_director_s0 $W/e620_loconav_ant_maze_xl_director_s0_BIG_j4697896"
      "e621_antmazeXL_director_s1 $W/e621_loconav_ant_maze_xl_director_s1_BIG_j4697897"
      "e622_antmazeXL_gaussian-eps_s0 $W/e622_loconav_ant_maze_xl_som_lipvq_line_relu_gaussian_eps_s0_BIG_j4697898"
      "e623_antmazeXL_gaussian-eps_s1 $W/e623_loconav_ant_maze_xl_som_lipvq_line_relu_gaussian_eps_s1_BIG_j4697899"
    ) ;;
  pinpad)
    # 2M, not the 4M all four runs also have: by 4M three of the four seeds
    # score ~0 and the panels show a settled policy, while 2M is mid-training
    # and the goal-code edits are still moving.
    STEP="${STEP:-000002000000}"; TAG=2M
    RUNS=(
      "e624_pinpad6_director_s0 $W/e624_pinpad_six_director_s0_BIG_j4697904"
      "e625_pinpad6_director_s1 $W/e625_pinpad_six_director_s1_BIG_j4697905"
      "e626_pinpad6_gaussian-eps_s0 $W/e626_pinpad_six_som_lipvq_line_relu_gaussian_eps_s0_BIG_j4697906"
      "e627_pinpad6_gaussian-eps_s1 $W/e627_pinpad_six_som_lipvq_line_relu_gaussian_eps_s1_BIG_j4697907"
    ) ;;
  dmc)
    # The four DMC benchmarks under the same gaussian+eps arm, both seeds
    # (e612-e619). All finished, so 4M is the end of training.
    STEP="${STEP:-000004000000}"; TAG=4M
    RUNS=(
      "e612_cartpole-swingup_gaussian-eps_s0 $W/e612_dmc_cartpole_swingup_som_lipvq_line_relu_gaussian_eps_s0_BIG_j4697887"
      "e616_cartpole-swingup_gaussian-eps_s1 $W/e616_dmc_cartpole_swingup_som_lipvq_line_relu_gaussian_eps_s1_BIG_j4697891"
      "e613_hopper-stand_gaussian-eps_s0 $W/e613_dmc_hopper_stand_som_lipvq_line_relu_gaussian_eps_s0_BIG_j4697888"
      "e617_hopper-stand_gaussian-eps_s1 $W/e617_dmc_hopper_stand_som_lipvq_line_relu_gaussian_eps_s1_BIG_j4697892"
      "e615_hopper-hop_gaussian-eps_s0 $W/e615_dmc_hopper_hop_som_lipvq_line_relu_gaussian_eps_s0_BIG_j4697890"
      "e619_hopper-hop_gaussian-eps_s1 $W/e619_dmc_hopper_hop_som_lipvq_line_relu_gaussian_eps_s1_BIG_j4697894"
      "e614_cheetah-run_gaussian-eps_s0 $W/e614_dmc_cheetah_run_som_lipvq_line_relu_gaussian_eps_s0_BIG_j4697889"
      "e618_cheetah-run_gaussian-eps_s1 $W/e618_dmc_cheetah_run_som_lipvq_line_relu_gaussian_eps_s1_BIG_j4697893"
    ) ;;
  *) echo "unknown FAMILY=$FAMILY (antmaze|pinpad|dmc)" >&2; exit 1 ;;
esac

mkdir -p "$OUT"
rc=0
for entry in "${RUNS[@]}"; do
  name="${entry%% *}"; dir="${entry#* }"
  name="${name}_${TAG}"
  ckpt="$dir/logdir/ckpt_milestones/$STEP"
  echo "=============== $name  ($STEP)"
  if [ ! -d "$ckpt" ]; then
    echo "SKIP $name: no milestone at $ckpt" >&2; rc=1; continue
  fi
  # One failure must not lose the other renders.
  python -u "$REPO/experiments/mask_viz/render_mask_viz.py" \
    --run_dir "$dir" --ckpt_path "$ckpt" \
    --out "$OUT/$name" --steps "$STEPS" || { echo "FAILED $name" >&2; rc=1; }
done

echo "=============== done (rc=$rc)"
ls -la "$OUT" | tail -20
exit $rc
