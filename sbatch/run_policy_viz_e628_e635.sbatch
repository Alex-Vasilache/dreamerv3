#!/bin/bash -l
#SBATCH -J policy_viz_cauchy
#SBATCH -p gpu-v100
#SBATCH -t 0-04:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH -o /work/DoyaU/vasilache/work/slurm_logs/%x-%j.out
#SBATCH -e /work/DoyaU/vasilache/work/slurm_logs/%x-%j.err

# Policy mask-viz panels for the Student-t (Cauchy nu=1) manager arm, e628-e635,
# in the same form as experiments/manager_head/policy/e627_pinpad6_*.mp4: per
# frame, the observation, the running goal code with the blocks the manager
# changed at the last switch tinted, and the active goal decoded back to pixels.
#
# Same shape as run_policy_viz_e620_e627.sbatch; see that script for why this
# runs on gpu-v100 (read-only rollouts, a100 lane is full of live training) and
# why DREAMERV3_CONV_IMPL must stay at the env script's `reference`.
#
# ALL SIX ARM RUNS ARE STILL TRAINING, so this reads ckpt_milestones/ only --
# ckpt/latest is rewritten every 15 minutes and can be read mid-write, while a
# milestone is immutable.
#
#   sbatch --export=ALL,FAMILY=pinpad6 sbatch/run_policy_viz_e628_e635.sbatch
#   sbatch --export=ALL,FAMILY=cheetah sbatch/run_policy_viz_e628_e635.sbatch
#   sbatch --export=ALL,FAMILY=pinpad5 sbatch/run_policy_viz_e628_e635.sbatch
set -uo pipefail

REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3_somvae
W=/work/DoyaU/vasilache/work
OUT="$REPO/experiments/manager_head/policy"

source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
export MUJOCO_GL=egl

FAMILY="${FAMILY:?set FAMILY=pinpad6|cheetah|pinpad5}"
STEPS="${STEPS:-520}"

# "<output basename> <run dir>". The arm tag is `studentt-cauchy` so these sort
# next to, and read against, the `gaussian-eps` panels of the same task.
case "$FAMILY" in
  pinpad6)
    # 2M, matching the existing e626/e627 gaussian-eps panels exactly, even
    # though both runs already have a 2.5M milestone -- a matched step is worth
    # more here than a fresher one.
    STEP="${STEP:-000002000000}"; TAG=2M
    RUNS=(
      "e628_pinpad6_studentt-cauchy_s0 $W/e628_pinpad_six_som_lipvq_line_relu_studentt_s0_BIG_j4699190"
      "e629_pinpad6_studentt-cauchy_s1 $W/e629_pinpad_six_som_lipvq_line_relu_studentt_s1_BIG_j4699191"
    ) ;;
  cheetah)
    # Highest milestone both seeds have reached. The gaussian-eps cheetah panels
    # (e614/e618) are at 4M, so that pairing is NOT step-matched until these
    # runs get there -- rerender at 4M when they do.
    STEP="${STEP:-000002000000}"; TAG=2M
    RUNS=(
      "e630_cheetah-run_studentt-cauchy_s0 $W/e630_dmc_cheetah_run_som_lipvq_line_relu_studentt_s0_BIG_j4699192"
      "e631_cheetah-run_studentt-cauchy_s1 $W/e631_dmc_cheetah_run_som_lipvq_line_relu_studentt_s1_BIG_j4699193"
    ) ;;
  pinpad5)
    # 1.5M: the only milestone e635 has yet, and the Director baselines e632/
    # e633 both have it too, so all four pinpad-five runs stay comparable.
    STEP="${STEP:-000001500000}"; TAG=1.5M
    RUNS=(
      "e634_pinpad5_studentt-cauchy_s0 $W/e634_pinpad_five_som_lipvq_line_relu_studentt_s0_BIG_j4699194"
      "e635_pinpad5_studentt-cauchy_s1 $W/e635_pinpad_five_som_lipvq_line_relu_studentt_s1_BIG_j4699195"
    ) ;;
  *) echo "unknown FAMILY=$FAMILY (pinpad6|cheetah|pinpad5)" >&2; exit 1 ;;
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
