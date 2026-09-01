#!/bin/bash
# Resume the antmaze XL runs e620-e623, paused 2026-08-24 at ~1.8M steps.
#
# They were CANCELLED, NOT ARCHIVED: the checkpoint and the multi-GB replay
# buffer are both still in /work, which is what makes a real resume possible.
# Do NOT run copy_dreamerv3_runs_to_bucket.sh with SKIP_REPLAY=1 and delete
# these dirs -- that keeps the checkpoint but destroys the replay, and training
# would restart from an empty buffer.
#
# Each job re-pins RUN_DIR to the original directory, so it picks up from the
# last checkpoint rather than starting a new run. RUN_STEPS is 6M, the target
# they were switched to on 2026-08-24 (down from the original 10M).
#
# State at pause is in job_logs/e620_e623_paused.tsv. No watchdog tracked these
# (checked against both e510_e557 and e558_e573 state files), so nothing needed
# removing and nothing will resubmit them on its own.
#
#   DRY_RUN=1 sbatch/resume_e620_e623_antmazexl.sh
#   sbatch/resume_e620_e623_antmazexl.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/sbatch/run_v3_goal_ae_ablation_big_a100.sbatch"
W=/work/DoyaU/vasilache/work

PARTITION="${PARTITION:-gpu-a100}"
WALL="${WALL:-2-00:00:00}"
GRES="${GRES:-gpu:a100:1}"
RUN_STEPS="${RUN_STEPS:-6000000}"
DRY_RUN="${DRY_RUN:-0}"

# exp | seed | arm | run dir basename
RUNS=(
  "620|0|director|e620_loconav_ant_maze_xl_director_s0_BIG_j4697896"
  "621|1|director|e621_loconav_ant_maze_xl_director_s1_BIG_j4697897"
  "622|0|som_lipvq_line_relu_gaussian_eps|e622_loconav_ant_maze_xl_som_lipvq_line_relu_gaussian_eps_s0_BIG_j4697898"
  "623|1|som_lipvq_line_relu_gaussian_eps|e623_loconav_ant_maze_xl_som_lipvq_line_relu_gaussian_eps_s1_BIG_j4697899"
)

for row in "${RUNS[@]}"; do
  IFS='|' read -r exp seed arm dir <<< "$row"
  full="$W/$dir"
  if [ ! -d "$full/logdir/ckpt" ]; then
    echo "SKIP e$exp: no checkpoint at $full/logdir/ckpt" >&2
    continue
  fi
  if [ ! -d "$full/logdir/replay" ]; then
    echo "WARNING e$exp: replay is GONE from $full -- resuming would restart" \
         "from an empty buffer. Skipping." >&2
    continue
  fi
  ev="ALL,EXP_TAG=e$exp,TASK=loconav_ant_maze_xl,ARM=$arm,SEED=$seed"
  ev="$ev,RUN_STEPS=$RUN_STEPS,SAVE_EVERY=900,TRAIN_RATIO=256"
  ev="$ev,EXTRA_CONFIGS=loconav,RUN_DIR=$full"
  if [ "$DRY_RUN" = "1" ]; then
    printf '[dry-run] e%s %-34s seed %s  -> %sM\n' \
      "$exp" "$arm" "$seed" "$((RUN_STEPS / 1000000))"
  else
    out="$(sbatch -J "e${exp}_antmazexl_${arm}_s${seed}_resume" \
          -p "$PARTITION" -t "$WALL" --gres="$GRES" --requeue \
          --export="$ev" "$SCRIPT")"
    echo "$out  (e$exp $arm s$seed -> $((RUN_STEPS / 1000000))M)"
    sleep 2
  fi
done
