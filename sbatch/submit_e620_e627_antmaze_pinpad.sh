#!/bin/bash
# e620-e627: pure Director vs SOM-line+LiP/ReLU/Gaussian/eps on two
# exploration benchmarks, two seeds each, 10M steps.
#
#   antmaze XL   e620-e621 director      e622-e623 arm
#   pinpad six   e624-e625 director      e626-e627 arm
#
# These are the first tasks in this programme chosen because exploration is the
# binding constraint rather than control, which is what the eps-greedy index
# jump exists for. The five DMC benchmarks are all dense-reward control where
# Director already does well.
#
# Two settings differ from the DMC runs, and both had to be plumbed through the
# launcher first:
#
#   EXTRA_CONFIGS=loconav   antmaze needs its env repeat and the higher train
#                           ratio. Without the block it would silently run with
#                           the DMC settings.
#   TRAIN_RATIO=256         a CLI flag beats a config block, and the launcher
#                           previously hardcoded --run.train_ratio 64, which
#                           would have overridden loconav's 256.
#
# pinpad has no config block and takes the launcher defaults (train_ratio 64).
# It is a small image env, so it is far cheaper per step than antmaze.
#
# COST WARNING. antmaze at train_ratio 256 does 4x the gradient work per env
# step of the DMC runs, and 10M steps is 2.5x their length -- about 10x the
# total compute of a 4M DMC run, which takes ~26h on one A100. Expect these to
# run for days each and to requeue across several 48h slices (the launcher
# self-requeues on USR1). They are submitted HELD, behind e592-e619.
#
#   DRY_RUN=1 sbatch/submit_e620_e627_antmaze_pinpad.sh
#   sbatch/submit_e620_e627_antmaze_pinpad.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/sbatch/run_v3_goal_ae_ablation_big_a100.sbatch"
LOG="$REPO/job_logs/e620_e627_antmaze_pinpad.tsv"

PARTITION="${PARTITION:-gpu-a100}"
WALL="${WALL:-2-00:00:00}"
SAVE_EVERY="${SAVE_EVERY:-900}"
GRES="${GRES:-gpu:a100:1}"

SEEDS="${SEEDS:-0 1}"
NICE="${NICE:-6000}"
DRY_RUN="${DRY_RUN:-0}"

# task | label | extra config blocks | train ratio | steps
# Steps are per-case: antmaze is the long-horizon one and follows the existing
# antmaze convention of 10M, while pinpad runs 4M like the DMC benchmarks.
CASES=(
  "loconav_ant_maze_xl|antmazexl|loconav|256|10000000"
  "pinpad_six|pinpadsix||64|4000000"
)
ARMS=(director som_lipvq_line_relu_gaussian_eps)

exp=620
for case in "${CASES[@]}"; do
  IFS='|' read -r task label extra tr steps <<< "$case"
  for arm in "${ARMS[@]}"; do
    for seed in $SEEDS; do
      tag="e${exp}"; name="e${exp}_${label}_${arm}_s${seed}"
      ev="ALL,EXP_TAG=$tag,TASK=$task,ARM=$arm,SEED=$seed"
      ev="$ev,RUN_STEPS=$steps,SAVE_EVERY=$SAVE_EVERY,TRAIN_RATIO=$tr"
      [ -n "$extra" ] && ev="$ev,EXTRA_CONFIGS=$extra"
      if [ "$DRY_RUN" = "1" ]; then
        printf '[dry-run] %-58s %4sM tr=%-4s cfg=%s\n' "$name" "$((steps/1000000))" "$tr" "${extra:-<none>}"
      else
        out="$(sbatch -J "$name" -p "$PARTITION" -t "$WALL" --gres="$GRES" \
              --nice="$NICE" --requeue --hold --export="$ev" "$SCRIPT")"
        echo "$out  ($name, $((steps/1000000))M tr=$tr ${extra:+cfg=$extra})"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$(date -Is)" "${out##* }" "$tag" "$task" "$arm" "$seed" "$name" \
          "$PARTITION" "$steps" >> "$LOG"
        sleep 2
      fi
      exp=$((exp + 1))
    done
  done
done

echo
echo "log: $LOG   (all HELD; seed_major_releaser.sh lets them out by exp number)"
