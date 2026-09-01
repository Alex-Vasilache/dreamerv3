#!/bin/bash
# e636-e643: SOM-line + LiP + ReLU + categorical manager, with count-based
# novelty as a THIRD manager reward and critic.
#
# Four environments x two seeds, 4M steps, BIG/A100:
#
#   pinpad_five  e636 s0  e637 s1
#   pinpad_six   e638 s0  e639 s1
#   cheetah_run  e640 s0  e641 s1
#   cartpole     e642 s0  e643 s1
#
# The manager carries one task reward and TWO exploration rewards, each with its
# own critic: the existing goal-autoencoder reconstruction error (mgr_expl_val)
# and the new code-count novelty (mgr_novel_val). Both exploration advantages
# enter at mgr_expl_weight = 0.1, so the total exploration pull against the task
# reward is 0.2 rather than 0.1.
#
# WHY COUNTS. The reconstruction-error bonus is coupled to the quality of the
# goal autoencoder: as reconstruction improves the bonus decays everywhere, so a
# better autoencoder explores less by construction, and the signal fades over
# training exactly when the manager is most able to use it. Counting is
# independent of that. It also only needs the encoder to LABEL a state
# consistently, not to reconstruct it -- which is why it works where boosting
# the codes of high-reconstruction-error states does not, since a novel state's
# code is precisely the code that fails to decode back to it.
#
# THE MEMORY. Visit counts over the JOINT code (never per-block: shuffling each
# block independently, which leaves every per-block marginal identical, spreads
# the measured codes over 3-4x more regions, so the structure is in which
# COMBINATIONS occur). Two resolutions: 4 bins/block fine (65,536 cells) and
# 2 bins/block coarse (256). Both decay at exp(-env_steps_per_train/replay.size)
# with insertion weight 1/train_ratio, derived in agent.py -- so the memory
# forgets at exactly the rate replay forgets, each env step contributes exactly
# 1.0 however often replay resamples it, and a count reads as "env steps spent
# near this code inside the replay window".
#
# NOTE ON COMPARISON. ReLU + categorical has never been run in any arm: lip_relu
# has only ever shipped with mgr_gaussian (e592-e619) or mgr_studentt
# (e628-e635). There is therefore no matched baseline for these cells, and a
# difference against the silu arms (e534-e573) or the ReLU+Gaussian arms cannot
# be attributed to the exploration change alone. Running ARM=som_lipvq_line_relu
# on the same four tasks and seeds would supply that control.
#
#   DRY_RUN=1 sbatch/submit_e636_e643_relu_count.sh
#   sbatch/submit_e636_e643_relu_count.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/sbatch/run_v3_goal_ae_ablation_big_a100.sbatch"
LOG="$REPO/job_logs/e636_e643_relu_count.tsv"

PARTITION="${PARTITION:-gpu-a100}"
WALL="${WALL:-2-00:00:00}"
SAVE_EVERY="${SAVE_EVERY:-900}"
GRES="${GRES:-gpu:a100:1}"
STEPS="${STEPS:-4000000}"
SEEDS="${SEEDS:-0 1}"
DRY_RUN="${DRY_RUN:-0}"
WORK="${WORK:-/work/DoyaU/vasilache/work}"
ARM="${ARM:-som_lipvq_line_relu_count}"

TASKS=(pinpad_five pinpad_six dmc_cheetah_run dmc_cartpole_swingup)
LABELS=(pinpadfive pinpadsix cheetah cpswingup)

find_run_dir() {  # tag task arm seed
  python3 - "$WORK" "$1" "$2" "$3" "$4" <<'PY'
import glob, json, os, sys
work, tag, task, arm, seed = sys.argv[1:6]
best, bestdir = -1, ''
for d in glob.glob(f'{work}/{tag}_{task}_{arm}_s{seed}_BIG_j*'):
    p = os.path.join(d, 'logdir', 'metrics.jsonl')
    if not os.path.exists(p):
        continue
    step = -1
    with open(p, 'rb') as f:
        f.seek(0, 2); f.seek(max(0, f.tell() - 200_000))
        for line in f.read().decode('utf8', 'ignore').splitlines()[1:]:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if 'step' in row:
                step = max(step, int(row['step']))
    if step > best:
        best, bestdir = step, d
print(bestdir if best > 0 else '')
PY
}

# Task-major, seed-minor: both seeds of a task are adjacent, so an early stop
# leaves complete two-seed cells rather than eight half-finished environments.
exp=636
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"; label="${LABELS[$i]}"
  for seed in $SEEDS; do
    tag="e${exp}"; name="e${exp}_${label}_${ARM}_s${seed}"
    ev="ALL,EXP_TAG=$tag,TASK=$task,ARM=$ARM,SEED=$seed"
    ev="$ev,RUN_STEPS=$STEPS,SAVE_EVERY=$SAVE_EVERY"
    pinned="$(find_run_dir "$tag" "$task" "$ARM" "$seed")"
    [ -n "$pinned" ] && ev="$ev,RUN_DIR=$pinned"
    if [ "$DRY_RUN" = "1" ]; then
      printf '[dry-run] %s\n' "$name"
    else
      out="$(sbatch -J "$name" -p "$PARTITION" -t "$WALL" --gres="$GRES" \
            --requeue --export="$ev" "$SCRIPT")"
      echo "$out  ($name)"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Is)" "${out##* }" "$tag" "$task" "$ARM" "$seed" "$name" \
        "$PARTITION" "$STEPS" >> "$LOG"
      sleep 2
    fi
    exp=$((exp + 1))
  done
done

echo
echo "log: $LOG"
