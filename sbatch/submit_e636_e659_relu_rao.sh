#!/bin/bash
# e636-e659: does Rao's quadratic entropy on the manager help?
#
# Paired arms, launched together so the Rao term is the ONLY difference:
#
#   som_lipvq_line_relu       SOM-line + LiP + ReLU + categorical manager
#   som_lipvq_line_relu_rao   the same, + manager_rao held at normalized 0.5
#
# The control arm is not redundant. ReLU + categorical has never been run:
# `lip_relu` only ever shipped bundled with a different manager head
# (mgr_gaussian in e592-e619, mgr_studentt in e628-e635), so activation and
# head always moved together and neither was ever isolated. Without this
# control a difference here could not be attributed to Rao rather than ReLU.
#
# WHY. Measured on the e534-e573 checkpoints (see experiments/manager_rao):
# the trained manager holds normalized entropy 0.555, on target, but normalized
# Rao 0.125 -- 13.6% of the 0.925 attainable at that same entropy. A
# label-shuffling control puts SOM-line+LiP at ratio 0.43 against Director's
# 1.06, i.e. the manager's mass genuinely sits on ADJACENT classes. That is the
# SOM ordering working as designed, and it is also exactly what this term
# trades away: Rao asks the manager to spend the same entropy budget on classes
# that are FAR APART on the line instead of neighbouring ones.
#
# WHAT 0.5 MEANS. Normalized Rao 0.5 is ABOVE the uniform value of 0.429, so it
# cannot be met by widening a unimodal distribution -- it needs mass at both
# ends. At entropy 0.5 the solutions look like ~0.25 / 0.50 / 0.25 on classes
# 0 / middle / 7: the same ~3 effective classes the manager already uses, moved
# to opposite ends of the line. This is a strong setpoint (4x the measured
# value); if the entropy controller and the Rao controller end up fighting,
# MGR_RAO_TARGET below is the knob (0.25-0.30 is still 2x measured and stays
# under uniform).
#
# EXPECTATION. Genuinely open. Across the ten measured checkpoints, Rao does
# NOT predict return: of the five within-task seed pairs two favour higher Rao,
# three favour lower, and on hopper_hop two seeds with essentially identical
# Rao (0.088 vs 0.091) scored 303 vs 173. The headroom is real; that it should
# be closed is the hypothesis being tested, not a known result.
#
#   DRY_RUN=1 sbatch/submit_e636_e659_relu_rao.sh
#   sbatch/submit_e636_e659_relu_rao.sh
#   SEEDS="0 1" sbatch/submit_e636_e659_relu_rao.sh     # half-size first pass
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/sbatch/run_v3_goal_ae_ablation_big_a100.sbatch"
LOG="$REPO/job_logs/e636_e659_relu_rao.tsv"

PARTITION="${PARTITION:-gpu-a100}"
WALL="${WALL:-2-00:00:00}"
SAVE_EVERY="${SAVE_EVERY:-900}"
GRES="${GRES:-gpu:a100:1}"
STEPS="${STEPS:-4000000}"
SEEDS="${SEEDS:-0 1 2 3}"
DRY_RUN="${DRY_RUN:-0}"
WORK="${WORK:-/work/DoyaU/vasilache/work}"
MGR_RAO_TARGET="${MGR_RAO_TARGET:-}"   # empty = use the mgr_rao config block

# hopper_hop is deliberately absent: it resolves differences at about 1% power
# (5 seeds cannot separate a 15x effect there), so a cell of it costs 27 GPU-h
# per seed and buys nothing. cheetah_run is kept because it is where
# SOM-line+LiP beat Director and where the ReLU+Gaussian arm lost 33.6%, i.e.
# the benchmark that has actually discriminated between these arms.
TASKS=(dmc_cartpole_swingup dmc_hopper_stand dmc_cheetah_run)
LABELS=(cpswingup hopperstand cheetah)
ARMS=(som_lipvq_line_relu som_lipvq_line_relu_rao)

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

# Seed-major, and within a seed the two arms of a task are adjacent, so an
# early stop leaves complete A/B PAIRS rather than a control arm with no
# treatment to compare against.
exp=636
for seed in $SEEDS; do
  nice=$((1000 + 1000 * seed))
  for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"; label="${LABELS[$i]}"
    for arm in "${ARMS[@]}"; do
      tag="e${exp}"; name="e${exp}_${label}_${arm}_s${seed}"
      ev="ALL,EXP_TAG=$tag,TASK=$task,ARM=$arm,SEED=$seed"
      ev="$ev,RUN_STEPS=$STEPS,SAVE_EVERY=$SAVE_EVERY"
      if [ -n "$MGR_RAO_TARGET" ] && [ "$arm" = "som_lipvq_line_relu_rao" ]; then
        ev="$ev,MGR_RAO_TARGET=$MGR_RAO_TARGET"
      fi
      pinned="$(find_run_dir "$tag" "$task" "$arm" "$seed")"
      [ -n "$pinned" ] && ev="$ev,RUN_DIR=$pinned"
      if [ "$DRY_RUN" = "1" ]; then
        printf '[dry-run] nice=%-5s %s\n' "$nice" "$name"
      else
        out="$(sbatch -J "$name" -p "$PARTITION" -t "$WALL" --gres="$GRES" \
              --nice="$nice" --requeue --export="$ev" "$SCRIPT")"
        echo "$out  ($name, nice $nice)"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$(date -Is)" "${out##* }" "$tag" "$task" "$arm" "$seed" "$name" \
          "$PARTITION" "$STEPS" >> "$LOG"
        sleep 2
      fi
      exp=$((exp + 1))
    done
  done
done

echo
echo "log: $LOG"
echo "seed-major, A/B adjacent: an early stop leaves complete pairs."
