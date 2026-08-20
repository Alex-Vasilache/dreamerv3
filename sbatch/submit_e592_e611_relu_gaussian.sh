#!/bin/bash
# e592-e611: SOM-line + LiP (ReLU trunk) + discretized-Gaussian manager.
# Four environments x five seeds, queued SEED-MAJOR.
#
#   seed 0 -> e592 cartpole  e593 hopper_stand  e594 cheetah  e595 hopper_hop
#   seed 1 -> e596 ..e599      seed 2 -> e600..e603
#   seed 3 -> e604..e607       seed 4 -> e608..e611
#
# Ordering is enforced with --nice (1000 + 1000*seed), so the whole seed-0 row
# across all four environments runs before any seed-1 job starts. That way an
# early stop still leaves complete, balanced cells rather than four seeds of
# cartpole and none of hopper_hop.
#
# Three deliberate changes from the e534-e573 SOM-line+LiP runs:
#
#   ReLU trunk     `lip_relu`. prod_l softplus(c_l) bounds the network's
#                  Lipschitz constant only for 1-Lipschitz activations. relu is
#                  one; silu's slope peaks at 1.0998, so on this 4-layer trunk
#                  the silu arms' reported product understates the true
#                  constant by up to 1.0998^4 = 1.46. With relu the number IS
#                  the bound. Verified against the e534/e570 checkpoints that
#                  those runs really were silu, not just configured for it.
#   Gaussian mgr   `mgr_gaussian`. See the config block: at L=8, C=8 the
#                  Poisson's parameter-count advantage is worth 24k parameters
#                  and nothing else, while its variance-tied-to-mean coupling
#                  costs uniform confidence across classes and symmetric
#                  neighbour credit.
#   (unchanged)    straight-through ON (goal_som_lipvq_line_prod is the ste:'on'
#                  member of the family -- the ste:'off' arms are the only ones
#                  that reliably LOST), one scalar Lipschitz bound per layer
#                  (lip_per_row: False), lip_impl prod, strict_bound True.
#
# NOTE this is not a single-factor change from e534-e573: it moves the
# activation AND the manager head together. It is the "best configuration" arm,
# not an ablation. Isolating either factor needs its own run.
#
#   DRY_RUN=1 sbatch/submit_e592_e611_relu_gaussian.sh
#   sbatch/submit_e592_e611_relu_gaussian.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/sbatch/run_v3_goal_ae_ablation_big_a100.sbatch"
LOG="$REPO/job_logs/e592_e611_relu_gaussian.tsv"

PARTITION="${PARTITION:-gpu-a100}"
WALL="${WALL:-2-00:00:00}"
SAVE_EVERY="${SAVE_EVERY:-900}"
GRES="${GRES:-gpu:a100:1}"
STEPS="${STEPS:-4000000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
DRY_RUN="${DRY_RUN:-0}"
WORK="${WORK:-/work/DoyaU/vasilache/work}"
ARM=som_lipvq_line_relu_gaussian

# (task, short label) in the order they are launched within each seed
TASKS=(dmc_cartpole_swingup dmc_hopper_stand dmc_cheetah_run dmc_hopper_hop)
LABELS=(cpswingup hopperstand cheetah hopperhop)

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

exp=592
for seed in $SEEDS; do
  nice=$((1000 + 1000 * seed))
  for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"; label="${LABELS[$i]}"
    tag="e${exp}"; name="e${exp}_${label}_${ARM}_s${seed}"
    ev="ALL,EXP_TAG=$tag,TASK=$task,ARM=$ARM,SEED=$seed"
    ev="$ev,RUN_STEPS=$STEPS,SAVE_EVERY=$SAVE_EVERY"
    pinned="$(find_run_dir "$tag" "$task" "$ARM" "$seed")"
    [ -n "$pinned" ] && ev="$ev,RUN_DIR=$pinned"
    if [ "$DRY_RUN" = "1" ]; then
      printf '[dry-run] nice=%-5s %s\n' "$nice" "$name"
    else
      out="$(sbatch -J "$name" -p "$PARTITION" -t "$WALL" --gres="$GRES" \
            --nice="$nice" --requeue --export="$ev" "$SCRIPT")"
      echo "$out  ($name, nice $nice)"
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
echo "seed-major: the whole seed-0 row runs before any seed-1 job starts."
