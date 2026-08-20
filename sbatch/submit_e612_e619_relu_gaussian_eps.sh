#!/bin/bash
# e612-e619: the e592-e599 arm plus the 10% epsilon-greedy index jump.
#
#   seed 0 -> e612 cartpole  e613 hopper_stand  e614 cheetah  e615 hopper_hop
#   seed 1 -> e616 ..e619
#
# Differs from e592-e599 by `mgr_explore` ALONE (mgr_explore_eps 0 -> 0.1), so
# this is a clean single-factor test of the exploration mechanism: same
# SOM-line+LiP autoencoder, same ReLU trunk, same discretized-Gaussian manager,
# same seeds, same four environments.
#
# The jump fires only in the environment rollout, never in imagination or on
# the replay sequence, so the manager's REINFORCE gradient stays on-policy and
# the two arms differ in the DATA the actor collects rather than in the
# objective. See dreamerv3/hrl/explore.py.
#
# Two seeds, matching what e592-e599 will have. This resolves nothing on its
# own -- four seeds against four is the minimum for a p below 0.029 -- it is a
# paired screen on the same seeds.
#
# Seed-major via --nice, as before, and the seed-1 row is submitted HELD so a
# preempted seed-0 job cannot lose its slot to it (that is how e595 lost its
# GPU to e600 on 2026-08-20; --nice cannot protect a job that has momentarily
# left the queue). Release with sbatch/seed_major_releaser.sh.
#
#   DRY_RUN=1 sbatch/submit_e612_e619_relu_gaussian_eps.sh
#   sbatch/submit_e612_e619_relu_gaussian_eps.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/sbatch/run_v3_goal_ae_ablation_big_a100.sbatch"
LOG="$REPO/job_logs/e612_e619_relu_gaussian_eps.tsv"

PARTITION="${PARTITION:-gpu-a100}"
WALL="${WALL:-2-00:00:00}"
SAVE_EVERY="${SAVE_EVERY:-900}"
GRES="${GRES:-gpu:a100:1}"
STEPS="${STEPS:-4000000}"
SEEDS="${SEEDS:-0 1}"
DRY_RUN="${DRY_RUN:-0}"
ARM=som_lipvq_line_relu_gaussian_eps

TASKS=(dmc_cartpole_swingup dmc_hopper_stand dmc_cheetah_run dmc_hopper_hop)
LABELS=(cpswingup hopperstand cheetah hopperhop)

exp=612
for seed in $SEEDS; do
  nice=$((1000 + 1000 * seed))
  # Seed 0 runs as soon as a slot frees; every later seed waits held.
  hold=""
  [ "$seed" -gt 0 ] && hold="--hold"
  for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"; label="${LABELS[$i]}"
    tag="e${exp}"; name="e${exp}_${label}_${ARM}_s${seed}"
    ev="ALL,EXP_TAG=$tag,TASK=$task,ARM=$ARM,SEED=$seed"
    ev="$ev,RUN_STEPS=$STEPS,SAVE_EVERY=$SAVE_EVERY"
    if [ "$DRY_RUN" = "1" ]; then
      printf '[dry-run] nice=%-5s %-12s %s\n' "$nice" "${hold:-(eligible)}" "$name"
    else
      out="$(sbatch -J "$name" -p "$PARTITION" -t "$WALL" --gres="$GRES" \
            --nice="$nice" --requeue $hold --export="$ev" "$SCRIPT")"
      echo "$out  ($name, nice $nice ${hold:-eligible})"
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
