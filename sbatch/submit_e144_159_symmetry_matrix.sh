#!/bin/bash -l
# e144-e175: 4-group symmetry matrix testing the mask/duration Lagrangian mechanisms
# discussed in the loss-formulation review (see e57_losses.pdf discussion + follow-ups):
#   Group A (mask_entropy)  : MASK_MODE=entropy,      DUR_MODE=fixed       (e57 duration unchanged)
#   Group B (mask_probent)  : MASK_MODE=prob_entropy, DUR_MODE=fixed       (e57 duration unchanged)
#   Group C (dur_lagrange)  : MASK_MODE=prob,          DUR_MODE=lagrangian (e57 mask unchanged)
#   Group D (combined)      : MASK_MODE=prob_entropy, DUR_MODE=lagrangian (full symmetry:
#                              both mask and duration get folded-out, independently-scaled
#                              adaptive Lagrangians, both paired with entropy-style
#                              anti-collapse regularization at target 0.5)
# Each group x {cartpole, hopper} x {small (V100, size6m/32x32), BIG (A100, director_match
# 64x64)} = 4 x 2 x 2 = 16 jobs total (8 V100 + 8 A100 -- one job per GPU, fills the whole
# quota in a single wave, matching the 8+8 freed by cancelling the prior batch).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMALL="$SCRIPT_DIR/run_v3_prior_vargoal_small.sbatch"
BIG="$SCRIPT_DIR/run_v3_prior_vargoal_big_a100.sbatch"
LOG="$SCRIPT_DIR/e144_159_job_ids.tsv"
: > "$LOG"

declare -A GROUP_MASK_MODE=( [A]=entropy [B]=prob_entropy [C]=prob [D]=prob_entropy )
declare -A GROUP_DUR_MODE=(  [A]=fixed   [B]=fixed        [C]=lagrangian [D]=lagrangian )
declare -A GROUP_NAME=( [A]=mask_entropy [B]=mask_probent [C]=dur_lagrange [D]=combined )

N=144
for GROUP in A B C D; do
  MASK_MODE="${GROUP_MASK_MODE[$GROUP]}"
  DUR_MODE="${GROUP_DUR_MODE[$GROUP]}"
  GNAME="${GROUP_NAME[$GROUP]}"
  for TASK in dmc_cartpole_swingup dmc_hopper_hop; do
    TSHORT="${TASK#dmc_}"; TSHORT="${TSHORT%_swingup}"; TSHORT="${TSHORT%_hop}"
    for SCALE in small BIG; do
      EXP="e${N}"
      JNAME="${EXP}_${GNAME}_${TSHORT}_${SCALE}"
      if [ "$SCALE" = "small" ]; then
        JOBOUT=$(sbatch -J "$JNAME" \
          --export=ALL,EXP_TAG="$EXP",TASK="$TASK",MASK_MODE="$MASK_MODE",DUR_MODE="$DUR_MODE",DUR_TARGET=4.0,SEED=0 \
          "$SMALL")
      else
        JOBOUT=$(sbatch -J "$JNAME" \
          --export=ALL,EXP_TAG="$EXP",TASK="$TASK",MASK_MODE="$MASK_MODE",DUR_MODE="$DUR_MODE",DUR_TARGET=4.0,SEED=0 \
          "$BIG")
      fi
      JOBID="${JOBOUT##* }"
      echo -e "${EXP}\t${JNAME}\t${JOBID}\t${GNAME}\t${TASK}\t${SCALE}\t${MASK_MODE}\t${DUR_MODE}" | tee -a "$LOG"
      N=$((N+1))
    done
  done
done
echo "Submitted $((N-144)) jobs. Log: $LOG"
