#!/bin/bash -l
# Relaunch of the block-pooled set under the 2026-07-28 terminal-flag fix
# (EXPERIMENTS.md §7: replay ``term_down`` was a bool-cast continuation
# probability, so every manager decision read as terminal and the entire
# replay-side bootstrap was deleted -- 24x too-small value targets).
#
# e352/e353 (same-code pure Director, cheetah 337 / hopper 386 trailing-50 at
# ~2.3M) keep running as the targets; they never touched this code path.
#
# HOPPER IS THE SHARP TEST. Under the bug it scored exactly 0.0 with reward
# rate 0 while Director reached 386 -- sparse reward puts nearly all signal in
# the bootstrap, which the bug removed. Two seeds of the pinned hopper cell, so
# "hopper stops being zero" cannot be a single lucky seed.
#
# Usage: bash sbatch/submit_e360_e365_termfix.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SCRIPT=sbatch/run_v3_prior_vargoal_big_a100.sbatch
LOG=job_logs/e360_e365_termfix.tsv
mkdir -p job_logs
: > "$LOG"

submit() {  # submit <exp> <name> <exports...>
  local exp="$1"; shift
  local name="$1"; shift
  local out
  out=$(sbatch -J "$name" --export="ALL,EXP_TAG=$exp,RUN_STEPS=4000000,$*" "$SCRIPT")
  printf '%s\t%s\t%s\t%s\n' "$exp" "${out##* }" "$name" "$*" | tee -a "$LOG"
}

# Duration-pinned == fixed-K Director, reparameterized. Should now match e352/e353.
PIN="RECIPE=plain_vark,DUR_MODE=fixed,DUR_FIXED=8,DUR_TARGET=8.0,DUR_ACTENT_TARGET=0.0,MGR_REWARD_AGG=mean,VARIABLE_GOAL_BLOCK_REW=True"
submit e360 e360_hopper_pinned_termfix  "TASK=dmc_hopper_hop,SEED=0,$PIN"
submit e361 e361_cheetah_pinned_termfix "TASK=dmc_cheetah_run,SEED=0,$PIN"
submit e362 e362_hopper_pinned_seed1    "TASK=dmc_hopper_hop,SEED=1,$PIN"
submit e365 e365_cheetah_pinned_seed1   "TASK=dmc_cheetah_run,SEED=1,$PIN"

# Genuinely variable holds, best-performing variant from the e326-e341 matrix
# (relabel OFF). These are the actual research configuration, not a control.
LAGR="RECIPE=plain_vark,DUR_MODE=lagrangian,DUR_TARGET=8.0,MGR_REWARD_AGG=sum,VARIABLE_GOAL_BLOCK_REW=True,GOAL_DURATION_RELABEL_TRUNCATED=False"
submit e363 e363_cheetah_vark_termfix "TASK=dmc_cheetah_run,SEED=0,$LAGR"
submit e364 e364_hopper_vark_termfix  "TASK=dmc_hopper_hop,SEED=0,$LAGR"

echo "submitted; job ids in $LOG"
