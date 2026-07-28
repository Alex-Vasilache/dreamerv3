#!/bin/bash -l
# Relaunch of the cancelled e342-e349 A100 batch under the 2026-07-27
# Director-equivalence fixes to the block-pooled variable-K path
# (decision-count rescaling, valid-decision statistics, empty trailing
# decision, windowed worker credit under a pinned hold, inert pinned duration
# head -- see EXPERIMENTS.md §7 and embodied/tests/test_block_pooled_equivalence.py).
#
# Deviation from a 1:1 resubmit: the e348/e349 reruns are dropped in favour of
# two same-code pure-Director baselines (e352 cheetah, e353 hopper). The whole
# question this batch answers is whether the duration-pinned block-pooled path
# now reproduces Director, and the historical comparators (e191 cheetah 290.0,
# e124 hopper 279.4) predate many commits -- a Director run on the *current*
# code removes code drift as an alternative explanation either way.
#
# Usage: bash sbatch/submit_e350_e357_director_equiv.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SCRIPT=sbatch/run_v3_prior_vargoal_big_a100.sbatch
LOG=job_logs/e350_e357_director_equiv.tsv
mkdir -p job_logs
: > "$LOG"

submit() {  # submit <exp> <name> <exports...>
  local exp="$1"; shift
  local name="$1"; shift
  local out
  out=$(sbatch -J "$name" --export="ALL,EXP_TAG=$exp,RUN_STEPS=4000000,SEED=0,$*" "$SCRIPT")
  local jid="${out##* }"
  printf '%s\t%s\t%s\t%s\n' "$exp" "$jid" "$name" "$*" | tee -a "$LOG"
}

# --- duration-pinned Director control (the equivalence test itself) ----------
# Was e342/e343. plain_vark + goal_duration_fixed=8 == fixed-K Director,
# reparameterized. Target: e124 hopper 279.4 / e191 cheetah 290.0.
PIN="RECIPE=plain_vark,DUR_MODE=fixed,DUR_FIXED=8,DUR_TARGET=8.0,DUR_ACTENT_TARGET=0.0,MGR_REWARD_AGG=mean,VARIABLE_GOAL_BLOCK_REW=True"
submit e350 e350_hopper_pinned_BIG  "TASK=dmc_hopper_hop,$PIN"
submit e351 e351_cheetah_pinned_BIG "TASK=dmc_cheetah_run,$PIN"

# --- same-code pure Director baselines --------------------------------------
DIR="RECIPE=director,MGR_FREQ=8"
submit e352 e352_cheetah_director_BIG "TASK=dmc_cheetah_run,$DIR"
submit e353 e353_hopper_director_BIG  "TASK=dmc_hopper_hop,$DIR"

# --- free-duration matrix cells (reruns of e344-e347) -----------------------
# goal_duration_fixed=0 in all four, so only the padding-related fixes apply.
LAGR="RECIPE=plain_vark,DUR_MODE=lagrangian,DUR_TARGET=8.0,MGR_REWARD_AGG=sum,VARIABLE_GOAL_BLOCK_REW=True"
submit e354 e354_hopper_relabelON_BIG   "TASK=dmc_hopper_hop,$LAGR,GOAL_DURATION_RELABEL_TRUNCATED=True"
submit e355 e355_cheetah_relabelON_BIG  "TASK=dmc_cheetah_run,$LAGR,GOAL_DURATION_RELABEL_TRUNCATED=True"
submit e356 e356_hopper_relabelOFF_BIG  "TASK=dmc_hopper_hop,$LAGR,GOAL_DURATION_RELABEL_TRUNCATED=False"
submit e357 e357_cheetah_relabelOFF_BIG "TASK=dmc_cheetah_run,$LAGR,GOAL_DURATION_RELABEL_TRUNCATED=False"

echo "submitted; job ids in $LOG"
