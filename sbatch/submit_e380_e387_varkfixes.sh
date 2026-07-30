#!/bin/bash -l
# Relaunch of the six cancelled free-running var-K matrix cells under the three
# 2026-07-29 fixes (EXPERIMENTS.md §7):
#   1. relabel_truncated_last_duration's `avail` was STATES held, not
#      TRANSITIONS credited (off by one). Affects every relabel=True cell.
#   2. The Lagrangian duration prior's mean_abs_err was biased toward
#      sparse-hold rows (w bakes in the per-row rescale). Affects every
#      DUR_MODE=lagrangian cell.
#   3. worker_timed_goals countdown budget ignored a pinned hold -- dormant
#      here (flag is False in all of these), listed for completeness.
#
# All three are exclusive to GENUINELY VARIABLE holds: none can fire under a
# pinned/fixed-K config, which is why the duration-pinned equivalence pair
# (e360/e361, still running, 1 day in) is unaffected and was left alone --
# verified: neither has requeued since the refactor merge, so their code is
# also intact.
#
# Configs are 1:1 with the cancelled runs (verified against each archived
# job.env), differing only in the code fixes and the run/job ids.
#
# Usage: bash sbatch/submit_e380_e387_varkfixes.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SCRIPT=sbatch/run_v3_prior_vargoal_big_a100.sbatch
LOG=job_logs/e380_e387_varkfixes.tsv
mkdir -p job_logs
: > "$LOG"

submit() {  # submit <exp> <name> <exports...>
  local exp="$1"; shift
  local name="$1"; shift
  local out
  out=$(sbatch -J "$name" --export="ALL,EXP_TAG=$exp,RUN_STEPS=4000000,SEED=0,$*" "$SCRIPT")
  printf '%s\t%s\t%s\t%s\n' "$exp" "${out##* }" "$name" "$*" | tee -a "$LOG"
}

# --- relabel OFF (was e330/e331 -> e363/e364). e331 was the best cheetah cell
# in the original matrix; its relabel-ON twin collapsed late.
LAGR_OFF="RECIPE=plain_vark,DUR_MODE=lagrangian,DUR_TARGET=8.0,DUR_REG=0.01,MGR_REWARD_AGG=sum,GOAL_DURATION_RELABEL_TRUNCATED=False"
submit e380 e380_hopper_vark_relabOFF  "TASK=dmc_hopper_hop,$LAGR_OFF"
submit e381 e381_cheetah_vark_relabOFF "TASK=dmc_cheetah_run,$LAGR_OFF"

# --- relabel ON (was e326/e327 -> e366/e367). These are the cells fix #1
# actually changes; the off-by-one over-credited every truncated hold.
LAGR_ON="RECIPE=plain_vark,DUR_MODE=lagrangian,DUR_TARGET=8.0,DUR_REG=0.01,MGR_REWARD_AGG=sum,GOAL_DURATION_RELABEL_TRUNCATED=True"
submit e382 e382_hopper_vark_relabON  "TASK=dmc_hopper_hop,$LAGR_ON"
submit e383 e383_cheetah_vark_relabON "TASK=dmc_cheetah_run,$LAGR_ON"

# --- no duration prior (was e332/e333 -> e368/e369). DUR_MODE=fixed with
# DUR_REG=0 leaves the duration-entropy controller as the only shaping force;
# both originals showed hold-length collapse (to 5.0 / 3.55).
NOREG="RECIPE=plain_vark,DUR_MODE=fixed,DUR_REG=0.0,DUR_TARGET=8.0,MGR_REWARD_AGG=sum,GOAL_DURATION_RELABEL_TRUNCATED=True"
submit e384 e384_hopper_vark_nodurreg  "TASK=dmc_hopper_hop,$NOREG"
submit e385 e385_cheetah_vark_nodurreg "TASK=dmc_cheetah_run,$NOREG"

# --- Director baselines: queued, will start when a slot frees (the six above
# take every currently-free a100 slot). Fresh runs, not continuations -- the
# earlier e352/e353 replay buffers were destroyed by SKIP_REPLAY archiving.
submit e386 e386_cheetah_director_BIG "TASK=dmc_cheetah_run,RECIPE=director,MGR_FREQ=8"
submit e387 e387_hopper_director_BIG  "TASK=dmc_hopper_hop,RECIPE=director,MGR_FREQ=8"

echo "submitted; job ids in $LOG"
