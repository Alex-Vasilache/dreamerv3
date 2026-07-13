#!/bin/bash -l
# e162-e177: hopper-locality controls + Group D retry after the 2026-07-10 code fixes
# (valid-masked sparsity stats; symmetric |E[dur]-target| duration Lagrangian).
#
#   CTRL1 mask_fixedk (e162-e165): masking package WITHOUT var-K (fixed K=8, prob 0.3,
#     struct 200, Z-cond). If hopper dies here too, the masking+struct locality package
#     kills it independent of var-K; if hopper learns, var-K is implicated.
#   CTRL2 plain_vark (e166-e169): "e124 + var-K" -- full goal replacement, no mask,
#     no cond, struct 0, soft fixed duration prior (reg 0.01 -> target 4). If hopper
#     learns here, the locality package (mask+struct+running-goal) is confirmed as the
#     bottleneck; if it dies, the var-K machinery itself breaks hopper.
#   D-retry vark_masked prob_entropy+lagrangian (e170-e177): Group D rerun on the fixed
#     code (valid-masked mask stats; symmetric duration Lagrangian that can actually
#     relax), now also on cheetah + acrobot to widen the reward-density spectrum.
#
# 8 V100 small + 8 A100 BIG = exactly the full 8+8 quota in one wave.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMALL="$SCRIPT_DIR/run_v3_prior_vargoal_small.sbatch"
BIG="$SCRIPT_DIR/run_v3_prior_vargoal_big_a100.sbatch"
LOG="$SCRIPT_DIR/e162_177_job_ids.tsv"
: > "$LOG"

submit() {  # submit EXP JNAME SCALE EXPORTS
  local EXP="$1" JNAME="$2" SCALE="$3" EXPORTS="$4"
  local SCRIPT="$SMALL"
  [ "$SCALE" = "BIG" ] && SCRIPT="$BIG"
  local JOBOUT JOBID
  JOBOUT=$(sbatch -J "$JNAME" --export=ALL,EXP_TAG="$EXP",$EXPORTS "$SCRIPT")
  JOBID="${JOBOUT##* }"
  echo -e "${EXP}\t${JNAME}\t${JOBID}\t${SCALE}\t${EXPORTS}" | tee -a "$LOG"
}

# CTRL1: masked fixed-K8
submit e162 e162_maskfixedk_cartpole_small small "TASK=dmc_cartpole_swingup,RECIPE=mask_fixedk,MASK_MODE=prob,MGR_FREQ=8"
submit e163 e163_maskfixedk_cartpole_BIG   BIG   "TASK=dmc_cartpole_swingup,RECIPE=mask_fixedk,MASK_MODE=prob,MGR_FREQ=8"
submit e164 e164_maskfixedk_hopper_small   small "TASK=dmc_hopper_hop,RECIPE=mask_fixedk,MASK_MODE=prob,MGR_FREQ=8"
submit e165 e165_maskfixedk_hopper_BIG     BIG   "TASK=dmc_hopper_hop,RECIPE=mask_fixedk,MASK_MODE=prob,MGR_FREQ=8"

# CTRL2: plain var-K (full replacement, struct 0)
submit e166 e166_plainvark_cartpole_small small "TASK=dmc_cartpole_swingup,RECIPE=plain_vark,DUR_MODE=fixed,DUR_TARGET=4.0"
submit e167 e167_plainvark_cartpole_BIG   BIG   "TASK=dmc_cartpole_swingup,RECIPE=plain_vark,DUR_MODE=fixed,DUR_TARGET=4.0"
submit e168 e168_plainvark_hopper_small   small "TASK=dmc_hopper_hop,RECIPE=plain_vark,DUR_MODE=fixed,DUR_TARGET=4.0"
submit e169 e169_plainvark_hopper_BIG     BIG   "TASK=dmc_hopper_hop,RECIPE=plain_vark,DUR_MODE=fixed,DUR_TARGET=4.0"

# Group D retry on fixed code: cartpole, hopper, cheetah, acrobot
DFLAGS="RECIPE=vark_masked,MASK_MODE=prob_entropy,DUR_MODE=lagrangian,DUR_TARGET=4.0"
submit e170 e170_dfix_cartpole_small small "TASK=dmc_cartpole_swingup,$DFLAGS"
submit e171 e171_dfix_cartpole_BIG   BIG   "TASK=dmc_cartpole_swingup,$DFLAGS"
submit e172 e172_dfix_hopper_small   small "TASK=dmc_hopper_hop,$DFLAGS"
submit e173 e173_dfix_hopper_BIG     BIG   "TASK=dmc_hopper_hop,$DFLAGS"
submit e174 e174_dfix_cheetah_small  small "TASK=dmc_cheetah_run,$DFLAGS"
submit e175 e175_dfix_cheetah_BIG    BIG   "TASK=dmc_cheetah_run,$DFLAGS"
submit e176 e176_dfix_acrobot_small  small "TASK=dmc_acrobot_swingup,$DFLAGS"
submit e177 e177_dfix_acrobot_BIG    BIG   "TASK=dmc_acrobot_swingup,$DFLAGS"

echo "Submitted 16 jobs. Log: $LOG"
