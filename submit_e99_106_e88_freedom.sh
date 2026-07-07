#!/bin/bash -l
# e99-e106: e88 improvements (NO duration prior — full freedom) + multi-env port.
# Supersedes the misconfigured e91-e98 (which wrongly used a soft duration prior;
# cancelled within minutes, no data).
#
# e88 base recipe (best emergent-on-both-axes run, best-15 764 @ blk/step 0.16):
#   MGR_REWARD_AGG=sum DUR_REG=0.0 DUR_MAX=32 SWITCH_COST=0.02
#   MASK_MODE=prob PERBLOCK_CREDIT=True PERBLOCK_EDIT_COST=0.3
# Duration prior stays OFF (DUR_REG=0.0) in every run — K is fully emergent.
#
# Improvements under test:
#   (1) shorter emergent K -> lower the only length pressure: SWITCH_COST 0.02 -> 0.01.
#       No prior; K stays emergent. Pulls K (e88 ~18) shorter toward interior.
#   (3) prove emergent sparsity -> remove the (inert) sparsity target: MASK_MODE=none.
#       In e88 the prob-0.3 multiplier sat at its floor; per-block credit holds the
#       mask at ~0.35 on its own. MASK_MODE=none removes the target outright.
#
# e99 = (1) only          e100 = (3) only         e101 = (1)+(3)
# e102-e106 = (1)+(3) config ported to 5 DMC tasks.
#
# 8 jobs x 1 v100 = 8 GPUs = the gpu-v100 cap. e65 antmaze runs on a100 (separate).
set -euo pipefail
cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3

CART=run_v3_cartpole_freedom_sweep.sbatch
ENVSB=run_v3_freedom_sweep_env.sbatch

# Common e88 base knobs (NO duration prior; switch cost set per-run below).
BASE="MGR_REWARD_AGG=sum,DUR_REG=0.0,DUR_TARGET=4.0,DUR_MAX=32,PERBLOCK_CREDIT=True,PERBLOCK_EDIT_COST=0.3"

# --- cartpole ablations ---
# e99: (1) only  -- lower switch cost, keep prob target
sbatch -J e99_cartpole  --export=ALL,EXP_TAG=e99,${BASE},SWITCH_COST=0.01,MASK_MODE=prob "$CART"
# e100: (3) only -- remove sparsity target, e88 switch cost
sbatch -J e100_cartpole --export=ALL,EXP_TAG=e100,${BASE},SWITCH_COST=0.02,MASK_MODE=none "$CART"
# e101: (1)+(3) -- lower switch + emergent sparsity (full freedom on both axes)
sbatch -J e101_cartpole --export=ALL,EXP_TAG=e101,${BASE},SWITCH_COST=0.01,MASK_MODE=none "$CART"

# --- (1)+(3) ported to other DMC envs ---
sbatch -J e102_acrobot   --export=ALL,EXP_TAG=e102,ENV_NAME=acrobot,TASK=dmc_acrobot_swingup,${BASE},SWITCH_COST=0.01,MASK_MODE=none "$ENVSB"
sbatch -J e103_cheetah   --export=ALL,EXP_TAG=e103,ENV_NAME=cheetah,TASK=dmc_cheetah_run,${BASE},SWITCH_COST=0.01,MASK_MODE=none "$ENVSB"
sbatch -J e104_pendulum  --export=ALL,EXP_TAG=e104,ENV_NAME=pendulum,TASK=dmc_pendulum_swingup,${BASE},SWITCH_COST=0.01,MASK_MODE=none "$ENVSB"
sbatch -J e105_quadruped --export=ALL,EXP_TAG=e105,ENV_NAME=quadruped,TASK=dmc_quadruped_run,${BASE},SWITCH_COST=0.01,MASK_MODE=none "$ENVSB"
sbatch -J e106_hopper    --export=ALL,EXP_TAG=e106,ENV_NAME=hopper,TASK=dmc_hopper_hop,${BASE},SWITCH_COST=0.01,MASK_MODE=none "$ENVSB"

echo "submitted e99-e106"
