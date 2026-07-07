#!/bin/bash -l
# e91-e98: e88-derived recipe improvements + multi-env port.
#
# e88 base recipe (best emergent-on-both-axes run, best-15 764 @ blk/step 0.16):
#   MGR_REWARD_AGG=sum DUR_REG=0.0 DUR_TARGET=4.0 DUR_MAX=32 SWITCH_COST=0.02
#   MASK_MODE=prob PERBLOCK_CREDIT=True PERBLOCK_EDIT_COST=0.3
#
# Improvements under test:
#   (1) shorter emergent K -> activate the dormant soft duration prior: DUR_REG=0.01
#       (DUR_TARGET=4.0, e57's proven sweet spot). Pulls K (e88 ~18) toward interior.
#   (3) prove emergent sparsity -> remove the (inert) sparsity target: MASK_MODE=none.
#       In e88 the prob-0.3 multiplier sat at its floor; per-block credit holds the
#       mask at ~0.35 on its own. MASK_MODE=none removes the target outright.
#
# e91 = (1) only          e92 = (3) only          e93 = (1)+(3)
# e94-e98 = (1)+(3) config ported to 5 DMC tasks.
#
# 8 jobs x 1 v100 = 8 GPUs = the gpu-v100 cap. e65 antmaze runs on a100 (separate).
set -euo pipefail
cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3

CART=run_v3_cartpole_freedom_sweep.sbatch
ENVSB=run_v3_freedom_sweep_env.sbatch

# Common e88 base knobs (carried by every run below).
BASE="MGR_REWARD_AGG=sum,DUR_TARGET=4.0,DUR_MAX=32,SWITCH_COST=0.02,PERBLOCK_CREDIT=True,PERBLOCK_EDIT_COST=0.3"

# --- cartpole ablations ---
# e91: (1) only  -- activate soft prior, keep prob target
sbatch -J e91_cartpole --export=ALL,EXP_TAG=e91,${BASE},DUR_REG=0.01,MASK_MODE=prob "$CART"
# e92: (3) only  -- remove sparsity target, no prior
sbatch -J e92_cartpole --export=ALL,EXP_TAG=e92,${BASE},DUR_REG=0.0,MASK_MODE=none "$CART"
# e93: (1)+(3)   -- soft prior + emergent sparsity
sbatch -J e93_cartpole --export=ALL,EXP_TAG=e93,${BASE},DUR_REG=0.01,MASK_MODE=none "$CART"

# --- (1)+(3) ported to other DMC envs ---
sbatch -J e94_acrobot   --export=ALL,EXP_TAG=e94,ENV_NAME=acrobot,TASK=dmc_acrobot_swingup,${BASE},DUR_REG=0.01,MASK_MODE=none "$ENVSB"
sbatch -J e95_cheetah   --export=ALL,EXP_TAG=e95,ENV_NAME=cheetah,TASK=dmc_cheetah_run,${BASE},DUR_REG=0.01,MASK_MODE=none "$ENVSB"
sbatch -J e96_pendulum  --export=ALL,EXP_TAG=e96,ENV_NAME=pendulum,TASK=dmc_pendulum_swingup,${BASE},DUR_REG=0.01,MASK_MODE=none "$ENVSB"
sbatch -J e97_quadruped --export=ALL,EXP_TAG=e97,ENV_NAME=quadruped,TASK=dmc_quadruped_run,${BASE},DUR_REG=0.01,MASK_MODE=none "$ENVSB"
sbatch -J e98_hopper    --export=ALL,EXP_TAG=e98,ENV_NAME=hopper,TASK=dmc_hopper_hop,${BASE},DUR_REG=0.01,MASK_MODE=none "$ENVSB"

echo "submitted e91-e98"
