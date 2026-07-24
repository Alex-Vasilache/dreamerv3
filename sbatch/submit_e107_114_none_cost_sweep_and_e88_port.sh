#!/bin/bash -l
# e107-e114: cartpole perblock_edit_cost sweep (mask none) + e88-style multi-env port.
#
# e88 reference (best-15 764, blk/step 0.16 @ 4M):
#   MGR_REWARD_AGG=sum DUR_REG=0.0 DUR_MAX=32 SWITCH_COST=0.02
#   MASK_MODE=prob PERBLOCK_CREDIT=True PERBLOCK_EDIT_COST=0.3
#
# e107-e109: cartpole, mask none, sweep perblock_edit_cost at e88 switch_cost.
# e110-e114: e88 recipe verbatim on 5 DMC tasks (mask prob, not none).
set -euo pipefail
cd /apps/unit/DoyaU/vasilache/apps/code/dreamerv3

CART=sbatch/run_v3_cartpole_freedom_sweep.sbatch
ENVSB=sbatch/run_v3_freedom_sweep_env.sbatch

BASE="MGR_REWARD_AGG=sum,DUR_REG=0.0,DUR_TARGET=4.0,DUR_MAX=32,SWITCH_COST=0.02,PERBLOCK_CREDIT=True,MASK_MODE=none"
E88="MGR_REWARD_AGG=sum,DUR_REG=0.0,DUR_TARGET=4.0,DUR_MAX=32,SWITCH_COST=0.02,PERBLOCK_CREDIT=True,PERBLOCK_EDIT_COST=0.3,MASK_MODE=prob"

# --- cartpole: mask none, perblock cost sweep ---
sbatch -J e107_cartpole --export=ALL,EXP_TAG=e107,${BASE},PERBLOCK_EDIT_COST=0.1 "$CART"
sbatch -J e108_cartpole --export=ALL,EXP_TAG=e108,${BASE},PERBLOCK_EDIT_COST=0.3 "$CART"
sbatch -J e109_cartpole --export=ALL,EXP_TAG=e109,${BASE},PERBLOCK_EDIT_COST=1.0 "$CART"

# --- e88-style ports ---
sbatch -J e110_acrobot   --export=ALL,EXP_TAG=e110,ENV_NAME=acrobot,TASK=dmc_acrobot_swingup,${E88} "$ENVSB"
sbatch -J e111_cheetah   --export=ALL,EXP_TAG=e111,ENV_NAME=cheetah,TASK=dmc_cheetah_run,${E88} "$ENVSB"
sbatch -J e112_pendulum  --export=ALL,EXP_TAG=e112,ENV_NAME=pendulum,TASK=dmc_pendulum_swingup,${E88} "$ENVSB"
sbatch -J e113_quadruped --export=ALL,EXP_TAG=e113,ENV_NAME=quadruped,TASK=dmc_quadruped_run,${E88} "$ENVSB"
sbatch -J e114_hopper    --export=ALL,EXP_TAG=e114,ENV_NAME=hopper,TASK=dmc_hopper_hop,${E88} "$ENVSB"

echo "submitted e107-e114"
