#!/bin/bash -l
# Queue e57–e60 (cartpole duration-target sweep) + e61 (walker @ target 8).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CART="$SCRIPT_DIR/run_v3_cartpole_vargoal_masked_durreg_soft.sbatch"
WALKER="$SCRIPT_DIR/run_v3_e61_walker_vargoal_masked_durreg_soft8.sbatch"

submit_cart() {
  local exp="$1" target="$2" dur_max="${3:-16}"
  sbatch -J "${exp}_cartpole" \
    --export=ALL,EXP_TAG="$exp",DUR_TARGET="$target",DUR_MAX="$dur_max" \
    "$CART"
}

submit_cart e57 4.0
submit_cart e58 8.0
submit_cart e59 16.0
submit_cart e60 32.0 32

sbatch "$WALKER"
