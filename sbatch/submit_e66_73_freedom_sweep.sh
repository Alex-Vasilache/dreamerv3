#!/bin/bash -l
# Manager-freedom sweep e66-e73 (cartpole). 8 v100 runs (= v100 GPU cap).
#   Phase0 (e66): length freedom via SUM aggregation (prior off, sparsity target 0.3)
#   PhaseA (e67-69): length freedom via switch-cost on MEAN (prior off, sparsity 0.3)
#   PhaseB (e70-72): sparsity freedom via edit-cost (sparsity none, prior pins K=4)
#   PhaseC (e73): both freedoms combined (switch 0.05 + edit 0.1, both targets off)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$SCRIPT_DIR/run_v3_cartpole_freedom_sweep.sbatch"

sub() {  # sub <exp> <KEY=VAL ...>
  local exp="$1"; shift
  local exports="ALL,EXP_TAG=${exp}"
  for kv in "$@"; do exports="${exports},${kv}"; done
  sbatch -J "${exp}_cartpole" --export="$exports" "$S"
}

# Phase 0 — sum aggregation alone (preferred length fix)
sub e66 MGR_REWARD_AGG=sum DUR_REG=0.0 MASK_MODE=prob

# Phase A — switch-cost on mean (fallback length lever), sparsity target on
sub e67 MGR_REWARD_AGG=mean DUR_REG=0.0 MASK_MODE=prob SWITCH_COST=0.02
sub e68 MGR_REWARD_AGG=mean DUR_REG=0.0 MASK_MODE=prob SWITCH_COST=0.05
sub e69 MGR_REWARD_AGG=mean DUR_REG=0.0 MASK_MODE=prob SWITCH_COST=0.1

# Phase B — edit-cost alone (sparsity freedom), length pinned by prior (target 4)
sub e70 MGR_REWARD_AGG=mean DUR_REG=0.01 DUR_TARGET=4.0 MASK_MODE=none EDIT_COST=0.05
sub e71 MGR_REWARD_AGG=mean DUR_REG=0.01 DUR_TARGET=4.0 MASK_MODE=none EDIT_COST=0.1
sub e72 MGR_REWARD_AGG=mean DUR_REG=0.01 DUR_TARGET=4.0 MASK_MODE=none EDIT_COST=0.2

# Phase C — both costs combined, both targets off, no actent change
sub e73 MGR_REWARD_AGG=mean DUR_REG=0.0 MASK_MODE=none SWITCH_COST=0.05 EDIT_COST=0.1
