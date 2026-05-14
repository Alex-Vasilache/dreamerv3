#!/usr/bin/env bash
# Saion login node: TensorBoard over cartpole-vis runs — baseline (v3 defaults), v4
# (all flags), and single-flag A/Bs (ab_pmpo, ab_rms, ab_rollout).
# One-run command template: TENSORBOARD_SAION.md
#
#   ./tensorboard_all_last_runs.sh
#   PORT=6007 BACKGROUND=1 ./tensorboard_all_last_runs.sh
#
# Only the newest run per variant (one v4, one ab_pmpo, … + baseline):
#   TB_ONE_PER_VARIANT=1 ./tensorboard_all_last_runs.sh
#
# Override directory glob (default is the cartpole-vis experiment family):
#   RUN_GLOB='dreamerv3_online_vis1m32_1env_*' ./tensorboard_all_last_runs.sh

set -euo pipefail

# shellcheck source=/dev/null
source "/apps/unit/DoyaU/vasilache/rl_env/bin/activate"

BASE="${BASE:-/work/DoyaU/vasilache/work}"
RUN_GLOB="${RUN_GLOB:-dreamerv3_online_vis1m32_1env_*}"
PORT="${PORT:-6006}"
BACKGROUND="${BACKGROUND:-0}"
TBLOG="${BASE}/tensorboard_all_last_${PORT}.log"
TB_ONE_PER_VARIANT="${TB_ONE_PER_VARIANT:-0}"

variant_key() {
  local b="$1"
  if [[ "$b" == *'_v4_'* ]]; then echo v4
  elif [[ "$b" == *'_ab_pmpo_'* ]]; then echo ab_pmpo
  elif [[ "$b" == *'_ab_rms_'* ]]; then echo ab_rms
  elif [[ "$b" == *'_ab_rollout_'* ]]; then echo ab_rollout
  else echo baseline
  fi
}

declare -A seen=()
spec=""
# Newest-first so TB_ONE_PER_VARIANT keeps the latest per variant.
while IFS= read -r d; do
  [[ -d "$d" ]] || continue
  [[ -d "$d/logdir" ]] || continue
  b=$(basename "$d")
  if [[ "$TB_ONE_PER_VARIANT" == 1 ]]; then
    k=$(variant_key "$b")
    [[ -n "${seen[$k]:-}" ]] && continue
    seen[$k]=1
  fi
  spec+="${b}:$d/logdir,"
done < <(find "$BASE" -maxdepth 1 -type d -name "$RUN_GLOB" -printf '%T@\t%p\n' 2>/dev/null | sort -rn | cut -f2-)

spec="${spec%,}"
if [[ -z "$spec" ]]; then
  echo "No runs with logdir under $BASE matching $RUN_GLOB" >&2
  exit 1
fi

tb_cmd=(
  tensorboard --logdir_spec="$spec" --port "$PORT" --host 0.0.0.0
  --load_fast=false)

echo "Runs: $(echo "$spec" | tr ',' '\n' | wc -l)"
echo "URL:  http://127.0.0.1:${PORT}/"

if [[ "$BACKGROUND" == 1 ]]; then
  nohup "${tb_cmd[@]}" >"$TBLOG" 2>&1 &
  echo "TB_PID=$!"
  echo "Log:  $TBLOG"
else
  exec "${tb_cmd[@]}"
fi
