#!/usr/bin/env bash
# Saion login node (not Slurm). Matches the flow used before: source env, then
#   nohup tensorboard --logdir ... --port ... --host 0.0.0.0 --load_fast=false
# with stdout/stderr in the run logdir. If dreamerv3_env fails (e.g. GLIBC),
# try: module purge && module load python/3.7.3 && source /apps/unit/DoyaU/vasilache/apps/rl_env/bin/activate
#
# Foreground (default):
#   LOGDIR=.../logdir PORT=6016 ./tensorboard_login.sh
# Background like before:
#   LOGDIR=.../logdir PORT=6016 BACKGROUND=1 ./tensorboard_login.sh

set -euo pipefail

# shellcheck source=/dev/null
source /etc/profile.d/modules.sh 2>/dev/null || true
# shellcheck source=/dev/null
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh

LOGDIR="${LOGDIR:-/work/DoyaU/vasilache/work/dreamerv3_online_vis1m32_1env_v4_20260513_134842_d000kC/logdir}"
PORT="${PORT:-6016}"
TBLOG="${LOGDIR}/tensorboard_${PORT}.log"
BACKGROUND="${BACKGROUND:-0}"

tb_cmd=(tensorboard --logdir "$LOGDIR" --port "$PORT" --host 0.0.0.0 --load_fast=false)

echo "Logdir: $LOGDIR"
echo "URL:    http://127.0.0.1:${PORT}/"

if [[ "$BACKGROUND" == 1 ]]; then
  nohup "${tb_cmd[@]}" >"$TBLOG" 2>&1 &
  echo "TB_PID=$!"
  echo "Log:    $TBLOG"
else
  exec "${tb_cmd[@]}"
fi
