#!/usr/bin/env bash
# TensorBoard on the Saion login node (do not run inside Slurm jobs for this).
# Reference copy of the command: TENSORBOARD_SAION.md
#
# Canonical one-liner (same flags as the working setup):
#   source "/apps/unit/DoyaU/vasilache/rl_env/bin/activate" \
#     && tensorboard --logdir "/work/DoyaU/vasilache/work/<RUN_DIR>/" \
#        --port 6006 --host 0.0.0.0 --load_fast=false
#
# Event files live under <RUN_DIR>/logdir/; pointing --logdir at <RUN_DIR> is fine
# because TensorBoard recurses into subdirectories.
#
# Foreground (default):
#   LOGDIR=/work/.../my_run/ PORT=6006 ./tensorboard_login.sh
#
# Background with logs under the run directory:
#   LOGDIR=/work/.../my_run/ PORT=6006 BACKGROUND=1 ./tensorboard_login.sh
#
# If rl_env is missing on a host, fall back after adjusting paths/modules locally.

set -euo pipefail

# shellcheck source=/dev/null
source "/apps/unit/DoyaU/vasilache/rl_env/bin/activate"

LOGDIR="${LOGDIR:-/work/DoyaU/vasilache/work/dreamerv3_online_vis1m32_1env_v4_20260513_134842_d000kC/}"
PORT="${PORT:-6006}"
TBLOG="${LOGDIR%/}/tensorboard_${PORT}.log"
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
