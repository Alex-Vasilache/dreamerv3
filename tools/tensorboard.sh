#!/usr/bin/env bash
# TensorBoard for the robot runs.
#
# --reload_multifile matters here: the learner and actor both write event files
# into the same run directory, and TensorBoard otherwise reads only the most
# recent file per directory, so half the metrics never appear and the other
# half look frozen.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin/tensorboard" \
  --logdir "${1:-$HOME/logdir}" \
  --port "${PORT:-6006}" \
  --reload_interval 5 \
  --reload_multifile true
