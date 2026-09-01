#!/bin/bash
# Start the goal-code explorer on the login node (same pattern as
# tensorboard_login.sh). Extra args are passed to server.py.
#
#   ./tools/goal_demo/run_demo.sh              # port 8899
#   ./tools/goal_demo/run_demo.sh --port 8123
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /apps/unit/DoyaU/vasilache/apps/source_dreamerv3_env.sh
cd "$HERE"
exec python -u server.py "$@"
