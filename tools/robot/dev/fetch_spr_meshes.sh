#!/usr/bin/env bash
# Download the robot's STL meshes (body, phone, wheels) from tojo-soraai's
# rlSim2Real MuJoCo model, for sim_video3d.py --meshes. ~25 MB.
#
#   tools/robot/dev/fetch_spr_meshes.sh [DIR]     # default ~/.cache/spr_meshes
set -euo pipefail
DIR=${1:-$HOME/.cache/spr_meshes}
BASE=https://raw.githubusercontent.com/tojo-soraai/smartphone-robot-android/rl-sim2real/apps/rlSim2Real/mujoco/assets/meshes
mkdir -p "$DIR"
for f in mesh_spr_body.stl mesh_spr_phone.stl mesh_spr_wheel_left.stl mesh_spr_wheel_right.stl; do
  [ -s "$DIR/$f" ] || curl -fsSL -o "$DIR/$f" "$BASE/$f"
done
echo "$DIR"
