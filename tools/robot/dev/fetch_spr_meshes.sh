#!/usr/bin/env bash
# Fetch the robot's STL meshes (body, phone, wheels) from tojo-soraai's
# rlSim2Real MuJoCo model, for sim_video3d.py --meshes. ~25 MB. The repo is
# private: this uses git, so your GitHub credentials (gh auth) apply.
#
#   tools/robot/dev/fetch_spr_meshes.sh [DIR]     # default ~/.cache/spr_meshes
set -euo pipefail
DIR=${1:-$HOME/.cache/spr_meshes}
REPO=https://github.com/tojo-soraai/smartphone-robot-android.git
SUB=apps/rlSim2Real/mujoco/assets/meshes
mkdir -p "$DIR"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git clone -q --depth 1 -b rl-sim2real --filter=blob:none --sparse "$REPO" "$tmp/repo"
git -C "$tmp/repo" sparse-checkout set "$SUB"
for f in mesh_spr_body.stl mesh_spr_phone.stl mesh_spr_wheel_left.stl mesh_spr_wheel_right.stl; do
  cp "$tmp/repo/$SUB/$f" "$DIR/"
done
echo "$DIR"
