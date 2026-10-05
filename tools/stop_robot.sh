#!/usr/bin/env bash
# Stop a robot session started by tools/start_robot.sh. Does not cancel the
# Saion job -- that is `ssh saion scancel <job>` (saion-ext off-campus).
pkill -f 'dreamerv3/main.py'
pkill -f robot_bridge_sync
pkill -f fake_robot_phone
pkill -f 'bin/tensorboard'
# The tunnel helper execs into ssh, so it is not matchable by script name.
pkill -f 'ssh -N -o ControlPath=none'
sleep 1
echo "stopped; remaining: $(pgrep -f 'dreamerv3/main.py|robot_bridge_sync|bin/tensorboard' | wc -l)"
