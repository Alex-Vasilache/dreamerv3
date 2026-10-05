#!/usr/bin/env bash
# Point the phone at a robot job running on Saion.
#
# The compute node is not reachable from the lab WiFi, so this forwards the
# MacBook's port 3000 to the port the job listens on. The phone keeps its
# config.json pointing at this MacBook exactly as for a local run; only what
# sits behind port 3000 changes.
#
#   tools/robot_tunnel.sh            # follow the newest robot job
#   tools/robot_tunnel.sh 12345678   # a specific job id
#
# Leave it running for the length of the run. Ctrl-C closes the tunnel, which
# disconnects the robot but does not stop the job. Note that this script execs
# into ssh, so the running process is named `ssh -N ...` and NOT this script:
# kill it with `pkill -f 'ssh -N -o ControlPath=none'`, not by script name.
set -uo pipefail

# saion when the internal name resolves, saion-ext otherwise. Measured
# 2026-09-03: direct is 2.3 MB/s up / 2.1 MB/s down, while the login.oist.jp
# path is policed to ~10 KB/s aggregate -- 1, 4 and 8 parallel streams all sum
# to the same ~10 KB/s, so batching and concurrency cannot rescue it. Only the
# direct host clears the 93 KB/s the bridge needs.
if [ -z "${REMOTE:-}" ]; then
  if [ -n "$(dig +short saion.oist.jp 2>/dev/null)" ]; then
    REMOTE=saion
  else
    REMOTE=saion-ext
  fi
fi
LOCAL_PORT="${LOCAL_PORT:-3000}"
ENDPOINT_DIR="${ENDPOINT_DIR:-/work/DoyaU/vasilache/work/robot_endpoints}"
JOB="${1:-latest}"

endpoint=$(ssh "$REMOTE" "cat $ENDPOINT_DIR/${JOB/#[0-9]*/job_$JOB} 2>/dev/null")
if [ -z "$endpoint" ]; then
  echo "No robot endpoint at $REMOTE:$ENDPOINT_DIR/$JOB." >&2
  echo "Is the job running? Check with:  ssh $REMOTE squeue --me" >&2
  exit 1
fi
eval "$endpoint"

echo "Job:    $JOB on $NODE:$PORT (started $STARTED)"
echo "Tunnel: phone -> $(ipconfig getifaddr en0 2>/dev/null || echo this-mac):$LOCAL_PORT -> $NODE:$PORT"
echo "Logs:   ssh $REMOTE tail -f $RUN_DIR/logdir/metrics.jsonl"
echo

# 0.0.0.0 rather than the default localhost bind: the phone is a different
# machine, so a loopback-only forward would accept nothing. ServerAlive keeps a
# quiet tunnel from being dropped by a NAT timeout mid-episode.
# Reconnect rather than exec: a run is hours long, the hop goes through
# login.oist.jp, and a single dropped SSH session would silently end the
# experiment -- the job keeps burning GPU while the robot sits unreachable.
# The phone retries on its own, so a rebuilt tunnel heals the whole chain.
#
# ControlPath=none deliberately: multiplexed over a shared master, this tunnel
# dies whenever anything closes that master (ssh -O exit, a wedged socket), and
# the robot drops mid-episode with no obvious cause.
trap 'echo "tunnel closed by request"; exit 0' INT TERM
while :; do
  ssh -N \
    -o ControlPath=none \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -L "0.0.0.0:${LOCAL_PORT}:${NODE}:${PORT}" \
    "$REMOTE"
  echo "$(date +%H:%M:%S) tunnel dropped (rc=$?), reconnecting in 3s"
  sleep 3
done
