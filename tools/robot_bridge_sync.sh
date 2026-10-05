#!/usr/bin/env bash
# Carry the actor/learner split across the network, so the robot's control loop
# stays local while the learner runs on a Saion V100.
#
# Why not tunnel the robot to the cluster instead: measured 2026-09-03, phone ->
# WiFi -> MacBook -> ssh -> login.oist.jp -> compute node delivered a median 9ms
# but a p90 of 3.8s and stalls up to 203s, which collapsed the control loop to
# 0.45Hz with 95% of observations dropped. Latency the pipelined protocol can
# absorb; multi-second gaps it cannot.
#
# What crosses the slow link here instead tolerates seconds happily:
#
#   actor  -> learner   replay chunks, the actor's step counter, shutdown
#   learner -> actor    policy weights (the actor reloads every 30s), ready flag
#
# Usage, with the learner job already running on Saion:
#
#   tools/robot_bridge_sync.sh ~/logdir/robot_local <remote_run_dir>/logdir
#
# Leave it running for the length of the run.
set -uo pipefail

LOCAL="${1:?usage: robot_bridge_sync.sh <local logdir> <remote logdir>}"
REMOTE_DIR="${2:?usage: robot_bridge_sync.sh <local logdir> <remote logdir>}"
# REMOTE=local runs both sides on this machine with plain file copies, which is
# how to exercise the split without the cluster (and the OIST jump host is not
# always up).
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
PERIOD="${PERIOD:-8}"
# Its own multiplexed connection: each rsync over the jump host costs seconds to
# set up otherwise, and sharing a socket with anything else means one `ssh -O
# exit` takes the run down with it.
SOCK=~/.ssh/cm/bridge-%r@%h:%p
# ConnectTimeout and ServerAlive matter more than they look: the OIST jump host
# stalls for minutes at a time, and without them one wedged transfer freezes
# every later cycle -- including the learner_ready fetch the actor is waiting
# on. Better to abandon a cycle and retry than to hang.
# ControlPath=none, deliberately paying ~5s of setup per transfer.
# Multiplexing looked like the obvious win, but a stale control socket
# makes ssh block forever -- ConnectTimeout does not apply while waiting
# on the socket -- and the whole bridge wedges with no error at all. At an
# 8s period the setup cost is affordable; a silent stall is not.
SSH="ssh -o BatchMode=yes -o ControlPath=none -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"
if [ "$REMOTE" = local ]; then
  SSH=""
  rsh() { eval "$@"; }
  far() { echo "$1"; }            # no host prefix on a local path
  rs() { rsync "$@"; }
else
  rsh() { $SSH "$REMOTE" "$@"; }
  far() { echo "$REMOTE:$1"; }
  rs() { rsync --timeout=45 -e "$SSH" "$@"; }
fi

mkdir -p ~/.ssh/cm "$LOCAL/online_shared/experience" "$LOCAL/online_shared/policy"

echo "local:  $LOCAL"
echo "remote: $REMOTE:$REMOTE_DIR"
echo "period: ${PERIOD}s"

rsh "mkdir -p $REMOTE_DIR/online_shared/experience $REMOTE_DIR/online_shared/policy" || {
  echo "cannot reach $REMOTE" >&2; exit 1; }

# macOS has no timeout(1), so every transfer gets its own watchdog. This is the
# difference between a working bridge and a dead one: measured 2026-09-03, an
# rsync through the OIST jump host hung for four minutes while interactive ssh
# to the same host still answered in seconds, and rsync's own --timeout does not
# fire because the connection is alive, just idle. A stall must cost one cycle,
# never the run.
# `set -m` matters: without job control a background job shares the shell's
# process group, so killing $pid kills only the wrapper and the rsync it
# spawned keeps running -- which is how hung transfers piled up three deep
# while the loop believed it had timed them out. With job control each job
# gets its own group and `kill -- -$pid` takes the whole tree.
set -m
guard() {  # guard <seconds> <command...>
  local limit=$1; shift
  "$@" & local pid=$!
  ( sleep "$limit"; kill -9 -- -$pid 2>/dev/null ) 2>/dev/null & local dog=$!
  disown $dog 2>/dev/null   # else killing it prints "Killed: 9" into the log
  wait $pid 2>/dev/null; local rc=$?
  kill -9 -- -$dog 2>/dev/null
  return $rc
}

# Chunks go up a few at a time rather than in one sweep. A backlog otherwise
# means one enormous rsync that blocks the policy download behind it, and the
# actor then runs on stale weights while the bridge looks busy.
BATCH="${BATCH:-12}"

cycle=0
while :; do
  cycle=$((cycle + 1))
  t0=$SECONDS

  guard 30 rs -a "$(far "$REMOTE_DIR/online_shared/learner_ready")" \
    "$LOCAL/online_shared/" >/dev/null 2>&1

  pending=0
  if [ -d "$LOCAL/online_shared/experience" ]; then
    ( cd "$LOCAL/online_shared/experience" &&
      find . -name '*.npz' -type f ! -newermt '-5 seconds' 2>/dev/null |
        sed 's|^\./|experience/|' > /tmp/.bridge_all )
    # Only what the far side does not have yet, oldest first.
    comm -23 <(sort /tmp/.bridge_all 2>/dev/null) <(sort /tmp/.bridge_sent 2>/dev/null) \
      > /tmp/.bridge_todo 2>/dev/null
    pending=$(wc -l < /tmp/.bridge_todo | tr -d ' ')
    head -n "$BATCH" /tmp/.bridge_todo > /tmp/.bridge_now
    # actor_step rides along in the same session, and must be re-sent every
    # cycle rather than skipped as already-present: the learner paces itself
    # with should_train(actor_step), so a stale 0 means it takes exactly one
    # train step and then spins forever on `repeats <= 0` while chunks pile up
    # and everything else looks healthy. It is five bytes; send it always.
    cp /tmp/.bridge_now /tmp/.bridge_files
    [ -f "$LOCAL/online_shared/actor_step" ] && echo actor_step >> /tmp/.bridge_files
    if [ -s /tmp/.bridge_files ]; then
      if guard 120 rs -a --files-from=/tmp/.bridge_files \
           "$LOCAL/online_shared/" \
           "$(far "$REMOTE_DIR/online_shared/")" >/dev/null 2>&1; then
        cat /tmp/.bridge_now >> /tmp/.bridge_sent
        sent=$(wc -l < /tmp/.bridge_now | tr -d ' ')
      else
        sent=0
      fi
    else
      sent=0
    fi
  fi

  # Everything coming back in ONE ssh session. Setup through the jump host
  # costs ~8s and the payload is kilobytes, so five separate transfers cost
  # five setups and nothing else -- the whole cycle was 50s for ~2MB.
  stage="$LOCAL/.incoming"
  mkdir -p "$stage" "$LOCAL/learner_metrics"
  guard 60 rs -a \
    --include='online_shared/' --include='online_shared/policy/***' \
    --include='online_shared/learner_ready' \
    --include='metrics.jsonl' --include='scores.jsonl' --exclude='*' \
    "$(far "$REMOTE_DIR/")" "$stage/" >/dev/null 2>&1
  [ -d "$stage/online_shared/policy" ] &&
    cp -f "$stage/online_shared/policy/"* "$LOCAL/online_shared/policy/" 2>/dev/null
  [ -f "$stage/online_shared/learner_ready" ] &&
    cp -f "$stage/online_shared/learner_ready" "$LOCAL/online_shared/" 2>/dev/null
  for f in metrics.jsonl scores.jsonl; do
    [ -f "$stage/$f" ] && cp -f "$stage/$f" "$LOCAL/learner_metrics/$f" 2>/dev/null
  done
  pol=$(ls "$LOCAL/online_shared/policy" 2>/dev/null | wc -l | tr -d ' ')
  met=$( { wc -l < "$LOCAL/learner_metrics/metrics.jsonl"; } 2>/dev/null | tr -d ' ')

  echo "$(date +%H:%M:%S) cycle $cycle  sent ${sent:-0}/${pending:-0} pending  policy $pol  learner_metrics ${met:-0}  took $((SECONDS - t0))s"
  sleep "$PERIOD"
done
