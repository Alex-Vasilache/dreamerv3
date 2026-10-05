#!/usr/bin/env bash
# Characterise the MacBook -> Saion link for the robot bridge, in about two
# minutes, so the transport is chosen from measurements rather than guesses.
#
#   tools/test_saion_link.sh
#
# What matters: the actor writes one 390KB chunk every ~4.1s at 48Hz, so the
# bridge must sustain ~93KB/s. Anything slower and replay falls behind the robot
# for as long as the run lasts.
#
# Each measurement is guarded, so a stalled transfer costs its limit and the
# run continues -- a hang is a result, not a reason to wait.
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
SSH="ssh -o BatchMode=yes -o ControlPath=none -o ConnectTimeout=15"
FAR="${FAR:-/work/DoyaU/vasilache/work/linktest}"
SRC="${SRC:-$HOME/logdir/robot_local/online_shared/experience}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
set -m

guard() {  # guard <seconds> <command...>
  local limit=$1; shift
  "$@" & local pid=$!
  ( sleep "$limit"; kill -9 -- -$pid 2>/dev/null ) 2>/dev/null & local dog=$!
  wait $pid 2>/dev/null; local rc=$?
  kill -9 -- -$dog 2>/dev/null
  return $rc
}

ms() { python3 -c "import sys;print(f'{(float(sys.argv[2])-float(sys.argv[1]))*1000:.0f}')" "$1" "$2"; }
now() { python3 -c "import time;print(time.time())"; }
rate() { python3 -c "import sys;b,t=float(sys.argv[1]),float(sys.argv[2]);print(f'{b/1024/t:.0f}' if t>0 else 'na')" "$1" "$2"; }

echo "=== Saion link test  $(date +%H:%M:%S)"
echo "requirement: 93 KB/s sustained (390KB chunk every 4.1s at 48Hz)"
echo

echo "-- 1. ssh setup cost (no payload)"
for i in 1 2 3; do
  a=$(now); guard 30 $SSH "$REMOTE" true >/dev/null 2>&1; rc=$?; b=$(now)
  echo "   try $i: $(ms "$a" "$b") ms  rc=$rc"
done

ls -1 "$SRC"/*.npz > "$TMP/all" 2>/dev/null
n=$(wc -l < "$TMP/all" | tr -d ' ')
if [ "$n" -lt 25 ]; then
  echo "need at least 25 chunks in $SRC (have $n); run the robot briefly first" >&2
  exit 1
fi
guard 30 $SSH "$REMOTE" "rm -rf $FAR; mkdir -p $FAR" >/dev/null 2>&1

echo
echo "-- 2. one 390KB chunk per rsync (what a naive bridge does)"
for i in 1 2; do
  f=$(sed -n "${i}p" "$TMP/all")
  a=$(now); guard 60 rsync -a -e "$SSH" "$f" "$REMOTE:$FAR/" >/dev/null 2>&1; rc=$?; b=$(now)
  t=$(python3 -c "print(float('$b')-float('$a'))")
  echo "   try $i: $(ms "$a" "$b") ms  -> $(rate 390000 "$t") KB/s  rc=$rc"
done

echo
echo "-- 3. 12 chunks in one rsync (4.7MB, one setup)"
sed -n '3,14p' "$TMP/all" | sed "s|.*/||" > "$TMP/list12"
a=$(now)
guard 120 rsync -a --files-from="$TMP/list12" -e "$SSH" "$SRC/" "$REMOTE:$FAR/" >/dev/null 2>&1
rc=$?; b=$(now); t=$(python3 -c "print(float('$b')-float('$a'))")
echo "   $(ms "$a" "$b") ms  -> $(rate 4680000 "$t") KB/s  rc=$rc"

echo
echo "-- 4. 25 chunks streamed as tar (9.8MB, one session, no per-file protocol)"
sed -n '15,39p' "$TMP/all" | sed "s|.*/||" > "$TMP/list25"
a=$(now)
guard 120 bash -c "cd '$SRC' && tar cf - -T '$TMP/list25' 2>/dev/null | $SSH $REMOTE 'tar xf - -C $FAR' 2>/dev/null"
rc=$?; b=$(now); t=$(python3 -c "print(float('$b')-float('$a'))")
echo "   $(ms "$a" "$b") ms  -> $(rate 9750000 "$t") KB/s  rc=$rc"

echo
echo "-- 5. small download (policy/metrics shaped, one session)"
a=$(now); guard 60 rsync -a -e "$SSH" "$REMOTE:$FAR/" "$TMP/down/" >/dev/null 2>&1; rc=$?; b=$(now)
echo "   $(ms "$a" "$b") ms  rc=$rc  files=$(ls "$TMP/down" 2>/dev/null | wc -l | tr -d ' ')"

echo
echo "-- 6. stall check: 8 rapid setups, looking for outliers"
worst=0
for i in $(seq 1 8); do
  a=$(now); guard 30 $SSH "$REMOTE" true >/dev/null 2>&1; b=$(now)
  d=$(ms "$a" "$b"); [ "$d" -gt "$worst" ] && worst=$d
  printf '%s ' "$d"
done
echo
echo "   worst: ${worst} ms"

guard 30 $SSH "$REMOTE" "rm -rf $FAR" >/dev/null 2>&1
echo
echo "=== done $(date +%H:%M:%S)"
