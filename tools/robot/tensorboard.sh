#!/usr/bin/env bash
# Live TensorBoard on this computer for robot runs, local or on the cluster.
#
#   tools/robot/tensorboard.sh                  # the latest robot job
#   tools/robot/tensorboard.sh <run dir> [more run dirs ...]   # compare runs
#
# Run dirs are paths on the trainer host (ROBOT_TRAINER in robot.env). Runs
# write metrics.jsonl and scores.jsonl; this copies those every 20 s into
# ~/logdir/robot_tb/<run>/ and
# appends new rows to event files, split into `actor` (episodes, link: fps,
# dropped, policy age; x = robot steps) and `learner` (losses, train fps;
# x = trained samples), then serves them on the first free port from 6006.
# Ctrl-C stops both.
set -uo pipefail
source "$(dirname "$0")/common.sh"
PY=$(robot_python)
command -v tensorboard >/dev/null || [ -x "$(dirname "$PY")/tensorboard" ] \
  || robot_die "needs tensorboard and torch: $PY -m pip install tensorboard torch"
# First free port from 6006 up (Cursor's extension host sits on 6006).
PORT=${PORT:-$(python3 -c '
import socket
for p in range(6006, 6100):
  s = socket.socket()
  try:
    s.bind(("127.0.0.1", p)); print(p); break
  except OSError:
    pass
  finally:
    s.close()')}
OUT="$HOME/logdir/robot_tb"
if [ $# -eq 0 ]; then
  set -- "$(latest_run_dir)"
  [ -n "$1" ] || robot_die "no running robot job in $ROBOT_RUNS/robot_endpoints; pass a run dir"
fi
mkdir -p "$OUT"
echo "runs: $*"
echo "TensorBoard: http://localhost:$PORT"

SRC=; [ "$ROBOT_TRAINER" = slurm ] && SRC="$ROBOT_SSH:"
"$PY" - "$OUT" "$SRC" "$@" <<'PY' &
import json, pathlib, subprocess, sys, time
from torch.utils.tensorboard import SummaryWriter

out = pathlib.Path(sys.argv[1])
src = sys.argv[2]  # '' for this computer, 'host:' for the cluster
runs = sys.argv[3:]
seen, writers = {}, {}

def writer(name, part):
  key = (name, part)
  if key not in writers:
    writers[key] = SummaryWriter(str(out / name / part), flush_secs=5)
  return writers[key]

def learner_row(row):
  return any(k.startswith(('train/', 'fps/train', 'report/')) for k in row)

while True:
  for run in runs:
    name = pathlib.PurePosixPath(run).name
    local = out / name / 'src'
    local.mkdir(parents=True, exist_ok=True)
    for f in ('metrics.jsonl', 'scores.jsonl'):
      subprocess.run(['rsync', '-q', f'{src}{run}/logdir/{f}', str(local / f)],
                     stderr=subprocess.DEVNULL)
      path = local / f
      if not path.exists():
        continue
      lines = path.read_text().splitlines()
      for line in lines[seen.get((name, f), 0):]:
        try:
          row = json.loads(line)
        except json.JSONDecodeError:
          continue
        step = int(row.pop('step', 0))
        w = writer(name, 'learner' if learner_row(row) else 'actor')
        for k, v in row.items():
          if isinstance(v, (int, float)):
            w.add_scalar(k, float(v), step)
      seen[(name, f)] = len(lines)
  for w in writers.values():
    w.flush()
  time.sleep(20)
PY
SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT
"$( [ -x "$(dirname "$PY")/tensorboard" ] && echo "$(dirname "$PY")/tensorboard" || echo tensorboard)" --logdir "$OUT" --port "$PORT" --reload_interval 10 2>&1 \
  | grep --line-buffered -v -E "^W|TensorFlow installation not found"
