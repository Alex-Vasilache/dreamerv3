#!/usr/bin/env bash
# Live TensorBoard for a split run, with both halves in one place.
#
# The actor runs here and torch is installed, so it writes event files itself.
# The learner runs on Saion, where torch is not installed, so it writes only
# metrics.jsonl -- which tools/robot_bridge_sync.sh copies back into
# learner_metrics/. This tails that file and appends new scalars to event files
# as they arrive, then serves the whole logdir.
#
#   tools/robot_tensorboard.sh [logdir]     # default ~/logdir/robot_local
#
# Then open http://localhost:6006. Runs appear as `.` (actor) and
# `learner_tb` (learner losses, train ratio, throughput).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${1:-$HOME/logdir/robot_local}"
PORT="${PORT:-6006}"
mkdir -p "$LOGDIR/learner_metrics"

# Append-only converter: it remembers how many lines it has consumed, so a
# growing metrics.jsonl becomes a growing event file rather than being rewritten
# (which would duplicate every point on each pass).
"$ROOT/.venv/bin/python" - "$LOGDIR" <<'PY' &
import json, pathlib, sys, time
from torch.utils.tensorboard import SummaryWriter
logdir = pathlib.Path(sys.argv[1])
src = logdir / 'learner_metrics' / 'metrics.jsonl'
writer = SummaryWriter(str(logdir / 'learner_tb'), flush_secs=10)
seen = 0
while True:
    try:
        if src.exists():
            lines = src.read_text().splitlines()
            for line in lines[seen:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = int(row.get('step', 0))
                for key, value in row.items():
                    if key != 'step' and isinstance(value, (int, float)):
                        writer.add_scalar(key, float(value), step)
            seen = len(lines)
            writer.flush()
    except Exception:
        # A logging helper must never be the thing that stops a run.
        pass
    time.sleep(10)
PY
converter=$!
trap 'kill $converter 2>/dev/null' INT TERM EXIT

echo "TensorBoard on http://localhost:$PORT  (logdir $LOGDIR)"
# --reload_multifile: the actor and the converter each write their own event
# files into this tree, and TensorBoard otherwise reads only the newest per
# directory, so half the metrics would look frozen.
exec "$ROOT/.venv/bin/tensorboard" \
  --logdir "$LOGDIR" --port "$PORT" \
  --reload_interval 5 --reload_multifile true
