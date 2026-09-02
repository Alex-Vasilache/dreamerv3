"""Backfill TensorBoard event files from a run's metrics.jsonl.

Runs logged before `tensorboard` was added to logger.outputs still have their
full scalar history in metrics.jsonl; this replays it so old and new runs can be
compared in one TensorBoard.

  .venv/bin/python tools/jsonl_to_tensorboard.py ~/logdir/robot_*
"""
import json
import pathlib
import sys

from torch.utils.tensorboard import SummaryWriter  # noqa


def convert(logdir):
    logdir = pathlib.Path(logdir)
    src = logdir / 'metrics.jsonl'
    if not src.exists():
        return f'{logdir.name}: no metrics.jsonl'
    rows = []
    for line in src.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return f'{logdir.name}: metrics.jsonl empty'
    out = logdir / 'tb_backfill'
    writer = SummaryWriter(str(out))
    n = 0
    for row in rows:
        step = int(row.get('step', 0))
        for key, value in row.items():
            if key == 'step' or not isinstance(value, (int, float)):
                continue
            writer.add_scalar(key, float(value), step)
            n += 1
    writer.close()
    return f'{logdir.name}: {len(rows)} rows, {n} scalars -> {out.name}/'


for path in sys.argv[1:]:
    print(convert(path))
