"""Re-derive the seed choice in ``runs.py`` from the runs on /work.

``runs.py`` pins one seed per environment per arm: the better-scoring one at the
3M checkpoint the demo loads. Seeds keep training and new ones get launched, so
this recomputes the scores and says whether any pin has gone stale.

    python tools/goal_demo/pick_runs.py

Score is the mean episode return over env steps 2.7M-3.0M -- the 300k steps
ending at the checkpoint -- which is steadier than the value at the single step.
"""
import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
  sys.path.insert(0, _HERE)

import runs as runsmod  # noqa: E402

LO, HI = 2.7e6, 3.0e6
SCORE_KEYS = ('episode/score', 'eval/episode/score', 'stats/score')

# What a run directory name has between the task and the '_BIG_' suffix. These
# are anchored because 'dmc_cartpole_swingup' is a prefix of
# 'dmc_cartpole_swingup_sparse', which is a different task.
ARM_SUFFIX = {
    'director': re.compile(r'^_director_s\d+$'),
    'ours': re.compile(r'^_som_lipvq_line_relu_gaussian_s\d+$'),
}


def score_at_3m(run_dir):
  """Mean episode return over ``[LO, HI]``, or None if the run is not there yet."""
  path = os.path.join(run_dir, 'logdir', 'metrics.jsonl')
  if not os.path.exists(path):
    return None, 0
  vals, last = [], 0
  with open(path) as f:
    for line in f:
      try:
        row = json.loads(line)
      except ValueError:
        continue
      step = row.get('step')
      if step is None:
        continue
      last = max(last, step)
      if not LO <= step <= HI:
        continue
      for key in SCORE_KEYS:
        if key in row:
          vals.append(row[key])
          break
  return (sum(vals) / len(vals) if vals else None), last


def has_ckpt(run_dir):
  return os.path.exists(os.path.join(
      run_dir, 'logdir', 'ckpt_milestones', runsmod.STEP, 'agent.pkl'))


def candidates(task, arm):
  """Every run on /work for this task and arm, pinned or not."""
  out = []
  for path in sorted(glob.glob(os.path.join(runsmod.WORK, 'e*_BIG_*'))):
    body = os.path.basename(path).split('_BIG_')[0]
    match = re.match(r'^e\d+_(.*)$', body)
    if not match or not match.group(1).startswith(task):
      continue
    if ARM_SUFFIX[arm].match(match.group(1)[len(task):]):
      out.append(path)
  return out


def main():
  stale = 0
  for task in runsmod.TASKS:
    print(f'{runsmod.RUNS[task]["label"]}')
    for arm in runsmod.ARMS:
      pinned = runsmod.RUNS[task][arm][0]
      rows = []
      for path in candidates(task, arm):
        score, last = score_at_3m(path)
        rows.append((os.path.basename(path), score, last, has_ckpt(path)))
      usable = [r for r in rows if r[1] is not None and r[3]]
      usable.sort(key=lambda r: -r[1])
      for name, score, last, ckpt in sorted(rows, key=lambda r: -(r[1] or -1)):
        mark = '<-- pinned' if name == pinned else ''
        note = '' if ckpt else '  (no 3M checkpoint)'
        shown = f'{score:7.1f}' if score is not None else '      -'
        print(f'  {arm:9s} {shown}  {name[:62]:62s}'
              f' last {last/1e6:.2f}M{note} {mark}')
      if usable and usable[0][0] != pinned:
        print(f'  ** {arm}: {usable[0][0]} now scores higher than the pinned '
              f'{pinned}')
        stale += 1
    print()
  print('runs.py is up to date' if not stale else
        f'{stale} pinned seed(s) are no longer the best -- update runs.py')
  return 1 if stale else 0


if __name__ == '__main__':
  raise SystemExit(main())
