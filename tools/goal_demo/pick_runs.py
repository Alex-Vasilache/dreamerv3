"""Re-derive the seed choice in ``runs.py`` from the finished runs.

``runs.py`` pins one seed per environment per arm: the best-scoring one at the
checkpoint the demo loads. Seeds keep finishing, so this recomputes the scores
and says whether any pin has gone stale (or prints a fresh RUNS block).

    python tools/goal_demo/pick_runs.py           # check the pins
    python tools/goal_demo/pick_runs.py --emit    # print a RUNS dict to paste

Score is the mean episode return over the 300k env steps that end at the
checkpoint (``scores.jsonl``), which is steadier than the value at one step.
Runs are looked up on /work first, then in the bucket.
"""
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
  sys.path.insert(0, _HERE)

import runs as runsmod  # noqa: E402

WINDOW = 3e5


def score(run_dir, step):
  path = os.path.join(run_dir, 'logdir', 'scores.jsonl')
  if not os.path.exists(path):
    return None
  vals = []
  for line in open(path):
    try:
      row = json.loads(line)
    except ValueError:
      continue
    if row.get('episode/score') is not None and step - WINDOW <= row['step'] <= step:
      vals.append(row['episode/score'])
  return sum(vals) / len(vals) if vals else None


def candidates(task, arm):
  got = {}
  for root in (runsmod.BUCKET, runsmod.WORK):   # /work wins
    for d in glob.glob(os.path.join(root, runsmod.PATTERNS[arm].format(t=task))):
      if os.path.exists(os.path.join(d, 'logdir', 'ckpt_milestones', runsmod.STEP,
                                     'agent.pkl')):
        got[os.path.basename(d)] = d
  return sorted(got.items())


def best(task, arm):
  rows = [(name, score(d, int(runsmod.STEP))) for name, d in candidates(task, arm)]
  rows = [r for r in rows if r[1] is not None]
  return sorted(rows, key=lambda r: -r[1])


def main():
  emit = '--emit' in sys.argv
  stale, out = 0, {}
  for task in runsmod.TASK_LABELS:
    out[task] = {}
    print(runsmod.TASK_LABELS[task])
    for arm in runsmod.ARMS:
      rows = best(task, arm)
      pinned = runsmod.RUNS.get(task, {}).get(arm, ('',))[0]
      for name, s in rows:
        print(f'  {arm:8s} {s:7.1f}  {name[:70]}' + ('  <-- pinned' if name == pinned else ''))
      if rows:
        out[task][arm] = rows[0]
        if rows[0][0] != pinned:
          stale += 1
  if emit:
    print('\nRUNS = {')
    for task, arms in out.items():
      print(f"    '{task}': {{")
      for arm, (name, s) in arms.items():
        print(f"        '{arm}': ('{name}', {s:.1f}),")
      print('    },')
    print('}')
  print('runs.py is up to date' if not stale else f'{stale} pin(s) differ from the best seed')
  return 1 if stale else 0


if __name__ == '__main__':
  raise SystemExit(main())
