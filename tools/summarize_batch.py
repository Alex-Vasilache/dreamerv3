#!/usr/bin/env python3
"""Status and score table for a batch of runs, from /work or /bucket.

Pure standard library on purpose: it has to run on the login node, where the
JAX env cannot even be imported (glibc), and it is the tool the unattended
supervision loop reads every few hours.

  python3 tools/summarize_batch.py                       # the e510-e557 batch
  python3 tools/summarize_batch.py --pattern 'e50*_BIG_j*'
  python3 tools/summarize_batch.py --json                # machine-readable

Score convention follows EXPERIMENTS.md §1: last-N-episode mean and the best
trailing-N window ("peak"), N=15 by default.
"""
import argparse
import glob
import json
import os
import statistics
import sys

WORK = '/work/DoyaU/vasilache/work'
BUCKET = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'

# Health metrics worth seeing at a glance. Keys are as written in
# metrics.jsonl; the short names are what the table prints.
HEALTH = [
    ('train/goal/perplexity', 'perpl'),
    ('train/goal/used_frac', 'used'),
    ('train/goal/rec_q', 'rec_q'),
    ('train/goal/lip_penalty', 'lip_pen'),
    ('train/goal/lip_bound_max', 'lip_max'),
    ('train/wkr_goal_rew', 'wkr_rew'),
    ('train/mgr_extr_rew', 'mgr_rew'),
]


def read_job_env(run_dir):
  out = {}
  path = os.path.join(run_dir, 'job.env')
  if not os.path.exists(path):
    return out
  with open(path) as f:
    for line in f:
      for field in line.strip().split(' '):
        if '=' in field:
          k, _, v = field.partition('=')
          out.setdefault(k, v)
  return out


def tail_lines(path, nbytes=400_000):
  """Last chunk of a file as lines, dropping a possibly-truncated first line."""
  if not os.path.exists(path):
    return []
  with open(path, 'rb') as f:
    f.seek(0, 2)
    size = f.tell()
    f.seek(max(0, size - nbytes))
    chunk = f.read().decode('utf8', 'ignore')
  lines = chunk.splitlines()
  return lines[1:] if size > nbytes else lines


def read_scores(run_dir, last_n):
  path = os.path.join(run_dir, 'logdir', 'scores.jsonl')
  scores, steps = [], []
  for line in tail_lines(path):
    try:
      row = json.loads(line)
    except Exception:
      continue
    if 'episode/score' in row:
      scores.append(float(row['episode/score']))
      steps.append(int(row.get('step', 0)))
  if not scores:
    return None
  last = statistics.mean(scores[-last_n:])
  peak = max(
      statistics.mean(scores[i:i + last_n])
      for i in range(max(1, len(scores) - last_n + 1))) if scores else 0.0
  return dict(n_eps=len(scores), last=last, peak=peak,
              step=steps[-1] if steps else 0)


def read_health(run_dir):
  path = os.path.join(run_dir, 'logdir', 'metrics.jsonl')
  out, step = {}, 0
  for line in tail_lines(path):
    try:
      row = json.loads(line)
    except Exception:
      continue
    step = max(step, int(row.get('step', 0)))
    for key, short in HEALTH:
      if key in row:
        out[short] = float(row[key])
  out['step'] = step
  return out


def collect(patterns, roots, last_n):
  runs = []
  for root in roots:
    for pattern in patterns:
      for d in sorted(glob.glob(os.path.join(root, pattern))):
        if not os.path.isdir(d):
          continue
        env = read_job_env(d)
        health = read_health(d)
        scores = read_scores(d, last_n)
        runs.append(dict(
            dir=d,
            root=os.path.basename(root),
            exp=env.get('EXP_TAG', os.path.basename(d).split('_')[0]),
            task=env.get('TASK', '?'),
            arm=env.get('ARM', '?'),
            seed=env.get('SEED', '?'),
            step=max(health.get('step', 0), (scores or {}).get('step', 0)),
            last=(scores or {}).get('last'),
            peak=(scores or {}).get('peak'),
            n_eps=(scores or {}).get('n_eps', 0),
            health={k: v for k, v in health.items() if k != 'step'},
        ))
  runs.sort(key=lambda r: (r['exp'], r['seed']))
  return runs


def fmt(v, width=7, prec=1):
  return ' ' * width if v is None else f'{v:>{width}.{prec}f}'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--pattern', action='append', default=None)
  ap.add_argument('--roots', default=f'{WORK},{BUCKET}')
  ap.add_argument('--last-n', type=int, default=15)
  ap.add_argument('--target-steps', type=int, default=4_000_000)
  ap.add_argument('--json', action='store_true')
  args = ap.parse_args()

  patterns = args.pattern or ['e5[1-5]*_BIG_j*']
  roots = [r for r in args.roots.split(',') if os.path.isdir(r)]
  runs = collect(patterns, roots, args.last_n)

  if args.json:
    json.dump(runs, sys.stdout, indent=2)
    print()
    return

  if not runs:
    print('no runs matched', patterns)
    return

  shorts = [s for _, s in HEALTH if any(s in r['health'] for r in runs)]
  head = (f'{"exp":<6} {"task":<26} {"arm":<26} {"s":<2} {"step":>9} '
          f'{"%":>5} {"last"+str(args.last_n):>8} {"peak":>8}')
  head += ''.join(f' {s:>8}' for s in shorts)
  print(head)
  print('-' * len(head))
  for r in runs:
    pct = 100.0 * r['step'] / args.target_steps
    line = (f'{r["exp"]:<6} {r["task"].replace("dmc_", ""):<26} {r["arm"]:<26} '
            f'{r["seed"]:<2} {r["step"]:>9,} {pct:>4.0f}% '
            f'{fmt(r["last"], 8)} {fmt(r["peak"], 8)}')
    line += ''.join(f' {fmt(r["health"].get(s), 8, 3)}' for s in shorts)
    print(line)

  print()
  by_cell = {}
  for r in runs:
    if r['last'] is not None:
      by_cell.setdefault((r['task'], r['arm']), []).append(r['last'])
  if by_cell:
    print(f'{"task":<26} {"arm":<26} {"n":>2} {"mean":>8} {"std":>8} '
          f'{"min":>8} {"max":>8}')
    for (task, arm), vals in sorted(by_cell.items()):
      std = statistics.stdev(vals) if len(vals) > 1 else 0.0
      print(f'{task.replace("dmc_", ""):<26} {arm:<26} {len(vals):>2} '
            f'{statistics.mean(vals):>8.1f} {std:>8.1f} '
            f'{min(vals):>8.1f} {max(vals):>8.1f}')


if __name__ == '__main__':
  main()
