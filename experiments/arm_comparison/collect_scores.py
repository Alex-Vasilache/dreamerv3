#!/usr/bin/env python3
"""Per-seed score curves and endpoint statistics for the e502-e549 comparison.

Reads `scores.jsonl` from every run directory of every arm, bins the episode
returns onto a common step grid so seeds and arms are directly comparable, and
writes one JSON that the figure script and the paper text both read.

Standard library only: this runs on the login node, where neither JAX nor
matplotlib will load.

Statistics, and their limits. Four seeds against four is the design, so the
exact two-sided permutation test over all C(8,4)=70 splits cannot return a
p below 2/70 = 0.0286, and reaches it only when the two groups do not overlap
at all. p is therefore reported next to the effect size and the per-arm seed
spread, never on its own -- with n=4 the spread is the more honest summary, and
a "significant" label at this sample size would be an overclaim.

  python3 collect_scores.py --out results/arms.json
  python3 collect_scores.py --at-step 2000000 --out results/arms_2M.json
"""
import argparse
import glob
import itertools
import json
import os
import statistics
import sys

WORK = '/work/DoyaU/vasilache/work'
BUCKET = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'

# Display order and labels. Director first: it is the reference every other row
# is tested against.
ARMS = [
    ('director', 'Director'),
    ('som_line', 'SOM-line'),
    ('som_orig_line', 'SOM-line-OG'),
    ('lipvq_prod', 'LiP'),
    ('som_lipvq_line_prod', 'SOM-line + LiP'),
    ('som_orig_lipvq_line_prod', 'SOM-line-OG + LiP'),
]
TASKS = [
    ('dmc_cartpole_swingup', 'Cartpole Swingup'),
    ('dmc_hopper_stand', 'Hopper Stand'),
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


def read_curve(run_dir):
  """(steps, scores) for every episode the run logged, in order."""
  path = os.path.join(run_dir, 'logdir', 'scores.jsonl')
  steps, scores = [], []
  if not os.path.exists(path):
    return steps, scores
  with open(path) as f:
    for line in f:
      try:
        row = json.loads(line)
      except Exception:
        continue
      if 'episode/score' in row:
        steps.append(int(row.get('step', 0)))
        scores.append(float(row['episode/score']))
  return steps, scores


def bin_curve(steps, scores, grid):
  """Mean episode return in each bin of `grid`, forward-filled.

  Episodes land at irregular steps and at different rates per arm, so a common
  grid is the only way to average across seeds without one fast run dominating
  the mean at a given x.
  """
  out, j, last = [], 0, float('nan')
  for i, edge in enumerate(grid):
    lo = grid[i - 1] if i else 0
    vals = []
    while j < len(steps) and steps[j] <= edge:
      if steps[j] > lo:
        vals.append(scores[j])
      j += 1
    if vals:
      last = statistics.mean(vals)
    out.append(last)
  return out


def endpoint(steps, scores, at_step, last_n):
  """Mean of the last `last_n` episodes completed at or before `at_step`."""
  vals = [s for st, s in zip(steps, scores) if at_step is None or st <= at_step]
  if not vals:
    return None
  return statistics.mean(vals[-last_n:])


def permutation_p(a, b, n_max=200000):
  """Exact two-sided permutation p on the difference of means.

  Exact whenever C(n+m, n) is small, which at 4 vs 4 it always is (70).
  """
  a, b = list(a), list(b)
  obs = abs(statistics.mean(a) - statistics.mean(b))
  pool = a + b
  n = len(a)
  combos = list(itertools.combinations(range(len(pool)), n))
  if len(combos) > n_max:
    return None, len(combos)
  hits = 0
  for idx in combos:
    left = [pool[i] for i in idx]
    right = [pool[i] for i in range(len(pool)) if i not in idx]
    if abs(statistics.mean(left) - statistics.mean(right)) >= obs - 1e-12:
      hits += 1
  return hits / len(combos), len(combos)


def collect(roots, at_step, last_n, n_bins, min_frac=0.99):
  """Per-run curves and endpoints.

  `min_frac` is what keeps this honest. `endpoint()` takes the last `last_n`
  episodes recorded at or before `at_step`, which for a run that has only
  reached 700k of a 4M target silently returns its 700k score and lets it into
  the cell mean as though it were a finished seed. Reading the batch at 4M
  while round 3 was two hours old produced exactly that: hopper `som_line` came
  out at 573.7 +/- 436 from two finished seeds at ~810 and one that had barely
  started. Runs short of `min_frac * at_step` are therefore dropped and counted,
  never averaged in.
  """
  grid_max = at_step or 4_000_000
  grid = [int(grid_max * (i + 1) / n_bins) for i in range(n_bins)]
  runs = []
  for root in roots:
    for d in sorted(glob.glob(os.path.join(root, 'e5*_BIG_j*'))):
      if not os.path.isdir(d):
        continue
      env = read_job_env(d)
      arm, task = env.get('ARM'), env.get('TASK')
      if arm is None or task is None:
        continue
      if task not in dict(TASKS) or arm not in dict(ARMS):
        continue
      steps, scores = read_curve(d)
      if not steps:
        continue
      runs.append(dict(
          exp=env.get('EXP_TAG', '?'), arm=arm, task=task,
          seed=int(env.get('SEED', -1)), dir=d, max_step=max(steps),
          n_eps=len(steps),
          curve=bin_curve(steps, scores, grid),
          final=endpoint(steps, scores, at_step, last_n)))

  # One run per (task, arm, seed): a relaunch can leave an older, shorter
  # directory behind, and the furthest one is the real run.
  best = {}
  for r in runs:
    key = (r['task'], r['arm'], r['seed'])
    if key not in best or r['max_step'] > best[key]['max_step']:
      best[key] = r
  runs = sorted(best.values(), key=lambda r: (r['task'], r['arm'], r['seed']))

  incomplete = []
  if at_step:
    need = at_step * min_frac
    incomplete = [r for r in runs if r['max_step'] < need]
    runs = [r for r in runs if r['max_step'] >= need]

  cells = {}
  for (task, _), (arm, _) in itertools.product(TASKS, ARMS):
    sel = [r for r in runs if r['task'] == task and r['arm'] == arm]
    vals = [r['final'] for r in sel if r['final'] is not None]
    if not vals:
      continue
    cells[f'{task}|{arm}'] = dict(
        task=task, arm=arm, n=len(vals), seeds=[r['seed'] for r in sel],
        values=vals, mean=statistics.mean(vals),
        std=statistics.stdev(vals) if len(vals) > 1 else 0.0,
        spread=max(vals) - min(vals), min=min(vals), max=max(vals),
        max_step=min(r['max_step'] for r in sel),
        curves=[r['curve'] for r in sel])

  for (task, _) in TASKS:
    ref = cells.get(f'{task}|director')
    if not ref:
      continue
    for (arm, _) in ARMS:
      cell = cells.get(f'{task}|{arm}')
      if not cell or arm == 'director':
        continue
      p, n_perm = permutation_p(ref['values'], cell['values'])
      cell['p_vs_director'] = p
      cell['n_permutations'] = n_perm
      cell['p_floor'] = 2.0 / n_perm if n_perm else None
      cell['delta_vs_director'] = cell['mean'] - ref['mean']

  return dict(grid=grid, at_step=at_step, last_n=last_n, min_frac=min_frac,
              arms=ARMS, tasks=TASKS, cells=cells,
              incomplete=[{k: v for k, v in r.items() if k != 'curve'}
                          for r in incomplete],
              runs=[{k: v for k, v in r.items() if k != 'curve'}
                    for r in runs])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--roots', default=f'{WORK},{BUCKET}')
  ap.add_argument('--at-step', type=int, default=None,
                  help='evaluate every arm at this common step (default: end)')
  ap.add_argument('--last-n', type=int, default=15)
  ap.add_argument('--n-bins', type=int, default=80)
  ap.add_argument('--min-frac', type=float, default=0.99,
                  help='drop runs short of this fraction of --at-step')
  ap.add_argument('--out', default=None)
  a = ap.parse_args()

  roots = [r for r in a.roots.split(',') if os.path.isdir(r)]
  data = collect(roots, a.at_step, a.last_n, a.n_bins, a.min_frac)
  if data['incomplete']:
    print(f'excluded {len(data["incomplete"])} run(s) short of '
          f'{a.min_frac:.0%} of {a.at_step}: ' + ', '.join(
              f'{r["exp"]}({r["max_step"]//1000}k)'
              for r in data['incomplete']))

  for (task, tlabel) in TASKS:
    ref = data['cells'].get(f'{task}|director')
    print(f'\n{tlabel}  (at step {a.at_step or "end"}, '
          f'last-{a.last_n} episode mean)')
    print(f'  {"arm":<22} {"n":>2} {"mean":>7} {"std":>6} {"spread":>7} '
          f'{"delta":>7} {"p":>7}')
    for (arm, alabel) in data['arms']:
      c = data['cells'].get(f'{task}|{arm}')
      if not c:
        print(f'  {alabel:<22} --')
        continue
      d = c.get('delta_vs_director')
      p = c.get('p_vs_director')
      print(f'  {alabel:<22} {c["n"]:>2} {c["mean"]:>7.1f} {c["std"]:>6.1f} '
            f'{c["spread"]:>7.1f} '
            f'{("%+.1f" % d) if d is not None else "ref":>7} '
            f'{("%.3f" % p) if p is not None else "-":>7}')
    if ref and ref.get('n', 0) > 1:
      floor = 2.0 / 70
      print(f'  (n=4 vs 4: smallest attainable two-sided p is {floor:.3f})')

  if a.out:
    path = a.out
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
      json.dump(data, f)
    print(f'\nwrote {path}')


if __name__ == '__main__':
  sys.exit(main())
