#!/usr/bin/env python3
"""Endpoint scores for the discretized-Gaussian manager against its baselines.

The Gaussian arm is e592-e599: SOM-line+LiP with a ReLU trunk and a discretized
Gaussian manager head, no epsilon-greedy jump, 4 tasks x 2 seeds at 4M steps.
Its two references are the same ones the five-benchmark table uses -- Director
and the categorical SOM-line+LiP arm -- both at 4 seeds.

Endpoint and statistics follow arm_comparison/collect_scores.py exactly: the
mean of the last 15 episodes recorded at or before 4M, and the exact two-sided
permutation test on the difference of means. The sample sizes differ from that
table, though, and the difference matters: 2 seeds against 4 gives C(6,2) = 15
splits, so the smallest attainable two-sided p is 1/15 = 0.067. Note that this
floor is 1/C and not the 2/C of the five-benchmark table: that one compares 4
against 4, where every split is matched by its complement and hits therefore
come in pairs, which does not happen once the two groups have different sizes.
No comparison in this file can be significant at any conventional level, and
the per-seed values are printed so the spread can be read directly.

Standard library only, so it runs on the login node.

  python3 gaussian_scores.py --out results/gaussian_4M.json
"""
import argparse
import glob
import itertools
import json
import os
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
WORK = os.environ.get('WORK_DIR', '/work/DoyaU/vasilache/work')
BUCKET = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'

# (task, label, director exps, som-line+LiP exps, gaussian exps)
TASKS = [
    ('dmc_cartpole_swingup', 'Cartpole Swingup',
     range(502, 506), range(534, 538), (592, 596)),
    ('dmc_hopper_stand', 'Hopper Stand',
     range(506, 510), range(538, 542), (593, 597)),
    ('dmc_cheetah_run', 'Cheetah Run',
     range(554, 558), range(562, 566), (594, 598)),
    ('dmc_hopper_hop', 'Hopper Hop',
     range(566, 570), range(570, 574), (595, 599)),
]
ARMS = ['director', 'som_lipvq_line_prod', 'gaussian']
LABELS = {'director': 'Director', 'som_lipvq_line_prod': 'SOM-line + LiP',
          'gaussian': '+ Gaussian head'}
AT_STEP = 4_000_000
LAST_N = 15
MIN_FRAC = 0.99


def find_run(exp, task, roots):
  """The furthest run directory for one experiment number."""
  best, best_step = None, -1
  for root in roots:
    for d in sorted(glob.glob(os.path.join(root, f'e{exp}_{task}_*_BIG_j*'))):
      steps, _ = read_curve(d)
      if steps and max(steps) > best_step:
        best, best_step = d, max(steps)
  return best, best_step


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


def endpoint(steps, scores, at_step, last_n):
  vals = [s for st, s in zip(steps, scores) if st <= at_step]
  return statistics.mean(vals[-last_n:]) if vals else None


def permutation_p(a, b):
  """Exact two-sided permutation p on the difference of means.

  Returns (p, n_splits, floor). The floor is 1/n_splits for unequal group
  sizes and 2/n_splits for equal ones, where each split and its complement
  give the same |difference| and so always hit together.
  """
  a, b = list(a), list(b)
  obs = abs(statistics.mean(a) - statistics.mean(b))
  pool = a + b
  combos = list(itertools.combinations(range(len(pool)), len(a)))
  hits = 0
  for idx in combos:
    left = [pool[i] for i in idx]
    right = [pool[i] for i in range(len(pool)) if i not in idx]
    if abs(statistics.mean(left) - statistics.mean(right)) >= obs - 1e-12:
      hits += 1
  floor = (2.0 if len(a) == len(b) else 1.0) / len(combos)
  return hits / len(combos), len(combos), floor


def stratified_p(ref_by_task, arm_by_task):
  """Exact stratified permutation p on the aggregate normalized difference.

  Seeds are only exchangeable within a benchmark, so each benchmark is
  permuted independently and the aggregate difference is recomputed from the
  four permuted benchmarks together. With 4-vs-2 seeds in each of four
  benchmarks that is 15^4 = 50,625 assignments, enumerated exactly.
  """
  obs = abs(statistics.mean(
      [statistics.mean(arm_by_task[i]) - statistics.mean(ref_by_task[i])
       for i in range(len(ref_by_task))]))
  per_task = []
  for ref, arm in zip(ref_by_task, arm_by_task):
    pool = list(ref) + list(arm)
    n = len(ref)
    per_task.append([
        statistics.mean([pool[i] for i in range(len(pool)) if i not in idx]) -
        statistics.mean([pool[i] for i in idx])
        for idx in itertools.combinations(range(len(pool)), n)])
  hits = total = 0
  for combo in itertools.product(*per_task):
    total += 1
    if abs(statistics.mean(combo)) >= obs - 1e-12:
      hits += 1
  return hits / total, total


def cell(exps, task, roots, dropped):
  vals, seeds = [], []
  for e in exps:
    d, max_step = find_run(e, task, roots)
    if not d:
      continue
    if max_step < AT_STEP * MIN_FRAC:
      dropped.append(f'e{e}({max_step // 1000}k)')
      continue
    steps, scores = read_curve(d)
    v = endpoint(steps, scores, AT_STEP, LAST_N)
    if v is not None:
      vals.append(v)
      seeds.append(e)
  if not vals:
    return None
  # Population std, which is what the paper's five-benchmark table reports:
  # its Director cartpole cell is 753.6 +/- 72.2, and the sample std of those
  # four seeds is 83.6. Mixing the two conventions across neighbouring tables
  # would make the arms look differently spread for no reason. At n = 2 this
  # is exactly half the range between the two seeds.
  return dict(n=len(vals), exps=seeds, values=vals, mean=statistics.mean(vals),
              std=statistics.pstdev(vals) if len(vals) > 1 else 0.0,
              std_sample=statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--roots', default=f'{WORK},{BUCKET}')
  ap.add_argument('--out', default=str(HERE / 'results' / 'gaussian_4M.json'))
  a = ap.parse_args()
  roots = [r for r in a.roots.split(',') if os.path.isdir(r)]

  dropped = []
  out = {}
  for task, label, dexp, mexp, gexp in TASKS:
    cells = dict(zip(ARMS, (cell(dexp, task, roots, dropped),
                            cell(mexp, task, roots, dropped),
                            cell(gexp, task, roots, dropped))))
    print(f'\n{label}   (last-{LAST_N} episode mean at {AT_STEP:,} steps)')
    print(f'  {"arm":<18} {"n":>2} {"mean":>7} {"std":>6}   per-seed')
    for arm in ARMS:
      c = cells[arm]
      if not c:
        print(f'  {LABELS[arm]:<18} --')
        continue
      per = ' '.join('%.0f' % v for v in sorted(c['values'], reverse=True))
      print(f'  {LABELS[arm]:<18} {c["n"]:>2} {c["mean"]:>7.1f} '
            f'{c["std"]:>6.1f}   {per}')
    # The Gaussian head is a change to the SOM-line+LiP arm, so that is the
    # reference; the Director delta is reported too since the paper's other
    # table uses it.
    g = cells['gaussian']
    for ref in ('som_lipvq_line_prod', 'director'):
      r = cells[ref]
      if g and r:
        p, n_perm, floor = permutation_p(r['values'], g['values'])
        d = g['mean'] - r['mean']
        cells.setdefault('stats', {})[ref] = dict(
            delta=d, pct=100 * d / r['mean'], p=p, n_permutations=n_perm,
            p_floor=floor, at_floor=abs(p - floor) < 1e-9)
        print(f'  vs {LABELS[ref]:<15} delta {d:+7.1f}  '
              f'({100 * d / r["mean"]:+.1f}%)  p={p:.3f}  '
              f'(floor {floor:.3f}, {n_perm} splits)')
    out[task] = dict(label=label, cells=cells)

  # Aggregate over the four benchmarks, on normalized return (score/1000), the
  # same normalization the five-benchmark table uses.
  print('\nAggregate (normalized, return/1000, mean over the four benchmarks)')
  agg = {}
  for arm in ARMS:
    per_task = [out[t]['cells'][arm]['mean'] / 1000
                for t, *_ in TASKS if out[t]['cells'][arm]]
    if len(per_task) == len(TASKS):
      agg[arm] = statistics.mean(per_task)
      print(f'  {LABELS[arm]:<18} {agg[arm]:.3f}')
  agg_stats = {}
  for ref in ('som_lipvq_line_prod', 'director'):
    if 'gaussian' in agg and ref in agg:
      d = agg['gaussian'] - agg[ref]
      ref_vals = [[v / 1000 for v in out[t]['cells'][ref]['values']]
                  for t, *_ in TASKS]
      arm_vals = [[v / 1000 for v in out[t]['cells']['gaussian']['values']]
                  for t, *_ in TASKS]
      p, n_perm = stratified_p(ref_vals, arm_vals)
      agg_stats[ref] = dict(delta=d, pct=100 * d / agg[ref], p=p,
                            n_permutations=n_perm)
      print(f'  vs {LABELS[ref]:<15} {d:+.3f}  ({100 * d / agg[ref]:+.1f}%)  '
            f'p={p:.3f}  ({n_perm:,} stratified assignments)')

  if dropped:
    print('\nexcluded (short of %d%% of %d steps): %s'
          % (MIN_FRAC * 100, AT_STEP, ', '.join(dropped)))

  path = pathlib.Path(a.out)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(dict(at_step=AT_STEP, last_n=LAST_N,
                                  tasks=out, aggregate=agg,
                                  aggregate_stats=agg_stats,
                                  dropped=dropped), indent=1))
  print(f'\nwrote {path}')


if __name__ == '__main__':
  sys.exit(main())
