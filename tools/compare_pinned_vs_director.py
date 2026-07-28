"""Same-step comparison of the duration-pinned control against its references.

Three-way readout for the 2026-07-27 block-pooled Director-equivalence question:

  e351 (pinned, post-fix)  vs  e352 (pure Director, SAME code)   -> did the fixes work?
  e351 (pinned, post-fix)  vs  e343 (pinned, PRE-fix, archived)  -> did anything change?

Comparing at matched env steps matters: the manager's return normalizer warms up
from zero over the first ~20k steps, so "latest row vs latest row" between runs
at different speeds reads as a large spurious gap.

Usage:
  python tools/compare_pinned_vs_director.py [--steps 100000 200000 350000]
"""

import argparse
import io
import json
import os

WORK = '/work/DoyaU/vasilache/work'
BUCKET = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'

RUNS = [
    ('e351 cheetah pinned  (post-fix)', WORK, 'e351_dmc_cheetah_run_plain_vark_BIG_j4671355'),
    ('e352 cheetah Director (same code)', WORK, 'e352_dmc_cheetah_run_director_BIG_j4671356'),
    ('e343 cheetah pinned  (PRE-fix)', BUCKET, 'e343_dmc_cheetah_run_plain_vark_BIG_j4671226'),
    ('e350 hopper  pinned  (post-fix)', WORK, 'e350_dmc_hopper_hop_plain_vark_BIG_j4671354'),
    ('e353 hopper  Director (same code)', WORK, 'e353_dmc_hopper_hop_director_BIG_j4671357'),
    ('e342 hopper  pinned  (PRE-fix)', BUCKET, 'e342_dmc_hopper_hop_plain_vark_BIG_j4671225'),
]

KEYS = [
    ('score', 'episode/score'),
    ('rew_rate', 'epstats/reward_rate'),
    ('mgr_adv', 'train/mgr_extr_adv'),
    ('mgr_pol', 'train/loss/mgr_policy'),
    ('wkr_rew', 'train/wkr_goal_rew'),
]


def load(root, name):
  path = os.path.join(root, name, 'logdir', 'metrics.jsonl')
  rows = []
  if not os.path.exists(path):
    return rows
  for line in io.open(path):
    line = line.strip()
    if not line:
      continue
    try:
      rows.append(json.loads(line))
    except ValueError:
      pass
  return rows


def at_step(rows, target):
  """Forward-filled snapshot at the logged step closest to ``target``."""
  carry, best, bd = {}, None, None
  for r in rows:
    carry.update(r)
    if 'step' in r:
      d = abs(r['step'] - target)
      if bd is None or d < bd:
        bd, best = d, dict(carry)
  return best


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--steps', type=int, nargs='+',
                  default=[100000, 200000, 350000])
  args = ap.parse_args()

  data = [(label, load(root, name)) for label, root, name in RUNS]
  for label, rows in data:
    steps = [r['step'] for r in rows if 'step' in r]
    print('%-36s rows=%-5d last_step=%s' % (
        label, len(rows), max(steps) if steps else 'none'))
  print()

  for target in args.steps:
    print('=== near step %d ===' % target)
    for label, rows in data:
      snap = at_step(rows, target)
      if not snap:
        print('  %-36s no data' % label)
        continue
      got = snap.get('step', -1)
      flag = '' if abs(got - target) < 0.15 * max(target, 1) else '  (far)'
      cells = '  '.join(
          '%s=%.4g' % (short, snap[k]) for short, k in KEYS if k in snap)
      print('  %-36s step=%-8g %s%s' % (label, got, cells, flag))
    print()


if __name__ == '__main__':
  main()
