#!/usr/bin/env python3
"""Report a run's health at fixed step checkpoints, with an explicit verdict.

Written for the manager-side arms (e478/e479/e482/e483), where the failure mode
is not a crash but a manager that never learns while every other part of the
agent looks fine. Each checkpoint prints the metrics that discriminate the known
failures and a PASS/FAIL per check, so "is this working" does not depend on
eyeballing a chart.

The checks encode what the earlier failures actually looked like:

  ent_live    mgr_ent_loss != 0. e478 ran 1.6M steps with it identically 0
              because ManagerRingHead did not advertise minent/maxent and the
              adapter's hasattr guard skipped the head without raising.
  ent_target  normalized entropy near manager_actent_target (0.5). e478 sat at
              0.98 (uniform); a healthy manager tracks the target.
  ent_ctrl    the adapter multiplier is not railed. e479 ran it to ~1e4x the
              healthy value fighting a smoothing floor that coincided with the
              target.
  mgr_reward  mgr_extr_rew above a floor and rising. This is the one that
              separates a broken manager from a merely slow one: e479 and e482
              both sat at ~1e-5, flat, for their whole lives, while working arms
              climb from ~1e-4 to ~1e-2 over the same span.
  codebook    used_frac / perplexity, the VQ collapse watch.

Usage:
  tools/watch_arm_checkpoints.py e483 --at 100000 200000 --ref e481
"""

import argparse
import glob
import json
import os
import sys
import time

# Reference values from e481 (som_lipvq_line, the closest working control) at
# the same steps. Used only to contextualize, never to gate.
REF = {
    100_000: {'score': 6.5, 'mgr_extr_rew': 2.56e-4},
    200_000: {'score': 11.0, 'mgr_extr_rew': 2.41e-3},
}

# A manager that has found nothing at all sits here. Both failed smoothing arms
# were at ~1e-5 for their entire runs; working arms pass 1e-4 well before 100k.
DEAD_MGR_REWARD = 5e-5

# The level alone does not separate them: e479 was at 5.02e-5 at 200k, just over
# the floor, and an earlier version of this script called it HEALTHY. What
# separates a manager that is learning from one that is not is GROWTH. Measured
# against the value at a quarter of the checkpoint step:
#   e481 (works)  100k: 42x    200k: 157x
#   e479 (broken) 100k: 0.5x   200k: 0.4x
# 3x is well clear of both.
MIN_MGR_REWARD_GROWTH = 3.0

# Entropy band. Wide enough that the controller's early transient does not trip
# it (e479 sat at 0.714 at 100k and still recovered), narrow enough to catch the
# failure it exists for: e478's unregularized head sat at 0.98.
ENT_TOLERANCE = 0.25


def find_metrics(tag):
  hits = sorted(glob.glob(f'/work/DoyaU/vasilache/work/{tag}_*/logdir/metrics.jsonl'))
  return hits[-1] if hits else None


def load(path):
  rows = []
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError:
        continue          # a partially written trailing row
  return rows


def at_step(rows, key, step, window=30_000):
  """Last value of `key` at or before `step`, within `window`."""
  vals = [(r['step'], r[key]) for r in rows if key in r and r['step'] <= step]
  if not vals:
    return None
  s, v = vals[-1]
  return v if step - s <= window else None


def mean_recent(rows, key, step, n=20):
  vals = [r[key] for r in rows if key in r and r['step'] <= step]
  return sum(vals[-n:]) / len(vals[-n:]) if vals else None


def report(tag, step, rows, blocks, classes, target):
  import math
  maxent = blocks * math.log(classes)
  g = lambda k: at_step(rows, k, step)
  ent = g('train/mgr_ent/skill')
  entn = g('train/mgr_ent_norm_skill_mean')
  if entn is None and ent is not None:
    entn = ent / maxent
  loss = g('train/mgr_ent_loss')
  mult = g('train/mgr_actent_skill_scale_mean')
  # Smoothed: mgr_extr_rew is noisy enough per-row that a single sample can
  # straddle the threshold either way.
  rew = mean_recent(rows, 'train/mgr_extr_rew', step, n=5)
  rew_early = mean_recent(
      rows, 'train/mgr_extr_rew', max(step // 4, 20_000), n=5)
  score = mean_recent(rows, 'episode/score', step)
  wkr = g('train/wkr_goal_rew')
  perp = g('train/goal/perplexity')
  used = g('train/goal/used_frac')
  corr = g('train/goal/struct_corr_code')

  checks = []
  checks.append(('ent_live', loss is not None and loss != 0.0,
                 f'mgr_ent_loss={loss}' if loss is not None else 'missing'))
  checks.append(('ent_target',
                 entn is not None and abs(entn - target) < ENT_TOLERANCE,
                 f'norm_ent={entn:.3f} vs target {target}' if entn is not None
                 else 'missing'))
  checks.append(('ent_ctrl', mult is not None and mult < 10.0,
                 f'multiplier={mult:.3g}' if mult is not None else 'missing'))
  rew_half = mean_recent(rows, 'train/mgr_extr_rew', step // 2, n=5)
  if rew is None or rew_early is None:
    checks.append(('mgr_reward', False, 'missing'))
  else:
    growth = rew / rew_early if rew_early > 0 else float('inf')
    checks.append((
        'mgr_reward',
        rew > DEAD_MGR_REWARD and growth >= MIN_MGR_REWARD_GROWTH,
        f'mgr_extr_rew={rew:.3g} (was {rew_early:.3g}, {growth:.1f}x; '
        f'need >{DEAD_MGR_REWARD:.0e} and >{MIN_MGR_REWARD_GROWTH:.0f}x)'))
    # Growth measured from step/4 is flattered by a low starting point: e484
    # scored 382x at 200k while its last 50k were flat (1.003x). Check the most
    # recent half separately -- e480, the slow starter that later took off, grew
    # 1.3x over that window at 200k and 6x by 300k.
    if rew_half is not None and rew_half > 0:
      recent = rew / rew_half
      checks.append((
          'mgr_not_stalled', recent >= 1.25,
          f'{recent:.2f}x over the last half ({rew_half:.3g} -> {rew:.3g})'))

  # The entropy multiplier relative to the working controls. A large value means
  # the entropy term is competing with REINFORCE for the manager's gradient:
  # e479 sat at 1.46 here at 200k and railed at 100 by 500k, against 0.0005 to
  # 0.003 for the arms that learned.
  if mult is not None:
    checks.append((
        'ent_ctrl_quiet', mult < 0.5,
        f'multiplier={mult:.3g} (controls run 5e-4 to 3e-3)'))
  # Perplexity threshold set from the arms that learn, not from total collapse.
  # At 100k: 6.4 to 6.5 with the straight-through estimator, 5.6 for pure
  # SOM-VAE plus Lipschitz, 3.9 for pure SOM-VAE at alpha=0.25, 1.8 at alpha=1.
  # An earlier threshold of 2.0 only caught the last of these and called a
  # half-dead codebook healthy.
  # A fixed bar alone is the wrong test for the no-estimator arms, which sit at
  # 4-5.5 and climb slowly rather than reaching the 6.4-7.6 of the estimator
  # arms. Pass on EITHER a healthy level or a codebook still filling up; fail
  # only when it is both low and no longer moving, which is what the ring
  # collapse looked like (1.66, 1.49, 1.60, 1.81 over 25k-100k, then flat).
  perp_half = at_step(rows, 'train/goal/perplexity', step // 2)
  if perp is None or used is None:
    checks.append(('codebook', False, 'missing'))
  else:
    rising = perp_half is not None and perp > 1.05 * perp_half
    detail = f'used_frac={used:.3f}, perplexity={perp:.3g}'
    if perp_half is not None:
      detail += f' ({perp_half:.3g} at {step // 2000}k, {perp / perp_half:.2f}x)'
    checks.append(('codebook', (used > 0.9 and perp > 5.5) or rising,
                   detail + '  [want >5.5, or still rising]'))

  out = [f'=== {tag} @ {step // 1000}k steps ===']
  for name, ok, detail in checks:
    out.append(f'  [{"PASS" if ok else "FAIL"}] {name:<11} {detail}')
  ref = REF.get(step, {})
  out.append(f'  score={score:.1f} (ref e481 {ref.get("score", float("nan")):.1f})'
             if score is not None else '  score=missing')
  out.append(f'  wkr_goal_rew={wkr:.3g}  struct_corr_code='
             f'{corr if corr is None else round(corr, 3)}')
  failed = [n for n, ok, _ in checks if not ok]
  out.append(f'  VERDICT: {"HEALTHY" if not failed else "BROKEN -> " + ",".join(failed)}')
  return '\n'.join(out), not failed


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('tag')
  ap.add_argument('--at', type=int, nargs='+', default=[100_000, 200_000])
  ap.add_argument('--blocks', type=int, default=8)
  ap.add_argument('--classes', type=int, default=8)
  ap.add_argument('--target', type=float, default=0.5)
  ap.add_argument('--poll', type=int, default=180)
  ap.add_argument('--once', action='store_true')
  args = ap.parse_args()

  pending = sorted(args.at)
  while pending:
    path = find_metrics(args.tag)
    if not path:
      if args.once:
        print(f'{args.tag}: no metrics yet', flush=True)
        return
      time.sleep(args.poll)
      continue
    rows = load(path)
    cur = max((r['step'] for r in rows), default=0)
    while pending and cur >= pending[0]:
      step = pending.pop(0)
      text, _ = report(args.tag, step, rows, args.blocks, args.classes, args.target)
      print(text, flush=True)
    if args.once:
      if pending:
        print(f'{args.tag}: at {cur // 1000}k, next checkpoint '
              f'{pending[0] // 1000}k', flush=True)
      return
    if pending:
      time.sleep(args.poll)


if __name__ == '__main__':
  main()
