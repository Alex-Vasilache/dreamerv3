"""Live invariant check for a duration-pinned block-pooled run (e350/e351 class).

With ``goal_duration_fixed=K`` the variable-K rollout must switch on exactly the
fixed-K grid, so a handful of quantities in ``metrics.jsonl`` are not merely
"reasonable" but *determined*. Checking them on a live run catches a
configuration or plumbing regression in minutes instead of after a 4M-step run.

Usage:
  python tools/check_pinned_director_invariants.py <logdir> [--imag-length 16] [--hold 8]

Runs under any Python 3 (stdlib only) -- no need for the training env.
"""

import argparse
import json
import os
import sys


def last_values(path, keys):
  out = {k: None for k in keys}
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError:
        continue
      for k in keys:
        if k in row:
          out[k] = row[k]
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('logdir')
  ap.add_argument('--imag-length', type=int, default=16)
  ap.add_argument('--hold', type=int, default=8)
  args = ap.parse_args()

  path = os.path.join(args.logdir, 'metrics.jsonl')
  if not os.path.exists(path):
    print(f'no metrics yet at {path}')
    return 2

  # Decisions owning realized transitions in an H-step rollout at a fixed hold:
  # switches at 0, hold, 2*hold, ... and the one landing on step H owns nothing.
  expected_decisions = args.imag_length // args.hold

  keys = [
      'train/mgr_valid_decisions',
      'train/goal/mgr_duration_mean',
      'train/goal/mgr_duration_std',
      'train/mgr_extr_adv',
      'train/loss/mgr_policy',
      'train/wkr_goal_rew',
      'train/mgr_duration_reg_loss',
      'train/mgr_ent/duration',
  ]
  vals = last_values(path, keys)

  checks = []
  checks.append((
      'valid decisions per rollout',
      vals['train/mgr_valid_decisions'],
      lambda v: v is not None and abs(v - expected_decisions) < 1e-6,
      f'== {expected_decisions} (padded-axis rescaling active)'))
  checks.append((
      'mean hold length',
      vals['train/goal/mgr_duration_mean'],
      lambda v: v is not None and abs(v - args.hold) < 1e-3,
      f'== {args.hold} (duration head bypassed)'))
  checks.append((
      'hold length spread',
      vals['train/goal/mgr_duration_std'],
      lambda v: v is not None and abs(v) < 1e-3,
      '== 0 (every hold identical)'))
  checks.append((
      'duration prior loss',
      vals['train/mgr_duration_reg_loss'],
      lambda v: v is None,
      'absent (prior disabled under a pinned hold)'))

  ok = True
  for name, value, pred, want in checks:
    good = pred(value)
    ok = ok and good
    print(f'[{"ok " if good else "FAIL"}] {name}: {value!r}  (want {want})')

  print('\ncontext (not asserted):')
  for k in ('train/mgr_extr_adv', 'train/loss/mgr_policy', 'train/wkr_goal_rew'):
    print(f'  {k}: {vals[k]!r}')
  return 0 if ok else 1


if __name__ == '__main__':
  sys.exit(main())
