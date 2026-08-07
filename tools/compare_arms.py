#!/usr/bin/env python3
"""Side-by-side comparison of arms, every number a moving average.

Reuses watch_arm_checkpoints' windowed accessor so the numbers here and in the
monitor are computed the same way. Reports at a common step (the furthest point
every arm has reached) so the columns are comparable, and again at each arm's
own latest step.

  tools/compare_arms.py e486 e487 e488 e489 e490 e494 --ref e481
"""
import argparse
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    'watch', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'watch_arm_checkpoints.py'))
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)

COLS = [
    ('score', 'episode/score'),
    ('codes', 'train/goal/perplexity'),
    ('used', 'train/goal/used_frac'),
    ('wkr', 'train/wkr_goal_rew'),
    ('mgr_rew', 'train/mgr_extr_rew'),
    ('geom', 'train/goal/struct_corr_code'),
]
FMT = {'score': '{:>8.1f}', 'codes': '{:>7.2f}', 'used': '{:>7.3f}',
       'wkr': '{:>7.3f}', 'mgr_rew': '{:>9.4f}', 'geom': '{:>7.3f}'}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('tags', nargs='+')
  ap.add_argument('--ref', default=None, help='reference arm, listed last')
  ap.add_argument('--window', type=int, default=100)
  ap.add_argument('--labels', default='', help='comma-separated, matching tags')
  args = ap.parse_args()

  tags = list(args.tags) + ([args.ref] if args.ref else [])
  labels = dict(zip(args.tags, args.labels.split(','))) if args.labels else {}

  data = {}
  for t in tags:
    p = W.find_metrics(t)
    if not p:
      print(f'{t}: no metrics'); continue
    rows = W.load(p)
    if rows:
      data[t] = (rows, max(r['step'] for r in rows))

  if not data:
    return
  common = min(s for _, s in data.values())

  def table(title, step_for):
    print(f'\n{title}')
    print(f'  {"arm":<26}{"step":>7}' + ''.join(f'{n:>{len(FMT[n].split(":")[1][:-2].lstrip(">"))+1}}'
          if False else f'{n:>9}' for n, _ in COLS))
    for t in tags:
      if t not in data:
        continue
      rows, mx = data[t]
      s = step_for(mx)
      cells = []
      for name, key in COLS:
        v = W.at_step(rows, key, s, args.window, stale=60_000)
        cells.append('        -' if v is None else FMT[name].format(v).rjust(9))
      lab = labels.get(t, '')
      print(f'  {t + " " + lab:<26}{s // 1000:>6}k' + ''.join(cells))

  print(f'All values are means over the last {args.window} logged rows '
        f'(~{args.window * 5}k steps).')
  table(f'AT A COMMON STEP ({common // 1000}k) -- columns comparable',
        lambda mx: common)
  table('AT EACH ARM\'S OWN LATEST STEP -- columns NOT comparable',
        lambda mx: mx)
  print('\ncodes = perplexity of the code distribution, out of 8.')


if __name__ == '__main__':
  main()
