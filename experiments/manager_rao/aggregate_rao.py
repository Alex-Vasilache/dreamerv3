"""Collect diag_rao.py per-run JSON summaries into one comparison."""
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

RESULTS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                       str(pathlib.Path(__file__).resolve().parent / 'results'))

COLS = [
    ('ent_norm', 'entropy'),
    ('rao_self', 'Rao self'),
    ('rao_ceiling', 'Rao max@H'),
    ('rao_headroom', 'frac of max'),
    ('rao_cross', 'Rao cross'),
    ('overlap', 'overlap'),
    ('step_idx', '|idx step|'),
    ('changed_frac', 'blk changed'),
    ('rao_marginal', 'Rao marg'),
    ('r_idx_goal', 'r(idx,goal)'),
]

rows = []
for p in sorted(RESULTS.glob('*.json')):
  d = json.loads(p.read_text())
  arm, task, seed = p.stem.split('_')[0], p.stem.split('_')[1], p.stem.split('_')[-1]
  rows.append((arm, task, seed, d))

if not rows:
  print(f'no results in {RESULTS}')
  sys.exit(1)

print(f'{"run":<30}' + ''.join(f'{lab:>13}' for _, lab in COLS))
print('-' * (30 + 13 * len(COLS)))
for arm, task, seed, d in rows:
  name = f'{arm}/{task}/{seed}'
  print(f'{name:<30}' + ''.join(f'{d[k]:>13.3f}' for k, _ in COLS))

print()
print('=== per-arm mean over tasks and seeds ===')
by_arm = defaultdict(list)
for arm, task, seed, d in rows:
  by_arm[arm].append(d)
print(f'{"arm":<30}' + ''.join(f'{lab:>13}' for _, lab in COLS))
print('-' * (30 + 13 * len(COLS)))
for arm, ds in sorted(by_arm.items()):
  print(f'{arm + f" (n={len(ds)})":<30}' + ''.join(
      f'{np.mean([d[k] for d in ds]):>13.3f}' for k, _ in COLS))

print()
print('=== reference points (C=8 line, squared distance, normalized) ===')
print('  all mass on one class          Rao 0.000')
print('  three adjacent classes, equal  Rao 0.054')
print('  uniform over all 8             Rao 0.429')
print('  half at each end               Rao 1.000  (but entropy 0.333 < target 0.5)')
print()
print('=== per-block detail (first run of each arm) ===')
seen = set()
for arm, task, seed, d in rows:
  if arm in seen:
    continue
  seen.add(arm)
  print(f'\n{arm}/{task}/{seed}  per-block Rao / entropy:')
  for l, (r, e) in enumerate(zip(d['per_block_rao_self'], d['per_block_ent'])):
    marg = np.array(d['per_block_marginal'][l])
    bar = ''.join('.:-=+*#@'[min(int(v * 8 / max(marg.max(), 1e-9)), 7)]
                  for v in marg)
    print(f'  block {l}: Rao {r:.3f}  H {e:.3f}  marginal [{bar}]')
