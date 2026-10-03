"""wm_norepgrad (e1080-e1094) vs the benchmark's director_og baselines (e919-e936).

Per run: sustained-50 crossing (pinpad only; first 50k bin from which the next
4 bins all average >= 50) and mean episode score over 1M windows.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wmfix_status import find

B = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
TASKS = [('pinpad_five', 1080), ('pinpad_six', 1083), ('dmc_cartpole_swingup', 1086),
         ('dmc_cheetah_run', 1089), ('dmc_hopper_hop', 1092)]
WIN = [(0, 1e6), (1e6, 2e6), (2e6, 3e6), (3e6, 4e6)]


def stats(d, pinpad):
  S = np.array([(r['step'], r['episode/score']) for r in map(json.loads, open(f'{d}/logdir/scores.jsonl'))])
  out = []
  if pinpad:
    bw = 5e4; nb = int(S[:, 0].max() // bw) + 1
    b = np.array([S[(S[:, 0] // bw) == i, 1].mean() if ((S[:, 0] // bw) == i).any() else np.nan for i in range(nb)])
    c = next((i * bw / 1e6 for i in range(nb - 3) if np.all(b[i:i + 4] >= 50)), None)
    out.append(f'{c:4.2f}M' if c is not None else ' never')
  for lo, hi in WIN:
    w = S[(S[:, 0] >= lo) & (S[:, 0] < hi), 1]
    out.append(f'{w.mean():5.0f}' if len(w) > 20 else '    -')
  return '  '.join(out) + f'   to {S[:, 0].max() / 1e6:.2f}M'


for task, n0 in TASKS:
  pin = task.startswith('pinpad')
  print(f'== {task}   ({"cross50  " if pin else ""}0-1M 1-2M 2-3M 3-4M)')
  for d in sorted(glob.glob(f'{B}/e9[1-3][0-9]_{task}_director-director_og_s*')):
    print(f'  base {os.path.basename(d)[:4]}  ' + stats(d, pin))
  for k in range(3):
    d = find(f'e{n0 + k}')
    if d:
      print(f'  fix  e{n0 + k}  ' + stats(d, pin))
