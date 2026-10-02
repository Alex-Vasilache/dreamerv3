"""Per-run summary for the wm-fix study: sustained-50 crossing + window means.

Sustained-50 crossing = first 50k bin from which the next 4 bins (200k steps)
all average >= 50. Window means over 0.5-1.0M, 1.0-1.5M, 1.5-2.0M (and to 4M if run).
"""
import sys, json, numpy as np
sys.path.insert(0, 'tools/collapse_study')
from wmfix_status import find

ARMS = [('control (old code)', ['e1057', 'e1058', 'e1059']),
        ('wm_realrew', ['e1068', 'e1069', 'e1070']),
        ('wm_norepgrad', ['e1071', 'e1072', 'e1073']),
        ('wm_b64 wm_oldrew', ['e1074', 'e1075', 'e1076']),
        ('wm_norepgrad wm_b64', ['e1077', 'e1078', 'e1079'])]
WIN = [(0.5e6, 1e6), (1e6, 1.5e6), (1.5e6, 2e6), (2e6, 3e6), (3e6, 4e6)]

def row(e):
  d = find(e)
  S = np.array([(json.loads(l)['step'], json.loads(l)['episode/score']) for l in open(f'{d}/logdir/scores.jsonl')])
  bw = 5e4
  nb = int(S[:, 0].max() // bw) + 1
  b = np.array([S[(S[:, 0] // bw) == i, 1].mean() if ((S[:, 0] // bw) == i).any() else np.nan for i in range(nb)])
  cross = next((i * bw for i in range(nb - 3) if np.all(b[i:i + 4] >= 50)), None)
  wins = [S[(S[:, 0] >= lo) & (S[:, 0] < hi), 1] for lo, hi in WIN]
  wins = [f'{w.mean():5.0f}' if len(w) > 20 else '    -' for w in wins]
  return f'{e}  cross50 {cross / 1e6 if cross is not None else float("nan"):4.2f}M  ' + ' '.join(wins) + f'   max_step {S[:, 0].max() / 1e6:.2f}M'

print('window means: 0.5-1M 1-1.5M 1.5-2M 2-3M 3-4M')
for name, exps in ARMS:
  print(name)
  for e in exps:
    print('  ' + row(e))
