"""Emotion signals around pushes and bumper hits (sim_phone runs' emotion.csv).

  python fig_emotrig.py OUT.png EXP [EXP ...]

z-scored over each run's last two thirds (after it learnt to stand), averaged
over events and runs, from 0.5 s before to 1 s after. 'salience' = |TD
valence| + pain, the event signal that replaced arousal.
"""

import csv
import glob
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOTS = ['/work/DoyaU/vasilache/work', '/bucket/DoyaU/vasilache/bucket/results/dreamerv3']
DT = 4 / 65   # one row every FEEL_EVERY steps
LAGS = np.arange(-8, 17)
SIG = [('arousal', '#000000'), ('salience', '#D55E00'), ('pain', '#E69F00'),
       ('td_valence', '#0072B2'), ('empowerment', '#009E73')]


def load(n):
  d = next(g for r in ROOTS for g in glob.glob(f'{r}/e{n}_*'))
  rows = list(csv.DictReader(open(d + '/logdir/emotion.csv')))
  rows = rows[len(rows) // 3:]
  X = {k: np.array([float(x[k] or 0) for x in rows]) for k in (
      'arousal', 'pain', 'empowerment', 'td', 'theta', 'push')}
  ms, tv = X['td'][0] ** 2, []
  for t in X['td']:
    ms = (1 - 1 / 162.5) * ms + t * t / 162.5
    tv.append(np.tanh(t / np.sqrt(ms + 1e-8)))
  X['td_valence'] = np.array(tv)
  X['salience'] = np.abs(X['td_valence']) + X['pain']
  th = np.degrees(X['theta'])
  push = X['push'] != 0
  ev = dict(
      push=[i for i in range(1, len(th)) if push[i] and not push[i - 1]],
      hit=[i for i in range(1, len(th)) if (th[i] > 6.5 >= th[i - 1]) or (th[i] < -5 <= th[i - 1])])
  return X, ev


def main():
  out, exps = sys.argv[1], sys.argv[2:]
  fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
  data = [load(n) for n in exps]
  for ax, kind in zip(axes, ('push', 'hit')):
    for key, color in SIG:
      curves = []
      for X, ev in data:
        z = (X[key] - X[key].mean()) / (X[key].std() + 1e-9)
        idx = [i for i in ev[kind] if 8 <= i < len(z) - 17]
        curves.append(np.mean([z[i + LAGS] for i in idx], 0))
      ax.plot(LAGS * DT, np.mean(curves, 0), color=color, label=key.replace('_', ' '), lw=1.6)
    ax.axvline(0, color='#999999', lw=0.8, ls='--')
    ax.set_ylabel(f'z, around a {"push" if kind == "push" else "bumper hit"}')
    ax.spines[['top', 'right']].set_visible(False)
  axes[0].legend(frameon=False, fontsize=8, ncol=3)
  axes[1].set_xlabel(f's from the event (n={len(exps)} runs)')
  plt.tight_layout()
  plt.savefig(out, dpi=130)


if __name__ == '__main__':
  main()
