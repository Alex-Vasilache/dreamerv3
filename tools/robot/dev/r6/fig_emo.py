"""Emotion experiments E1/E2 (and E3): live balance recipe moved from carpet to the table at minute 6.
Per minute: time upright (|tilt| < 3 deg), bumper crossings (+6.5 / -5 deg), tilt-rate SD; mean +- std over seeds.
  python fig_emo.py OUT_PREFIX"""
import glob, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'axes.titlesize': 15, 'axes.labelsize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})
W = '/work/DoyaU/vasilache/work'
E = lambda a, b: list(range(a, b + 1))
ARMS = [('control', '#000000', '-', E(19340, 19342) + E(19373, 19376)),
        ('imagination, unexpected surprise w1', '#56B4E9', '-', E(19343, 19345)),
        ('imagination, raw KL w1', '#009E73', '--', E(19346, 19348)),
        ('imagination, unexpected surprise w3', '#0072B2', '-', E(19349, 19351) + E(19377, 19380)),
        ('real replay, model-loss priority .5', '#D55E00', '-', E(19352, 19354)),
        ('real replay, recency .3', '#E69F00', '--', E(19355, 19357)),
        ('real replay, priority + recency', '#CC79A7', ':', E(19358, 19360))]


def per_min(n):
  runs = glob.glob(f'{W}/e{n}_*')
  if not runs:
    return None
  L = [np.load(f) for f in sorted(glob.glob(runs[0] + '/logdir/online_shared/experience/*.npz'))]
  if not L:
    return None
  O = np.concatenate([d['orientation'] for d in L]); th = O[:, 0] * 0.18 * 57.3; rate = O[:, 1] * 3 * 57.3
  out = []
  for i in range(len(th) // 3900):
    t = th[i * 3900:(i + 1) * 3900]
    hits = np.sum((t[1:] > 6.5) & (t[:-1] <= 6.5)) + np.sum((t[1:] < -5) & (t[:-1] >= -5))
    out.append((np.mean(np.abs(t) < 3), hits, rate[i * 3900:(i + 1) * 3900].std()))
  return np.array(out)


fig, axes = plt.subplots(2, 3, figsize=(22, 9.5))
for row, keep in enumerate([lambda l: 'real' not in l, lambda l: 'imagination' not in l]):
  for c, (j, lab) in enumerate([(0, 'time upright'), (1, 'bumper crossings per minute'), (2, 'tilt-rate SD (deg/s)')]):
    ax = axes[row, c]
    for name, col, ls, runs in ARMS:
      if not keep(name):
        continue
      xs = [x for x in (per_min(n) for n in runs) if x is not None and len(x) >= 20]
      if not xs:
        continue
      y = np.stack([x[:20, j] for x in xs]); m = np.arange(1, 21)
      ax.plot(m, y.mean(0), color=col, ls=ls, lw=2, label=f'{name} (n={len(xs)})')
      ax.fill_between(m, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=col, alpha=0.12)
    ax.axvline(6.5, color='#999999', lw=1)
    ax.set_xlabel('minute of training (carpet, then the table from minute 7)'); ax.set_ylabel(lab)
    ax.set_title(('emotion in imagination' if row == 0 else 'emotion on real data') + f': {lab}')
    ax.grid(color='#d5d5d2', lw=0.7)
    if c == 0:
      ax.legend(loc='lower right')
plt.tight_layout()
for ext in ('png', 'pdf'):
  plt.savefig(f'{sys.argv[1]}.{ext}', dpi=200, facecolor='white')
