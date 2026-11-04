"""Emotion over a learning run (sim runs with ckpt_milestones every 5 min): per minute, the means of
the face signals and arousal's parts, beside how well it balances.
  python tools/robot/dev/fig_emotion_learning.py OUT_PREFIX NAME=TRACE.npz [...]"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
plt.rcParams.update({'axes.titlesize': 15, 'axes.labelsize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})
OI = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#CC79A7', '#000000']
fig, ax = plt.subplots(2, 3, figsize=(22, 9))
P = [('upright', 'time upright (|tilt| < 3 deg)'), ('mood', 'mood (value, z)'), ('arousal', 'arousal'),
     ('empowerment', 'empowerment (nats)'), ('unexpected', 'unexpected surprise (nats)'), ('novelty', 'novelty, prior entropy (nats)')]
for i, arg in enumerate(sys.argv[2:]):
  name, path = arg.split('=', 1)
  d = dict(np.load(path))
  d['upright'] = (np.abs(np.degrees(d['theta'])) < 3).astype(float)
  t = (d['t'] - d['t'][0]) / 60 + 5      # traces start at the first milestone (5 min)
  m = np.floor(t).astype(int)
  for a, (k, lab) in zip(ax.flat, P):
    mins = np.unique(m)
    y = [d[k][m == u].mean() for u in mins]
    a.plot(mins + 0.5, y, 'o-', color=OI[i], lw=2, label=name)
    a.set_title(lab); a.set_xlabel('minute of training'); a.grid(color='#d5d5d2', lw=0.7)
for a in ax.flat:
  for s in (10, 15):
    a.axvline(s, color='#bbbbbb', lw=0.8, ls=':')
ax[0, 0].legend(loc='lower right')
plt.tight_layout()
for ext in ('png', 'pdf'):
  plt.savefig(f'{sys.argv[1]}.{ext}', dpi=200, facecolor='white')
