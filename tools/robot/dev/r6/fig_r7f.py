"""Round 7f: speed-cap ramps, slow leash, geofence; all with the brake. Round 7 (live balance from scratch, sim, 15 min): per minute, how far the robot roams (range of its position, m),
time upright (|tilt| < 3 deg) and time in full-drive runaways (net drive beyond 0.7 one way for >= 1 s). mean ± std over seeds.
  python fig_r7.py r7all.json"""
import json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11})
D = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}
M = 0.045 / 548 * 1000   # phone mm/s -> m/s of wheel rim, times the script's /1000
E = lambda a, b: list(range(a, b + 1))
ROWS = [('table', [('brake only', '#000000', '-', E(19269, 19271)),
                   ('cap 0.4 -> 1 over 5 min', '#D55E00', '-', E(19251, 19253)),
                   ('cap 0.4 -> 1 over 10 min', '#E69F00', '--', E(19254, 19256)),
                   ('cap 0.6 -> 1 over 5 min', '#0072B2', '-', E(19257, 19259)),
                   ('leash 0.3, 120 s leak', '#009E73', '-', E(19260, 19262)),
                   ('cap 0.4 + leash', '#CC79A7', '--', E(19263, 19265)),
                   ('geofence 0.5 m + leash', '#999999', ':', E(19266, 19268))]),
        ('carpet', [('brake only', '#000000', '-', E(19236, 19238)),
                    ('cap 0.4 -> 1 over 5 min', '#D55E00', '-', E(19272, 19274))])]
K = [('rng', 'roaming range per minute (m)', M), ('up', 'time upright', 1), ('run', 'time in runaways (%)', 1)]
fig, axes = plt.subplots(2, 3, figsize=(20, 9.5))
for r, (floor, arms) in enumerate(ROWS):
  for c, (k, lab, f) in enumerate(K):
    ax = axes[r, c]
    for name, col, ls, runs in arms:
      n = min(len(D[e]) for e in runs); x = np.arange(1, n + 1)
      y = np.array([[D[e][i][k] * f for i in range(n)] for e in runs])
      ax.plot(x, y.mean(0), color=col, ls=ls, lw=2, label=name); ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=col, alpha=0.12)
    ax.set_title(floor); ax.set_xlabel('minute of training'); ax.set_ylabel(lab); ax.grid(color='#d5d5d2', lw=0.7)
    if c == 0: ax.legend(loc='upper right')
plt.tight_layout()
for ext in ('png', 'pdf'): plt.savefig(f'/work/DoyaU/vasilache/work/sysid/r6/figs/r7f.{ext}', dpi=200, facecolor='white')
