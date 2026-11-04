"""Round 7e: position leash against the brake and control. Round 7 (live balance from scratch, sim, 15 min): per minute, how far the robot roams (range of its position, m),
time upright (|tilt| < 3 deg) and time in full-drive runaways (net drive beyond 0.7 one way for >= 1 s). mean ± std over seeds.
  python fig_r7.py r7all.json"""
import json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11})
D = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}
M = 0.045 / 548 * 1000   # phone mm/s -> m/s of wheel rim, times the script's /1000
E = lambda a, b: list(range(a, b + 1))
ROWS = [('table', [('CAPS spatial 0.3, no brake', '#000000', '-', E(19200, 19202) + E(19212, 19214)),
                   ('runaway brake', '#0072B2', '-', E(19233, 19235)),
                   ('brake + leash 0.3', '#009E73', '-', E(19239, 19241)),
                   ('brake + leash 1.0', '#D55E00', '--', E(19242, 19244))]),
        ('carpet', [('CAPS spatial 0.3, no brake', '#000000', '-', E(19206, 19208)),
                    ('runaway brake', '#0072B2', '-', E(19236, 19238)),
                    ('brake + leash 0.3', '#009E73', '-', E(19245, 19247)),
                    ('brake + leash 1.0', '#D55E00', '--', E(19248, 19250))])]
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
for ext in ('png', 'pdf'): plt.savefig(f'/work/DoyaU/vasilache/work/sysid/r6/figs/r7e.{ext}', dpi=200, facecolor='white')
