"""Round 6i: level and checkpoint-to-checkpoint swing of fitted-sim tracking per arm (mean ± std over 2 seeds).
  python fig_r6i.py"""
import glob, json, re, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 11, 'ytick.labelsize': 12, 'legend.fontsize': 12})
EV = '/work/DoyaU/vasilache/work/sysid/r6/ev'
ARMS = [('6d recipe\n(lr 3e-4)', '#999999', ['e8180', 'e8181']), ('actor\nlr 1e-4', '#56B4E9', ['e8182', 'e8183']),
        ('model + actor\nlr 1e-4', '#0072B2', ['e8184', 'e8185']), ('tilt lag\n0-15 ms', '#D55E00', ['e8186', 'e8187'])]


def run(tag, exp):
  pts = {}
  for f in glob.glob(f'{EV}/{tag}/{exp}@*_*.json'):
    pts.setdefault(int(re.search(r'@(\d+)_', f).group(1)), {}).update(json.load(open(f)))
  c = np.array([pts[s]['fit'].get('corr_f', np.nan) for s in sorted(pts)])
  mc = np.array([np.nanmin([v.get('corr_f', np.nan) for v in pts[s].values()]) for s in sorted(pts)])
  return dict(level=np.nanmean(c), swing=np.nanstd(c), minc=np.nanmean(mc))


M = [('level', 'tracking, mean over 5 ckpts'), ('swing', 'tracking SD over 5 ckpts'), ('minc', 'worst-variant tracking')]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for row, (floor, tag) in enumerate([('carpet', 'sel6i_c'), ('table', 'sel6i_t')]):
  for j, (k, ylab) in enumerate(M):
    ax = axes[row, j]
    for i, (lab, c, exps) in enumerate(ARMS):
      v = [run(tag, e)[k] for e in exps]
      ax.bar(i, np.mean(v), 0.7, yerr=np.std(v), color=c, capsize=3, label=lab.replace('\n', ' '))
    ax.set_xticks(range(len(ARMS)), [a[0] for a in ARMS]); ax.set_ylabel(ylab); ax.set_title(floor)
    ax.grid(axis='y', color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
plt.tight_layout()
for ext in ('png', 'pdf'): plt.savefig(f'figs/r6i_lr_lag.{ext}', dpi=200, facecolor='white')
