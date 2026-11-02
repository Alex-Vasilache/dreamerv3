"""Round 6t: balance policies, mixed floors vs single floor. Wobble (tilt-rate SD), falls and worst-variant wobble
over the last 5 checkpoints, mean ± std over seeds.  python fig_bal6t.py"""
import glob, json, re, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})
EV = '/work/DoyaU/vasilache/work/sysid/r6/ev'
ARMS = {'c': [('mixed', '#0072B2', [f'e{n}' for n in range(9670, 9674)]), ('carpet only', '#E69F00', ['e9674', 'e9675'])],
        't': [('mixed', '#0072B2', [f'e{n}' for n in range(9670, 9674)]), ('table only', '#009E73', ['e9676', 'e9677'])]}


def run(tag, exp):
  pts = {}
  for f in glob.glob(f'{EV}/{tag}/{exp}@*_*.json'):
    pts.setdefault(int(re.search(r'@(\d+)_', f).group(1)), {}).update(json.load(open(f)))
  st = sorted(pts); fit = [pts[s]['fit'] for s in st]
  return dict(rate=np.mean([f['rate_std'] for f in fit]), falls=np.mean([f['dips_per_min'] for f in fit]),
              worst=np.mean([max(v['rate_std'] for v in pts[s].values()) for s in st]))


M = [('rate', 'wobble: tilt-rate SD (deg/s)'), ('falls', 'falls / min'), ('worst', 'worst-variant wobble (deg/s)')]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for row, (sfx, floor) in enumerate([('c', 'carpet'), ('t', 'table')]):
  for j, (k, ylab) in enumerate(M):
    ax = axes[row, j]
    for i, (lab, col, exps) in enumerate(ARMS[sfx]):
      v = [run(f'bal6t_{sfx}', e)[k] for e in exps]
      ax.bar(i, np.mean(v), 0.6, yerr=np.std(v), color=col, capsize=4)
      ax.scatter([i] * len(v), v, color='black', s=12, zorder=3)
    ax.set_xticks(range(len(ARMS[sfx])), [a[0] for a in ARMS[sfx]]); ax.set_ylabel(ylab); ax.set_title(f'{floor}')
    ax.set_ylim(bottom=0); ax.grid(axis='y', color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
plt.tight_layout()
for ext in ('png', 'pdf'): plt.savefig(f'figs/bal6t.{ext}', dpi=200, facecolor='white')
