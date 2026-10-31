"""Rounds 6l-6s: mixed-floor arms (smoothing, sensor-lag randomisation, stiction range), level / swing / falls over
the last 5 checkpoints, mean ± std over seeds.  python fig_r6arms.py"""
import glob, json, re, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 12, 'legend.fontsize': 12})
EV = '/work/DoyaU/vasilache/work/sysid/r6/ev'
ARMS = [('control', '#999999', [('6l', 'e8210'), ('6l', 'e8211')]),
        ('spatial .3\nnoise .3', '#56B4E9', [('6l', 'e8216'), ('6l', 'e8217'), ('6o', 'e8240'), ('6o', 'e8241'), ('6o', 'e8242')]),
        ('lag', '#E69F00', [('6m', 'e8220'), ('6m', 'e8221'), ('6m', 'e8222')]),
        ('lag + spatial\nnoise .3', '#009E73', [('6n', f'e{n}') for n in range(8230, 8234)]),
        ('lag 0-30 ms +\nspatial noise .3', '#0072B2', [('6o', 'e8243'), ('6o', 'e8244'), ('6o', 'e8245')]),
        ('lag + spatial\nnoise .5', '#D55E00', [('6o', 'e8246'), ('6o', 'e8247')] + [('6q', f'e{n}') for n in range(8260, 8264)] + [('6r', f'e{n}') for n in range(8270, 8274)]),
        ('lag + spatial .6\nnoise .5', '#D55E00', [('6r', 'e8274'), ('6r', 'e8275')], '//'),
        ('recipe +\ntilt penalty .4', '#D55E00', [('6s', f'e{n}') for n in range(8280, 8283)], 'xx'),
        ('recipe +\naction rate .25', '#D55E00', [('6s', f'e{n}') for n in range(8283, 8286)], '..'),
        ('recipe +\nsmooth .15', '#D55E00', [('6s', 'e8286'), ('6s', 'e8287')], '\\\\'),
        ('lag + spatial\nnoise 1.0', '#F0E442', [('6q', 'e8264'), ('6q', 'e8265')]),
        ('spatial noise .5\n(no lag)', '#000000', [('6q', 'e8266'), ('6q', 'e8267')]),
        ('stiction 5-12\n(spatial / lag)', '#CC79A7', [('6p', f'e{n}') for n in range(8250, 8256)])]


def run(tag, exp):
  pts = {}
  for f in glob.glob(f'{EV}/{tag}/{exp}@*_*.json'):
    pts.setdefault(int(re.search(r'@(\d+)_', f).group(1)), {}).update(json.load(open(f)))
  fit = [pts[s]['fit'] for s in sorted(pts)]
  c = np.array([f.get('corr_f', np.nan) for f in fit]); d = np.array([f.get('dips_per_min', np.nan) for f in fit])
  return dict(level=np.nanmean(c), swing=np.nanstd(c), falls=np.nanmean(d))


M = [('level', 'tracking, mean over 5 ckpts'), ('swing', 'tracking SD over 5 ckpts'), ('falls', 'falls / min')]
fig, axes = plt.subplots(2, 3, figsize=(24, 9.5))
for row, (floor, sfx) in enumerate([('carpet', 'c'), ('table', 't')]):
  for j, (k, ylab) in enumerate(M):
    ax = axes[row, j]
    for i, (lab, col, exps, *h) in enumerate(ARMS):
      v = [run(f'sel{r}_{sfx}', e)[k] for r, e in exps]
      ax.bar(i, np.mean(v), 0.7, yerr=np.std(v), color=col, capsize=3, hatch=h[0] if h else None, edgecolor='white', label=lab.replace('\n', ' '))
    ax.set_xticks(range(len(ARMS)), [a[0] for a in ARMS], rotation=35, ha='right'); ax.set_ylabel(ylab); ax.set_title(floor)
    if k == 'level': ax.set_ylim(0.4, 1)
    ax.grid(axis='y', color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
plt.tight_layout()
for ext in ('png', 'pdf'): plt.savefig(f'figs/r6_arms.{ext}', dpi=200, facecolor='white')
