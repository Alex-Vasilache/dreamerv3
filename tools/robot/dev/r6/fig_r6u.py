"""Rounds 6u-6w: one policy for command and balance without a fixed balance point. 10-min evals per policy: tracking on the
standard and hard sets, falls with the stick and at a zero stick, zero-stick roll. Bars: mean over the set's variants, error: SD over
variants.  python fig_r6u.py"""
import glob, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 11, 'ytick.labelsize': 12, 'legend.fontsize': 12})
EV = '/work/DoyaU/vasilache/work/sysid/r6/ev'
POL = [('r6_mixed\n(e8261, current)', 'e8261@838500', '#999999'), ('6u C\n(e10025)', 'e10025@760500', '#56B4E9'),
       ('6v H\n(e10054)', 'e10054@916500', '#D55E00'), ('6v J\n(e10058)', 'e10058@838500', '#009E73')]
EXTRA = [('6w K\n(e10071)', 'e10071@682500', '#CC79A7')]


def get(tag, pol):
  d = {}
  for f in glob.glob(f'{EV}/{tag}/{pol}_*.json'):
    d.update(json.load(open(f)))
  return d


def nz(x):
  return 0.0 if x is None or not np.isfinite(x) else x


M = [('c10_{}', lambda v: nz(v.get('corr_f')), 'tracking corr,\nstandard set', False),
     ('c10h_{}', lambda v: nz(v.get('corr_f')), 'tracking corr,\nhard set', False),
     ('c10h_{}', lambda v: v['dips_per_min'], 'falls / min,\nhard set, stick', False),
     ('c10h_z{}', lambda v: v['dips_per_min'], 'falls / min,\nhard set, zero stick', True),
     ('c10h_z{}', lambda v: abs(v['fwd_mean']) if np.isfinite(v['fwd_mean']) else 2000, 'roll |mm/s|,\nhard set, zero stick', True)]
pols = POL + EXTRA
fig, axes = plt.subplots(2, len(M), figsize=(26, 9.5))
for row, (floor, sfx) in enumerate([('carpet', 'c'), ('table', 't')]):
  for j, (tag, fn, ylab, skipfit) in enumerate(M):
    ax = axes[row, j]
    for i, (lab, pol, col) in enumerate(pols):
      d = get(tag.format(sfx), pol)
      v = [fn(x) for k, x in d.items()]
      if v:
        ax.bar(i, np.mean(v), 0.7, yerr=np.std(v), color=col, capsize=3, edgecolor='white')
    ax.set_xticks(range(len(pols)), [p[0] for p in pols], rotation=30, ha='right'); ax.set_ylabel(ylab); ax.set_title(floor)
    ax.set_ylim(bottom=0); ax.grid(axis='y', color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
plt.tight_layout()
for ext in ('png', 'pdf'): plt.savefig(f'/work/DoyaU/vasilache/work/sysid/r6/figs/r6u.{ext}', dpi=200, facecolor='white')
