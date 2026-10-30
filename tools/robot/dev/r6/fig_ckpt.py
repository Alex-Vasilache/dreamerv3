"""Fitted-sim tracking over the last saved checkpoints of one run (ckptsweep.sh), carpet and table.
  python fig_ckpt.py"""
import glob, json, re, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})
EV = '/work/DoyaU/vasilache/work/sysid/r6/ev'
RUNS = [('e8132', 'carpet-only (6d)', '#E69F00', '-'), ('e8167', 'carpet-only (6g)', '#E69F00', '--'), ('e8140', 'carpet-only, no CAPS (6e)', '#0072B2', '-'),
        ('e8136', 'mixed (6d)', '#D55E00', '-'), ('e8137', 'mixed (6d)', '#D55E00', '--'), ('e8145', 'mixed, no CAPS (6e)', '#56B4E9', '-'),
        ('e8163', 'mixed sigma .6 tilt .25 (6g)', '#000000', '-'), ('e8164', 'mixed sigma .6 tilt .25 (6g)', '#000000', '--')]


def sweep(tags, exp):
  pts = {}
  for tag in tags:
    for f in glob.glob(f'{EV}/{tag}/{exp}@*_*.json'):
      step = int(re.search(r'@(\d+)_', f).group(1))
      pts.setdefault(step, {}).update(json.load(open(f)))
  s = sorted(pts)
  return np.array(s) / 1e3, [pts[k]['fit'].get('corr_f', np.nan) for k in s], [pts[k]['fit']['dips_per_min'] for k in s]


fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
for row, (floor, tags) in enumerate([('carpet', ['sel_c', 'sel6g_c']), ('table', ['sel_t', 'sel6g_t'])]):
  for exp, lab, c, ls in RUNS:
    x, corr, dips = sweep(tags, exp)
    if not len(x):
      continue
    axes[row, 0].plot(x, corr, ls, color=c, marker='o', lw=2, label=f'{exp} {lab}')
    axes[row, 1].plot(x, dips, ls, color=c, marker='o', lw=2, label=f'{exp} {lab}')
  axes[row, 0].set_ylabel('forward tracking (corr)'); axes[row, 1].set_ylabel('falls below -6 deg / min'); axes[row, 1].set_yscale('symlog', linthresh=1)
  for ax in axes[row]:
    ax.set_title(floor); ax.set_xlabel('checkpoint (k env steps)'); ax.grid(color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
h, l = [], []
for ax in axes.flat:
  for hh, ll in zip(*ax.get_legend_handles_labels()):
    if ll not in l: h.append(hh); l.append(ll)
fig.legend(h, l, loc='lower center', ncol=3, frameon=False)
plt.tight_layout(rect=(0, 0.1, 1, 1))
for ext in ('png', 'pdf'): plt.savefig(f'figs/ckpt_sweep.{ext}', dpi=200, facecolor='white')
