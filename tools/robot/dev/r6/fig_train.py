"""Round 6 training arms in the fitted sim (v6 + 9 robustness variants, Alex's 10-30 stick replayed): mean ± std over seeds.
  python fig_train.py"""
import glob, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 11, 'ytick.labelsize': 12, 'legend.fontsize': 12})
EV = '/work/DoyaU/vasilache/work/sysid/r6/ev'


def load(tag, exp):
  out = {}
  for f in glob.glob(f'{EV}/{tag}/{exp}_*.json'):
    out.update(json.load(open(f)))
  return out


def stats(d):
  c = [v.get('corr_f', np.nan) for v in d.values()]
  f = d['fit']
  return dict(corr=f['corr_f'], rate=f['rate_std'], dips=f['dips_per_min'], minc=np.nanmin(c))


# (label, colour, carpet runs (tag, exp), table runs)
ARMS = [
  ('round 5\n(e6031 / e6033)', '#999999', [('ref6', 'e6031')], [('ref6', 'e6033')]),
  ('6a carpet\nSM (CAPS)', '#CC79A7', [('r6a_final', 'e8105'), ('r6a_final', 'e8103')], []),
  ('6b table\nSM', '#009E73', [], [('r6b_final_t', 'e8114'), ('r6b_final_t', 'e8115')]),
  ('6c carpet\nstick 1500', '#0072B2', [('r6c_final', 'e8124'), ('r6c_final', 'e8125')], []),
  ('6d/6g carpet\nSM+progress', '#E69F00', [('r6d_final', 'e8132'), ('r6d_final', 'e8133'), ('r6g_final', 'e8167')], []),
  ('6e mixed\nno CAPS', '#56B4E9', [('r6e_final', e) for e in ('e8145', 'e8146', 'e8147')], [('r6e_final_t', e) for e in ('e8145', 'e8146', 'e8147')]),
  ('6d/6g mixed\nSM+progress', '#D55E00', [('r6d_final', 'e8136'), ('r6d_final', 'e8137')] + [('r6g_final', e) for e in ('e8160', 'e8161', 'e8162')],
   [('r6d_final_t', 'e8136'), ('r6d_final_t', 'e8137')] + [('r6g_final_t', e) for e in ('e8160', 'e8161', 'e8162')]),
  ('6g mixed SM+progress\nsigma .6, tilt .25', '#000000', [('r6g_final', 'e8163'), ('r6g_final', 'e8164')], [('r6g_final_t', 'e8163'), ('r6g_final_t', 'e8164')]),
]
M = [('corr', 'forward tracking (corr)'), ('rate', 'tilt-rate SD (deg/s)'), ('dips', 'falls below -6 deg / min'), ('minc', 'worst-variant tracking')]
fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))
for row, (floor, idx) in enumerate([('carpet', 2), ('table', 3)]):
  for j, (k, ylab) in enumerate(M):
    ax = axes[row, j]
    for i, arm in enumerate(ARMS):
      runs = arm[idx]
      if not runs:
        continue
      v = [stats(load(t, e))[k] for t, e in runs]
      ax.bar(i, np.mean(v), 0.75, yerr=np.std(v) if len(v) > 1 else None, color=arm[1], capsize=3, label=arm[0].replace('\n', ' '))
    ax.set_xticks([]); ax.set_ylabel(ylab); ax.set_title(f'{floor}')
    ax.grid(axis='y', color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
h, l = [], []
for ax in axes.flat:
  for hh, ll in zip(*ax.get_legend_handles_labels()):
    if ll not in l: h.append(hh); l.append(ll)
order = [a[0].replace('\n', ' ') for a in ARMS]
h = [h[l.index(o)] for o in order if o in l]; l = [o for o in order if o in l]
fig.legend(h, l, loc='lower center', ncol=4, frameon=False)
plt.tight_layout(rect=(0, 0.1, 1, 1))
for ext in ('png', 'pdf'): plt.savefig(f'figs/train_r6.{ext}', dpi=200, facecolor='white')
