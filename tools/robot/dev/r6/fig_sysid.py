"""Closed-loop sysid: 7 real-robot recordings vs sim variants (rate SD, forward tracking, total loss)."""
import json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
sys.path.insert(0, '.')
from loss import load, real, POL, score
plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})
res = load(['A', 'B', 'B2', 'C', 'D'])
V = [('i4', 'old sim\n(inertia 4)', '#999999'), ('i1', 'inertia 1', '#56B4E9'), ('i2_bl015', 'inertia 2\nbacklash .015', '#0072B2'),
     ('v6', 'v6: + stick .3\ngear_k 1', '#009E73'), ('v6_rs7', 'v6 + carpet\nstiction 7x', '#E69F00')]
P = list(POL); lab = {'e2994': 'e2994\ncarpet bal', 'e2608': 'e2608\ncarpet bal', 'e2995': 'e2995\ntable bal', 'e2503': 'e2503\ntable bal',
                       'e2992': 'e2992\ncarpet cmd', 'e6031': 'e6031\ncarpet cmd', 'e6033': 'e6033\ntable cmd'}
fig = plt.figure(figsize=(15, 9.5)); gs = fig.add_gridspec(2, 3)
x = np.arange(len(P)); w = 0.14
ax = fig.add_subplot(gs[0, :2])
for k, (v, name, c) in enumerate(V):
  ax.bar(x + (k - 2.5) * w, [res[v][p]['rate_std'] for p in P], w, color=c, label=name.replace('\n', ' '))
ax.bar(x + 2.5 * w, [real[POL[p][1].split('/')[-1]]['rate_std'] for p in P], w, color='#D55E00', label='robot')
ax.set_xticks(x, [lab[p] for p in P]); ax.set_ylabel('tilt-rate SD (deg/s)'); ax.set_title('same policy, robot vs sim (eval mode, recorded stick)')
ax.grid(axis='y', color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
ax2 = fig.add_subplot(gs[0, 2])
tot = [np.mean([score(res[v][p], real[POL[p][1].split('/')[-1]]) for p in P]) for v, _, _ in V]
ax2.bar(range(len(V)), tot, color=[c for _, _, c in V]); ax2.set_yscale('log')
ax2.set_xticks(range(len(V)), [n for _, n, _ in V], fontsize=9, rotation=35, ha='right'); ax2.set_ylabel('mismatch (mean over 7 policies)'); ax2.set_title('closed-loop fit')
ax2.grid(axis='y', color='#d5d5d2', lw=0.7); ax2.set_axisbelow(True)
C = ['e2992', 'e6031', 'e6033']
for j, (key, ylab) in enumerate([('corr_f', 'forward tracking (corr)'), ('dips_per_min', 'falls below -6 deg / min'), ('fwd_fwd_med', 'median speed, stick forward (mm/s)')]):
  ax = fig.add_subplot(gs[1, j]); xc = np.arange(len(C))
  for k, (v, name, c) in enumerate(V):
    ax.bar(xc + (k - 2.5) * w, [res[v][p].get(key, np.nan) for p in C], w, color=c)
  ax.bar(xc + 2.5 * w, [real[POL[p][1].split('/')[-1]].get(key, np.nan) for p in C], w, color='#D55E00')
  ax.set_xticks(xc, [lab[p] for p in C]); ax.set_ylabel(ylab)
  ax.grid(axis='y', color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
h, l = fig.axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=6, frameon=False)
plt.tight_layout(rect=(0, 0.05, 1, 1))
for ext in ('png', 'pdf'): plt.savefig(f'figs/sysid_closedloop.{ext}', dpi=200, facecolor='white')
