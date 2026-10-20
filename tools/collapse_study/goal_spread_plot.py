"""Figure: how widely the manager proposes goals, OG+SOM vs UCB (env) vs UCB (imagination).

From goal_spread_probe.py: per run and milestone, K=16 manager goal samples at
each of 800 fixed probe states. Two measures, same for every arm:

  search width   mean summed class-index distance between two goals proposed
                 from the SAME state (0-56): how widely the manager searches
                 from one place.
  regions        distinct cells of the 4-bins-per-block grid (65,536) hit by
                 all 12,800 proposals: how much of the goal space it proposes
                 across all states.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/goal_spread_plot.py pinpad_five
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/work/DoyaU/vasilache/work/collapse_figs/goal_spread'
TASK = sys.argv[1] if len(sys.argv) > 1 else 'pinpad_five'
TITLE = {'pinpad_five': 'Pin Pad Five'}.get(TASK, TASK)
# ``<task>_flat.npz``: one plain int8 array per '<run>|<milestone>' (the probe's
# own file holds numpy-2 objects that this numpy-1 env cannot unpickle).
z = np.load(os.path.join(D, f'{TASK}_flat.npz'))
ARMS = [('som', 'OG + SOM', '#CC79A7', lambda n: 'goal_som_line-pubmatch' in n),
        ('env', 'OG + SOM + UCB (env)', '#000000', lambda n: 'mgr_ucb' in n),
        ('imagA', 'OG + SOM + UCB imag A', '#0072B2', lambda n: 'ucb_imag_a' in n),
        ('imagB', 'OG + SOM + UCB imag B', '#56B4E9', lambda n: 'ucb_imag_b' in n)]
INK, MUTED = '#1c1e22', '#5d636e'


def measures(ids):
  ids = ids.astype(np.int32)                                   # (S, K, L)
  d = np.abs(ids[:, :, None] - ids[:, None]).sum(-1)           # (S, K, K)
  k = ids.shape[1]
  width = d.sum((1, 2)) / (k * (k - 1))
  q = np.minimum(ids * 4 // 8, 3).reshape(-1, ids.shape[-1])
  cells = len(np.unique((q * (4 ** np.arange(q.shape[1]))).sum(-1)))
  return width.mean(), cells


rows = {}
for key in z.files:
  name, ck = key.split('|')
  arm = next(a for a, _, _, f in ARMS if f(name))
  rows.setdefault(arm, {}).setdefault(int(ck), []).append(measures(z[key]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
for ax, j, ylab, title in (
    (axes[0], 0, 'class-index distance (0-56)', 'Search width: goals proposed from the same state'),
    (axes[1], 1, 'distinct regions (of 65,536)', 'Regions proposed across all probe states')):
  for arm, label, color, _ in ARMS:
    if arm not in rows:
      continue
    st = sorted(rows[arm])
    x = np.array(st) / 1e6
    vals = [np.array([r[j] for r in rows[arm][s]]) for s in st]
    m = np.array([v.mean() for v in vals]); n = np.array([len(v) for v in vals])
    sd = np.array([v.std(ddof=1) if len(v) > 1 else 0 for v in vals])
    if (n > 1).any():
      ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.13, lw=0)
    ax.plot(x, m, color=color, lw=2.4, marker='o', ms=5, label=f'{label} (n={n.max()})')
  ax.set_title(title, fontsize=15, color=INK, loc='left')
  ax.set_xlabel('environment steps (M)', fontsize=15, color=INK)
  ax.set_ylabel(ylab, fontsize=15, color=INK)
  ax.set_xlim(0, 4.1)
  ax.tick_params(labelsize=13, colors=MUTED)
  ax.grid(True, color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
  for sp in ax.spines.values():
    sp.set_color('#999999')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, fontsize=13, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f'Which goals the manager proposes, {TITLE}: 16 samples at each of 800 fixed probe '
             'states, untilted policy samples', fontsize=14, color=MUTED, x=0.07, ha='left')
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
out = os.path.join(os.path.dirname(D), f'goal_spread_{TASK}')
fig.savefig(out + '.png', dpi=200, facecolor='white'); fig.savefig(out + '.pdf')
print(out + '.png')
for arm in rows:
  for s in sorted(rows[arm]):
    v = np.array(rows[arm][s])
    print(f'{arm:6s} {s / 1e6:3.1f}M  width {v[:, 0].mean():5.1f}  regions {v[:, 1].mean():7.0f}  n={len(v)}')
