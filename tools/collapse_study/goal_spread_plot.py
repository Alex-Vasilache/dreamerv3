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

ARMS = [a for a in ARMS if a[0] != 'imagB']   # arm B: one seed at 0.5M only, dropped
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
fig.subplots_adjust(left=0.07, right=0.9, top=0.8, bottom=0.13, wspace=0.42)
PANELS = (
    (axes[0], 0, 'goal distance (summed class gap, 0-56)',
     'How far apart are goals the manager\nproposes from the same state?'),
    (axes[1], 1, 'regions of goal space (4 bins per block)',
     'How many regions of goal space does it\npropose across all probe states?'))
for ax, j, ylab, title in PANELS:
  ends = []
  for arm, label, color, _ in ARMS:
    if arm not in rows:
      continue
    st = sorted(rows[arm])
    x = np.array(st) / 1e6
    vals = [np.array([r[j] for r in rows[arm][s]]) for s in st]
    m = np.array([v.mean() for v in vals]); n = np.array([len(v) for v in vals])
    sd = np.array([v.std(ddof=1) if len(v) > 1 else 0 for v in vals])
    if (n > 1).sum() > 1:
      ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.13, lw=0)
    ax.plot(x, m, color=color, lw=2.6 if arm == 'imagA' else 2.0, marker='o', ms=4)
    if arm == 'imagA':
      # The 0.5M point averages 3 seeds; from 1M on it is one run (e1821).
      v0 = vals[0]
      ax.scatter(np.full(len(v0), x[0]), v0, s=14, color=color, alpha=0.5, zorder=3)
    short = {'som': 'SOM', 'env': 'UCB env', 'imagA': 'UCB imag'}[arm]
    n_txt = f'{n.max()} seeds' if arm != 'imagA' else '1 seed'
    ends.append([m[-1], f'{short}\n({n_txt})', color])
  # Direct labels at the right end, nudged apart so they do not overlap.
  ends.sort(key=lambda e: e[0])
  span = ax.get_ylim()[1] - ax.get_ylim()[0]
  for i in range(1, len(ends)):
    if ends[i][0] - ends[i - 1][0] < 0.12 * span:
      ends[i][0] = ends[i - 1][0] + 0.12 * span
  for yv, lab, color in ends:
    ax.text(4.1, yv, lab, color=color, fontsize=12, va='center', linespacing=1.1)
  ax.set_title(title, fontsize=15, color=INK, loc='left')
  ax.set_xlabel('environment steps (M)', fontsize=15, color=INK)
  ax.set_ylabel(ylab, fontsize=14, color=INK)
  ax.set_xlim(0.4, 4.0)
  ax.tick_params(labelsize=13, colors=MUTED)
  ax.grid(True, color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
  for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
  for sp in ('left', 'bottom'):
    ax.spines[sp].set_color('#999999')
fig.suptitle(f'{TITLE}: goals the manager proposes from 800 fixed probe states (16 per state), '
             'same probe for every arm', fontsize=13, color=MUTED, x=0.02, ha='left')
out = os.path.join(os.path.dirname(D), f'goal_spread_{TASK}')
fig.savefig(out + '.png', dpi=200, facecolor='white', bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight')
print(out + '.png')
for arm in rows:
  for s in sorted(rows[arm]):
    v = np.array(rows[arm][s])
    print(f'{arm:6s} {s / 1e6:3.1f}M  width {v[:, 0].mean():5.1f}  regions {v[:, 1].mean():7.0f}  n={len(v)}')
