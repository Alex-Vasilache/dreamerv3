"""Figure: the from-to adaptive weight w, lean and novelty over training (e6200-e6229).

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/fromto_weight_plot.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import TASKS, runs, OUT  # noqa: E402

PAT = 'e62[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-pubmatch_s*'
KEYS = [('train/fromto/weight', 'weight w (log scale)'), ('train/fromto/lean', 'lean (policy / random novelty)'),
        ('train/fromto/novelty', 'novelty of policy goals')]
BW = 2.5e5
EDGES = np.arange(0, 4e6 + BW, BW)
X = (EDGES[:-1] + BW / 2) / 1e6


def curve(d, key):
  a = []
  for line in open(f'{d}/logdir/metrics.jsonl'):
    if 'fromto/weight' in line:
      r = json.loads(line)
      a.append((r['step'], r[key]))
  a = np.array(a)
  idx = np.digitize(a[:, 0], EDGES) - 1
  return np.array([a[idx == b, 1].mean() if (idx == b).any() else np.nan for b in range(len(X))])


fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))
rows = {}
for t, title, color in TASKS:
  cs = {k: [curve(d, k) for d in runs(PAT.format(t=t))] for k, _ in KEYS}
  for ax, (k, label) in zip(axes, KEYS):
    with np.errstate(all='ignore'):
      m = np.nanmedian(cs[k], 0) if k.endswith('weight') else np.nanmean(cs[k], 0)
    ax.plot(X, m, color=color, lw=2.3, label=title)
    rows.setdefault(t, {})[k] = m
for ax, (k, label) in zip(axes, KEYS):
  ax.set_title(label, fontsize=16, loc='left')
  ax.set_xlabel('environment steps (M)', fontsize=14)
  ax.tick_params(labelsize=13); ax.grid(True, color='#d5d5d2', lw=0.7); ax.set_xlim(0, 4)
  for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
axes[0].set_yscale('log'); axes[0].axhline(10, color='#8a8880', ls='--', lw=1)
axes[1].axhline(1.0, color='#8a8880', ls='--', lw=1)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=6, fontsize=14, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(f'{OUT}/fromto_weight.png', dpi=200, facecolor='white'); fig.savefig(f'{OUT}/fromto_weight.pdf')
print(f'{OUT}/fromto_weight.png')
pick = [0.25, 0.5, 1, 2, 3, 3.75]
print('median w per task at', pick, 'M')
for t, title, _ in TASKS:
  w = rows[t]['train/fromto/weight']
  print(f'  {title:17s} ' + '  '.join(f'{w[int(p / 0.25)]:5.2f}' for p in pick))
