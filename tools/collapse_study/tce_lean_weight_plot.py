"""Lean and TCE weight over training on pin pad five / six, for each TCE variant.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/tce_lean_weight_plot.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import runs, OUT  # noqa: E402

B = 'director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir'
ARMS = [
    ('TCE (0.5M half-life, adaptive w)', '#0072B2', '-', f'e62[0-2]?_{{t}}_{B}-pubmatch_s*'),
    ('TCE-anneal (no decay, w=2)', '#E69F00', ':', f'e70[0-2]?_{{t}}_{B}-fromto_anneal-pubmatch_s*'),
    ('TCE-grow (h=t/2, w=2)', '#CC79A7', '-.', f'e80[0-2]?_{{t}}_{B}-tce_grow-pubmatch_s*'),
    ('WM hash 1x16, adaptive w', '#009E73', '-', f'e95[0-2]?_{{t}}_{B}-tce_hash-pubmatch_s*'),
    ('WM hash 4x12, w=2', '#D55E00', '--', f'e1[09]0[0-3]?_{{t}}_{B}-tce_hash4-pubmatch_s*'),
    ('WM hash 4x12, adaptive w', '#000000', '-.', f'e191[0-3]?_{{t}}_{B}-tce_hash4a-pubmatch_s*'),
]
TASKS = [('pinpad_five', 'Pin Pad Five'), ('pinpad_six', 'Pin Pad Six')]
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


fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
for j, (t, title) in enumerate(TASKS):
  for name, col, ls, pat in ARMS:
    ds = runs(pat.format(t=t))
    if not ds:
      continue
    for i, key in enumerate(['train/fromto/lean', 'train/fromto/weight']):
      with np.errstate(all='ignore'):
        m = np.nanmedian([curve(d, key) for d in ds], 0)
      axes[i, j].plot(X, m, color=col, ls=ls, lw=2.6, label=f'{name} (n={len(ds)})' if j == 1 and i == 0 else name)
  axes[0, j].set_title(title, fontsize=17)
  axes[0, j].axhline(1.0, color='#8a8880', lw=1, ls='--')
  axes[1, j].set_yscale('log')
  axes[1, j].set_xlabel('environment steps (M)', fontsize=15)
  for ax in axes[:, j]:
    ax.set_xlim(0, 4); ax.tick_params(labelsize=13); ax.grid(True, color='#d5d5d2', lw=0.7)
    for sp in ('top', 'right'):
      ax.spines[sp].set_visible(False)
axes[0, 0].set_ylabel('lean = policy / random novelty', fontsize=15)
axes[1, 0].set_ylabel('TCE weight w (log)', fontsize=15)
h, l = axes[0, 1].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, fontsize=13, frameon=False)
fig.text(0.5, 0.115, 'median over seeds; lean > 1 = manager prefers rarely tried start->goal cells', ha='center', fontsize=13)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(f'{OUT}/tce_lean_weight.png', dpi=200, facecolor='white'); fig.savefig(f'{OUT}/tce_lean_weight.pdf')
print(f'{OUT}/tce_lean_weight.png')
