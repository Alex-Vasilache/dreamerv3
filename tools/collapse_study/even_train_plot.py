"""Figure: class-step geometry over training in V + SOM + evenness (e2000-e2059).

From the loss's own logged diagnostics (train/goal/even_*), one line per task,
mean over seeds. The V + SOM arm does not compute them; its final-checkpoint
values come from ucb_kernel_plot.py someven.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/even_train_plot.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import TASKS, runs, OUT  # noqa: E402

PAT = 'e20[0-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-pubmatch_s*'
KEYS = [('train/goal/even_step_cv', 'step-size variation (CV)', 0.0),
        ('train/goal/even_end_ratio', 'end steps / mean step', 1.0),
        ('train/goal/even_block_ratio', 'strongest / weakest block', 1.0)]
BW = 2e5
EDGES = np.arange(0, 4e6 + BW, BW)
X = (EDGES[:-1] + BW / 2) / 1e6


def curves(d):
  out = {k: [] for k, *_ in KEYS}
  f = f'{d}/logdir/metrics.jsonl'
  if not os.path.exists(f):
    return None
  for line in open(f):
    if 'even_step_cv' not in line:
      continue
    r = json.loads(line)
    for k, *_ in KEYS:
      out[k].append((r['step'], r[k]))
  res = {}
  for k, v in out.items():
    a = np.array(v)
    idx = np.digitize(a[:, 0], EDGES) - 1
    res[k] = np.array([a[idx == b, 1].mean() if (idx == b).any() else np.nan for b in range(len(X))])
  return res


def main():
  fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
  for t, title, color in TASKS:
    cs = [c for c in (curves(d) for d in runs(PAT.format(t=t))) if c]
    if not cs:
      continue
    for ax, (k, label, ideal) in zip(axes, KEYS):
      with np.errstate(all='ignore'):
        ax.plot(X, np.nanmean([c[k] for c in cs], 0), color=color, lw=2.3, label=title)
  for ax, (k, label, ideal) in zip(axes, KEYS):
    ax.axhline(ideal, color='#8a8880', ls='--', lw=1.2)
    ax.set_title(label + f'   (even = {ideal:g})', fontsize=16, loc='left')
    ax.set_xlabel('environment steps (M)', fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(True, color='#d5d5d2', lw=0.7)
    ax.set_xlim(0, 4)
    for sp in ('top', 'right'):
      ax.spines[sp].set_visible(False)
  axes[2].set_yscale('log')
  h, l = axes[0].get_legend_handles_labels()
  fig.legend(h, l, loc='lower center', ncol=6, fontsize=14, frameon=False)
  fig.tight_layout(rect=(0, 0.08, 1, 1))
  out = f'{OUT}/even_train'
  fig.savefig(out + '.png', dpi=200, facecolor='white'); fig.savefig(out + '.pdf')
  print(out + '.png')


if __name__ == '__main__':
  main()
