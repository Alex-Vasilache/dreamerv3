"""Figure: how much the entropy bonus weighs against the advantage in the worker update.

Worker policy loss per imagined step:  -(A * log pi(a) + eta * H(pi)).
With a fixed eta (fixed-entropy, R, V and flat DreamerV3: 3e-4) the ratio
eta / mean|A| says how strongly the entropy term pulls against the return
signal. A is the advantage exactly as it enters the loss (after return and
advantage normalization; train/wkr_goal_adv_mag, flat: train/adv_mag).

  fixed entropy, R   advantage normalized to unit std -> |A| ~ 0.6
  V, flat DreamerV3  advantage / (P95 - P5 of returns), no further scaling

OG adapts eta to hold normalized entropy at 0.5, so it has no fixed eta and is
drawn only in the entropy row.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/wkr_ent_balance.py

Reads wkr_diag_cache.json (written by wkr_diag.py) plus the flat runs' metrics.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import TASKS, runs, scores, OUT  # noqa: E402
from wkr_diag import ARMS as WARMS  # noqa: E402

ETA = 3e-4
ARMS = WARMS + [('dv3', 'DreamerV3 (flat)', '#000000', 'e12[3-8]?_{t}_dreamerv3-pubmatch_s*')]
BW = 2e5
EDGES = np.arange(0, 4e6 + BW, BW)
X = (EDGES[:-1] + BW / 2) / 1e6


def flat_metrics(d):
  out = {'adv': [], 'ent': []}
  for line in open(f'{d}/logdir/metrics.jsonl'):
    if 'adv_mag' not in line:
      continue
    r = json.loads(line)
    out['adv'].append((r['step'], r['train/adv_mag']))
    out['ent'].append((r['step'], r['train/ent/action']))
  return out


def binned(pairs):
  a = np.array(pairs) if len(pairs) else np.zeros((0, 2))
  idx = np.digitize(a[:, 0], EDGES) - 1 if len(a) else np.array([], int)
  return np.array([a[idx == b, 1].mean() if (idx == b).any() else np.nan for b in range(len(X))])


def main():
  cache = json.load(open(f'{OUT}/wkr_diag_cache.json'))
  curves, late = {}, {}
  for key, label, color, pat in ARMS:
    for t, *_ in TASKS:
      for d in runs(pat.format(t=t)):
        name = os.path.basename(d)
        if key == 'dv3':
          m = flat_metrics(d)
          adv, ent = m['adv'], m['ent']
        elif name in cache:
          adv, ent = cache[name]['train/wkr_goal_adv_mag'], cache[name]['train/wkr_ent/action']
        else:
          continue
        curves.setdefault((key, t), []).append((binned(adv), binned(ent)))
        s = scores(d)
        if len(s) and s[:, 0].max() >= 3.98e6:
          a = np.array(adv)
          sel = a[:, 0] >= 3e6
          late.setdefault((key, t), []).append((s[s[:, 0] >= 3e6, 1].mean(), a[sel, 1].mean()))

  fig, axes = plt.subplots(2, 6, figsize=(26, 9.5), sharex=True)
  for j, (t, title, _) in enumerate(TASKS):
    for key, label, color, _ in ARMS:
      if (key, t) not in curves:
        continue
      adv = np.nanmean([c[0] for c in curves[key, t]], 0)
      ent = np.nanmean([c[1] for c in curves[key, t]], 0)
      if key != 'og':
        # Flat DreamerV3 on Pin Pad Six never sees reward: A = 0, entropy only.
        axes[0, j].plot(X, 100 * ETA / np.where(adv > 1e-6, adv, np.nan), color=color, lw=2.3, label=label)
      axes[1, j].plot(X, ent, color=color, lw=2.3, label=label)
    axes[0, j].set_yscale('log')
    axes[0, j].set_title(title, fontsize=17)
    for ax in axes[:, j]:
      ax.tick_params(labelsize=12)
      ax.grid(True, color='#d5d5d2', lw=0.7)
      for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    axes[1, j].set_xlabel('environment steps (M)', fontsize=14)
  axes[0, 0].set_ylabel('entropy weight / mean |advantage| (%)', fontsize=14)
  axes[1, 0].set_ylabel('worker policy entropy (nats)', fontsize=14)
  h, l = axes[1, 0].get_legend_handles_labels()
  fig.legend(h, l, loc='lower center', ncol=5, fontsize=15, frameon=False)
  fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=2)
  fig.savefig(f'{OUT}/wkr_ent_balance.png', dpi=200, facecolor='white')
  fig.savefig(f'{OUT}/wkr_ent_balance.pdf')
  print(f'{OUT}/wkr_ent_balance.png')
  print('late 3-4M: score | entropy weight as % of mean |A|')
  for t, title, _ in TASKS:
    print(f'{title:18s} ' + '  '.join(
        f'{k} {np.mean([v[0] for v in late[k, t]]):5.0f} | {100 * ETA / np.mean([v[1] for v in late[k, t]]):6.2f}%'
        for k, *_ in ARMS if (k, t) in late and k != 'og'))


if __name__ == '__main__':
  main()
