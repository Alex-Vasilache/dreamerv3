"""Worker/manager training diagnostics for OG vs fixed worker entropy vs worker-v3 (R, V).

Per arm, task and run: the mean of each metric over 3-4M env steps (finished
runs only), plus binned curves. Answers *why* the arms differ: how random the
worker acts, how well it reaches its goals, how large its advantages are, and
what the manager does on top.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/wkr_diag.py

Writes collapse_figs/wkr_diag.{png,pdf} and wkr_diag_table.md.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import TASKS, runs, OUT, FULL  # noqa: E402

ARMS = [
    ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
    ('ent', 'OG + fixed worker entropy', '#999999',
     'e1[89][089]?_{t}_director-director_og-wkr_ent_fixed-pubmatch_s*'),
    ('R', '+ worker v3 return norm (R)', '#0072B2',
     'e19[1-6]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3_retnorm-pubmatch_s*'),
    ('V', '+ worker v3, all (V)', '#009E73',
     'e19[1-6]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-pubmatch_s*'),
]
KEYS = [('train/wkr_ent/action', 'worker entropy'),
        ('train/wkr_goal_rew', 'worker goal reward'),
        ('train/wkr_goal_adv_mag', 'worker |advantage|'),
        ('train/wkr_goal_adv_std', 'worker advantage std'),
        ('train/wkr_goal_ret', 'worker return'),
        ('train/mgr_ent/skill', 'manager entropy'),
        ('train/mgr_extr_ret', 'manager task return'),
        ('train/opt/wkr_update_rms', 'worker update size')]
LO, HI, BW = 3e6, 4e6, 2e5
CACHE = f'{OUT}/wkr_diag_cache.json'


def read(d):
  path = f'{d}/logdir/metrics.jsonl'
  out = {k: [] for k, _ in KEYS}
  if not os.path.exists(path):
    return out
  for line in open(path):
    if 'wkr_goal_rew' not in line:
      continue
    try:
      r = json.loads(line)
    except ValueError:
      continue
    for k, _ in KEYS:
      if r.get(k) is not None:
        out[k].append((r['step'], r[k]))
  return out


def main():
  cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
  data = {}
  for key, *_ , pat in ARMS:
    for t, *_ in TASKS:
      for d in runs(pat.format(t=t)):
        name = os.path.basename(d)
        done = os.path.exists(f'{d}/logdir/scores.jsonl') and d.startswith('/bucket')
        if name not in cache or not done:
          cache[name] = read(d)
        data.setdefault((key, t), []).append(cache[name])
  json.dump(cache, open(CACHE, 'w'))

  # Table: late mean per arm/task (only runs past 3.98M), averaged over seeds.
  rows = []
  late = {}
  for k, label in KEYS:
    for t, title, _ in TASKS:
      cells = []
      for key, *_ in ARMS:
        vals = []
        for m in data.get((key, t), []):
          a = np.array(m[k]) if m[k] else np.zeros((0, 2))
          if len(a) and a[:, 0].max() >= FULL:
            sel = (a[:, 0] >= LO) & (a[:, 0] < HI)
            vals.append(a[sel, 1].mean())
        late[k, t, key] = vals
        cells.append(f'{np.mean(vals):.3g} (n={len(vals)})' if vals else '-')
      rows.append(f'| {label} | {title} | ' + ' | '.join(cells) + ' |')
  head = '| metric | task | ' + ' | '.join(l for _, l, *_ in ARMS) + ' |'
  table = '\n'.join([head, '|---' * (len(ARMS) + 2) + '|'] + rows)
  open(f'{OUT}/wkr_diag_table.md', 'w').write(table + '\n')
  print(table)

  # Figure: curves of the four metrics that separate the arms, one row per metric
  # and one column per task.
  show = KEYS[:3] + [KEYS[6]]
  fig, axes = plt.subplots(len(show), 6, figsize=(24, 14), sharex=True)
  edges = np.arange(0, HI + BW, BW)
  x = (edges[:-1] + BW / 2) / 1e6
  for i, (k, label) in enumerate(show):
    for j, (t, title, _) in enumerate(TASKS):
      ax = axes[i, j]
      for key, alabel, color, _ in ARMS:
        curves = []
        for m in data.get((key, t), []):
          a = np.array(m[k]) if m[k] else np.zeros((0, 2))
          if not len(a):
            continue
          idx = np.digitize(a[:, 0], edges) - 1
          curves.append([a[idx == b, 1].mean() if (idx == b).any() else np.nan for b in range(len(x))])
        if curves:
          with np.errstate(all='ignore'):
            mean = np.nanmean(np.array(curves), 0)
          ax.plot(x, mean, color=color, lw=2.2, label=alabel)
      if i == 0:
        ax.set_title(title, fontsize=17)
      if j == 0:
        ax.set_ylabel(label, fontsize=15)
      ax.tick_params(labelsize=12)
      ax.grid(True, color='#d5d5d2', lw=0.7)
      for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
  for ax in axes[-1]:
    ax.set_xlabel('environment steps (M)', fontsize=14)
  h, l = axes[0, 0].get_legend_handles_labels()
  fig.legend(h, l, loc='lower center', ncol=4, fontsize=15, frameon=False)
  fig.tight_layout(rect=(0, 0.04, 1, 1))
  fig.savefig(f'{OUT}/wkr_diag.png', dpi=200, facecolor='white')
  fig.savefig(f'{OUT}/wkr_diag.pdf')
  print(f'{OUT}/wkr_diag.png')


if __name__ == '__main__':
  main()
