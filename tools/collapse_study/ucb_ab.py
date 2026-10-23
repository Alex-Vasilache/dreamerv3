"""UCB in imagination: arm A (goal 8 bins/block) vs arm B (start 2 x goal 4 bins/block), c=3.

Both batches were cancelled mid-run, so most seeds stop between 0.05M and 1M
and only a few reach 2.5-4M. Every comparison here is therefore at MATCHED
steps: for each window end (0.5M, 1M, 2.5M, 4M) a run counts only if it
reached that step, and it is scored by its mean return over the 0.5M before.
OG+SOM and OG+SOM+UCB (env) give the reference at the same windows.

Diagnostics (from metrics.jsonl, per run, binned 0.1M):
  logp_shift     log-prob of the picked goal minus the mean candidate (nats):
                 how far the tilt pulls the manager away from its own policy.
  novelty_gain   bonus of the picked goal / mean candidate bonus.
  pick_bonus     1/sqrt(n+1) of the picked cell: 1 = never tried, small = well tried.
  eff_cells      exp(entropy) of the count table: how many cells hold the mass.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/ucb_ab.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import TASKS, runs, scores, OUT, perm_p  # noqa: E402

ARMS = [
    ('som', 'OG + SOM', '#CC79A7', 'e1[34]??_{t}_director-director_og-goal_som_line-pubmatch_s*'),
    ('env', 'OG + SOM + UCB (env)', '#000000',
     'e14[3-5]?_{t}_director-director_og-goal_som_line-mgr_ucb-pubmatch_s*'),
    ('A', 'UCB imag A (goal 8 bins)', '#0072B2',
     'e18[2-4]?_{t}_director-director_og-goal_som_line-ucb_imag_a-ucb_imag_c3-pubmatch_s*'),
    ('B', 'UCB imag B (start 2 x goal 4 bins)', '#E69F00',
     'e18[5-7]?_{t}_director-director_og-goal_som_line-ucb_imag_b-ucb_imag_c3-pubmatch_s*'),
]
ENDS = [5e5, 1e6, 2.5e6, 4e6]
WIN = 5e5
DIAG = [('train/imag_ucb/logp_shift', 'log-prob shift of pick (nats)'),
        ('train/imag_ucb/novelty_gain', 'novelty gain (pick / candidates)'),
        ('train/imag_ucb/pick_bonus', 'bonus of picked cell'),
        ('train/imag_ucb/eff_cells', 'effective cells (of 16.7M)'),
        ('train/mgr_ent/skill', 'manager entropy (nats)'),
        ('train/wkr_goal_rew', 'worker goal reward')]
MAXRET = {'pinpad_four': 411, 'pinpad_five': 352, 'pinpad_six': 292,
          'dmc_cartpole_swingup': 1000, 'dmc_cheetah_run': 1000, 'dmc_hopper_hop': 1000}


def metrics(d):
  out = {k: [] for k, _ in DIAG}
  path = f'{d}/logdir/metrics.jsonl'
  if not os.path.exists(path):
    return out
  for line in open(path):
    if 'wkr_goal_rew' not in line:
      continue
    try:
      r = json.loads(line)
    except ValueError:
      continue
    for k, _ in DIAG:
      if r.get(k) is not None:
        out[k].append((r['step'], r[k]))
  return {k: np.array(v) if v else np.zeros((0, 2)) for k, v in out.items()}


def main():
  data = {(k, t): [(os.path.basename(d), scores(d)) for d in runs(p.format(t=t))]
          for k, _, _, p in ARMS for t, *_ in TASKS}
  # Matched-window table.
  lines = ['| task | window | ' + ' | '.join(l for _, l, *_ in ARMS) + ' |', '|---' * (len(ARMS) + 2) + '|']
  norm = {}
  for t, title, _ in TASKS:
    for hi in ENDS:
      cells, vals = [], {}
      for k, *_ in ARMS:
        v = [a[(a[:, 0] >= hi - WIN) & (a[:, 0] < hi), 1].mean() for _, a in data[k, t]
             if len(a) and a[:, 0].max() >= hi - 2e4]
        vals[k] = v
      if not vals['A'] and not vals['B']:
        continue
      for k, *_ in ARMS:
        v = vals[k]
        p = f', p={perm_p(vals["A"], v):.2f}' if k == 'B' and len(v) >= 2 and len(vals['A']) >= 2 else ''
        cells.append(f'{np.mean(v):.0f} (n={len(v)}{p})' if v else '-')
        if v:
          norm.setdefault((k, hi), {})[t] = np.mean(v) / MAXRET[t]
      lines.append(f'| {title} | {(hi - WIN) / 1e6:.1f}-{hi / 1e6:.1f}M | ' + ' | '.join(cells) + ' |')
  table = '\n'.join(lines)
  print(table)
  # Overall normalized score per window, only over tasks where both A and B have data.
  print('\nnormalized mean over tasks where A and B both have runs:')
  for hi in ENDS:
    both = [t for t, *_ in TASKS if t in norm.get(('A', hi), {}) and t in norm.get(('B', hi), {})]
    if both:
      print(f'  {hi / 1e6:.1f}M ({len(both)} tasks): ' + '  '.join(
          f'{k} {100 * np.mean([norm[k, hi][t] for t in both]):.0f}%' for k, *_ in ARMS
          if all(t in norm.get((k, hi), {}) for t in both)))
  open(f'{OUT}/ucb_ab_table.md', 'w').write(table + '\n')

  # Diagnostics: one row per metric, pin pads vs DMC pooled per arm (A, B only).
  diag = {}
  for k in ('A', 'B'):
    for t, *_ in TASKS:
      for d in runs(dict((a[0], a[3]) for a in ARMS)[k].format(t=t)):
        diag.setdefault(k, []).append((t, os.path.basename(d), metrics(d)))
  fig, axes = plt.subplots(2, 3, figsize=(18, 10))
  bw = 1e5
  edges = np.arange(0, 4e6 + bw, bw)
  x = (edges[:-1] + bw / 2) / 1e6
  summary = []
  for ax, (key, label) in zip(axes.ravel(), DIAG):
    for k, color, ls_by in (('A', '#0072B2', None), ('B', '#E69F00', None)):
      for dom, ls in (('pinpad', '-'), ('dmc', '--')):
        curves = []
        for t, name, m in diag.get(k, []):
          if not t.startswith(dom) or not len(m[key]):
            continue
          a = m[key]
          idx = np.digitize(a[:, 0], edges) - 1
          curves.append([a[idx == b, 1].mean() if (idx == b).any() else np.nan for b in range(len(x))])
        if not curves:
          continue
        c = np.array(curves)
        with np.errstate(all='ignore'):
          mean = np.nanmean(c, 0)
          n = (~np.isnan(c)).sum(0)
        mean[n < 1] = np.nan
        ax.plot(x, mean, color=color, ls=ls, lw=2.2,
                label=f'{"A" if k == "A" else "B"} {"pin pads" if dom == "pinpad" else "DMC"}')
        early = np.nanmean(mean[2:5])
        summary.append(f'{label:34s} {k} {dom:6s} 0.2-0.5M {early:10.4g}   last {mean[~np.isnan(mean)][-1]:10.4g}')
    ax.set_title(label, fontsize=15)
    ax.set_xlim(0, 4)
    ax.tick_params(labelsize=13)
    ax.grid(True, color='#d5d5d2', lw=0.7)
    for sp in ('top', 'right'):
      ax.spines[sp].set_visible(False)
    if 'eff_cells' in key:
      ax.set_yscale('log')
  for ax in axes[1]:
    ax.set_xlabel('environment steps (M)', fontsize=14)
  h, l = axes[0, 0].get_legend_handles_labels()
  fig.legend(h, l, loc='lower center', ncol=4, fontsize=14, frameon=False)
  fig.tight_layout(rect=(0, 0.05, 1, 1))
  fig.savefig(f'{OUT}/ucb_ab_diag.png', dpi=200, facecolor='white')
  fig.savefig(f'{OUT}/ucb_ab_diag.pdf')
  print('\n'.join(summary))
  print(f'{OUT}/ucb_ab_diag.png')


if __name__ == '__main__':
  main()
