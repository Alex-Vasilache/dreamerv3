"""Return curves + normalized scores for arms that are still running (matched steps).

Finished-run comparisons (compare_late.py, make_arm_curves.py) score the 3-4M
window. While some arms are mid-run that would compare different stages of
training, so here every arm is scored over the SAME window per task: the 0.5M
env steps ending at the step the least-advanced arm with data has reached on
that task (capped at 4M). An arm with no data at that step is left out of the
task. Scores are normalized by the maximum achievable return (DMC 1000; pin pad
411 / 352 / 292, shortest-path policy), averaged over its seeds, then over tasks.
The legend gives each arm's overall normalized score and its change relative to
OG on the same tasks and windows.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/compare_matched.py [set]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import TASKS, runs, scores, binned, style, OUT, XMAX, BW, perm_p  # noqa: E402

MAXRET = {'pinpad_four': 411, 'pinpad_five': 352, 'pinpad_six': 292,
          'dmc_cartpole_swingup': 1000, 'dmc_cheetah_run': 1000, 'dmc_hopper_hop': 1000}
WINDOW = 5e5
MIN_REACH = 1e6
SETS = {
    'ucbimag': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('som', 'OG + SOM', '#CC79A7', 'e1[34]??_{t}_director-director_og-goal_som_line-pubmatch_s*'),
        ('env', 'OG + SOM + UCB (env)', '#000000', 'e14[3-5]?_{t}_director-director_og-goal_som_line-mgr_ucb-pubmatch_s*'),
        ('A', 'OG + SOM + UCB imag A', '#0072B2', 'e18[2-4]?_{t}_director-director_og-goal_som_line-ucb_imag_a-ucb_imag_c3-pubmatch_s*'),
        ('B', 'OG + SOM + UCB imag B', '#56B4E9', 'e18[5-7]?_{t}_director-director_og-goal_som_line-ucb_imag_b-ucb_imag_c3-pubmatch_s*'),
    ],
    'wkrv3': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('dv3', 'DreamerV3 (flat)', '#000000', 'e12[3-8]?_{t}_dreamerv3-pubmatch_s*'),
        ('ent', 'OG + fixed worker entropy', '#999999',
         'e1[89][089]?_{t}_director-director_og-wkr_ent_fixed-pubmatch_s*'),
        ('R', '+ worker v3 return norm (R)', '#0072B2',
         'e19[1-6]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3_retnorm-pubmatch_s*'),
        ('V', '+ worker v3, all (V)', '#009E73',
         'e19[1-6]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-pubmatch_s*'),
    ],
}


def window_mean(a, lo, hi):
  m = (a[:, 0] >= lo) & (a[:, 0] < hi)
  return a[m, 1].mean() if m.sum() > 3 else np.nan


def main(which):
  arms = SETS[which]
  data = {(k, t): [scores(d) for d in runs(p.format(t=t))] for k, _, _, p in arms for t, _, _ in TASKS}
  data = {key: [a for a in v if len(a)] for key, v in data.items()}
  fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
  x = (np.arange(0, XMAX, BW) + BW / 2) / 1e6
  norm = {k: {} for k, *_ in arms}          # arm -> task -> list of normalized seed scores
  rows = []
  for ax, (task, title, _) in zip(axes.ravel(), TASKS):
    # An arm counts on a task once its most advanced run passes MIN_REACH;
    # earlier than that it would drag the shared window back to the first steps.
    reach = {k: min(max(a[:, 0].max() for a in data[k, task]), XMAX) for k, *_ in arms if data[k, task]}
    counted = {k for k, r in reach.items() if r >= MIN_REACH}
    hi = min(reach[k] for k in counted); lo = hi - WINDOW
    cells = []
    for key, label, color, _ in arms:
      seeds = data[key, task]
      if seeds:
        st = np.stack([binned(a) for a in seeds])
        with np.errstate(all='ignore'):
          n_ok = (~np.isnan(st)).sum(0)
          m = np.where(n_ok >= 1, np.nanmean(st, 0), np.nan)
          sd = np.where(n_ok >= 2, np.nanstd(st, 0, ddof=1), np.nan)
        ax.fill_between(x, m - sd, m + sd, color=color, lw=0, alpha=0.13)
        ax.plot(x, m, color=color, lw=2.4, ls='-' if len(seeds) >= 2 else '--', label=label)
      w = [window_mean(a, lo, hi) for a in seeds] if key in counted else []
      w = [v for v in w if not np.isnan(v)]
      if w:
        norm[key][task] = [v / MAXRET[task] for v in w]
        p = '' if key == 'og' or len(w) < 2 else f', p={perm_p(norm["og"][task], norm[key][task]):.2f}'
        cells.append(f'{np.mean(w):.0f} ± {np.std(w, ddof=1) if len(w) > 1 else 0:.0f} (n={len(w)}{p})')
      else:
        cells.append('-')
    ax.axvspan(lo / 1e6, hi / 1e6, color='#d5d5d2', alpha=0.35, lw=0)
    ax.set_xlim(0, 4); ax.set_ylim(bottom=0)
    style(ax, title)
    rows.append(f'| {title} | {lo / 1e6:.1f}-{hi / 1e6:.1f}M | ' + ' | '.join(cells) + ' |')
  for ax in axes[:, 0]:
    ax.set_ylabel('episode return', fontsize=15)
  # Legend: overall normalized score and change vs OG on the same tasks/windows.
  h, l = axes[0, 0].get_legend_handles_labels()
  labels = []
  for key, label, *_ in arms:
    tasks = [t for t in norm[key] if t in norm['og']]
    if not tasks:
      labels.append(f'{label}\nno data yet'); continue
    a = np.mean([np.mean(norm[key][t]) for t in tasks])
    b = np.mean([np.mean(norm['og'][t]) for t in tasks])
    ntask = '' if len(tasks) == len(TASKS) else f', {len(tasks)} tasks'
    labels.append(f'{label}\n{100 * a:.0f}%' if key == 'og' else
                  f'{label}\n{100 * a:.0f}% ({100 * (a - b) / b:+.0f}% vs OG{ntask})')
  order = [l.index(lab) for _, lab, *_ in arms if lab in l]
  fig.legend([h[i] for i in order], labels, loc='lower center', ncol=3, fontsize=12.5,
             frameon=False, bbox_to_anchor=(0.5, 0.0))
  fig.text(0.5, 0.165, 'environment steps (M)   (shaded: scoring window; dashed: one seed so far)',
           ha='center', fontsize=14)
  fig.tight_layout(rect=(0, 0.19, 1, 1), h_pad=2.0)
  out = f'{OUT}/{which}_matched'
  fig.savefig(out + '.png', dpi=200, facecolor='white'); fig.savefig(out + '.pdf')
  head = '| task | window | ' + ' | '.join(label for _, label, *_ in arms) + ' |'
  table = '\n'.join([head, '|---' * (len(arms) + 2) + '|'] + rows)
  open(out + '_table.md', 'w').write(table + '\n')
  print(out + '.png'); print(table)
  for key, label, *_ in arms:
    if norm[key]:
      print(f'{label:32s} ' + '  '.join(f'{t[:10]} {100 * np.mean(v):.0f}%' for t, v in norm[key].items()))


if __name__ == '__main__':
  main(sys.argv[1] if len(sys.argv) > 1 else 'wkrv3')
