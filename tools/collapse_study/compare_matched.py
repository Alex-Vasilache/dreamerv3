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
from compare_late import TASKS, runs, scores, binned, style, OUT, XMAX, BW, perm_p, B, FULL  # noqa: E402
from figs import figdir  # noqa: E402

MAXRET = {'pinpad_four': 411, 'pinpad_five': 352, 'pinpad_six': 292,
          'dmc_cartpole_swingup': 1000, 'dmc_cheetah_run': 1000, 'dmc_hopper_hop': 1000}
WINDOW = 5e5
MIN_REACH = 1e6
SETS = {
    'base3': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('dv3', 'DreamerV3 (flat)', '#000000', 'e12[3-8]?_{t}_dreamerv3-pubmatch_s*'),
        ('ent', 'OG + fixed worker entropy', '#0072B2',
         'e1[89][089]?_{t}_director-director_og-wkr_ent_fixed-pubmatch_s*'),
    ],
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
    'ogv': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('dv3', 'DreamerV3 (flat)', '#000000', 'e12[3-8]?_{t}_dreamerv3-pubmatch_s*'),
        ('ent', 'OG + fixed worker entropy', '#999999',
         'e1[89][089]?_{t}_director-director_og-wkr_ent_fixed-pubmatch_s*'),
        ('V', 'V = OG + fixed ent. + worker v3', '#009E73',
         'e19[1-6]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-pubmatch_s*'),
    ],
    'ucbkde': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('V', 'V (OG + fixed ent. + worker v3)', '#000000',
         'e19[1-6]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-pubmatch_s*'),
        ('som', 'V + SOM', '#D55E00',
         'e20[0-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-pubmatch_s*'),
        ('kde', 'V + SOM + UCB-B full res.', '#0072B2',
         'e26[3-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-ucb_imag_kde-pubmatch_s*'),
    ],
    'ucbtab': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('som', 'V + SOM', '#D55E00',
         'e20[0-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-pubmatch_s*'),
        ('even', 'V + US-SOM', '#009E73',
         'e20[0-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-pubmatch_s*'),
        ('tab', 'V + US-SOM + UCB-B table', '#0072B2',
         'e28[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-ucb_imag_btab-pubmatch_s*'),
    ],
    'fromto': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('som', 'V + SOM', '#D55E00',
         'e20[0-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-pubmatch_s*'),
        ('even', 'V + US-SOM', '#009E73',
         'e20[0-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-pubmatch_s*'),
        ('ft', 'V + US-SOM + TCE', '#0072B2',
         'e62[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-pubmatch_s*'),
    ],
    'tceanneal': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('even', 'V + US-SOM', '#009E73',
         'e20[0-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-pubmatch_s*'),
        ('ft', 'V + US-SOM + TCE', '#0072B2',
         'e62[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-pubmatch_s*'),
        ('fta', 'V + US-SOM + TCE-anneal', '#CC79A7',
         'e70[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-fromto_anneal-pubmatch_s*'),
    ],
    'tcegrow': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('even', 'V + US-SOM', '#009E73',
         'e20[0-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-pubmatch_s*'),
        ('ft', 'V + US-SOM + TCE', '#0072B2',
         'e62[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-pubmatch_s*'),
        ('ftg', 'V + US-SOM + TCE-grow', '#CC79A7',
         'e80[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_grow-pubmatch_s*'),
    ],
    'somtce': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('som', 'V + SOM', '#D55E00',
         'e20[0-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-pubmatch_s*'),
        ('ft', 'V + US-SOM + TCE', '#0072B2',
         'e62[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-pubmatch_s*'),
        ('fts', 'V + SOM + TCE', '#CC79A7',
         'e86[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-fromto_dir-pubmatch_s*'),
    ],
    'hash16': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('even', 'V + US-SOM', '#009E73',
         'e20[0-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-pubmatch_s*'),
        ('ft', 'V + US-SOM + TCE (codebook start)', '#0072B2',
         'e62[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-pubmatch_s*'),
        ('hash', 'V + US-SOM + TCE (1x16-bit WM hash start)', '#CC79A7',
         'e95[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash-pubmatch_s*'),
    ],
    'hash4e': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('ft', 'TCE (codebook start)', '#0072B2',
         'e62[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-pubmatch_s*'),
        ('hash', 'TCE, 1x16 hash', '#009E73',
         'e95[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash-pubmatch_s*'),
        ('h4e', 'TCE, 4x12 hash, smoothed w (US-SOM)', '#CC79A7',
         'h00[0-3]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash4e-pubmatch_s*'),
        ('h4es', 'TCE, 4x12 hash, smoothed w (plain SOM)', '#D55E00',
         'h00[3-7]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-fromto_dir-tce_hash4e-pubmatch_s*'),
    ],
    'weekend': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('ft', 'V + US-SOM + TCE', '#0072B2',
         'e62[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-pubmatch_s*'),
        ('hash', 'TCE, world-model hash start', '#009E73',
         'e95[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash-pubmatch_s*'),
        ('hash4', 'TCE, 4x12-bit hash, w=2', '#CC79A7',
         'e1[09]0[0-3]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash4-pubmatch_s*'),
        ('hash4a', 'TCE, 4x12-bit hash, adaptive w', '#000000',
         'e191[0-3]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash4a-pubmatch_s*'),
        ('noexpl', 'TCE w=2, Director expl reward off', '#D55E00',
         'e96[3-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-no_gae_expl-tce_wfix2-pubmatch_s*'),
    ],
    'final4x12': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('hash', 'TCE, 1x16 hash, adaptive w', '#009E73',
         'e95[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash-pubmatch_s*'),
        ('hash4a', 'TCE, 4x12 hash, adaptive w', '#000000',
         'e191[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash4a-pubmatch_s*'),
        ('h4e', 'TCE, 4x12 hash, smoothed w (US-SOM)', '#CC79A7',
         'h00[0-3]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash4e-pubmatch_s*'),
        ('h4es', 'TCE, 4x12 hash, smoothed w (plain SOM)', '#D55E00',
         'h00[3-7]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-fromto_dir-tce_hash4e-pubmatch_s*'),
    ],
    'tcekl': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('hash', 'TCE sqrt bonus, adaptive w (1x16)', '#009E73',
         'e95[0-2]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash-pubmatch_s*'),
        ('kl1', 'TCE-KL c=1', '#999999',
         'h00[7-9]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_kl-pubmatch_s*'),
        ('kl10', 'TCE-KL c=10', '#0072B2',
         'h01??_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_kl-tce_kl_c10-pubmatch_s*'),
        ('kl30', 'TCE-KL c=30', '#D55E00',
         'h01??_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_kl-tce_kl_c30-pubmatch_s*'),
    ],
    'someven': [
        ('og', 'Director (OG)', '#E69F00', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('V', 'V (OG + fixed ent. + worker v3)', '#000000',
         'e19[1-6]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-pubmatch_s*'),
        ('som', 'V + SOM', '#D55E00',
         'e20[0-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-pubmatch_s*'),
        ('even', 'V + US-SOM', '#0072B2',
         'e20[0-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-pubmatch_s*'),
    ],
}


def window_mean(a, lo, hi):
  m = (a[:, 0] >= lo) & (a[:, 0] < hi)
  return a[m, 1].mean() if m.sum() > 3 else np.nan


# Line style as a second cue besides colour (reference arms).
STYLE = {('someven', 'V'): '--', ('someven', 'og'): ':', ('ucbkde', 'V'): '--', ('ucbkde', 'og'): ':', ('ucbtab', 'og'): ':', ('fromto', 'og'): ':', ('tceanneal', 'og'): ':', ('tceanneal', 'fta'): '-.', ('tcegrow', 'og'): ':', ('tcegrow', 'ftg'): '-.', ('somtce', 'og'): ':', ('somtce', 'som'): '--', ('somtce', 'fts'): '-.', ('weekend', 'og'): ':', ('weekend', 'hash'): '-.', ('weekend', 'noexpl'): '--', ('hash4e', 'og'): ':', ('hash4e', 'hash'): '--', ('hash4e', 'h4es'): '-.', ('hash16', 'og'): ':', ('hash16', 'even'): '--', ('hash16', 'hash'): '-.', ('weekend', 'hash4'): (0, (5, 1, 1, 1)), ('final4x12', 'og'): ':', ('final4x12', 'hash'): '--', ('final4x12', 'h4es'): '-.', ('tcekl', 'og'): ':', ('tcekl', 'hash'): '--', ('tcekl', 'kl1'): '-.'}
LATE = {'final4x12': 1e6, 'base3': 1e6, 'someven': 1e6, 'ogv': 1e6}   # window length for sets whose runs are all finished


def main(which):
  arms = SETS[which]
  window = LATE.get(which, WINDOW)
  paths = {(k, t): runs(p.format(t=t)) for k, _, _, p in arms for t, _, _ in TASKS}
  data = {key: [scores(d) for d in v] for key, v in paths.items()}
  # A stopped arm (every run archived in the bucket, none reached 4M) is drawn but does
  # not set the shared window; otherwise one cancelled arm drags every arm back to its last step.
  stopped = {key for key, v in paths.items()
             if v and all(d.startswith(B) for d in v) and max(len(a) and a[:, 0].max() for a in data[key]) < FULL}
  data = {key: [a for a in v if len(a)] for key, v in data.items()}
  fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
  x = (np.arange(0, XMAX, BW) + BW / 2) / 1e6
  norm = {k: {} for k, *_ in arms}          # arm -> task -> list of normalized seed scores
  rows = []
  for ax, (task, title, _) in zip(axes.ravel(), TASKS):
    # An arm counts on a task once its most advanced run passes MIN_REACH;
    # earlier than that it would drag the shared window back to the first steps.
    reach = {k: min(max(a[:, 0].max() for a in data[k, task]), XMAX) for k, *_ in arms if data[k, task]}
    counted = {k for k, r in reach.items() if r >= MIN_REACH and (k, task) not in stopped}
    hi = min(reach[k] for k in counted); lo = hi - window
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
        ls = STYLE.get((which, key), '-')
        ax.plot(x, m, color=color, lw=2.4, ls=ls, label=label)
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
      labels.append(f'{label}\nno data in a scored window'); continue
    a = np.mean([np.mean(norm[key][t]) for t in tasks])
    b = np.mean([np.mean(norm['og'][t]) for t in tasks])
    ntask = '' if len(tasks) == len(TASKS) else f', {len(tasks)} tasks'
    ns = [len(data[key, t]) for t, *_ in TASKS if data[key, t]]
    label = f'{label} ({min(ns)}-{max(ns)} seeds)' if min(ns) != max(ns) else f'{label} ({ns[0]} seed{"s" * (ns[0] > 1)})'
    labels.append(f'{label}\n{100 * a:.0f}%' if key == 'og' else
                  f'{label}\n{100 * a:.0f}% ({100 * (a - b) / b:+.0f}% vs OG{ntask})')
  order = [l.index(lab) for _, lab, *_ in arms if lab in l]
  fig.legend([h[i] for i in order], labels, loc='lower center', ncol=3, fontsize=12.5,
             frameon=False, bbox_to_anchor=(0.5, 0.0))
  fig.text(0.5, 0.165, 'environment steps (M)' + (
      '   (shaded: scoring window, ending where the least-advanced live arm is)' if which not in LATE else
      '   (scores: mean over 3-4M, % of max achievable)'), ha='center', fontsize=14)
  fig.tight_layout(rect=(0, 0.19, 1, 1), h_pad=2.0)
  out = f'{figdir(which)}/{which}_matched'
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
