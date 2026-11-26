"""Director paper vs our Director (OG) vs DreamerV3 vs done cap 8 + surprise + US-SOM.

Late window 3.5-4.0M. Director paper = Hafner et al. 2022 curves read from the
vector figures (Fig. 5 pin pad, Fig. J.1 control suite), mean +- 1 sd over 5 seeds.
  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/compare_paper.py
"""
import csv, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import TASKS, runs, scores, binned, style, XMAX, BW, perm_p, FULL  # noqa
from compare_matched import SETS, MAXRET  # noqa
from figs import figdir  # noqa

PUB = '/apps/unit/DoyaU/vasilache/apps/code/26_04_HRL-paper/figures/motivation/director_published'
PUBNAME = {'pinpad_four': 'Pin Pad Four', 'pinpad_five': 'Pin Pad Five', 'pinpad_six': 'Pin Pad Six',
           'dmc_cartpole_swingup': 'Cartpole Swingup', 'dmc_cheetah_run': 'Cheetah Run', 'dmc_hopper_hop': 'Hopper Hop'}
LO, HI = 3.5e6, 4.0e6
arms = [a for a in SETS['ussom'] if a[0] in ('og', 'dv3', 'us')]
arms = [a for k in ('og', 'dv3', 'us') for a in arms if a[0] == k]

def paper(task):
  f = 'fig5_pinpad.csv' if task.startswith('pinpad') else 'figJ1_control_suite.csv'
  r = [row for row in csv.DictReader(open(f'{PUB}/{f}')) if row['figure_panel'] == PUBNAME[task]]
  return np.array([[float(x['step']), float(x['mean']), float(x['lo']), float(x['hi'])] for x in r])

fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
x = (np.arange(0, XMAX, BW) + BW / 2) / 1e6
rows, norm = [], {k: [] for k in ['paper'] + [a[0] for a in arms]}
for ax, (task, title, _) in zip(axes.ravel(), TASKS):
  p = paper(task); g = np.linspace(LO, HI, 11)
  pm, psd = np.interp(g, p[:, 0], p[:, 1]).mean(), np.interp(g, p[:, 0], (p[:, 3] - p[:, 2]) / 2).mean()
  ax.fill_between(p[:, 0] / 1e6, p[:, 2], p[:, 3], color='#56B4E9', lw=0, alpha=0.18)
  ax.plot(p[:, 0] / 1e6, p[:, 1], color='#56B4E9', lw=2.4, ls='-.', label='Director (paper)')
  cells = [f'{pm:.0f} ± {psd:.0f}']; norm['paper'].append(pm / MAXRET[task]); ref = None
  for key, label, color, pat in arms:
    seeds = [a for a in (scores(d) for d in runs(pat.format(t=task))) if len(a) and a[:, 0].max() >= FULL]
    if not seeds: cells.append('-'); continue
    st = np.stack([binned(a) for a in seeds])
    with np.errstate(all='ignore'):
      m = np.nanmean(st, 0); sd = np.nanstd(st, 0, ddof=1)
    ax.fill_between(x, m - sd, m + sd, color=color, lw=0, alpha=0.13)
    ax.plot(x, m, color=color, lw=2.4, ls={'og': ':', 'dv3': '--'}.get(key, '-'), label=label)
    w = np.array([a[(a[:, 0] >= LO) & (a[:, 0] < HI), 1].mean() for a in seeds])
    norm[key].append(w.mean() / MAXRET[task])
    if key == 'og': ref = w
    pv = '' if key == 'og' else f', p={perm_p(ref / MAXRET[task], w / MAXRET[task]):.2f}'
    cells.append(f'{w.mean():.0f} ± {w.std(ddof=1):.0f} (n={len(w)}{pv})')
  ax.axvspan(LO / 1e6, HI / 1e6, color='#d5d5d2', alpha=0.35, lw=0)
  ax.set_xlim(0, 4); ax.set_ylim(bottom=0); style(ax, title)
  rows.append(f'| {title} | ' + ' | '.join(cells) + ' |')
for ax in axes[:, 0]: ax.set_ylabel('episode return', fontsize=15)
for ax in axes[1]: ax.set_xlabel('env steps (M)', fontsize=15)
h, l = axes[0, 0].get_legend_handles_labels()
names = ['paper'] + [a[0] for a in arms]
l = [f'{lab}  ({100 * np.mean(norm[k]):.0f}% of max)' for lab, k in zip(l, names)]
fig.legend(h, l, loc='lower center', ncol=2, fontsize=14, frameon=False)
fig.tight_layout(rect=(0, 0.11, 1, 1))
out = figdir('misc'); fig.savefig(f'{out}/paper_vs_ours.png', dpi=200); fig.savefig(f'{out}/paper_vs_ours.pdf')
hdr = '| task (3.5-4.0M) | Director (paper) | ' + ' | '.join(a[1] for a in arms) + ' |'
tab = '\n'.join([hdr, '|' + '---|' * (len(arms) + 2)] + rows +
                ['| mean % of max | ' + ' | '.join(f'{100 * np.mean(norm[k]):.0f}%' for k in names) + ' |'])
open(f'{out}/paper_vs_ours_table.md', 'w').write(tab + '\n'); print(tab); print(f'{out}/paper_vs_ours.png')
