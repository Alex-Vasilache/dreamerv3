"""Same figure as make_figure.py, but for the SOFT (differentiable softmax
probability) goal code instead of the hard (argmax) code the manager actually
samples via REINFORCE.

Reads the same ``results/{tag}_{short_task}_seed{seed}.npz`` files, but plots
``sz_pairs`` (soft-code pairwise cosine_max similarity) against ``sd_pairs``
(goal-space similarity), and reports ``corr_soft_mean``/``corr_soft_std``.

Unlike the hard-code figure, ``diag_goal_struct_corr.py`` does not save a
fixed-bin trend curve for the soft code (only ``trend_x``/``trend_y``, which
are always the HARD-code trend, computed from a separate 4096-state
subsample -- see that script's ``fixed_bin_trend(sd_trend, sh_trend)`` call).
So this script recomputes an equivalent trend from the batch-0 pairwise data
that IS saved for both quantities (``sd_pairs``/``sz_pairs``, ~binom(1024,2)
~524k pairs per seed -- more than enough to fill the same 22 fixed bins).
This means the two figures' trend curves are computed from different-sized
pools (524k pairs from 1 batch vs. ~8.4M pairs from a 4096-state subsample),
but the same fixed bin edges, so they remain visually comparable.

Usage:
  python3 make_figure_soft.py            # reads ./results/*.npz, writes
                                          # ./goal_code_struct_corr_soft.{svg,pdf}
                                          # and overwrites the copy in
                                          # code/26_04_HRL-paper/figures/
"""
import argparse
import pathlib
import re
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from baselines_common import TASK_COLORS, TASK_LABELS, TASKS  # noqa: E402

# Must match diag_goal_struct_corr.py's TREND_XMIN/XMAX/NBINS exactly so this
# figure's bins line up with the hard-code figure's, even though the pool
# sizes differ.
TREND_XMIN, TREND_XMAX, TREND_NBINS = -0.2, 1.05, 22


def fixed_bin_trend(sd, sz, xmin=TREND_XMIN, xmax=TREND_XMAX, nbins=TREND_NBINS):
  edges = np.linspace(xmin, xmax, nbins + 1)
  centers = 0.5 * (edges[:-1] + edges[1:])
  bin_idx = np.clip(np.digitize(sd, edges) - 1, 0, nbins - 1)
  means = np.full(nbins, np.nan)
  for b in range(nbins):
    m = bin_idx == b
    if m.sum() >= 10:
      means[b] = sz[m].mean()
  return centers, means


ap = argparse.ArgumentParser()
ap.add_argument('--results_dir', default=str(HERE / 'results'))
ap.add_argument('--out_svg', default=str(HERE / 'goal_code_struct_corr_soft.svg'))
ap.add_argument('--out_pdf', default=str(HERE / 'goal_code_struct_corr_soft.pdf'))
ap.add_argument('--paper_pdf', default=str(
    HERE.parent.parent.parent / '26_04_HRL-paper' / 'figures' / 'motivation'
    / 'goal_code_struct_corr_soft.pdf'),
    help='also overwrite this copy (the one the paper actually includes); pass "" to skip')
args = ap.parse_args()
base = pathlib.Path(args.results_dir)

SHORT_TASK = {
    'dmc_cartpole_swingup': 'cartpole',
    'dmc_hopper_hop': 'hopper',
    'dmc_acrobot_swingup': 'acrobot',
    'dmc_cheetah_run': 'cheetah',
}
TASK_OF_SHORT = {v: k for k, v in SHORT_TASK.items()}
name_re = re.compile(r'^(e\d+)_([a-z]+)_seed(\d+)$')

by_task = {t: [] for t in TASKS}
for p in sorted(base.glob('*.npz')):
  m = name_re.match(p.stem)
  if not m:
    continue
  short = m.group(2)
  task = TASK_OF_SHORT.get(short)
  if task is None:
    continue
  by_task[task].append(p)

missing = [t for t in TASKS if not by_task[t]]
if missing:
  raise SystemExit(f'no matching results/*.npz found for: {missing}')

TEXTWIDTH_PT = 397.0
INCLUDE_FRAC = 0.95
TARGET_PT = INCLUDE_FRAC * TEXTWIDTH_PT

RES_SCALE = 6
PANEL = 260 * RES_SCALE
GAP = 34 * RES_SCALE
PAD_L, PAD_R = 50 * RES_SCALE, 8 * RES_SCALE
CELL = PANEL + PAD_L + PAD_R
NPANELS = len(TASKS)
FIG_W = NPANELS * CELL + (NPANELS - 1) * GAP
RATIO = FIG_W / TARGET_PT


def pt(final_pt):
  return final_pt * RATIO


F_TITLE = pt(9.0)
F_SUB = pt(7.0)
F_TICK = pt(7.2)
F_AXIS = pt(7.0)
TICK_LEN = pt(4.5)

PAD_T = F_TITLE * 1.3 + F_SUB * 1.9
PAD_B = TICK_LEN + F_TICK * 1.6 + F_AXIS * 2.2
FIG_H = PANEL + PAD_T + PAD_B

xmin, xmax = -0.2, 1.05
ymin, ymax = -0.05, 1.05


def sx(v, x0):
  return x0 + PAD_L + (v - xmin) / (xmax - xmin) * PANEL


def sy(v):
  return PAD_T + (1 - (v - ymin) / (ymax - ymin)) * PANEL


rng = np.random.RandomState(0)
parts_svg = []
parts_svg.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{FIG_W:.0f}" height="{FIG_H:.0f}" '
    f'viewBox="0 0 {FIG_W:.0f} {FIG_H:.0f}" font-family="Helvetica,Arial,sans-serif">')
parts_svg.append(f'<rect x="0" y="0" width="{FIG_W:.0f}" height="{FIG_H:.0f}" fill="white"/>')

TOTAL_SCATTER_PTS = 6000

for i, task in enumerate(TASKS):
  title = TASK_LABELS[task]
  color = TASK_COLORS[task]
  files = by_task[task]
  n_seeds = len(files)
  x0 = i * (CELL + GAP)

  seed_data = [np.load(f) for f in files]
  corr_means = np.array([float(d['corr_soft_mean']) for d in seed_data])
  r_mean = float(corr_means.mean())
  r_std = float(corr_means.std()) if n_seeds > 1 else 0.0

  per_seed_n = max(1, TOTAL_SCATTER_PTS // n_seeds)
  sd_all, sz_all = [], []
  trend_ys = []
  trend_x = None
  for d in seed_data:
    sd, sz = d['sd_pairs'], d['sz_pairs']
    idx = rng.choice(len(sd), min(per_seed_n, len(sd)), replace=False)
    sd_all.append(sd[idx])
    sz_all.append(sz[idx])
    tx, ty = fixed_bin_trend(sd, sz)
    trend_x = tx
    trend_ys.append(ty)
  sd_all = np.concatenate(sd_all)
  sz_all = np.concatenate(sz_all)

  trend_ys = np.stack(trend_ys, axis=0)
  with np.errstate(invalid='ignore'):
    trend_mean = np.nanmean(trend_ys, axis=0)
    trend_std = np.nanstd(trend_ys, axis=0) if n_seeds > 1 else np.zeros_like(trend_mean)
  valid = ~np.isnan(trend_mean)

  ax0, ay0 = sx(xmin, x0), sy(ymin)
  ax1, ay1 = sx(xmax, x0), sy(ymax)

  for t in (0.0, 1.0):
    gx = sx(t, x0)
    parts_svg.append(f'<line x1="{gx:.1f}" y1="{ay1:.1f}" x2="{gx:.1f}" y2="{ay0:.1f}" '
                      f'stroke="#bbb" stroke-width="{pt(0.5):.1f}" stroke-dasharray="{pt(2):.1f},{pt(2):.1f}"/>')
    gy = sy(t)
    parts_svg.append(f'<line x1="{ax0:.1f}" y1="{gy:.1f}" x2="{ax1:.1f}" y2="{gy:.1f}" '
                      f'stroke="#bbb" stroke-width="{pt(0.5):.1f}" stroke-dasharray="{pt(2):.1f},{pt(2):.1f}"/>')

  dlo, dhi = max(xmin, ymin), min(xmax, ymax)
  parts_svg.append(f'<line x1="{sx(dlo, x0):.1f}" y1="{sy(dlo):.1f}" '
                    f'x2="{sx(dhi, x0):.1f}" y2="{sy(dhi):.1f}" '
                    f'stroke="#888" stroke-width="{pt(1.0):.1f}" '
                    f'stroke-dasharray="{pt(3.5):.1f},{pt(2.5):.1f}"/>')

  pts = []
  for xv, yv in zip(sd_all, sz_all):
    px, py = sx(xv, x0), sy(yv)
    pts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{pt(0.8):.2f}" fill="{color}" '
               f'fill-opacity="0.16"/>')
  parts_svg.append(''.join(pts))

  vx = trend_x[valid]
  vmean = trend_mean[valid]
  vstd = trend_std[valid]
  if len(vx) >= 2:
    upper = [(x, m + s) for x, m, s in zip(vx, vmean, vstd)]
    lower = [(x, m - s) for x, m, s in zip(vx, vmean, vstd)]
    band = upper + lower[::-1]
    poly = ' '.join(f'{sx(x, x0):.2f},{sy(y):.2f}' for x, y in band)
    parts_svg.append(f'<polygon points="{poly}" fill="{color}" fill-opacity="0.25" stroke="none"/>')
    poly_mean = ' '.join(f'{sx(x, x0):.2f},{sy(y):.2f}' for x, y in zip(vx, vmean))
    parts_svg.append(f'<polyline points="{poly_mean}" fill="none" stroke="#1A1A1A" '
                      f'stroke-width="{pt(1.6):.2f}" stroke-linejoin="round" '
                      f'stroke-linecap="round"/>')

  tick_text_y = ay0 + TICK_LEN + F_TICK * 1.3
  for t in (0.0, 1.0):
    gx = sx(t, x0)
    parts_svg.append(f'<line x1="{gx:.1f}" y1="{ay0:.1f}" x2="{gx:.1f}" y2="{ay0+TICK_LEN:.1f}" '
                      f'stroke="black" stroke-width="{pt(1.8):.1f}"/>')
    parts_svg.append(f'<text x="{gx:.1f}" y="{tick_text_y:.1f}" font-size="{F_TICK:.1f}" '
                      f'font-weight="bold" text-anchor="middle" fill="black">{t:g}</text>')
    gy = sy(t)
    parts_svg.append(f'<line x1="{ax0-TICK_LEN:.1f}" y1="{gy:.1f}" x2="{ax0:.1f}" y2="{gy:.1f}" '
                      f'stroke="black" stroke-width="{pt(1.8):.1f}"/>')
    if i == 0:
      parts_svg.append(f'<text x="{ax0-TICK_LEN-F_TICK*0.5:.1f}" y="{gy+F_TICK*0.35:.1f}" font-size="{F_TICK:.1f}" '
                        f'font-weight="bold" text-anchor="end" fill="black">{t:g}</text>')

  parts_svg.append(f'<text x="{x0+CELL/2:.1f}" y="{F_TITLE*1.15:.1f}" font-size="{F_TITLE:.1f}" '
                    f'font-weight="bold" text-anchor="middle" fill="black">{title}</text>')
  parts_svg.append(f'<text x="{x0+CELL/2:.1f}" y="{F_TITLE*1.15+F_SUB*1.7:.1f}" font-size="{F_SUB:.1f}" '
                    f'text-anchor="middle" fill="#333">'
                    f'r = {r_mean:.2f} &#177; {r_std:.2f} (n={n_seeds} seeds)</text>')
  parts_svg.append(f'<text x="{x0+CELL/2:.1f}" y="{tick_text_y+F_AXIS*1.7:.1f}" font-size="{F_AXIS:.1f}" '
                    f'text-anchor="middle" fill="#333">goal-space similarity</text>')

cy = PAD_T + PANEL / 2
parts_svg.append(
    f'<text x="{F_AXIS*1.3:.1f}" y="{cy:.1f}" font-size="{F_AXIS:.1f}" text-anchor="middle" '
    f'fill="#333" transform="rotate(-90 {F_AXIS*1.3:.1f} {cy:.1f})">goal-code similarity (soft)</text>')

parts_svg.append('</svg>')

with open(args.out_svg, 'w') as f:
  f.write('\n'.join(parts_svg))
print('wrote', args.out_svg, '| FIG_W', FIG_W, '| FIG_H', round(FIG_H),
      '| target_pt', TARGET_PT, '| ratio', round(RATIO, 3))
for t in TASKS:
  print(f'  {t}: {len(by_task[t])} seed(s) -- {[p.name for p in by_task[t]]}')

subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', args.out_pdf, args.out_svg], check=True)
print('wrote', args.out_pdf)
if args.paper_pdf:
  pathlib.Path(args.paper_pdf).write_bytes(pathlib.Path(args.out_pdf).read_bytes())
  print('copied to', args.paper_pdf, '-- rebuild the paper (pdflatex x2) to see it')
