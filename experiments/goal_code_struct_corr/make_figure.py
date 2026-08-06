"""Hand-rolled SVG scatter plot (no matplotlib -- rl_env's install is broken:
missing libpng15 on this cluster). Converted to PDF afterwards with rsvg-convert.

Reads one ``results/{tag}_{short_task}_seed{seed}.npz`` per (task, seed)
baseline (written by ``../run_diag_goal_struct_corr.sbatch`` via the manifest
from ``submit_diag_goal_struct_corr.sh``), groups by task, and for each of
the 4 tasks (cartpole/hopper/acrobot/cheetah) plots:
  - the pooled scatter cloud (goal-space similarity vs goal-code similarity)
    from all available seeds, evenly subsampled per seed;
  - the MEAN trend curve across seeds, +/- 1 std shaded band, binned on the
    hard code's own discrete Y-axis (Hamming block-overlap similarity only
    takes L+1=9 exact values, k/8 for k=0..8, visible as horizontal stripes
    in the scatter) rather than on the continuous X-axis: for each of the 9
    levels, the conditional mean/std of goal-space similarity is computed
    per seed from that seed's full batch-0 pairwise pool (``sd_pairs``/
    ``sh_pairs``), then averaged across seeds. (Does NOT use
    diag_goal_struct_corr.py's saved ``trend_x``/``trend_y``, which are
    X-binned -- those remain relevant only to a continuous-Y code, e.g. the
    soft code in make_figure_soft.py.);
  - r = mean +/- std of the hard-code Pearson correlation ACROSS SEEDS (each
    seed's own value is already an average over its 10 measurement batches),
    and the number of seeds contributing.

This figure is included at width=INCLUDE_FRAC*\\textwidth in a NeurIPS
single-column paper (\\textwidth ~= 397pt), so the *final* printed width is
TARGET_PT points. Everything drawn here shrinks by TARGET_PT / FIG_W when
LaTeX places it, so font sizes are chosen so that final_pt = font_svg_px *
TARGET_PT / FIG_W lands around normal caption/tick sizes (~9pt title,
~7-7.5pt labels/ticks). RES_SCALE inflates the horizontal coordinate system
(and therefore every stroke/marker/font size in lockstep, since font sizes
are derived from RATIO = FIG_W / TARGET_PT) so the vector geometry is defined
at higher internal resolution without changing the final printed size or
aspect ratio.

Usage:
  python3 make_figure.py                 # reads ./results/*.npz, writes
                                          # ./goal_code_struct_corr.{svg,pdf}
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

ap = argparse.ArgumentParser()
ap.add_argument('--results_dir', default=str(HERE / 'results'))
ap.add_argument('--out_svg', default=str(HERE / 'goal_code_struct_corr.svg'))
ap.add_argument('--out_pdf', default=str(HERE / 'goal_code_struct_corr.pdf'))
ap.add_argument('--paper_pdf', default=str(
    HERE.parent.parent.parent / '26_04_HRL-paper' / 'figures' / 'motivation'
    / 'goal_code_struct_corr.pdf'),
    help='also overwrite this copy (the one the paper actually includes); pass "" to skip')
ap.add_argument('--title_suffix', default='',
                 help='appended to each panel title, e.g. " (~580k steps)" for a '
                      'mid-training snapshot variant of this figure')
ap.add_argument('--step_suffix', default=None,
                 help='select results/{tag}_{short}_seed{N}_step{THIS}.npz snapshot '
                      'files (watch_and_diag_running.py) instead of the default final '
                      '(complete-training) results/{tag}_{short}_seed{N}.npz files')
args = ap.parse_args()
base = pathlib.Path(args.results_dir)

# Group results/*.npz by task from the file name.
SHORT_TASK = {
    'dmc_cartpole_swingup': 'cartpole',
    'dmc_hopper_hop': 'hopper',
    'dmc_acrobot_swingup': 'acrobot',
    'dmc_cheetah_run': 'cheetah',
}
TASK_OF_SHORT = {v: k for k, v in SHORT_TASK.items()}

if args.step_suffix:
  # "{tag}_{short}_seed{N}_step{SUFFIX}.npz" -- mid-training snapshots.
  name_re = re.compile(r'^(e\d+)_([a-z]+)_seed(\d+)_step' + re.escape(args.step_suffix) + r'$')
else:
  # "{tag}_{short}_seed{N}.npz" -- final, complete-training results only
  # (explicitly excludes any "..._step*" snapshot file).
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
  raise SystemExit(
      f'no matching results/*.npz found for: {missing} (step_suffix={args.step_suffix!r}) -- '
      f'run the appropriate diag job(s) first')

TEXTWIDTH_PT = 397.0
INCLUDE_FRAC = 0.95  # 4 panels now, vs 3 before -- slightly wider to keep per-panel size up
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
  corr_means = np.array([float(d['corr_hard_mean']) for d in seed_data])
  # Cross-seed stats: mean/std of each seed's own (within-seed, over-batches)
  # mean correlation -- the seed-to-seed spread is the relevant error bar for
  # "how much does this vary across independently trained baselines".
  r_mean = float(corr_means.mean())
  r_std = float(corr_means.std()) if n_seeds > 1 else 0.0

  # Pooled scatter cloud: same target point budget regardless of n_seeds, so
  # panels stay equally legible as more seeds land.
  per_seed_n = max(1, TOTAL_SCATTER_PTS // n_seeds)
  sd_all, sh_all = [], []
  for d in seed_data:
    sd, sh = d['sd_pairs'], d['sh_pairs']
    idx = rng.choice(len(sd), min(per_seed_n, len(sd)), replace=False)
    sd_all.append(sd[idx])
    sh_all.append(sh[idx])
  sd_all = np.concatenate(sd_all)
  sh_all = np.concatenate(sh_all)

  # Trend binned on Y, not X: the hard code's Hamming-similarity y-axis is
  # not continuous -- with L=8 blocks it only takes L+1=9 exact values
  # (k/8, k=0..8), visible as the horizontal stripes in the scatter cloud
  # below. Binning the continuous x-axis (goal-space similarity) into
  # arbitrary bins and averaging the already-discrete y over each (the
  # previous approach, matching diag_goal_struct_corr.py's saved
  # trend_x/trend_y) fights that structure. Instead, group pairs by their
  # exact y-level and report the conditional mean/std of x (goal-space
  # similarity) per level -- using each seed's full batch-0 pairwise pool
  # (~n_states choose 2, not the scatter's subsample), since with 9 levels
  # every bin has ample support.
  L = int(seed_data[0]['L'])
  levels = np.arange(L + 1) / L
  per_seed_level_means = np.full((n_seeds, L + 1), np.nan)
  for si, d in enumerate(seed_data):
    sd, sh = d['sd_pairs'], d['sh_pairs']
    level_idx = np.rint(sh * L).astype(int)
    for k in range(L + 1):
      m = level_idx == k
      if m.sum() >= 10:
        per_seed_level_means[si, k] = sd[m].mean()
  with np.errstate(invalid='ignore'):
    trend_mean = np.nanmean(per_seed_level_means, axis=0)
    trend_std = (np.nanstd(per_seed_level_means, axis=0) if n_seeds > 1
                 else np.zeros_like(trend_mean))
  trend_x = levels  # now holds the (discrete) Y positions of the trend, despite the name
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
  for xv, yv in zip(sd_all, sh_all):
    px, py = sx(xv, x0), sy(yv)
    pts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{pt(0.8):.2f}" fill="{color}" '
               f'fill-opacity="0.16"/>')
  parts_svg.append(''.join(pts))

  # Shaded +/-1 std band (only meaningful once n_seeds > 1; degenerates to a
  # zero-width band -- effectively invisible -- for a single seed). Binned on
  # Y (the discrete code-similarity level, ascending), so the band/curve run
  # horizontally: at each level, the mean/std are of X (goal-space
  # similarity), not Y.
  vy = trend_x[valid]      # discrete levels (0, 1/8, ..., 1), still ascending
  vmean = trend_mean[valid]  # mean goal-space similarity at that level
  vstd = trend_std[valid]
  if len(vy) >= 2:
    upper = [(m + s, y) for y, m, s in zip(vy, vmean, vstd)]
    lower = [(m - s, y) for y, m, s in zip(vy, vmean, vstd)]
    band = upper + lower[::-1]
    poly = ' '.join(f'{sx(x, x0):.2f},{sy(y):.2f}' for x, y in band)
    parts_svg.append(f'<polygon points="{poly}" fill="{color}" fill-opacity="0.25" stroke="none"/>')
    poly_mean = ' '.join(f'{sx(x, x0):.2f},{sy(y):.2f}' for x, y in zip(vmean, vy))
    parts_svg.append(f'<polyline points="{poly_mean}" fill="none" stroke="#1A1A1A" '
                      f'stroke-width="{pt(1.6):.2f}" stroke-linejoin="round" '
                      f'stroke-linecap="round"/>')
    # Markers at each discrete level -- only 9 possible values (L+1), so the
    # trend is genuinely a handful of points, not a dense curve.
    for x, y in zip(vmean, vy):
      parts_svg.append(f'<circle cx="{sx(x, x0):.2f}" cy="{sy(y):.2f}" r="{pt(1.8):.2f}" '
                        f'fill="#1A1A1A" stroke="white" stroke-width="{pt(0.4):.2f}"/>')

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
                    f'r = {r_mean:.2f} &#177; {r_std:.2f} (n={n_seeds} seeds){args.title_suffix}</text>')
  parts_svg.append(f'<text x="{x0+CELL/2:.1f}" y="{tick_text_y+F_AXIS*1.7:.1f}" font-size="{F_AXIS:.1f}" '
                    f'text-anchor="middle" fill="#333">goal-space similarity</text>')

cy = PAD_T + PANEL / 2
parts_svg.append(
    f'<text x="{F_AXIS*1.3:.1f}" y="{cy:.1f}" font-size="{F_AXIS:.1f}" text-anchor="middle" '
    f'fill="#333" transform="rotate(-90 {F_AXIS*1.3:.1f} {cy:.1f})">goal-code similarity</text>')

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
