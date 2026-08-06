"""Hand-rolled SVG line plot (no matplotlib -- see ../goal_code_struct_corr/
make_figure.py for why: this cluster's rl_env has a broken matplotlib
install, missing system libpng15 on both login and compute nodes).

Reads one ``results/{tag}_{short_task}_seed{N}.json`` per (task, seed)
baseline (written by ``extract_worker_success.py``), groups by task, and for
each of the 4 tasks plots the MEAN ``train/wkr_goal_rew`` curve +/- 1 std
shaded band across seeds. Each seed logs at a slightly different step
cadence, so every seed's curve is first linearly interpolated onto a common
step grid per task (0 to the shortest seed's final step) before averaging --
this is what makes "mean and std across seeds" well-defined pointwise.

Usage:
  python3 make_figure.py     # reads ./results/*.json, writes
                              # ./worker_success.{svg,pdf} and overwrites the
                              # copy in code/26_04_HRL-paper/figures/motivation/
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from baselines_common import TASK_COLORS, TASK_LABELS, TASKS  # noqa: E402

SHORT_TASK = {
    'dmc_cartpole_swingup': 'cartpole',
    'dmc_hopper_hop': 'hopper',
    'dmc_acrobot_swingup': 'acrobot',
    'dmc_cheetah_run': 'cheetah',
}
TASK_OF_SHORT = {v: k for k, v in SHORT_TASK.items()}

ap = argparse.ArgumentParser()
ap.add_argument('--results_dir', default=str(HERE / 'results'))
ap.add_argument('--out_svg', default=str(HERE / 'worker_success.svg'))
ap.add_argument('--out_pdf', default=str(HERE / 'worker_success.pdf'))
ap.add_argument('--paper_pdf', default=str(
    HERE.parent.parent.parent / '26_04_HRL-paper' / 'figures' / 'motivation'
    / 'worker_success.pdf'),
    help='also overwrite this copy (the one the paper actually includes); pass "" to skip')
ap.add_argument('--n_grid', type=int, default=200,
                 help='number of points on the common step grid used for cross-seed averaging')
args = ap.parse_args()
base = pathlib.Path(args.results_dir)

by_task = {t: [] for t in TASKS}
for p in sorted(base.glob('*.json')):
  if p.name == 'summary.json':
    continue
  parts = p.stem.split('_')
  if len(parts) < 3 or not parts[-1].startswith('seed'):
    continue
  short = parts[-2]
  task = TASK_OF_SHORT.get(short)
  if task is None:
    continue
  by_task[task].append(p)

missing = [t for t in TASKS if not by_task[t]]
if missing:
  raise SystemExit(
      f'no results/*.json found for: {missing} -- run extract_worker_success.py first')

TEXTWIDTH_PT = 397.0
INCLUDE_FRAC = 0.95
TARGET_PT = INCLUDE_FRAC * TEXTWIDTH_PT

RES_SCALE = 6
PANEL_W = 300 * RES_SCALE
PANEL_H = 180 * RES_SCALE
PAD_L, PAD_R = 55 * RES_SCALE, 15 * RES_SCALE
FIG_W = PANEL_W + PAD_L + PAD_R
RATIO = FIG_W / TARGET_PT


def pt(final_pt):
  return final_pt * RATIO


F_TITLE = pt(9.0)
F_LEGEND = pt(7.3)
F_TICK = pt(7.5)
F_AXIS = pt(7.3)
TICK_LEN = pt(4.5)

PAD_T = F_TITLE * 1.3 + F_LEGEND * 1.9
PAD_B = TICK_LEN + F_TICK * 1.6 + F_AXIS * 2.2
FIG_H = PANEL_H + PAD_T + PAD_B

ymin, ymax = 0.0, 1.0

# Per-task: load every seed, interpolate onto a common grid spanning
# [0, min over seeds of that seed's final step] (the overlap all seeds cover,
# so the band is defined everywhere it's drawn -- no extrapolation past a
# shorter seed's run), then compute the pointwise mean/std across seeds.
task_curves = {}
for task in TASKS:
  seed_series = []
  for p in by_task[task]:
    with open(p) as f:
      d = json.load(f)
    seed_series.append((np.array(d['steps']), np.array(d['values']), d['seed']))
  xmax_common = min(s[0][-1] for s in seed_series)
  grid = np.linspace(0, xmax_common, args.n_grid)
  interped = np.stack([np.interp(grid, s, v) for s, v, _ in seed_series], axis=0)
  mean = interped.mean(axis=0)
  std = interped.std(axis=0) if len(seed_series) > 1 else np.zeros_like(mean)
  task_curves[task] = dict(grid=grid, mean=mean, std=std, n_seeds=len(seed_series),
                            final_mean=float(mean[-1]))

xmax_step = max(c['grid'][-1] for c in task_curves.values())
xmin, xmax = 0, xmax_step


def sx(v):
  return PAD_L + (v - xmin) / (xmax - xmin) * PANEL_W


def sy(v):
  return PAD_T + (1 - (v - ymin) / (ymax - ymin)) * PANEL_H


ax0, ay0 = sx(xmin), sy(ymin)
ax1, ay1 = sx(xmax), sy(ymax)

parts = []
parts.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{FIG_W:.0f}" height="{FIG_H:.0f}" '
    f'viewBox="0 0 {FIG_W:.0f} {FIG_H:.0f}" font-family="Helvetica,Arial,sans-serif">')
parts.append(f'<rect x="0" y="0" width="{FIG_W:.0f}" height="{FIG_H:.0f}" fill="white"/>')

for t in (0.0, 0.25, 0.5, 0.75, 1.0):
  gy = sy(t)
  parts.append(f'<line x1="{ax0:.1f}" y1="{gy:.1f}" x2="{ax1:.1f}" y2="{gy:.1f}" '
               f'stroke="#ddd" stroke-width="{pt(0.5):.1f}"/>')
  parts.append(f'<line x1="{ax0-TICK_LEN:.1f}" y1="{gy:.1f}" x2="{ax0:.1f}" y2="{gy:.1f}" '
               f'stroke="black" stroke-width="{pt(1.8):.1f}"/>')
  parts.append(f'<text x="{ax0-TICK_LEN-F_TICK*0.5:.1f}" y="{gy+F_TICK*0.35:.1f}" '
               f'font-size="{F_TICK:.1f}" font-weight="bold" text-anchor="end" '
               f'fill="black">{t:g}</text>')

parts.append(f'<line x1="{ax0:.1f}" y1="{sy(1.0):.1f}" x2="{ax1:.1f}" y2="{sy(1.0):.1f}" '
             f'stroke="#888" stroke-width="{pt(1.0):.1f}" '
             f'stroke-dasharray="{pt(3.5):.1f},{pt(2.5):.1f}"/>')

step_ticks = [s for s in (0, 1_000_000, 2_000_000, 3_000_000, 4_000_000) if s <= xmax]
tick_text_y = ay0 + TICK_LEN + F_TICK * 1.3
for s in step_ticks:
  gx = sx(s)
  parts.append(f'<line x1="{gx:.1f}" y1="{ay0:.1f}" x2="{gx:.1f}" y2="{ay0+TICK_LEN:.1f}" '
               f'stroke="black" stroke-width="{pt(1.8):.1f}"/>')
  label = '0' if s == 0 else f'{s // 1_000_000}M'
  parts.append(f'<text x="{gx:.1f}" y="{tick_text_y:.1f}" font-size="{F_TICK:.1f}" '
               f'font-weight="bold" text-anchor="middle" fill="black">{label}</text>')

legend_x = ax0 + F_LEGEND
legend_y = F_TITLE * 1.15 + F_LEGEND * 1.7
for i, task in enumerate(TASKS):
  color = TASK_COLORS[task]
  c = task_curves[task]
  grid, mean, std = c['grid'], c['mean'], c['std']

  upper = list(zip(grid, mean + std))
  lower = list(zip(grid, mean - std))
  band = upper + lower[::-1]
  poly = ' '.join(f'{sx(x):.2f},{sy(max(ymin, min(ymax, y))):.2f}' for x, y in band)
  parts.append(f'<polygon points="{poly}" fill="{color}" fill-opacity="0.20" stroke="none"/>')

  poly_mean = ' '.join(f'{sx(x):.2f},{sy(y):.2f}' for x, y in zip(grid, mean))
  parts.append(f'<polyline points="{poly_mean}" fill="none" stroke="{color}" '
               f'stroke-width="{pt(1.6):.2f}" stroke-linejoin="round" '
               f'stroke-linecap="round" opacity="0.95"/>')

  lx = legend_x + (i % 2) * (PANEL_W / 2)
  ly = legend_y + (i // 2) * (F_LEGEND * 1.7)
  parts.append(f'<line x1="{lx:.1f}" y1="{ly - F_LEGEND*0.35:.1f}" '
               f'x2="{lx+F_LEGEND*1.8:.1f}" y2="{ly - F_LEGEND*0.35:.1f}" '
               f'stroke="{color}" stroke-width="{pt(1.8):.1f}"/>')
  label = TASK_LABELS[task]
  parts.append(f'<text x="{lx+F_LEGEND*2.2:.1f}" y="{ly:.1f}" font-size="{F_LEGEND:.1f}" '
               f'text-anchor="start" fill="#333">{label} '
               f'(n={c["n_seeds"]}, final {c["final_mean"]:.2f})</text>')

parts.append(f'<text x="{PAD_L+PANEL_W/2:.1f}" y="{F_TITLE*1.15:.1f}" font-size="{F_TITLE:.1f}" '
             f'font-weight="bold" text-anchor="middle" fill="black">'
             f'Worker goal reward (cosine_max) across training, mean &#177; std over seeds</text>')
parts.append(f'<text x="{PAD_L+PANEL_W/2:.1f}" y="{tick_text_y+F_AXIS*1.7:.1f}" '
             f'font-size="{F_AXIS:.1f}" text-anchor="middle" fill="#333">env steps</text>')

cy = PAD_T + PANEL_H / 2
parts.append(
    f'<text x="{F_AXIS*1.3:.1f}" y="{cy:.1f}" font-size="{F_AXIS:.1f}" text-anchor="middle" '
    f'fill="#333" transform="rotate(-90 {F_AXIS*1.3:.1f} {cy:.1f})">worker goal reward</text>')

parts.append('</svg>')

with open(args.out_svg, 'w') as f:
  f.write('\n'.join(parts))
print('wrote', args.out_svg, '| FIG_W', FIG_W, '| FIG_H', round(FIG_H))
for task in TASKS:
  c = task_curves[task]
  print(f'  {task}: n_seeds={c["n_seeds"]} final_mean={c["final_mean"]:.3f}')

subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', args.out_pdf, args.out_svg], check=True)
print('wrote', args.out_pdf)
if args.paper_pdf:
  pathlib.Path(args.paper_pdf).parent.mkdir(parents=True, exist_ok=True)
  pathlib.Path(args.paper_pdf).write_bytes(pathlib.Path(args.out_pdf).read_bytes())
  print('copied to', args.paper_pdf, '-- rebuild the paper (pdflatex x2) to see it')
