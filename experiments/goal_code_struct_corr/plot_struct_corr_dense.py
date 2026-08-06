"""Plots the fully-logged in-training goal-code/goal-space hard-code
correlation trajectory for seed 4 (e406-e409), the one baseline run with
``goal_struct_diag: true`` set for its entire training -- see
extract_struct_corr_dense.py. One line per environment, densely sampled
(~820 points over 4M steps) rather than the handful of snapshot points used
in struct_corr_vs_step.py for seeds 2-3.

A light rolling-mean smoothing is applied (raw step-to-step std is
comparable to the training-long trend itself) purely for display -- the
underlying committed JSON keeps the raw per-step values.

Usage:
  python3 plot_struct_corr_dense.py     # reads ./results/*_seed4_dense.json,
                                         # writes ./struct_corr_dense.{svg,pdf}
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

ap = argparse.ArgumentParser()
ap.add_argument('--results_dir', default=str(HERE / 'results'))
ap.add_argument('--out_svg', default=str(HERE / 'struct_corr_dense.svg'))
ap.add_argument('--out_pdf', default=str(HERE / 'struct_corr_dense.pdf'))
ap.add_argument('--smooth_window', type=int, default=21)
args = ap.parse_args()
base = pathlib.Path(args.results_dir)


def smooth(y, w):
  if w <= 1 or len(y) < w:
    return y
  kernel = np.ones(w) / w
  pad = w // 2
  yp = np.pad(y, (pad, pad), mode='edge')
  return np.convolve(yp, kernel, mode='valid')[:len(y)]


series = {}
for p in sorted(base.glob('*_seed4_dense.json')):
  d = json.loads(p.read_text())
  task = d['task']
  steps = np.array(d['steps'])
  hard = np.array(d['corr_hard'])
  series[task] = (steps, smooth(hard, args.smooth_window))

if not series:
  raise SystemExit('no results found -- run extract_struct_corr_dense.py first')

TEXTWIDTH_PT = 397.0
INCLUDE_FRAC = 0.95
TARGET_PT = INCLUDE_FRAC * TEXTWIDTH_PT
RES_SCALE = 6
PANEL_W = 300 * RES_SCALE
PANEL_H = 190 * RES_SCALE
PAD_L, PAD_R = 55 * RES_SCALE, 15 * RES_SCALE
FIG_W = PANEL_W + PAD_L + PAD_R
RATIO = FIG_W / TARGET_PT


def pt(final_pt):
  return final_pt * RATIO


F_TITLE = pt(9.0)
F_LEGEND = pt(7.0)
F_TICK = pt(7.5)
F_AXIS = pt(7.3)
TICK_LEN = pt(4.5)
PAD_T = F_TITLE * 1.3 + F_LEGEND * 1.4
PAD_B = TICK_LEN + F_TICK * 1.6 + F_AXIS * 2.2
FIG_H = PANEL_H + PAD_T + PAD_B

ymin, ymax = 0.0, 1.0
xmax_step = max(s[-1] for s, _ in series.values())
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
legend_y = F_TITLE * 1.15 + F_LEGEND * 1.4
li = 0
for task in TASKS:
  if task not in series:
    continue
  color = TASK_COLORS[task]
  steps, vals = series[task]
  poly = ' '.join(f'{sx(s):.2f},{sy(r):.2f}' for s, r in zip(steps, vals))
  parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" '
               f'stroke-width="{pt(1.8):.2f}" stroke-linejoin="round" '
               f'stroke-linecap="round"/>')
  lx = legend_x + (li % 2) * (PANEL_W / 2)
  ly = legend_y + (li // 2) * (F_LEGEND * 1.6)
  parts.append(f'<line x1="{lx:.1f}" y1="{ly - F_LEGEND*0.35:.1f}" '
               f'x2="{lx+F_LEGEND*1.8:.1f}" y2="{ly - F_LEGEND*0.35:.1f}" '
               f'stroke="{color}" stroke-width="{pt(1.8):.1f}"/>')
  parts.append(f'<text x="{lx+F_LEGEND*2.2:.1f}" y="{ly:.1f}" font-size="{F_LEGEND:.1f}" '
               f'text-anchor="start" fill="#333">{TASK_LABELS[task]}</text>')
  li += 1

parts.append(f'<text x="{PAD_L+PANEL_W/2:.1f}" y="{F_TITLE*1.15:.1f}" font-size="{F_TITLE:.1f}" '
             f'font-weight="bold" text-anchor="middle" fill="black">'
             f'Goal-code/goal-space correlation, densely logged throughout training (seed 4)</text>')
parts.append(f'<text x="{PAD_L+PANEL_W/2:.1f}" y="{tick_text_y+F_AXIS*1.7:.1f}" '
             f'font-size="{F_AXIS:.1f}" text-anchor="middle" fill="#333">env steps</text>')

cy = PAD_T + PANEL_H / 2
parts.append(
    f'<text x="{F_AXIS*1.3:.1f}" y="{cy:.1f}" font-size="{F_AXIS:.1f}" text-anchor="middle" '
    f'fill="#333" transform="rotate(-90 {F_AXIS*1.3:.1f} {cy:.1f})">hard-code Pearson r</text>')

parts.append('</svg>')

with open(args.out_svg, 'w') as f:
  f.write('\n'.join(parts))
print('wrote', args.out_svg)
for task, (steps, vals) in series.items():
  print(f'  {task}: n={len(steps)} first={vals[0]:.3f} last={vals[-1]:.3f} '
        f'min={vals.min():.3f} max={vals.max():.3f}')

subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', args.out_pdf, args.out_svg], check=True)
print('wrote', args.out_pdf)

paper_pdf = (HERE.parent.parent.parent / '26_04_HRL-paper'
             / 'figures' / 'motivation' / 'struct_corr_dense.pdf')
if paper_pdf.parent.exists():
  import shutil
  shutil.copy(args.out_pdf, paper_pdf)
  print('copied to', paper_pdf, '-- rebuild the paper (pdflatex x2) to see it')
