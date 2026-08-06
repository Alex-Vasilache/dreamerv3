"""Plots hard-code goal-code/goal-space Pearson r vs. training step, using:
  - mid-training snapshots from watch_and_diag_running.py
    (results/{tag}_{short}_seed{N}_step{1,2,3}M.npz, ``train_step`` field set), and
  - the final, complete-training measurements from submit_diag_goal_struct_corr.sh
    (results/{tag}_{short}_seed{N}.npz, no ``train_step`` -- treated as the run's
    logged final step, read from its metrics.jsonl).

One line per (task, seed) pair -- unlike make_figure.py's cross-seed mean/std
(which needs the SAME set of seeds at each point), the milestone snapshots
only exist for whichever seeds happen to have been "currently training" when
watch_and_diag_running.py was running (seeds 2-3 as of 2026-08-03), so seeds
are plotted individually here rather than pooled.

Usage:
  python3 plot_struct_corr_vs_step.py     # reads ./results/*.npz, writes
                                           # ./struct_corr_vs_step.{svg,pdf}
"""
import argparse
import json
import pathlib
import re
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
FINAL_NAME_RE = re.compile(r'^(e\d+)_([a-z]+)_seed(\d+)$')
SNAPSHOT_NAME_RE = re.compile(r'^(e\d+)_([a-z]+)_seed(\d+)_step(\d+)M$')

ap = argparse.ArgumentParser()
ap.add_argument('--results_dir', default=str(HERE / 'results'))
ap.add_argument('--out_svg', default=str(HERE / 'struct_corr_vs_step.svg'))
ap.add_argument('--out_pdf', default=str(HERE / 'struct_corr_vs_step.pdf'))
args = ap.parse_args()
base = pathlib.Path(args.results_dir)

# {(task, seed): [(step, r_hard), ...]}, sorted by step.
series = {}
for p in sorted(base.glob('*.npz')):
  m = SNAPSHOT_NAME_RE.match(p.stem)
  is_snapshot = m is not None
  if not is_snapshot:
    m = FINAL_NAME_RE.match(p.stem)
  if not m:
    continue
  tag, short, seed = m.group(1), m.group(2), int(m.group(3))
  task = TASK_OF_SHORT.get(short)
  if task is None:
    continue
  d = np.load(p)
  if is_snapshot:
    step = int(d['train_step'])
    if step < 0:
      continue
  else:
    step = int(d['train_step']) if 'train_step' in d and int(d['train_step']) >= 0 else None
    if step is None:
      # Final measurement predates the train_step field -- fall back to the
      # run's own metrics.jsonl final step.
      import sys as _sys
      _sys.path.insert(0, str(HERE.parent))
      from baselines_common import _find_run_dir  # noqa
      run_dir = _find_run_dir(int(tag[1:]))
      if run_dir is None:
        continue
      logdir = run_dir / 'logdir'
      logdir = logdir if logdir.exists() else run_dir
      metrics = logdir / 'metrics.jsonl'
      if not metrics.exists():
        continue
      last = None
      with open(metrics) as f:
        for line in f:
          line = line.strip()
          if not line:
            continue
          try:
            dd = json.loads(line)
          except json.JSONDecodeError:
            continue
          if 'step' in dd:
            last = dd['step']
      step = last
  series.setdefault((task, seed), []).append((step, float(d['corr_hard_mean'])))

for k in series:
  series[k].sort()

if not series:
  raise SystemExit(
      'no results found -- run submit_diag_goal_struct_corr.sh (final) and/or '
      'wait for watch_and_diag_running.py to capture milestone snapshots first')

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
PAD_T = F_TITLE * 1.3 + F_LEGEND * 1.9
PAD_B = TICK_LEN + F_TICK * 1.6 + F_AXIS * 2.2
FIG_H = PANEL_H + PAD_T + PAD_B

ymin, ymax = 0.0, 1.0
xmax_step = max(s for pts in series.values() for s, _ in pts)
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
legend_y = F_TITLE * 1.15 + F_LEGEND * 1.7
DASH = {0: 'none', 1: f'{pt(3):.1f},{pt(2):.1f}', 2: f'{pt(1):.1f},{pt(1.5):.1f}',
        3: f'{pt(5):.1f},{pt(1.5):.1f},{pt(1):.1f},{pt(1.5):.1f}'}
li = 0
for task in TASKS:
  color = TASK_COLORS[task]
  seeds = sorted({seed for (t, seed) in series if t == task})
  for seed in seeds:
    pts = series[(task, seed)]
    dash = DASH.get(seed, 'none')
    poly = ' '.join(f'{sx(s):.2f},{sy(r):.2f}' for s, r in pts)
    dash_attr = f'stroke-dasharray="{dash}"' if dash != 'none' else ''
    parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" '
                 f'stroke-width="{pt(1.6):.2f}" {dash_attr} stroke-linejoin="round" '
                 f'stroke-linecap="round"/>')
    for s, r in pts:
      parts.append(f'<circle cx="{sx(s):.2f}" cy="{sy(r):.2f}" r="{pt(2.2):.2f}" '
                   f'fill="{color}" stroke="white" stroke-width="{pt(0.5):.2f}"/>')
    lx = legend_x + (li % 2) * (PANEL_W / 2)
    ly = legend_y + (li // 2) * (F_LEGEND * 1.6)
    parts.append(f'<line x1="{lx:.1f}" y1="{ly - F_LEGEND*0.35:.1f}" '
                 f'x2="{lx+F_LEGEND*1.8:.1f}" y2="{ly - F_LEGEND*0.35:.1f}" '
                 f'stroke="{color}" stroke-width="{pt(1.8):.1f}" {dash_attr}/>')
    parts.append(f'<text x="{lx+F_LEGEND*2.2:.1f}" y="{ly:.1f}" font-size="{F_LEGEND:.1f}" '
                 f'text-anchor="start" fill="#333">{TASK_LABELS[task]} seed{seed}</text>')
    li += 1

parts.append(f'<text x="{PAD_L+PANEL_W/2:.1f}" y="{F_TITLE*1.15:.1f}" font-size="{F_TITLE:.1f}" '
             f'font-weight="bold" text-anchor="middle" fill="black">'
             f'Goal-code/goal-space hard-code correlation (r) across training</text>')
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
for (task, seed), pts in sorted(series.items()):
  print(f'  {task} seed{seed}: {[(s, round(r,3)) for s, r in pts]}')

subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', args.out_pdf, args.out_svg], check=True)
print('wrote', args.out_pdf)
