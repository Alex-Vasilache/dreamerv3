"""Hand-rolled SVG line plot (no matplotlib -- see ../goal_code_struct_corr/
make_figure.py for why: this cluster's rl_env has a broken matplotlib
install, missing system libpng15 on both login and compute nodes).

Plots ``train/wkr_goal_rew`` (worker cosine_max goal reward) vs. training
step for the three pure-Director baselines, answering the paper's \\todo on
worker success. Converted to PDF with rsvg-convert.

Usage:
  python3 make_figure.py     # reads ./results/*.json, writes
                              # ./worker_success.{svg,pdf} and overwrites the
                              # copy in code/26_04_HRL-paper/figures/introduction/
"""
import argparse
import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent

OI_BLUE = '#0072B2'
OI_VERMILLION = '#D55E00'
OI_GREEN = '#009E73'

envs = [
    ('e124_hopper', 'Hopper Hop', OI_BLUE),
    ('e123_cheetah', 'Cheetah Run', OI_VERMILLION),
    ('e180_acrobot', 'Acrobot Swingup', OI_GREEN),
]

ap = argparse.ArgumentParser()
ap.add_argument('--results_dir', default=str(HERE / 'results'))
ap.add_argument('--out_svg', default=str(HERE / 'worker_success.svg'))
ap.add_argument('--out_pdf', default=str(HERE / 'worker_success.pdf'))
ap.add_argument('--paper_pdf', default=str(
    HERE.parent.parent.parent / '26_04_HRL-paper' / 'figures' / 'introduction'
    / 'worker_success.pdf'),
    help='also overwrite this copy (the one the paper actually includes); pass "" to skip')
args = ap.parse_args()
base = pathlib.Path(args.results_dir)

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

data = {}
for tag, title, color in envs:
  with open(base / f'{tag}.json') as f:
    data[tag] = json.load(f)

xmax_step = max(max(d['steps']) for d in data.values())
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

# Gridlines at y = 0, 0.25, 0.5, 0.75, 1 (perfect goal achievement).
for t in (0.0, 0.25, 0.5, 0.75, 1.0):
  gy = sy(t)
  parts.append(f'<line x1="{ax0:.1f}" y1="{gy:.1f}" x2="{ax1:.1f}" y2="{gy:.1f}" '
               f'stroke="#ddd" stroke-width="{pt(0.5):.1f}"/>')
  parts.append(f'<line x1="{ax0-TICK_LEN:.1f}" y1="{gy:.1f}" x2="{ax0:.1f}" y2="{gy:.1f}" '
               f'stroke="black" stroke-width="{pt(1.8):.1f}"/>')
  parts.append(f'<text x="{ax0-TICK_LEN-F_TICK*0.5:.1f}" y="{gy+F_TICK*0.35:.1f}" '
               f'font-size="{F_TICK:.1f}" font-weight="bold" text-anchor="end" '
               f'fill="black">{t:g}</text>')

# y=1 reference line (perfect worker goal achievement) emphasized dashed.
parts.append(f'<line x1="{ax0:.1f}" y1="{sy(1.0):.1f}" x2="{ax1:.1f}" y2="{sy(1.0):.1f}" '
             f'stroke="#888" stroke-width="{pt(1.0):.1f}" '
             f'stroke-dasharray="{pt(3.5):.1f},{pt(2.5):.1f}"/>')

# X ticks at 0, 1M, 2M, 3M, 4M steps (whatever fits within xmax).
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
for i, (tag, title, color) in enumerate(envs):
  d = data[tag]
  steps, vals = d['steps'], d['values']
  poly = ' '.join(f'{sx(s):.2f},{sy(v):.2f}' for s, v in zip(steps, vals))
  parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" '
               f'stroke-width="{pt(1.4):.2f}" stroke-linejoin="round" '
               f'stroke-linecap="round" opacity="0.9"/>')
  lx = legend_x + i * (PANEL_W / 3)
  parts.append(f'<line x1="{lx:.1f}" y1="{legend_y - F_LEGEND*0.35:.1f}" '
               f'x2="{lx+F_LEGEND*1.8:.1f}" y2="{legend_y - F_LEGEND*0.35:.1f}" '
               f'stroke="{color}" stroke-width="{pt(1.8):.1f}"/>')
  parts.append(f'<text x="{lx+F_LEGEND*2.2:.1f}" y="{legend_y:.1f}" font-size="{F_LEGEND:.1f}" '
               f'text-anchor="start" fill="#333">{title} '
               f'(final {vals[-1]:.2f})</text>')

parts.append(f'<text x="{PAD_L+PANEL_W/2:.1f}" y="{F_TITLE*1.15:.1f}" font-size="{F_TITLE:.1f}" '
             f'font-weight="bold" text-anchor="middle" fill="black">'
             f'Worker goal reward (cosine_max) across training</text>')
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

subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', args.out_pdf, args.out_svg], check=True)
print('wrote', args.out_pdf)
if args.paper_pdf:
  pathlib.Path(args.paper_pdf).parent.mkdir(parents=True, exist_ok=True)
  pathlib.Path(args.paper_pdf).write_bytes(pathlib.Path(args.out_pdf).read_bytes())
  print('copied to', args.paper_pdf, '-- rebuild the paper (pdflatex x2) to see it')
