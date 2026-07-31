"""Hand-rolled SVG scatter plot (no matplotlib -- rl_env's install is broken:
missing libpng15 on this cluster). Converted to PDF afterwards with rsvg-convert.

This figure is included at width=INCLUDE_FRAC*\\textwidth in a NeurIPS
single-column paper (\\textwidth ~= 397pt), so the *final* printed width is
TARGET_PT points. Everything drawn here shrinks by TARGET_PT / FIG_W when
LaTeX places it, so font sizes are chosen so that final_pt = font_svg_px *
TARGET_PT / FIG_W lands around normal caption/tick sizes (~9pt title,
~7-7.5pt labels/ticks). RES_SCALE inflates the horizontal coordinate system
(and therefore every stroke/marker/font size in lockstep, since font sizes
are derived from RATIO = FIG_W / TARGET_PT) so the vector geometry is defined
at higher internal resolution without changing the final printed size or
aspect ratio. Vertical padding is computed *after* the font sizes so it
always has enough room for the title/ticks regardless of RES_SCALE/
INCLUDE_FRAC -- otherwise shrinking INCLUDE_FRAC (denser DPI at same layout
padding) makes text collide with the plot box.

Why hand-rolled SVG instead of matplotlib: this cluster's rl_env (Python 3.7,
the env normally used for paper/make_figures.py) has a broken matplotlib
install (missing system libpng15 -- ImportError on every raster/vector
backend, login node and compute nodes alike). Plain SVG text + shapes, then
`rsvg-convert -f pdf` (present on the login node), sidesteps it entirely.

Usage:
  python3 make_figure.py                 # reads ./results/*.npz, writes
                                          # ./goal_code_struct_corr.{svg,pdf}
                                          # and overwrites the copy in
                                          # code/26_04_HRL-paper/figures/
"""
import argparse
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

OI_BLUE = '#0072B2'
OI_VERMILLION = '#D55E00'
OI_GREEN = '#009E73'
TREND_COLOR = '#1A1A1A'

envs = [
    ('e124_hopper', 'Hopper Hop', OI_BLUE),
    ('e123_cheetah', 'Cheetah Run', OI_VERMILLION),
    ('e180_acrobot', 'Acrobot Swingup', OI_GREEN),
]

ap = argparse.ArgumentParser()
ap.add_argument('--results_dir', default=str(HERE / 'results'))
ap.add_argument('--out_svg', default=str(HERE / 'goal_code_struct_corr.svg'))
ap.add_argument('--out_pdf', default=str(HERE / 'goal_code_struct_corr.pdf'))
ap.add_argument('--paper_pdf', default=str(
    HERE.parent.parent.parent / '26_04_HRL-paper' / 'figures' / 'introduction'
    / 'goal_code_struct_corr.pdf'),
    help='also overwrite this copy (the one the paper actually includes); pass "" to skip')
args = ap.parse_args()
base = args.results_dir

TEXTWIDTH_PT = 397.0
INCLUDE_FRAC = 0.72  # smaller final printed size
TARGET_PT = INCLUDE_FRAC * TEXTWIDTH_PT

RES_SCALE = 6  # internal resolution multiplier (higher "DPI"); final print size unaffected
PANEL = 300 * RES_SCALE
GAP = 40 * RES_SCALE
PAD_L, PAD_R = 55 * RES_SCALE, 10 * RES_SCALE
CELL = PANEL + PAD_L + PAD_R
FIG_W = 3 * CELL + 2 * GAP
RATIO = FIG_W / TARGET_PT  # svg_px per final pt


def pt(final_pt):
  return final_pt * RATIO


F_TITLE = pt(9.0)
F_SUB = pt(7.3)
F_TICK = pt(7.5)
F_AXIS = pt(7.3)
TICK_LEN = pt(4.5)

# Vertical padding derived from the actual font metrics so text never
# collides with the plot box, independent of RES_SCALE / INCLUDE_FRAC.
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
parts = []
parts.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{FIG_W:.0f}" height="{FIG_H:.0f}" '
    f'viewBox="0 0 {FIG_W:.0f} {FIG_H:.0f}" font-family="Helvetica,Arial,sans-serif">')
parts.append(f'<rect x="0" y="0" width="{FIG_W:.0f}" height="{FIG_H:.0f}" fill="white"/>')

for i, (tag, title, color) in enumerate(envs):
  x0 = i * (CELL + GAP)
  d = np.load(f'{base}/{tag}.npz')
  sd, sz = d['sd_pairs'], d['sh_pairs']  # hard (argmax, REINFORCE-sampled) code; batch 0
  corr_mean = float(d['corr_hard_mean'])
  corr_std = float(d['corr_hard_std'])
  n_batches = int(d['n_batches'])
  n = len(sd)
  idx = rng.choice(n, min(6000, n), replace=False)

  ax0, ay0 = sx(xmin, x0), sy(ymin)
  ax1, ay1 = sx(xmax, x0), sy(ymax)

  # Reference gridlines at 0 and 1 (dashed, light) so it's unambiguous where
  # those values fall inside the scatter cloud -- drawn before the points so
  # the points sit visually on top of the guide lines, not the other way round.
  for t in (0.0, 1.0):
    gx = sx(t, x0)
    parts.append(f'<line x1="{gx:.1f}" y1="{ay1:.1f}" x2="{gx:.1f}" y2="{ay0:.1f}" '
                 f'stroke="#bbb" stroke-width="{pt(0.5):.1f}" stroke-dasharray="{pt(2):.1f},{pt(2):.1f}"/>')
    gy = sy(t)
    parts.append(f'<line x1="{ax0:.1f}" y1="{gy:.1f}" x2="{ax1:.1f}" y2="{gy:.1f}" '
                 f'stroke="#bbb" stroke-width="{pt(0.5):.1f}" stroke-dasharray="{pt(2):.1f},{pt(2):.1f}"/>')

  # y=x reference: the hypothetical fully proportional mapping (goal-code
  # similarity exactly tracking goal-space similarity). Not a fitted line --
  # a fixed visual reference for "what perfect structure preservation would
  # look like", to judge the trend curve against.
  dlo, dhi = max(xmin, ymin), min(xmax, ymax)
  parts.append(f'<line x1="{sx(dlo, x0):.1f}" y1="{sy(dlo):.1f}" '
               f'x2="{sx(dhi, x0):.1f}" y2="{sy(dhi):.1f}" '
               f'stroke="#888" stroke-width="{pt(1.0):.1f}" '
               f'stroke-dasharray="{pt(3.5):.1f},{pt(2.5):.1f}"/>')

  pts = []
  for xv, yv in zip(sd[idx], sz[idx]):
    px, py = sx(xv, x0), sy(yv)
    pts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{pt(0.9):.2f}" fill="{color}" '
                f'fill-opacity="0.22"/>')
  parts.append(''.join(pts))

  # Trend line: mean goal-code similarity per goal-space-similarity bin,
  # computed over the *full* pair set (not the plotted subsample).
  nbins = 22
  edges = np.linspace(sd.min(), sd.max(), nbins + 1)
  bin_idx = np.clip(np.digitize(sd, edges) - 1, 0, nbins - 1)
  cx_list, cy_list = [], []
  for b in range(nbins):
    m = bin_idx == b
    if m.sum() < 10:
      continue
    cx_list.append(0.5 * (edges[b] + edges[b + 1]))
    cy_list.append(float(sz[m].mean()))
  if len(cx_list) >= 2:
    poly = ' '.join(f'{sx(cx, x0):.2f},{sy(cy):.2f}' for cx, cy in zip(cx_list, cy_list))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="{TREND_COLOR}" '
                 f'stroke-width="{pt(1.6):.2f}" stroke-linejoin="round" '
                 f'stroke-linecap="round"/>')

  # Tick marks + labels at 0 and 1 on both axes -- bold, dark, drawn on top of
  # everything so they stay clearly visible against the scatter/gridlines.
  # No surrounding box; the ticks alone mark where the axes are.
  tick_text_y = ay0 + TICK_LEN + F_TICK * 1.3
  for t in (0.0, 1.0):
    gx = sx(t, x0)
    parts.append(f'<line x1="{gx:.1f}" y1="{ay0:.1f}" x2="{gx:.1f}" y2="{ay0+TICK_LEN:.1f}" '
                 f'stroke="black" stroke-width="{pt(1.8):.1f}"/>')
    parts.append(f'<text x="{gx:.1f}" y="{tick_text_y:.1f}" font-size="{F_TICK:.1f}" '
                 f'font-weight="bold" text-anchor="middle" fill="black">{t:g}</text>')
    gy = sy(t)
    parts.append(f'<line x1="{ax0-TICK_LEN:.1f}" y1="{gy:.1f}" x2="{ax0:.1f}" y2="{gy:.1f}" '
                 f'stroke="black" stroke-width="{pt(1.8):.1f}"/>')
    if i == 0:
      parts.append(f'<text x="{ax0-TICK_LEN-F_TICK*0.5:.1f}" y="{gy+F_TICK*0.35:.1f}" font-size="{F_TICK:.1f}" '
                   f'font-weight="bold" text-anchor="end" fill="black">{t:g}</text>')

  parts.append(f'<text x="{x0+CELL/2:.1f}" y="{F_TITLE*1.15:.1f}" font-size="{F_TITLE:.1f}" '
               f'font-weight="bold" text-anchor="middle" fill="black">{title}</text>')
  parts.append(f'<text x="{x0+CELL/2:.1f}" y="{F_TITLE*1.15+F_SUB*1.7:.1f}" font-size="{F_SUB:.1f}" '
               f'text-anchor="middle" fill="#333">'
               f'r = {corr_mean:.2f} &#177; {corr_std:.2f} '
               f'(n={n_batches})</text>')
  parts.append(f'<text x="{x0+CELL/2:.1f}" y="{tick_text_y+F_AXIS*1.7:.1f}" font-size="{F_AXIS:.1f}" '
               f'text-anchor="middle" fill="#333">goal-space similarity</text>')

cy = PAD_T + PANEL / 2
parts.append(
    f'<text x="{F_AXIS*1.3:.1f}" y="{cy:.1f}" font-size="{F_AXIS:.1f}" text-anchor="middle" '
    f'fill="#333" transform="rotate(-90 {F_AXIS*1.3:.1f} {cy:.1f})">goal-code similarity</text>')

parts.append('</svg>')

with open(args.out_svg, 'w') as f:
  f.write('\n'.join(parts))
print('wrote', args.out_svg, '| FIG_W', FIG_W, '| FIG_H', round(FIG_H),
      '| target_pt', TARGET_PT, '| ratio', round(RATIO, 3),
      '| title_svg_px', round(F_TITLE, 1), '| tick_svg_px', round(F_TICK, 1),
      '| PAD_T', round(PAD_T), '| PAD_B', round(PAD_B))

subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', args.out_pdf, args.out_svg], check=True)
print('wrote', args.out_pdf)
if args.paper_pdf:
  pathlib.Path(args.paper_pdf).write_bytes(pathlib.Path(args.out_pdf).read_bytes())
  print('copied to', args.paper_pdf, '-- rebuild the paper (pdflatex x2) to see it')
