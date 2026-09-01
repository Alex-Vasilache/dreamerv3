"""Goal-code vs goal-space geometry under MSE instead of ``cosine_max``.

``make_figure.py``/``make_figure_soft.py`` plot pairwise ``cosine_max``
similarity, which is the right axis for the worker (its reward literally IS
``cosine_max(goal, deter)``) but the wrong one for the question the Lipschitz
argument asks. ``cosine_max`` is scale-free: two goals pointing the same way at
different magnitudes score as identical, so a decoder that preserves directions
while stretching lengths looks perfect. MSE is the metric the autoencoder's own
reconstruction loss is written in and it does see magnitude.

Both axes are therefore DISTANCES here, not similarities, and they run the same
way: a structure-preserving code puts far-apart goals on far-apart codes, so the
correlation stays positive and a perfect code would trace a straight line
through the origin. That line is drawn (least squares through zero, pooled over
seeds) as the reference the dashed y=x diagonal plays in the cosine figures --
it cannot be y=x here because the two axes carry different units.

  python3 make_figure_mse.py --code hard    # -> goal_code_struct_corr_mse.pdf
  python3 make_figure_mse.py --code soft    # -> goal_code_struct_corr_mse_soft.pdf

Layout, sizing and fonts follow make_figure_soft.py so the four figures can be
read side by side in the paper.
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

NBINS = 22

ap = argparse.ArgumentParser()
ap.add_argument('--code', choices=['hard', 'soft'], default='hard')
ap.add_argument('--results_dir', default=str(HERE / 'results'))
ap.add_argument('--out', default=None, help='basename; default derives from --code')
ap.add_argument('--paper_dir', default=str(
    HERE.parent.parent.parent / '26_04_HRL-paper' / 'figures' / 'motivation'),
    help='also copy the pdf here; pass "" to skip')
args = ap.parse_args()

HARD = args.code == 'hard'
YKEY = 'mh_pairs' if HARD else 'mz_pairs'
RKEY = 'corr_hard_mse_mean' if HARD else 'corr_soft_mse_mean'
stem = 'goal_code_struct_corr_mse' + ('' if HARD else '_soft')
out_base = args.out or str(HERE / stem)
out_svg, out_pdf = out_base + '.svg', out_base + '.pdf'

SHORT_TASK = {
    'dmc_cartpole_swingup': 'cartpole',
    'dmc_hopper_hop': 'hopper',
    'dmc_acrobot_swingup': 'acrobot',
    'dmc_cheetah_run': 'cheetah',
}
TASK_OF_SHORT = {v: k for k, v in SHORT_TASK.items()}
name_re = re.compile(r'^(e\d+)_([a-z]+)_seed(\d+)$')

by_task = {t: [] for t in TASKS}
for p in sorted(pathlib.Path(args.results_dir).glob('*.npz')):
  m = name_re.match(p.stem)
  if m and TASK_OF_SHORT.get(m.group(2)) in by_task:
    with np.load(p) as d:
      if YKEY not in d:   # measured before the MSE fields existed
        continue
    by_task[TASK_OF_SHORT[m.group(2)]].append(p)

missing = [t for t in TASKS if not by_task[t]]
if missing:
  raise SystemExit(f'no results/*.npz carrying {YKEY} for: {missing} -- '
                   're-run diag_goal_struct_corr.py on those seeds')

TEXTWIDTH_PT = 397.0
TARGET_PT = 0.95 * TEXTWIDTH_PT
RES_SCALE = 6
# 2 x 2, not the cosine figures' 1 x 4. Those tick at "0" and "1"; these carry
# real magnitudes on both axes, and four panels in a row leaves no room for the
# labels -- rendered that way, the rotated y-axis title printed straight through
# the tick numbers. Two rows give each panel four times the area at the same
# printed width, which is what makes the axis legible at all.
NCOL, NROW = 2, 2
PANEL = 300 * RES_SCALE
GAP_X, GAP_Y = 30 * RES_SCALE, 46 * RES_SCALE
PAD_L, PAD_R = 92 * RES_SCALE, 14 * RES_SCALE
CELL = PANEL + PAD_L + PAD_R
FIG_W = NCOL * CELL + (NCOL - 1) * GAP_X
RATIO = FIG_W / TARGET_PT


def pt(final_pt):
  return final_pt * RATIO


F_TITLE, F_SUB, F_TICK, F_AXIS = pt(9.0), pt(7.0), pt(7.2), pt(7.0)
TICK_LEN = pt(4.5)
PAD_T = F_TITLE * 1.3 + F_SUB * 1.9
PAD_B = TICK_LEN + F_TICK * 1.6 + F_AXIS * 2.2
ROW_H = PANEL + PAD_T + PAD_B
FIG_H = NROW * ROW_H + (NROW - 1) * GAP_Y


def ticks(hi, n=4):
  """Round ticks spanning [0, hi]."""
  if hi <= 0:
    return [0.0]
  raw = hi / float(n)
  mag = 10.0 ** np.floor(np.log10(raw))
  step = min([m * mag for m in (1, 2, 2.5, 5, 10)],
             key=lambda s: abs(s - raw))
  return [v for v in np.arange(0, hi * 1.0001, step)]


def fmt(v):
  if v == 0:
    return '0'
  return ('%.4g' % v)


rng = np.random.RandomState(0)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{FIG_W:.0f}" '
       f'height="{FIG_H:.0f}" viewBox="0 0 {FIG_W:.0f} {FIG_H:.0f}" '
       f'font-family="Helvetica,Arial,sans-serif">',
       f'<rect x="0" y="0" width="{FIG_W:.0f}" height="{FIG_H:.0f}" '
       f'fill="white"/>']

TOTAL_SCATTER_PTS = 6000
summary = []

for i, task in enumerate(TASKS):
  files = by_task[task]
  n_seeds = len(files)
  color = TASK_COLORS[task]

  col, row = i % NCOL, i // NCOL
  x0 = col * (CELL + GAP_X)
  y0 = row * (ROW_H + GAP_Y)

  seeds = [np.load(f) for f in files]
  rs = np.array([float(d[RKEY]) for d in seeds])
  r_mean, r_std = float(rs.mean()), float(rs.std()) if n_seeds > 1 else 0.0

  xs = [np.asarray(d['md_pairs'], np.float64) for d in seeds]
  ys = [np.asarray(d[YKEY], np.float64) for d in seeds]
  # Axis range from a high quantile, not the max: MSE has a long right tail
  # (the few most distant state pairs), and scaling to it would squash every
  # panel into its bottom-left corner.
  xmax = float(np.percentile(np.concatenate(xs), 99.0))
  ymax = float(np.percentile(np.concatenate(ys), 99.9)) if not HARD else \
      float(max(y.max() for y in ys))
  xmax = xmax * 1.04 or 1.0
  ymax = ymax * 1.04 or 1.0

  def sx(v, x0=x0):
    return x0 + PAD_L + np.clip(v / xmax, 0, 1.02) * PANEL

  def sy(v, y0=y0):
    return y0 + PAD_T + (1 - np.clip(v / ymax, 0, 1.02)) * PANEL

  ax0, ay0, ax1, ay1 = sx(0), sy(0), sx(xmax), sy(ymax)

  # Trend. The hard code's MSE takes only L+1 exact values (2k/(L*C) for k
  # blocks differing), so it is binned on its own discrete y-axis -- the
  # conditional mean of goal MSE at each level -- exactly as make_figure.py
  # does for the cosine version. The soft code is continuous, so it is binned
  # on x like any scatter.
  if HARD:
    levels = np.unique(np.concatenate([np.unique(y) for y in ys]))
    per_seed = []
    for xv, yv in zip(xs, ys):
      per_seed.append([
          xv[np.isclose(yv, lv)].mean() if np.isclose(yv, lv).sum() >= 10
          else np.nan for lv in levels])
    tvals = levels
  else:
    edges = np.linspace(0, xmax, NBINS + 1)
    tvals = 0.5 * (edges[:-1] + edges[1:])
    per_seed = []
    for xv, yv in zip(xs, ys):
      idx = np.clip(np.digitize(xv, edges) - 1, 0, NBINS - 1)
      per_seed.append([yv[idx == b].mean() if (idx == b).sum() >= 10
                       else np.nan for b in range(NBINS)])
  arr = np.array(per_seed, float)
  with np.errstate(invalid='ignore'):
    tmean = np.nanmean(arr, 0)
    tstd = np.nanstd(arr, 0) if n_seeds > 1 else np.zeros_like(tmean)
  ok = ~np.isnan(tmean)

  # Reference: proportionality. Least squares through the origin on the pooled
  # pairs -- what the cloud would lie on if code distance tracked goal distance
  # exactly. Slope only; the claim is the SHAPE, not the units.
  xf, yf = np.concatenate(xs), np.concatenate(ys)
  slope = float((xf * yf).sum() / max((xf * xf).sum(), 1e-30))
  xe = min(xmax, ymax / slope) if slope > 0 else xmax
  svg.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(xe):.1f}" '
             f'y2="{sy(slope * xe):.1f}" stroke="#888" '
             f'stroke-width="{pt(1.0):.1f}" '
             f'stroke-dasharray="{pt(3.5):.1f},{pt(2.5):.1f}"/>')

  per_seed_n = max(1, TOTAL_SCATTER_PTS // n_seeds)
  dots = []
  for xv, yv in zip(xs, ys):
    sel = rng.choice(len(xv), min(per_seed_n, len(xv)), replace=False)
    for a, b in zip(xv[sel], yv[sel]):
      if a <= xmax * 1.02 and b <= ymax * 1.02:
        dots.append(f'<circle cx="{sx(a):.2f}" cy="{sy(b):.2f}" '
                    f'r="{pt(0.8):.2f}" fill="{color}" fill-opacity="0.10"/>')
  svg.append(''.join(dots))

  if ok.sum() >= 2:
    vx, vm, vs = tvals[ok], tmean[ok], tstd[ok]
    if HARD:   # y is the binned axis: band runs horizontally
      band = ([(m + s, y) for m, s, y in zip(vm, vs, vx)] +
              [(m - s, y) for m, s, y in zip(vm, vs, vx)][::-1])
    else:
      band = ([(x, m + s) for x, m, s in zip(vx, vm, vs)] +
              [(x, m - s) for x, m, s in zip(vx, vm, vs)][::-1])
    svg.append('<polygon points="%s" fill="%s" fill-opacity="0.25" '
               'stroke="none"/>' % (' '.join(
                   f'{sx(a):.2f},{sy(b):.2f}' for a, b in band), color))
    line = ([(m, y) for m, y in zip(vm, vx)] if HARD else
            list(zip(vx, vm)))
    svg.append('<polyline points="%s" fill="none" stroke="#1A1A1A" '
               'stroke-width="%.2f" stroke-linejoin="round" '
               'stroke-linecap="round"/>' % (' '.join(
                   f'{sx(a):.2f},{sy(b):.2f}' for a, b in line), pt(1.6)))

  # Axis rules. Without them the tick marks float unattached below the cloud.
  svg.append(f'<line x1="{ax0:.1f}" y1="{ay0:.1f}" x2="{ax1:.1f}" '
             f'y2="{ay0:.1f}" stroke="#8a8880" stroke-width="{pt(0.7):.2f}"/>')
  svg.append(f'<line x1="{ax0:.1f}" y1="{ay0:.1f}" x2="{ax0:.1f}" '
             f'y2="{ay1:.1f}" stroke="#8a8880" stroke-width="{pt(0.7):.2f}"/>')

  tick_text_y = ay0 + TICK_LEN + F_TICK * 1.3
  for v in ticks(xmax):
    gx = sx(v)
    svg.append(f'<line x1="{gx:.1f}" y1="{ay0:.1f}" x2="{gx:.1f}" '
               f'y2="{ay0 + TICK_LEN:.1f}" stroke="black" '
               f'stroke-width="{pt(1.8):.1f}"/>')
    svg.append(f'<text x="{gx:.1f}" y="{tick_text_y:.1f}" '
               f'font-size="{F_TICK:.1f}" font-weight="bold" '
               f'text-anchor="middle" fill="black">{fmt(v)}</text>')
  # The hard code's MSE is discrete -- one-hot blocks give exactly L+1 possible
  # distances, 2k/(L*C) -- so tick at the values it can take, one per stripe,
  # rather than at round numbers.
  for v in (levels if HARD else ticks(ymax)):
    gy = sy(v)
    svg.append(f'<line x1="{ax0 - TICK_LEN:.1f}" y1="{gy:.1f}" '
               f'x2="{ax0:.1f}" y2="{gy:.1f}" stroke="black" '
               f'stroke-width="{pt(1.8):.1f}"/>')
    svg.append(f'<text x="{ax0 - TICK_LEN - F_TICK * 0.5:.1f}" '
               f'y="{gy + F_TICK * 0.35:.1f}" '
               f'font-size="{F_TICK * (0.86 if HARD else 1.0):.1f}" '
               f'font-weight="bold" text-anchor="end" fill="black">'
               f'{("%.2f" % v).lstrip("0") if HARD and v else fmt(v)}</text>')

  svg.append(f'<text x="{x0 + CELL / 2:.1f}" y="{y0 + F_TITLE * 1.15:.1f}" '
             f'font-size="{F_TITLE:.1f}" font-weight="bold" '
             f'text-anchor="middle" fill="black">{TASK_LABELS[task]}</text>')
  svg.append(f'<text x="{x0 + CELL / 2:.1f}" '
             f'y="{y0 + F_TITLE * 1.15 + F_SUB * 1.7:.1f}" '
             f'font-size="{F_SUB:.1f}" '
             f'text-anchor="middle" fill="#333">r = {r_mean:.2f} &#177; '
             f'{r_std:.2f} (n={n_seeds} seed{"s" if n_seeds != 1 else ""})'
             f'</text>')
  svg.append(f'<text x="{x0 + CELL / 2:.1f}" '
             f'y="{tick_text_y + F_AXIS * 1.7:.1f}" font-size="{F_AXIS:.1f}" '
             f'text-anchor="middle" fill="#333">goal-space MSE</text>')
  summary.append((task, n_seeds, r_mean, r_std, xmax, ymax))

label = 'goal-code MSE (hard)' if HARD else 'goal-code MSE (soft)'
for row in range(NROW):
  cy = row * (ROW_H + GAP_Y) + PAD_T + PANEL / 2
  cx = F_AXIS * 1.2
  svg.append(f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="{F_AXIS:.1f}" '
             f'text-anchor="middle" fill="#333" '
             f'transform="rotate(-90 {cx:.1f} {cy:.1f})">{label}</text>')
svg.append('</svg>')

pathlib.Path(out_svg).write_text('\n'.join(svg))
print(f'wrote {out_svg}  ({FIG_W:.0f} x {FIG_H:.0f})')
for task, n, rm, rs_, xm, ym in summary:
  print(f'  {task:24s} n={n}  r={rm:.3f}+/-{rs_:.3f}  '
        f'xmax={xm:.4g} ymax={ym:.4g}')
subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', out_pdf, out_svg],
               check=True)
print('wrote', out_pdf)
if args.paper_dir:
  dst = pathlib.Path(args.paper_dir) / pathlib.Path(out_pdf).name
  dst.write_bytes(pathlib.Path(out_pdf).read_bytes())
  print('copied to', dst)
