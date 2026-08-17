#!/usr/bin/env python3
"""Decoded-goal shift against code distance, on two different x-axes.

Both are built from the same draws (``index_total/*`` in the geometry npz), in
which m of the L blocks are moved so that the index shifts sum to exactly D.

``--x total`` (default): x is the TOTAL index distance
D = sum_l |c'_l - c_l|, from 0 to L*(C-1) = 56. Eight blocks each moved one
class is D=8; eight blocks each moved seven classes is D=56. This is the
distance between two codes if the index is a coordinate, and it is the only
axis on which "one small step in every block" and "one huge step in one block"
are different things.

Not every D is reachable from every reference code: a block at class c can move
at most max(c, C-1-c), so a reference whose classes sit mid-range tops out well
below 56, and only a code with every class at 0 or C-1 reaches the maximum. The
curve is drawn solid while at least ``--min-coverage`` of reference states can
supply the distance and faded beyond, rather than quietly averaging over a
shrinking and increasingly extreme subset.

``--x codemse``: x is the goal-code MSE the architecture actually presents to
the manager. The decoder's input is the one-hot code for every arm, so that
distance is 2 x (blocks that differ) / (L*C) -- it counts blocks and is blind
to how far each one moved. Plotting goal MSE against it puts both axes in the
same units as the pairwise-similarity figures, and the vertical spread at each
x is precisely the goal movement the code's own metric cannot represent.

  python3 make_figure_index_total.py --x total   --out goal_code_index_total
  python3 make_figure_index_total.py --x codemse --out goal_code_mse_vs_mse
"""
import argparse
import ast
import glob
import os
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
REF = '#52514e'
INK, INK2, INK3, GRID = '#0b0b0b', '#52514e', '#8a8880', '#e6e5e1'

ARMS = [('director', 'Director'), ('som_line', 'SOM-line'),
        ('som_orig_line', 'SOM-line-OG'), ('lipvq_prod', 'LiP'),
        ('som_lipvq_line_prod', 'SOM-line + LiP'),
        ('som_orig_lipvq_line_prod', 'SOM-line-OG + LiP')]
# The two estimator-free arms (STE off) are measured but not plotted: their
# codebooks collapsed, so their curves say more about a dead encoder than about
# the geometry question, and six series in one panel crowded the readable ones.
# Their numbers are still in the npz and in the summary tables below; pass
# --include-all to draw them.
EXCLUDE = ('som_orig_line', 'som_orig_lipvq_line_prod')

TASKS = [('dmc_cartpole_swingup', 'Cartpole Swingup'),
         ('dmc_hopper_stand', 'Hopper Stand')]

TEXTWIDTH_PT = 397.0
TARGET_PT = 0.95 * TEXTWIDTH_PT
RES = 6
PANEL_W, PANEL_H = 205 * RES, 200 * RES
GAP = 168 * RES
PAD_L, PAD_R = 62 * RES, 138 * RES
PAD_T_EXTRA = 4 * RES


def pt(v, w):
  return v * w / TARGET_PT


def esc(s):
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def arm_of(name, meta):
  for key, _ in sorted(ARMS, key=lambda a: -len(a[0])):
    if f'_{key}_s' in name:
      return key
  return 'director' if meta.get('impl') == 'director' else None


def load(results_dir, mode):
  """(task, arm) -> (per-seed curves, per-seed coverage, x values)."""
  out = {}
  for f in sorted(glob.glob(os.path.join(results_dir, '*.npz'))):
    d = np.load(f, allow_pickle=True)
    if 'index_total/mse' not in d:
      continue
    meta = ast.literal_eval(str(d['meta']))
    arm = arm_of(os.path.basename(f), meta)
    if arm is None:
      continue
    if mode == 'total':
      with np.errstate(invalid='ignore'):
        curve = np.nanmean(np.asarray(d['index_total/mse'], float), 0)
      xs = np.asarray(d['index_total/d'], float)
      cov = np.asarray(d['index_total/coverage'], float)
      lo = hi = None
    else:
      # bin the per-draw records on the code's own metric; its L+1 exact
      # values are the natural bins, so no binning choice is made here
      cm = np.asarray(d['index_total/rec_code_mse'], float)
      gm = np.asarray(d['index_total/rec_goal_mse'], float)
      xs = np.unique(cm)
      sel = [cm == v for v in xs]
      curve = np.array([gm[s].mean() for s in sel])
      # the band is the spread WITHIN a seed at fixed code distance, not the
      # spread across seeds: it is the goal movement the code's metric does not
      # determine, which is the point of plotting this axis at all
      lo = np.array([np.percentile(gm[s], 10) for s in sel])
      hi = np.array([np.percentile(gm[s], 90) for s in sel])
      cov = np.ones_like(xs)
      xs = np.concatenate([[0.0], xs])
      curve = np.concatenate([[0.0], curve])
      lo = np.concatenate([[0.0], lo])
      hi = np.concatenate([[0.0], hi])
      cov = np.concatenate([[1.0], cov])
    out.setdefault((meta['task'], arm), []).append((curve, cov, xs, lo, hi))
  return out


def ticks(hi, n=5):
  raw = hi / float(n)
  mag = 10.0 ** np.floor(np.log10(raw)) if raw > 0 else 1.0
  step = min([m * mag for m in (1, 2, 2.5, 5, 10)], key=lambda s: abs(s - raw))
  return [v for v in np.arange(0, hi * 1.0001, step)]


def fmt(v):
  if v == 0:
    return '0'
  return ('%g' % v) if abs(v - round(v)) < 1e-9 else ('%.3g' % v)


def build(data, mode, ymax=None, min_cov=0.5, band=True):
  fig_w = 2 * PANEL_W + GAP + PAD_L + PAD_R
  f_title, f_tick = pt(9.0, fig_w), pt(7.0, fig_w)
  f_axis, f_lab = pt(7.5, fig_w), pt(6.4, fig_w)
  pad_t = f_title * 1.5 + PAD_T_EXTRA
  pad_b = f_tick * 1.8 + f_axis * 1.8
  fig_h = PANEL_H + pad_t + pad_b

  cells = {}
  for key, seeds in data.items():
    xs = seeds[0][2]
    curves = np.stack([s[0] for s in seeds])
    cov = np.stack([s[1] for s in seeds]).mean(0)
    with np.errstate(invalid='ignore'):
      mean = np.nanmean(curves, 0)
      if seeds[0][3] is None:      # total: band is the spread across seeds
        lo = mean - np.nanstd(curves, 0)
        hi = mean + np.nanstd(curves, 0)
      else:                        # codemse: within-seed 10-90 spread
        lo = np.nanmean(np.stack([s[3] for s in seeds]), 0)
        hi = np.nanmean(np.stack([s[4] for s in seeds]), 0)
      cells[key] = (xs, mean, np.maximum(lo, 0.0), hi, cov, len(seeds))
  xmax = max(x[-1] for x, *_ in cells.values()) if mode != 'total' else \
      float(max(x[-1] for x, *_ in cells.values()))
  # Per-task y range; a shared axis wasted half of the Hopper panel.
  ymax_of = {}
  for task, _ in TASKS:
    vals = [np.nanmax(np.where(c[4] >= min_cov, c[3] if band else c[1],
                               np.nan))
            for (t, _a), c in cells.items() if t == task]
    ymax_of[task] = ymax if ymax else (max(vals) * 1.08 if vals else 1.0)

  out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{fig_w:.0f}" '
         f'height="{fig_h:.0f}" viewBox="0 0 {fig_w:.0f} {fig_h:.0f}" '
         f'font-family="Helvetica,Arial,sans-serif">',
         f'<rect x="0" y="0" width="{fig_w:.0f}" height="{fig_h:.0f}" '
         f'fill="white"/>']

  xlabel = ('total index distance across all blocks' if mode == 'total'
            else 'goal-code MSE')
  for ti, (task, tlabel) in enumerate(TASKS):
    x0 = PAD_L + ti * (PANEL_W + GAP)
    ymax_t = ymax_of[task]

    def sx(v, x0=x0):
      return x0 + min(v / xmax, 1.02) * PANEL_W

    def sy(v, ymax_t=ymax_t):
      return pad_t + (1.0 - min(v / ymax_t, 1.02)) * PANEL_H

    out.append(f'<text x="{x0 + PANEL_W / 2:.1f}" y="{f_title * 1.1:.1f}" '
               f'font-size="{f_title:.1f}" fill="{INK}" text-anchor="middle">'
               f'{esc(tlabel)}</text>')
    for v in ticks(ymax_t):
      y = sy(v)
      out.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + PANEL_W:.1f}" '
                 f'y2="{y:.1f}" stroke="{GRID}" '
                 f'stroke-width="{pt(0.5, fig_w):.2f}"/>')
      out.append(f'<text x="{x0 - pt(4, fig_w):.1f}" '
                 f'y="{y + f_tick * 0.36:.1f}" font-size="{f_tick:.1f}" '
                 f'fill="{INK2}" text-anchor="end">{fmt(v)}</text>')
    # On the code axis the quantity is discrete -- one-hot blocks give exactly
    # L+1 possible distances, 2m/(L*C) -- so it ticks at the values it can take
    # rather than at round numbers, one tick per plotted point.
    if mode == 'codemse':
      xt = list(next(iter(cells.values()))[0])
      f_xt = f_tick * 0.82        # 9 labels in a panel sized for 6
      xfmt = lambda v: '0' if v == 0 else ('%.2f' % v).lstrip('0')
    else:
      xt, f_xt, xfmt = ticks(xmax, 7), f_tick, fmt
    for v in xt:
      out.append(f'<line x1="{sx(v):.1f}" y1="{sy(0):.1f}" x2="{sx(v):.1f}" '
                 f'y2="{sy(0) + pt(2.0, fig_w):.1f}" stroke="{INK3}" '
                 f'stroke-width="{pt(0.7, fig_w):.2f}"/>')
      out.append(f'<text x="{sx(v):.1f}" y="{sy(0) + f_tick * 1.6:.1f}" '
                 f'font-size="{f_xt:.1f}" fill="{INK2}" '
                 f'text-anchor="middle">{xfmt(v)}</text>')
    out.append(f'<text x="{x0 + PANEL_W / 2:.1f}" '
               f'y="{sy(0) + f_tick * 1.6 + f_axis * 1.6:.1f}" '
               f'font-size="{f_axis:.1f}" fill="{INK}" text-anchor="middle">'
               f'{xlabel}</text>')
    if ti == 0:
      yc = pad_t + PANEL_H / 2
      out.append(f'<text x="{pt(9, fig_w):.1f}" y="{yc:.1f}" '
                 f'font-size="{f_axis:.1f}" fill="{INK}" text-anchor="middle" '
                 f'transform="rotate(-90 {pt(9, fig_w):.1f} {yc:.1f})">'
                 f'decoded goal shift (MSE)</text>')
    out.append(f'<line x1="{x0:.1f}" y1="{sy(0):.1f}" x2="{x0 + PANEL_W:.1f}" '
               f'y2="{sy(0):.1f}" stroke="{INK3}" '
               f'stroke-width="{pt(0.7, fig_w):.2f}"/>')

    ends = []
    for ai, (arm, alabel) in enumerate(ARMS):
      if (task, arm) not in cells:
        continue
      xs, mean, lo, hi, cov, n = cells[(task, arm)]
      color = REF if arm == 'director' else SERIES[(ai - 1) % len(SERIES)]
      solid = [i for i in range(len(xs))
               if cov[i] >= min_cov and mean[i] == mean[i]]
      faded = [i for i in range(len(xs))
               if 0 < cov[i] < min_cov and mean[i] == mean[i]]
      if not solid:
        continue
      if band:
        poly = ([(sx(xs[i]), sy(hi[i])) for i in solid] +
                [(sx(xs[i]), sy(lo[i])) for i in solid][::-1])
        out.append('<polygon points="%s" fill="%s" fill-opacity="0.16" '
                   'stroke="none"/>' % (
                       ' '.join(f'{x:.1f},{y:.1f}' for x, y in poly), color))
      # the sparsely-reachable tail, drawn but visibly weaker
      if faded:
        tail = [solid[-1]] + faded
        d = 'M' + ' L'.join(f'{sx(xs[i]):.1f},{sy(mean[i]):.1f}' for i in tail)
        out.append(f'<path d="{d}" fill="none" stroke="{color}" '
                   f'stroke-width="{pt(1.2, fig_w):.2f}" stroke-opacity="0.35" '
                   f'stroke-dasharray="{pt(2.5, fig_w):.1f},'
                   f'{pt(2.0, fig_w):.1f}"/>')
      d = 'M' + ' L'.join(f'{sx(xs[i]):.1f},{sy(mean[i]):.1f}' for i in solid)
      out.append(f'<path d="{d}" fill="none" stroke="{color}" '
                 f'stroke-width="{pt(1.7, fig_w):.2f}" '
                 f'stroke-linejoin="round"/>')
      step = max(1, len(solid) // 12)
      for i in solid[::step] + [solid[-1]]:
        out.append(f'<circle cx="{sx(xs[i]):.1f}" cy="{sy(mean[i]):.1f}" '
                   f'r="{pt(1.7, fig_w):.1f}" fill="{color}" stroke="white" '
                   f'stroke-width="{pt(0.7, fig_w):.2f}"/>')
      ends.append([sy(mean[solid[-1]]), color, alabel])

    ends.sort()
    minsep = f_lab * 1.15
    for i in range(1, len(ends)):
      if ends[i][0] - ends[i - 1][0] < minsep:
        ends[i][0] = ends[i - 1][0] + minsep
    for y, color, alabel in ends:
      out.append(f'<text xml:space="preserve" '
                 f'x="{x0 + PANEL_W + pt(4, fig_w):.1f}" '
                 f'y="{y + f_lab * 0.36:.1f}" font-size="{f_lab:.1f}" '
                 f'fill="{color}">{esc(alabel)}</text>')

  out.append('</svg>')
  return '\n'.join(out), fig_w, fig_h


def summarize(data, mode, min_cov):
  if mode == 'total':
    print(f'\n{"task":22s} {"arm":26s} {"n":>2s} '
          f'{"Dlim":>10s} {"maxD":>5s} {"MSE D=8":>10s} {"MSE Dlim":>10s} '
          f'{"Dlim/D8":>12s}   (Dlim = largest D at coverage >= %.1f)'
          % min_cov)
    for key in sorted(data):
      seeds = data[key]
      xs = seeds[0][2]
      c = np.stack([x[0] for x in seeds])
      cov = np.stack([x[1] for x in seeds]).mean(0)
      lim = max(np.flatnonzero(cov >= min_cov))
      mx = int(xs[max(np.flatnonzero(cov > 0))])
      d8 = c[:, 8]
      top = c[:, lim]
      ratio = top / np.maximum(d8, 1e-30)
      print(f'{key[0].replace("dmc_", ""):22s} {key[1]:26s} {len(c):2d} '
            f'{int(xs[lim]):10d} {mx:5d} {d8.mean():10.4g} '
            f'{np.nanmean(top):10.4g} {np.nanmean(ratio):6.2f}'
            f'+/-{np.nanstd(ratio):<5.2f}')
  else:
    # At the largest code distance every block has changed, so the code-space
    # MSE is identical for every draw; whatever goal-shift spread remains is
    # movement the code's own metric cannot represent.
    print(f'\n{"task":22s} {"arm":26s} {"n":>2s} {"goal MSE":>10s} '
          f'{"p10":>9s} {"p90":>9s} {"p90/p10":>9s}   (at full code distance)')
    for key in sorted(data):
      seeds = data[key]
      mean = np.stack([x[0] for x in seeds]).mean(0)[-1]
      lo = np.stack([x[3] for x in seeds]).mean(0)[-1]
      hi = np.stack([x[4] for x in seeds]).mean(0)[-1]
      print(f'{key[0].replace("dmc_", ""):22s} {key[1]:26s} {len(seeds):2d} '
            f'{mean:10.4g} {lo:9.4g} {hi:9.4g} '
            f'{hi / max(lo, 1e-30):9.2f}')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--results', default=str(HERE / 'results'))
  ap.add_argument('--x', choices=['total', 'codemse'], default='total')
  ap.add_argument('--out', default=None)
  ap.add_argument('--ymax', type=float, default=None)
  ap.add_argument('--min-coverage', type=float, default=0.5)
  ap.add_argument('--include-all', action='store_true',
                  help='also draw the estimator-free arms (see EXCLUDE)')
  ap.add_argument('--copy-to', default=None)
  a = ap.parse_args()

  out_base = a.out or str(HERE / ('goal_code_index_total' if a.x == 'total'
                                  else 'goal_code_mse_vs_mse'))
  data_all = load(a.results, a.x)
  # Colour comes from each arm's fixed slot in ARMS, not from its position in
  # the filtered list, so hiding an arm never repaints the others.
  data = (data_all if a.include_all else
          {k: v for k, v in data_all.items() if k[1] not in EXCLUDE})
  if not data:
    print('no npz with index_total/* in', a.results)
    return
  print('cells:', ', '.join(f'{t.replace("dmc_", "")}/{arm}={len(v)}'
                            for (t, arm), v in sorted(data.items())))
  summarize(data_all, a.x, a.min_coverage)
  svg, w, h = build(data, a.x, a.ymax, a.min_coverage)
  svg_path, pdf_path = out_base + '.svg', out_base + '.pdf'
  with open(svg_path, 'w') as f:
    f.write(svg)
  try:
    subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', pdf_path, svg_path],
                   check=True)
  except Exception as exc:
    print('rsvg-convert failed:', exc)
    pdf_path = None
  print(f'wrote {svg_path} ({w:.0f}x{h:.0f}) -> {pdf_path}')
  if a.copy_to and pdf_path:
    os.makedirs(a.copy_to, exist_ok=True)
    subprocess.run(['cp', pdf_path, os.path.join(
        a.copy_to, os.path.basename(pdf_path))], check=True)
    print('copied to', a.copy_to)


if __name__ == '__main__':
  main()
