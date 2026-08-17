#!/usr/bin/env python3
"""Learning curves for the goal-AE arms, as small multiples.

One panel per (task, arm), each showing that arm's mean across seeds against
the Director baseline in grey. Small multiples rather than six lines in one
axes: six overlapping curves need six hues that separate under colour-vision
deficiency at *any* pairing, which the reference palette explicitly does not
guarantee past three slots -- and the question the reader has is "does this arm
beat Director", which is a two-series question. Faceting answers it once per
panel with two series, and never asks anyone to untangle a spaghetti plot.

Bands are +/- 1 standard deviation across seeds (population std, the same
convention as the endpoint figure and the tables).

  python3 make_figure_curves.py --json results/arms_3M.json --out arm_curves
"""
import argparse
import json
import math
import os
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
EXCLUDE = ('som_orig_line', 'som_orig_lipvq_line_prod')
REF_LINE = '#8a8880'
REF_BAND = '#ebeae6'
INK = '#0b0b0b'
INK2 = '#52514e'
INK3 = '#8a8880'
GRID = '#eeede9'

TEXTWIDTH_PT = 397.0
TARGET_PT = 0.98 * TEXTWIDTH_PT
RES = 6
PANEL_W, PANEL_H = 128 * RES, 96 * RES
# GAP_X has to clear two x-tick labels meeting at a panel edge; PAD_L has to
# hold a 4-digit tick plus the rotated axis title; PAD_R has to hold the
# right-most panel's direct label, which overhangs its panel.
GAP_X, GAP_Y = 20 * RES, 30 * RES
PAD_L, PAD_R = 48 * RES, 26 * RES


def pt(v, fig_w):
  return v * fig_w / TARGET_PT


def esc(s):
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def finite(seq):
  return [v for v in seq if v is not None and not math.isnan(v)]


def agg(curves):
  """Per-bin mean and +/- 1 std across seeds, skipping bins nobody reached.

  Population std (divide by n), not the sample estimate: these four seeds are
  the whole population of runs for this cell, not a sample drawn from a larger
  one, and it is the same convention the endpoint figure and every table in
  EXPERIMENTS.md use.
  """
  if not curves:
    return [], [], []
  n = len(curves[0])
  mean, lo, hi = [], [], []
  for i in range(n):
    vals = finite([c[i] for c in curves])
    if vals:
      m = sum(vals) / len(vals)
      sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
      mean.append(m); lo.append(m - sd); hi.append(m + sd)
    else:
      mean.append(None); lo.append(None); hi.append(None)
  return mean, lo, hi


def path_of(xs, ys, sx, sy):
  d, pen = [], False
  for x, y in zip(xs, ys):
    if y is None:
      pen = False
      continue
    d.append(f'{"M" if not pen else "L"}{sx(x):.1f},{sy(y):.1f}')
    pen = True
  return ' '.join(d)


def band_of(xs, lo, hi, sx, sy):
  pts_hi = [(x, v) for x, v in zip(xs, hi) if v is not None]
  pts_lo = [(x, v) for x, v in zip(xs, lo) if v is not None]
  if len(pts_hi) < 2:
    return ''
  fwd = ' '.join(f'{sx(x):.1f},{sy(v):.1f}' for x, v in pts_hi)
  back = ' '.join(f'{sx(x):.1f},{sy(v):.1f}' for x, v in reversed(pts_lo))
  return f'{fwd} {back}'


def build(data, ymax):
  tasks = data['tasks']
  # The estimator-free arms are measured and tabulated but not drawn: their
  # codebooks collapsed, and the figure is about the arms still in contention.
  arms = [a for a in data['arms']
          if a[0] != 'director' and a[0] not in EXCLUDE]
  grid = data['grid']
  xmax = grid[-1] if grid else 4e6

  ncol, nrow = len(arms), len(tasks)
  fig_w = PAD_L + ncol * PANEL_W + (ncol - 1) * GAP_X + PAD_R
  f_title = pt(7.4, fig_w)
  f_tick = pt(6.2, fig_w)
  f_axis = pt(7.0, fig_w)
  f_note = pt(6.0, fig_w)
  pad_t = f_title * 2.3
  row_h = pad_t + PANEL_H + f_tick * 2.6
  fig_h = nrow * row_h + (nrow - 1) * GAP_Y + f_note * 2.4

  out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{fig_w:.0f}" '
         f'height="{fig_h:.0f}" viewBox="0 0 {fig_w:.0f} {fig_h:.0f}" '
         f'font-family="Helvetica,Arial,sans-serif">',
         f'<rect x="0" y="0" width="{fig_w:.0f}" height="{fig_h:.0f}" '
         f'fill="white"/>']

  for r, (task, tlabel) in enumerate(tasks):
    row_top = r * (row_h + GAP_Y)
    ref = data['cells'].get(f'{task}|director')
    rmean, rlo, rhi = agg(ref['curves']) if ref else ([], [], [])

    out.append(f'<text xml:space="preserve" x="{PAD_L:.1f}" '
               f'y="{row_top + f_title * 1.0:.1f}" '
               f'font-size="{f_title:.1f}" fill="{INK}">{esc(tlabel)}'
               f'<tspan fill="{INK2}" font-size="{f_note:.1f}">'
               f'   —  grey: Director baseline, {ref["n"] if ref else 0} seeds'
               f'</tspan></text>')

    for c, (arm, alabel) in enumerate(arms):
      x0 = PAD_L + c * (PANEL_W + GAP_X)
      top = row_top + pad_t
      color = SERIES[c % len(SERIES)]

      def sx(v, x0=x0):
        return x0 + (v / xmax) * PANEL_W

      def sy(v, top=top):
        return top + (1.0 - v / ymax) * PANEL_H

      for v in range(0, int(ymax) + 1, 250):
        y = sy(v)
        out.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + PANEL_W:.1f}" '
                   f'y2="{y:.1f}" stroke="{GRID}" '
                   f'stroke-width="{pt(0.5, fig_w):.2f}"/>')
        if c == 0:
          out.append(f'<text x="{x0 - pt(3, fig_w):.1f}" '
                     f'y="{y + f_tick * 0.36:.1f}" font-size="{f_tick:.1f}" '
                     f'fill="{INK2}" text-anchor="end">{v}</text>')

      if ref:
        poly = band_of(grid, rlo, rhi, sx, sy)
        if poly:
          out.append(f'<polygon points="{poly}" fill="{REF_BAND}"/>')
        d = path_of(grid, rmean, sx, sy)
        if d:
          out.append(f'<path d="{d}" fill="none" stroke="{REF_LINE}" '
                     f'stroke-width="{pt(1.0, fig_w):.2f}"/>')

      cell = data['cells'].get(f'{task}|{arm}')
      if cell:
        amean, alo, ahi = agg(cell['curves'])
        poly = band_of(grid, alo, ahi, sx, sy)
        if poly:
          out.append(f'<polygon points="{poly}" fill="{color}" '
                     f'fill-opacity="0.18"/>')
        d = path_of(grid, amean, sx, sy)
        if d:
          out.append(f'<path d="{d}" fill="none" stroke="{color}" '
                     f'stroke-width="{pt(1.5, fig_w):.2f}" '
                     f'stroke-linejoin="round"/>')

      # Direct label inside the panel: identity never rests on colour alone.
      n = cell['n'] if cell else 0
      out.append(f'<text x="{x0 + pt(3, fig_w):.1f}" '
                 f'y="{top - pt(2.5, fig_w):.1f}" font-size="{f_note:.1f}" '
                 f'fill="{color}">{esc(alabel)}'
                 f'<tspan fill="{INK3}"> (n={n})</tspan></text>')

      out.append(f'<line x1="{x0:.1f}" y1="{sy(0):.1f}" '
                 f'x2="{x0 + PANEL_W:.1f}" y2="{sy(0):.1f}" stroke="{INK3}" '
                 f'stroke-width="{pt(0.6, fig_w):.2f}"/>')
      # Drop the leading 0M on every panel but the first: next to the
      # previous panel's final tick it renders as one unreadable "2M0M".
      for frac in ((0.0, 0.5, 1.0) if c == 0 else (0.5, 1.0)):
        xv = frac * xmax
        out.append(f'<text x="{sx(xv):.1f}" '
                   f'y="{sy(0) + f_tick * 1.5:.1f}" font-size="{f_tick:.1f}" '
                   f'fill="{INK2}" text-anchor="middle">'
                   f'{xv / 1e6:.0f}M</text>')

    ycen = row_top + pad_t + PANEL_H / 2
    out.append(f'<text x="{pt(7, fig_w):.1f}" y="{ycen:.1f}" '
               f'font-size="{f_axis:.1f}" fill="{INK}" text-anchor="middle" '
               f'transform="rotate(-90 {pt(7, fig_w):.1f} {ycen:.1f})">'
               f'episode return</text>')

  out.append(f'<text x="{fig_w / 2:.1f}" y="{fig_h - pt(2, fig_w):.1f}" '
             f'font-size="{f_note:.1f}" fill="{INK3}" text-anchor="middle">'
             f'line: mean across seeds; band: +/-1 std across seeds; '
             f'x: environment steps</text>')
  out.append('</svg>')
  return '\n'.join(out), fig_w, fig_h


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--json', default=str(HERE / 'results' / 'arms_3M.json'))
  ap.add_argument('--out', default=str(HERE / 'arm_curves'))
  ap.add_argument('--ymax', type=float, default=1000.0)
  ap.add_argument('--copy-to', default=None)
  a = ap.parse_args()

  with open(a.json) as f:
    data = json.load(f)
  svg, w, h = build(data, a.ymax)
  svg_path, pdf_path = a.out + '.svg', a.out + '.pdf'
  os.makedirs(os.path.dirname(os.path.abspath(svg_path)), exist_ok=True)
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
