#!/usr/bin/env python3
"""Endpoint comparison figure: five goal-AE arms against the Director baseline.

Hand-rolled SVG, converted to PDF with rsvg-convert, because matplotlib does
not load anywhere on this cluster (rl_env is missing libpng15 and the JAX env
cannot run on the login node). Same approach as
experiments/goal_code_struct_corr/make_figure.py, and the same sizing
discipline: the figure is drawn at high internal resolution and shrunk to
TARGET_PT when LaTeX places it, so font sizes are chosen in FINAL points.

Form: one panel per task, arms along x, episode return on y, and every seed
drawn as its own dot. With four seeds a bar of the mean would hide the thing
the reader most needs -- whether the seeds overlap the baseline's -- so the
seeds are the marks and the mean is a tick across them. The Director baseline
is not a sixth series but the reference: a grey band spanning the panel at its
min-max across seeds, with its mean as a line. An arm beats it when its dots
sit clear of that band, which is exactly the pre-registered reading.

Colour: the reference palette's categorical slots in their fixed documented
order (blue, orange, aqua, yellow, magenta), Director in neutral grey. Identity
never rests on colour -- every arm is directly labelled on the x axis -- which
also discharges the contrast-relief rule for the lighter slots. Adjacent pairs
here are (blue,orange), (orange,aqua), (aqua,yellow), (yellow,magenta), all of
which the palette documents as passing; orange and yellow are never adjacent.

  python3 make_figure_arms.py --json results/arms_3M.json \
      --out arm_comparison --copy-to ../../../26_04_HRL-paper/figures/motivation
"""
import argparse
import json
import os
import math
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent

# Measured and tabulated, but not drawn: collapsed codebooks.
EXCLUDE = ('som_orig_line', 'som_orig_lipvq_line_prod')

# Reference palette, categorical slots 1-5, light mode, fixed order.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
REF_LINE = '#52514e'      # text-secondary: the Director mean
REF_BAND = '#e6e5e1'      # its seed min-max
INK = '#0b0b0b'
INK2 = '#52514e'
INK3 = '#8a8880'
GRID = '#e6e5e1'

TEXTWIDTH_PT = 397.0
INCLUDE_FRAC = 0.95
TARGET_PT = INCLUDE_FRAC * TEXTWIDTH_PT
# Panels are stacked VERTICALLY, one task per row, so each is the full text
# width. Side by side, a six-slot axis left ~50pt per arm and "SOM-line-OG"
# does not fit in that at a readable size; rotating the labels instead drove
# them through the p-value row underneath. Full width gives ~63pt a slot and
# every label sits flat inside its own column.
RES = 6
PANEL_W, PANEL_H = 660 * RES, 150 * RES
GAP = 34 * RES
# PAD_L holds a 4-digit tick label AND the rotated axis title. At 46*RES the
# title was placed at a negative x and rendered off-canvas entirely -- the
# figure had no y-axis label at all and it took rendering it to notice.
PAD_L, PAD_R = 68 * RES, 30 * RES


def pt(final_pt, fig_w):
  return final_pt * fig_w / TARGET_PT


def esc(s):
  return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def halo(x, y, size, fill, anchor, text, fig_w, weight=1.8):
  """Text with a white outline drawn underneath it.

  Two elements rather than `paint-order="stroke"`: rsvg-convert ignores
  paint-order and renders the stroke over the glyph, which turned every value
  label into a white blob.
  """
  common = (f'x="{x:.1f}" y="{y:.1f}" font-size="{size:.1f}" '
            f'text-anchor="{anchor}"')
  return (f'<text {common} fill="none" stroke="white" '
          f'stroke-width="{pt(weight, fig_w):.2f}" stroke-linejoin="round">'
          f'{text}</text>'
          f'<text {common} fill="{fill}">{text}</text>')


def yrange(data, task, ymax_cap):
  """Per-panel y range fitted to the data.

  The two panels are different tasks with different achievable returns, so a
  shared 0-1000 axis is not the comparison anyone wants: on hopper_stand every
  value sits between 750 and 830 and a full-height axis renders the one task
  that can resolve a difference as six identical rows. These are dot plots --
  position encodes the value, not length -- so a non-zero baseline is not the
  bar-chart sin it would be. What keeps it honest is the Director band, which
  is drawn on the same fitted axis: if an arm's dots sit inside the baseline's
  own seed spread, the reader sees that at whatever zoom.
  """
  vals = []
  for arm, _ in data['arms']:
    cell = data['cells'].get(f'{task}|{arm}')
    if cell:
      vals += list(cell['values'])
  if not vals:
    return 0.0, ymax_cap
  lo, hi = min(vals), max(vals)
  pad = max((hi - lo) * 0.22, 25.0)
  lo, hi = max(0.0, lo - pad), min(ymax_cap, hi + pad)
  step = 50 if (hi - lo) <= 400 else 200
  lo = step * math.floor(lo / step)
  hi = step * math.ceil(hi / step)
  return lo, hi


def build(data, ymax):
  tasks = [(t, l) for t, l in data['tasks']]
  arms = [(a, l) for a, l in data['arms'] if a not in EXCLUDE]
  n = len(tasks)
  fig_w = PANEL_W + PAD_L + PAD_R
  f_title = pt(9.0, fig_w)
  f_tick = pt(7.0, fig_w)
  f_xlab = pt(6.3, fig_w)   # 'SOM-line-OG + LiP' must fit one slot
  f_axis = pt(7.5, fig_w)
  f_note = pt(6.2, fig_w)
  # Per row: title, baseline subtitle, panel, one label line, one p line.
  pad_t = f_title * 1.5 + f_note * 1.9
  pad_b = f_tick * 1.6 + f_note * 1.6
  row_h = pad_t + PANEL_H + pad_b
  fig_h = n * row_h + (n - 1) * GAP + f_note * 3.6

  ranges = {t: yrange(data, t, ymax) for t, _ in tasks}

  def sy(v, top, task):
    lo, hi = ranges[task]
    return top + (1.0 - (v - lo) / (hi - lo)) * PANEL_H

  out = [
      f'<svg xmlns="http://www.w3.org/2000/svg" width="{fig_w:.0f}" '
      f'height="{fig_h:.0f}" viewBox="0 0 {fig_w:.0f} {fig_h:.0f}" '
      f'font-family="Helvetica,Arial,sans-serif">',
      f'<rect x="0" y="0" width="{fig_w:.0f}" height="{fig_h:.0f}" '
      f'fill="white"/>']

  x0 = PAD_L
  for i, (task, tlabel) in enumerate(tasks):
    row_top = i * (row_h + GAP)
    top = row_top + pad_t
    ref = data['cells'].get(f'{task}|director')

    out.append(f'<text x="{x0 + PANEL_W / 2:.1f}" '
               f'y="{row_top + f_title * 1.1:.1f}" '
               f'font-size="{f_title:.1f}" fill="{INK}" text-anchor="middle">'
               f'{esc(tlabel)}</text>')
    # The baseline is named above the plot, so the annotation can never land on
    # the data it is the reference for.
    if ref:
      out.append(
          f'<text x="{x0 + PANEL_W / 2:.1f}" '
          f'y="{row_top + f_title * 1.1 + f_note * 1.6:.1f}" '
          f'font-size="{f_note:.1f}" fill="{INK2}" text-anchor="middle">'
          f'Director baseline {ref["mean"]:.0f} '
          f'({ref["min"]:.0f}–{ref["max"]:.0f} across {ref["n"]} seeds, '
          f'shaded band)</text>')

    lo_r, hi_r = ranges[task]
    n_ticks = 5
    tick = (hi_r - lo_r) / n_ticks
    for i in range(n_ticks + 1):
      v = lo_r + i * tick
      y = sy(v, top, task)
      out.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + PANEL_W:.1f}" '
                 f'y2="{y:.1f}" stroke="{GRID}" '
                 f'stroke-width="{pt(0.5, fig_w):.2f}"/>')
      out.append(f'<text x="{x0 - pt(4, fig_w):.1f}" '
                 f'y="{y + f_tick * 0.36:.1f}" font-size="{f_tick:.1f}" '
                 f'fill="{INK2}" text-anchor="end">{v:.0f}</text>')
    ycen = top + PANEL_H / 2
    out.append(f'<text x="{pt(11, fig_w):.1f}" y="{ycen:.1f}" '
               f'font-size="{f_axis:.1f}" fill="{INK}" text-anchor="middle" '
               f'transform="rotate(-90 {pt(11, fig_w):.1f} {ycen:.1f})">'
               f'episode return</text>')

    if ref:
      ytop, ybot = sy(ref['max'], top, task), sy(ref['min'], top, task)
      out.append(f'<rect x="{x0:.1f}" y="{ytop:.1f}" width="{PANEL_W:.1f}" '
                 f'height="{max(ybot - ytop, 1):.1f}" fill="{REF_BAND}"/>')
      ym = sy(ref['mean'], top, task)
      out.append(f'<line x1="{x0:.1f}" y1="{ym:.1f}" x2="{x0 + PANEL_W:.1f}" '
                 f'y2="{ym:.1f}" stroke="{REF_LINE}" '
                 f'stroke-width="{pt(0.9, fig_w):.2f}" stroke-dasharray="'
                 f'{pt(3, fig_w):.1f},{pt(2.4, fig_w):.1f}"/>')

    slot = PANEL_W / len(arms)
    for k, (arm, alabel) in enumerate(arms):
      cx = x0 + slot * (k + 0.5)
      cell = data['cells'].get(f'{task}|{arm}')
      color = REF_LINE if arm == 'director' else SERIES[(k - 1) % len(SERIES)]

      out.append(f'<text x="{cx:.1f}" y="{top + PANEL_H + f_tick * 1.5:.1f}" '
                 f'font-size="{f_xlab:.1f}" fill="{INK}" '
                 f'text-anchor="middle">{esc(alabel)}</text>')

      if not cell:
        out.append(f'<text x="{cx:.1f}" y="{sy(ranges[task][0], top, task) - pt(7, fig_w):.1f}" '
                   f'font-size="{f_note:.1f}" fill="{INK3}" '
                   f'text-anchor="middle">not yet</text>')
        continue

      r = pt(2.5, fig_w)
      for j, v in enumerate(cell['values']):
        jitter = (j - (len(cell['values']) - 1) / 2.0) * pt(3.6, fig_w)
        out.append(f'<circle cx="{cx + jitter:.1f}" cy="{sy(v, top, task):.1f}" '
                   f'r="{r:.1f}" fill="{color}" stroke="white" '
                   f'stroke-width="{pt(1.0, fig_w):.2f}"/>')
      half = slot * 0.30
      ym = sy(cell['mean'], top, task)
      out.append(f'<line x1="{cx - half:.1f}" y1="{ym:.1f}" '
                 f'x2="{cx + half:.1f}" y2="{ym:.1f}" stroke="{color}" '
                 f'stroke-width="{pt(1.6, fig_w):.2f}"/>')
      out.append(halo(cx, ym - pt(5.0, fig_w), f_note, INK, 'middle',
                      f'{cell["mean"]:.0f}', fig_w))

      p = cell.get('p_vs_director')
      if p is not None:
        floor = cell.get('p_floor') or 0.029
        txt = f'p={p:.3f}' + ('*' if abs(p - floor) < 1e-9 else '')
        out.append(f'<text x="{cx:.1f}" '
                   f'y="{top + PANEL_H + f_tick * 1.5 + f_note * 1.5:.1f}" '
                   f'font-size="{f_note:.1f}" fill="{INK2}" '
                   f'text-anchor="middle">{txt}</text>')

    out.append(f'<line x1="{x0:.1f}" y1="{sy(ranges[task][0], top, task):.1f}" '
               f'x2="{x0 + PANEL_W:.1f}" y2="{sy(ranges[task][0], top, task):.1f}" '
               f'stroke="{INK3}" stroke-width="{pt(0.7, fig_w):.2f}"/>')

  # bottom note
  step = data.get('at_step')
  notes = [
      'each dot is one seed; bar is the cell mean; y range fitted per task',
      'p: exact two-sided permutation vs Director, floor 0.029 at 4 vs 4'
      + (f'; measured at {step / 1e6:.0f}M steps' if step else '')]
  for i, note in enumerate(notes):
    out.append(f'<text x="{fig_w / 2:.1f}" '
               f'y="{fig_h - pt(2.5, fig_w) - (len(notes) - 1 - i) * f_note * 1.25:.1f}" '
               f'font-size="{f_note:.1f}" fill="{INK3}" text-anchor="middle">'
               f'{esc(note)}</text>')
  out.append('</svg>')
  return '\n'.join(out), fig_w, fig_h


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--json', default=str(HERE / 'results' / 'arms_3M.json'))
  ap.add_argument('--out', default=str(HERE / 'arm_comparison'))
  ap.add_argument('--ymax', type=float, default=1000.0)
  ap.add_argument('--copy-to', default=None)
  a = ap.parse_args()

  with open(a.json) as f:
    data = json.load(f)
  svg, w, h = build(data, a.ymax)

  svg_path = a.out + '.svg'
  pdf_path = a.out + '.pdf'
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
    dest = os.path.join(a.copy_to, os.path.basename(pdf_path))
    os.makedirs(a.copy_to, exist_ok=True)
    subprocess.run(['cp', pdf_path, dest], check=True)
    print('copied to', dest)


if __name__ == '__main__':
  main()
