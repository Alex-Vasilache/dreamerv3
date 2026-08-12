#!/usr/bin/env python3
"""The code->goal landscape: is a code index a label or a coordinate?

For each arm: take a state's code, move ONE block k classes away, decode, and
measure how far the decoded goal moved. Plotted against k this is flat when
entry indices are arbitrary labels (Director, whose categorical head has no
topology) and a rising line when they are a coordinate (a SOM on an open path,
where adjacent entries are trained to decode nearby).

This is the figure that states motivation.tex's argument as a measurement.
No correlation, no p-value: a flat line and a ramp are different objects and
the reader can see which is which.

Six series in one panel is normally where a categorical palette runs out, but
identity here does not rest on hue -- the curves separate monotonically and
each is labelled at its own right-hand end, so position carries it and colour
only reinforces. Director is drawn in neutral grey as the reference, and the
palette's fixed slot order supplies the rest.

  python3 make_figure_landscape.py --out goal_code_landscape \
      --copy-to ../../../26_04_HRL-paper/figures/motivation
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
TASKS = [('dmc_cartpole_swingup', 'Cartpole Swingup'),
         ('dmc_hopper_stand', 'Hopper Stand')]

TEXTWIDTH_PT = 397.0
TARGET_PT = 0.95 * TEXTWIDTH_PT
RES = 6
PANEL_W, PANEL_H = 205 * RES, 200 * RES
GAP = 46 * RES
# The right pad has to hold the longest end-label in full
# ("SOM-line-OG + LiP (n=4)"); at 76*RES it clipped the second panel's.
PAD_L, PAD_R = 44 * RES, 138 * RES
PAD_T_EXTRA = 4 * RES


def pt(v, w):
  return v * w / TARGET_PT


def esc(s):
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def load(results_dir):
  """(task, arm) -> list of per-seed displacement curves."""
  out = {}
  for f in sorted(glob.glob(os.path.join(results_dir, '*.npz'))):
    d = np.load(f, allow_pickle=True)
    meta = ast.literal_eval(str(d['meta']))
    name = os.path.basename(f)
    arm = None
    for key, _ in sorted(ARMS, key=lambda a: -len(a[0])):
      if f'_{key}_s' in name:
        arm = key
        break
    if arm is None and meta.get('impl') == 'director':
      arm = 'director'
    if arm is None:
      continue
    out.setdefault((meta['task'], arm), []).append(np.array(d['sweep/curve'],
                                                           float))
  return out


def build(data, ymax=None):
  fig_w = 2 * PANEL_W + GAP + PAD_L + PAD_R
  f_title = pt(9.0, fig_w)
  f_tick = pt(7.0, fig_w)
  f_axis = pt(7.5, fig_w)
  f_lab = pt(6.4, fig_w)
  pad_t = f_title * 1.5 + PAD_T_EXTRA
  pad_b = f_tick * 1.8 + f_axis * 1.8
  fig_h = PANEL_H + pad_t + pad_b

  if ymax is None:
    vals = [v for curves in data.values() for c in curves for v in c[1:]
            if v == v]
    ymax = max(vals) * 1.08 if vals else 15.0

  out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{fig_w:.0f}" '
         f'height="{fig_h:.0f}" viewBox="0 0 {fig_w:.0f} {fig_h:.0f}" '
         f'font-family="Helvetica,Arial,sans-serif">',
         f'<rect x="0" y="0" width="{fig_w:.0f}" height="{fig_h:.0f}" '
         f'fill="white"/>']

  kmax = 7
  for ti, (task, tlabel) in enumerate(TASKS):
    x0 = PAD_L + ti * (PANEL_W + GAP)

    def sx(k, x0=x0):
      return x0 + (k - 1) / (kmax - 1) * PANEL_W

    def sy(v):
      return pad_t + (1.0 - v / ymax) * PANEL_H

    out.append(f'<text x="{x0 + PANEL_W / 2:.1f}" y="{f_title * 1.1:.1f}" '
               f'font-size="{f_title:.1f}" fill="{INK}" text-anchor="middle">'
               f'{esc(tlabel)}</text>')

    step = max(2, int(round(ymax / 6 / 2)) * 2)
    for v in range(0, int(ymax) + 1, step):
      y = sy(v)
      out.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + PANEL_W:.1f}" '
                 f'y2="{y:.1f}" stroke="{GRID}" '
                 f'stroke-width="{pt(0.5, fig_w):.2f}"/>')
      if ti == 0:
        out.append(f'<text x="{x0 - pt(4, fig_w):.1f}" '
                   f'y="{y + f_tick * 0.36:.1f}" font-size="{f_tick:.1f}" '
                   f'fill="{INK2}" text-anchor="end">{v}</text>')
    for k in range(1, kmax + 1):
      out.append(f'<text x="{sx(k):.1f}" y="{sy(0) + f_tick * 1.6:.1f}" '
                 f'font-size="{f_tick:.1f}" fill="{INK2}" '
                 f'text-anchor="middle">{k}</text>')
    out.append(f'<text x="{x0 + PANEL_W / 2:.1f}" '
               f'y="{sy(0) + f_tick * 1.6 + f_axis * 1.6:.1f}" '
               f'font-size="{f_axis:.1f}" fill="{INK}" text-anchor="middle">'
               f'code index distance k</text>')
    if ti == 0:
      yc = pad_t + PANEL_H / 2
      out.append(f'<text x="{pt(9, fig_w):.1f}" y="{yc:.1f}" '
                 f'font-size="{f_axis:.1f}" fill="{INK}" text-anchor="middle" '
                 f'transform="rotate(-90 {pt(9, fig_w):.1f} {yc:.1f})">'
                 f'decoded goal shift</text>')
    out.append(f'<line x1="{x0:.1f}" y1="{sy(0):.1f}" x2="{x0 + PANEL_W:.1f}" '
               f'y2="{sy(0):.1f}" stroke="{INK3}" '
               f'stroke-width="{pt(0.7, fig_w):.2f}"/>')

    ends = []
    for ai, (arm, alabel) in enumerate(ARMS):
      curves = data.get((task, arm))
      if not curves:
        continue
      m = np.nanmean(np.stack(curves), 0)
      color = REF if arm == 'director' else SERIES[(ai - 1) % len(SERIES)]
      pts = [(sx(k), sy(m[k])) for k in range(1, kmax + 1) if m[k] == m[k]]
      if len(pts) < 2:
        continue
      d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts)
      out.append(f'<path d="{d}" fill="none" stroke="{color}" '
                 f'stroke-width="{pt(1.7, fig_w):.2f}" '
                 f'stroke-linejoin="round"/>')
      for x, y in pts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{pt(1.7, fig_w):.1f}" '
                   f'fill="{color}" stroke="white" '
                   f'stroke-width="{pt(0.7, fig_w):.2f}"/>')
      ends.append([pts[-1][1], color, alabel, len(curves)])

    # Direct end-labels, pushed apart so overlapping curve ends stay readable.
    ends.sort()
    minsep = f_lab * 1.15
    for i in range(1, len(ends)):
      if ends[i][0] - ends[i - 1][0] < minsep:
        ends[i][0] = ends[i - 1][0] + minsep
    for y, color, alabel, n in ends:
      out.append(f'<text xml:space="preserve" x="{x0 + PANEL_W + pt(4, fig_w):.1f}" '
                 f'y="{y + f_lab * 0.36:.1f}" font-size="{f_lab:.1f}" '
                 f'fill="{color}">{esc(alabel)}'
                 f'<tspan fill="{INK3}"> (n={n})</tspan></text>')

  out.append('</svg>')
  return '\n'.join(out), fig_w, fig_h


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--results', default=str(HERE / 'results'))
  ap.add_argument('--out', default=str(HERE / 'goal_code_landscape'))
  ap.add_argument('--ymax', type=float, default=None)
  ap.add_argument('--copy-to', default=None)
  a = ap.parse_args()

  data = load(a.results)
  if not data:
    print('no npz results in', a.results)
    return
  print('cells:', ', '.join(f'{t.replace("dmc_", "")}/{arm}={len(v)}'
                            for (t, arm), v in sorted(data.items())))
  svg, w, h = build(data, a.ymax)
  svg_path, pdf_path = a.out + '.svg', a.out + '.pdf'
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
