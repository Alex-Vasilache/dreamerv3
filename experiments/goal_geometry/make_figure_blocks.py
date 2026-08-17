#!/usr/bin/env python3
"""How far does the decoded goal move when m of the L code blocks change?

The manager does not nudge a class index, it rewrites whole blocks: a REINFORCE
update raises or lowers the probability of the code it just sampled, and the
smallest move it can make from there is one block. So the axis that matters for
"can this policy make a small correction" is Hamming distance in blocks, and the
question is whether the decoded goal moves a little when one block changes and a
lot when all eight do.

For every reference state and every m = 0..L we draw codes at Hamming distance
exactly m -- a random subset of m blocks, each sent to a uniformly random other
class -- decode, and record the mean squared error against the reference goal.
m=0 is 0 by construction.

Two panels, one per task, six arms each. Absolute MSE, because the size of the
smallest available edit is half the claim; the shape alone would hide a code
whose every edit is huge. `--normalize` divides each curve by its own m=L value
to compare shapes when that is the question instead.

  python3 make_figure_blocks.py --out goal_code_blocks \
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
# The left panel's end-labels are drawn to the right of its plot area, so the
# inter-panel gap -- not just the figure's right pad -- has to hold the longest
# of them. At 46*RES they printed on top of the second panel. With per-task y
# scales the gap has to hold the second panel's tick labels as well.
GAP = 168 * RES
PAD_L, PAD_R = 62 * RES, 138 * RES
PAD_T_EXTRA = 4 * RES


def pt(v, w):
  return v * w / TARGET_PT


def esc(s):
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def load(results_dir):
  """(task, arm) -> list of per-seed (L+1,) mean MSE curves."""
  out = {}
  for f in sorted(glob.glob(os.path.join(results_dir, '*.npz'))):
    d = np.load(f, allow_pickle=True)
    if 'hamming/mse' not in d:
      continue
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
    # mean over reference states; the spread that matters for the paper is
    # across seeds, which is taken at the figure/summary stage.
    out.setdefault((meta['task'], arm), []).append(
        np.asarray(d['hamming/mse'], float).mean(0))
  return out


def ticks(hi, n=5):
  raw = hi / float(n)
  mag = 10.0 ** np.floor(np.log10(raw)) if raw > 0 else 1.0
  step = min([m * mag for m in (1, 2, 2.5, 5, 10)], key=lambda s: abs(s - raw))
  return [v for v in np.arange(0, hi * 1.0001, step)]


def fmt(v):
  return '0' if v == 0 else ('%.3g' % v)


def build(data, ymax=None, normalize=False, band=False):
  fig_w = 2 * PANEL_W + GAP + PAD_L + PAD_R
  f_title = pt(9.0, fig_w)
  f_tick = pt(7.0, fig_w)
  f_axis = pt(7.5, fig_w)
  f_lab = pt(6.4, fig_w)
  pad_t = f_title * 1.5 + PAD_T_EXTRA
  pad_b = f_tick * 1.8 + f_axis * 1.8
  fig_h = PANEL_H + pad_t + pad_b

  curves = {}
  for key, seeds in data.items():
    a = np.stack(seeds)
    if normalize:
      a = a / np.maximum(a[:, -1:], 1e-30)
    curves[key] = (a.mean(0), a.std(0), len(seeds))
  # Per-task y range. The two tasks differ in scale by ~2x, and a shared axis
  # spent half of the Hopper panel's height on empty space.
  ymax_of = {}
  for task, _ in TASKS:
    vals = [(m + (sd if band else 0)).max()
            for (t, _a), (m, sd, _n) in curves.items() if t == task]
    ymax_of[task] = ymax if ymax else (max(vals) * 1.08 if vals else 1.0)

  out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{fig_w:.0f}" '
         f'height="{fig_h:.0f}" viewBox="0 0 {fig_w:.0f} {fig_h:.0f}" '
         f'font-family="Helvetica,Arial,sans-serif">',
         f'<rect x="0" y="0" width="{fig_w:.0f}" height="{fig_h:.0f}" '
         f'fill="white"/>']

  mmax = 8
  for ti, (task, tlabel) in enumerate(TASKS):
    x0 = PAD_L + ti * (PANEL_W + GAP)
    ymax_t = ymax_of[task]

    def sx(m, x0=x0):
      return x0 + m / float(mmax) * PANEL_W

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
    for m in range(0, mmax + 1):
      out.append(f'<text x="{sx(m):.1f}" y="{sy(0) + f_tick * 1.6:.1f}" '
                 f'font-size="{f_tick:.1f}" fill="{INK2}" '
                 f'text-anchor="middle">{m}</text>')
    out.append(f'<text x="{x0 + PANEL_W / 2:.1f}" '
               f'y="{sy(0) + f_tick * 1.6 + f_axis * 1.6:.1f}" '
               f'font-size="{f_axis:.1f}" fill="{INK}" text-anchor="middle">'
               f'code blocks changed</text>')
    if ti == 0:
      yc = pad_t + PANEL_H / 2
      lab = ('decoded goal shift, MSE / MSE at 8' if normalize
             else 'decoded goal shift (MSE)')
      out.append(f'<text x="{pt(9, fig_w):.1f}" y="{yc:.1f}" '
                 f'font-size="{f_axis:.1f}" fill="{INK}" text-anchor="middle" '
                 f'transform="rotate(-90 {pt(9, fig_w):.1f} {yc:.1f})">'
                 f'{lab}</text>')
    out.append(f'<line x1="{x0:.1f}" y1="{sy(0):.1f}" x2="{x0 + PANEL_W:.1f}" '
               f'y2="{sy(0):.1f}" stroke="{INK3}" '
               f'stroke-width="{pt(0.7, fig_w):.2f}"/>')

    ends = []
    for ai, (arm, alabel) in enumerate(ARMS):
      if (task, arm) not in curves:
        continue
      mean, std, n = curves[(task, arm)]
      color = REF if arm == 'director' else SERIES[(ai - 1) % len(SERIES)]
      ms = list(range(mmax + 1))
      if band and n > 1:
        poly = ([(sx(m), sy(mean[m] + std[m])) for m in ms] +
                [(sx(m), sy(max(mean[m] - std[m], 0.0))) for m in ms][::-1])
        out.append('<polygon points="%s" fill="%s" fill-opacity="0.16" '
                   'stroke="none"/>' % (
                       ' '.join(f'{x:.1f},{y:.1f}' for x, y in poly), color))
      pts = [(sx(m), sy(mean[m])) for m in ms]
      d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts)
      out.append(f'<path d="{d}" fill="none" stroke="{color}" '
                 f'stroke-width="{pt(1.7, fig_w):.2f}" '
                 f'stroke-linejoin="round"/>')
      for x, y in pts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" '
                   f'r="{pt(1.7, fig_w):.1f}" fill="{color}" stroke="white" '
                   f'stroke-width="{pt(0.7, fig_w):.2f}"/>')
      ends.append([pts[-1][1], color, alabel, n])

    ends.sort()
    minsep = f_lab * 1.15
    for i in range(1, len(ends)):
      if ends[i][0] - ends[i - 1][0] < minsep:
        ends[i][0] = ends[i - 1][0] + minsep
    for y, color, alabel, n in ends:
      out.append(f'<text xml:space="preserve" '
                 f'x="{x0 + PANEL_W + pt(4, fig_w):.1f}" '
                 f'y="{y + f_lab * 0.36:.1f}" font-size="{f_lab:.1f}" '
                 f'fill="{color}">{esc(alabel)}</text>')

  out.append('</svg>')
  return '\n'.join(out), fig_w, fig_h


def summarize(data):
  """Per-cell statistics, seed mean +/- seed std.

  ``add`` is the additivity ratio MSE(L) / (L x MSE(1)): what changing all L
  blocks costs, against L times what changing one costs. It is 1 when blocks
  contribute independently, above 1 when they interact (the whole edit moves the
  goal further than its parts), below 1 when the goal shift saturates. This is
  the statistic that matters for the manager's credit assignment: REINFORCE
  raises the probability of the blocks it sampled, which only transfers to the
  next decision if a block's contribution does not depend on the other seven.
  """
  print(f'\n{"task":22s} {"arm":26s} {"n":>2s} {"MSE m=1":>18s} '
        f'{"MSE m=8":>18s} {"m1/m8":>14s} {"add":>14s}')
  for (task, arm) in sorted(data):
    a = np.stack(data[(task, arm)])
    L = a.shape[1] - 1
    m1, m8 = a[:, 1], a[:, -1]
    frac = m1 / np.maximum(m8, 1e-30)
    add = m8 / np.maximum(L * m1, 1e-30)
    print(f'{task.replace("dmc_", ""):22s} {arm:26s} {len(a):2d} '
          f'{m1.mean():9.4g}+/-{m1.std():<7.2g} '
          f'{m8.mean():9.4g}+/-{m8.std():<7.2g} '
          f'{frac.mean():6.3f}+/-{frac.std():<6.3f} '
          f'{add.mean():6.2f}+/-{add.std():<6.2f}')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--results', default=str(HERE / 'results'))
  ap.add_argument('--out', default=str(HERE / 'goal_code_blocks'))
  ap.add_argument('--ymax', type=float, default=None)
  ap.add_argument('--normalize', action='store_true')
  ap.add_argument('--band', action='store_true',
                  help='shade +/- 1 std across seeds')
  ap.add_argument('--copy-to', default=None)
  a = ap.parse_args()

  data = load(a.results)
  if not data:
    print('no npz with hamming/mse in', a.results)
    return
  print('cells:', ', '.join(f'{t.replace("dmc_", "")}/{arm}={len(v)}'
                            for (t, arm), v in sorted(data.items())))
  summarize(data)
  svg, w, h = build(data, a.ymax, a.normalize, a.band)
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
