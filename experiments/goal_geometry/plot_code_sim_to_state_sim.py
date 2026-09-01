#!/usr/bin/env python3
"""Director: code similarity -> goal-state similarity, on the code's 9 bins.

Square PNG in the style of code_to_state_sim_director_only.png, but with the
x-axis the paragraph actually describes: the L+1 = 9 similarities a hard code
admits, from 0 blocks different to all 8 different. One line per task, four
seeds each, band = +/- 1 std across seeds.

Reads code_sim/*.npz (code_sim_to_state_sim.py).

  python3 plot_code_sim_to_state_sim.py --out code_sim_to_state_sim_director
"""
import argparse
import glob
import math
import os
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

# Order fixes the panel order and the colours. A task measured but missing from
# this list is silently dropped from the figure -- which is exactly what
# happened to hopper_hop on 2026-08-20: the .npz files were there, the panels
# were not.
TASKS = [
    ('dmc_cartpole_swingup', 'Cartpole Swingup', '#2a78d6'),
    ('dmc_hopper_stand', 'Hopper Stand', '#eb6834'),
    ('dmc_cartpole_swingup_sparse', 'Cartpole Swingup sparse', '#1baf7a'),
    ('dmc_cheetah_run', 'Cheetah Run', '#eda100'),
    ('dmc_hopper_hop', 'Hopper Hop', '#7b3fa0'),
]

S = 1000
PAD_L, PAD_R, PAD_T, PAD_B = 250, 150, 250, 200
PLOT = S - PAD_L - PAD_R
INK, INK_2, GRID, BASELINE = '#000000', '#1a1a1a', '#b0afa8', '#5c5b56'


def esc(s):
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def load(results_dir):
  out = {}
  for f in sorted(glob.glob(os.path.join(results_dir, '*.npz'))):
    d = np.load(f, allow_pickle=True)
    out.setdefault(str(d['task']), []).append(np.asarray(d['curve'], float))
  return out


def load_q(results_dir):
  """Per-bin percentile spread over PAIRS, averaged over seeds.

  Distinct from the seed band: ``pair_q`` is the p5/p25/p50/p75/p95 of the
  100k individual code pairs inside one bin of one run, i.e. how much the goal
  similarity VARIES at a fixed code distance. That spread is the thing the
  manager actually faces -- the mean says a maximally different code lands
  ~0.67 away, but individual pairs run from 0.31 to 0.88.
  """
  out = {}
  for f in sorted(glob.glob(os.path.join(results_dir, '*.npz'))):
    d = np.load(f, allow_pickle=True)
    if 'pair_q' in d.files:
      out.setdefault(str(d['task']), []).append(np.asarray(d['pair_q'], float))
  return {k: np.mean(v, 0) for k, v in out.items()}


def build(data, title, sub):
  plot_h = S - PAD_T - PAD_B
  nb = 9

  # x is CODE SIMILARITY, the fraction of blocks that match: bin m (blocks
  # different) sits at 1 - m/L, so all 8 different is 0.0 and none different is
  # 1.0. Both axes are then similarities on the same scale and the dashed y=x
  # is the code whose similarities equal the goal space's.
  def sx(i):
    return PAD_L + (1.0 - i / (nb - 1)) * PLOT

  def sy(v):
    return PAD_T + (1 - v) * plot_h

  o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{S}" height="{S}" '
       f'viewBox="0 0 {S} {S}" font-family="Helvetica,Arial,sans-serif">',
       f'<rect width="{S}" height="{S}" fill="white"/>',
       f'<text x="{PAD_L}" y="70" font-size="27" font-weight="bold" '
       f'fill="{INK}">{esc(title)}</text>',
       f'<text x="{PAD_L}" y="112" font-size="20" fill="{INK_2}">'
       f'{esc(sub)}</text>']

  for v in (0.0, 0.25, 0.5, 0.75, 1.0):
    y = sy(v)
    col = BASELINE if v == 0.0 else GRID
    o.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L + PLOT}" '
             f'y2="{y:.1f}" stroke="{col}" stroke-width="2"/>')
    o.append(f'<text x="{PAD_L - 78}" y="{y + 8:.1f}" font-size="21" '
             f'fill="{INK_2}" text-anchor="end">{v:.2f}</text>')
  for i in range(nb):
    lab = '%.3g' % (1.0 - i / (nb - 1))
    o.append(f'<text x="{sx(i):.1f}" y="{sy(0) + 40:.1f}" font-size="21" '
             f'fill="{INK_2}" text-anchor="middle">{lab}</text>')
  # perfect alignment: goal similarity equal to code similarity
  o.append(f'<line x1="{sx(nb - 1):.1f}" y1="{sy(0):.1f}" x2="{sx(0):.1f}" '
           f'y2="{sy(1):.1f}" stroke="{BASELINE}" stroke-width="3" '
           f'stroke-dasharray="10,8"/>')
  o.append(f'<text x="{sx(3.2):.1f}" y="{sy(0.60):.1f}" font-size="18" '
           f'fill="{BASELINE}" text-anchor="middle" '
           f'transform="rotate({-math.degrees(math.atan2(plot_h, PLOT)):.1f} '
           f'{sx(3.2):.1f} {sy(0.60):.1f})">'
           f'perfect alignment</text>')

  # Aggregate over environments, drawn last so it sits on top. This is the
  # plain mean of the four per-environment means -- the y axis is cosine_max,
  # so a curve rescaled to its own range could not share it.
  # Short labels: at ~79pt printed, a panel fits roughly 12 characters, and
  # "Cartpole Swingup sparse" overran into its neighbour.
  SHORT = {'Cartpole Swingup': 'Cartpole', 'Hopper Stand': 'Hopper Stand',
           'Cartpole Swingup sparse': 'Cartpole sparse', 'Cheetah Run': 'Cheetah',
           'Hopper Hop': 'Hopper Hop'}
  present = [t for t, _, _ in TASKS if t in data]
  if len(present) > 1:
    per_env = np.stack([np.stack(data[t]).mean(0) for t in present])
    am, asd = per_env.mean(0), per_env.std(0)
    band = ([(sx(i), sy(min(am[i] + asd[i], 1.0))) for i in range(nb)] +
            [(sx(i), sy(max(am[i] - asd[i], 0.0))) for i in range(nb)][::-1])
    o.append('<polygon points="%s" fill="%s" fill-opacity="0.16"/>' %
             (' '.join(f'{x:.1f},{y:.1f}' for x, y in band), INK))
    pts = [(sx(i), sy(am[i])) for i in range(nb)]
    o.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="7" '
             'stroke-linejoin="round" stroke-linecap="round"/>' %
             (' '.join(f'{x:.1f},{y:.1f}' for x, y in pts), INK))
    for x, y in pts:
      o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{INK}" '
               f'stroke="white" stroke-width="3"/>')
    AGG = (am, asd, len(present))
  else:
    AGG = None

  ends = []
  for task, label, color in TASKS:
    curves = data.get(task)
    if not curves:
      continue
    a = np.stack(curves)
    m, sd = a.mean(0), a.std(0)
    band = ([(sx(i), sy(min(m[i] + sd[i], 1.0))) for i in range(nb)] +
            [(sx(i), sy(max(m[i] - sd[i], 0.0))) for i in range(nb)][::-1])
    o.append('<polygon points="%s" fill="%s" fill-opacity="0.18"/>' %
             (' '.join(f'{x:.1f},{y:.1f}' for x, y in band), color))
    pts = [(sx(i), sy(m[i])) for i in range(nb)]
    o.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="6" '
             'stroke-linejoin="round" stroke-linecap="round"/>' %
             (' '.join(f'{x:.1f},{y:.1f}' for x, y in pts), color))
    for x, y in pts:
      o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}" '
               f'stroke="white" stroke-width="3"/>')
    ends.append([pts[-1][1], color, label, m[-1], len(a)])
  if AGG is not None:
    ends.append([sy(AGG[0][-1]), INK, 'All environments', AGG[0][-1], AGG[2]])

  # Floor values, annotated at the LEFT end where they actually occur: every
  # curve is 1.00 at x=1 by construction, so the informative number is the
  # value at zero code similarity.
  ends.sort()
  for k in range(1, len(ends)):
    if ends[k][0] - ends[k - 1][0] < 26:
      ends[k][0] = ends[k - 1][0] + 26
  for y, color, label, val, n in ends:
    o.append(f'<text x="{PAD_L - 52:.1f}" y="{y + 7:.1f}" font-size="19" '
             f'font-weight="bold" fill="{color}" text-anchor="end">'
             f'{val:.2f}</text>')

  lx, ly = PAD_L, S - 76
  o.append(f'<text x="{PAD_L + PLOT / 2:.1f}" y="{sy(0) + 88:.1f}" '
           f'font-size="23" font-weight="bold" fill="{INK}" '
           f'text-anchor="middle">goal-code similarity '
           f'(fraction of blocks matching)</text>')
  cy = PAD_T + plot_h / 2
  o.append(f'<text x="46" y="{cy:.1f}" font-size="23" font-weight="bold" '
           f'fill="{INK}" text-anchor="middle" '
           f'transform="rotate(-90 46 {cy:.1f})">goal-state similarity '
           f'(cosine_max)</text>')
  entries = [(l, c) for t, l, c in TASKS if t in data]
  if AGG is not None:
    entries.append((f'All environments (mean of {AGG[2]})', INK))
  for i, (label, color) in enumerate(entries):
    x = lx + (i % 2) * 380
    y = ly + (i // 2) * 30
    o.append(f'<rect x="{x}" y="{y - 12}" width="22" height="7" fill="{color}"/>')
    o.append(f'<text x="{x + 30}" y="{y - 3}" font-size="19" fill="{INK_2}">'
             f'{esc(label)}</text>')
  o.append('</svg>')
  return '\n'.join(o)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--results', default=str(HERE / 'code_sim'))
  ap.add_argument('--out', default=str(HERE / 'code_sim_to_state_sim_director'))
  ap.add_argument('--layout', choices=['single', 'row'], default='single')
  ap.add_argument('--xlabel', default=None)
  ap.add_argument('--copy-to', default=None)
  ap.add_argument('--spread', choices=['pairs', 'seeds'], default='pairs',
                  help='pairs: p5-p95/p25-p75 of the individual code pairs at '
                       'each bin (the diversity at a fixed code distance). '
                       'seeds: +/-1 std across runs (agreement between seeds).')
  ap.add_argument('--only-all', action='store_true',
                  help='row layout: draw only the All Envs panel. One panel is '
                       'wide enough for ticks every 0.25 rather than every 0.5.')
  ap.add_argument('--target-pt', type=float, default=0.98 * 397.0,
                  help='printed width in points the fonts are sized for; '
                       'match the \\includegraphics width (textwidth=397pt).')
  a = ap.parse_args()
  data = load(a.results)
  if not data:
    print('no npz in', a.results)
    return
  for t, v in data.items():
    arr = np.stack(v)
    print('%-30s n=%d  m=0 %.3f  m=8 %.3f' % (t, len(arr), arr[:, 0].mean(),
                                              arr[:, -1].mean()))
  n = min(len(v) for v in data.values())
  q = load_q(a.results) if a.spread == 'pairs' else {}
  sub = (f'Director, {n} seeds per environment, bands = p5-p95 and p25-p75 '
         f'over code pairs' if q else
         f'Director, {n} seeds per environment, band = +/-1 std across seeds')
  if q:
    for t in sorted(q):
      print('   %-30s bin8 spread p5-p95  %.3f - %.3f' %
            (t, q[t][-1, 0], q[t][-1, 4]))
  if a.layout == 'row':
    svg, W, H = build_row(data, sub, a.xlabel, q, a.only_all, a.target_pt)
    print('row layout %dx%d svg units' % (W, H))
  else:
    svg = build(data,
                'Goal-code similarity vs goal-state similarity (Director)', sub)
  pathlib.Path(a.out + '.svg').write_text(svg)
  # rsvg-convert 2.42 ignores -d/-p for an SVG with a pixel width, so the PNG
  # comes out at 1:1 with the SVG units. The six-panel row is 2276 units wide
  # and so lands at ~420 dpi by accident; one panel is 378 units and would
  # print at 70 dpi. Ask for the pixel width that gives 600 dpi at the size the
  # figure is actually included at.
  cmd = ['rsvg-convert', '-f', 'png', '-d', '600', '-p', '600']
  if a.layout == 'row' and a.only_all:
    cmd += ['-w', str(int(round(a.target_pt / 72.0 * 600)))]
  subprocess.run(cmd + ['-o', a.out + '.png', a.out + '.svg'], check=True)
  print('wrote', a.out + '.png')
  if a.copy_to:
    for ext in ('.png', '.svg'):
      subprocess.run(['cp', a.out + ext,
                      os.path.join(a.copy_to, os.path.basename(a.out) + ext)],
                     check=True)
    print('copied to', a.copy_to)




# --------------------------------------------------------------- row layout

def build_row(data, sub, xlabel=None, quant=None, only_all=False,
              target_pt=0.98 * 397.0):
  """Five square panels in a row, styled after the DreamerV3/Director figures.

  That style is: a boxed axes frame, light gridlines on both axes, outward
  ticks, plain-weight titles, thin lines with no markers, and a translucent
  band. Panels sit close together and the axis labels are shared.

  Each panel prints at about a fifth of 397pt, so every font is a fraction of
  the SVG width and the render is 600 dpi; an absolute font size at this scale
  is either invisible in print or enormous on screen. Ticks are cut to 0/0.5/1
  on x and labelled once on y, which is what fits.
  """
  n_panels = 1 if only_all else len(TASKS) + 1
  P = 300
  # GAP and PR grew when hopper_hop made this six panels rather than five:
  # at GAP=30/PR=22 the titles of neighbouring panels touched and the last
  # panel's title ran off the canvas.
  GAP = 56
  # PB carries the tick row AND the shared x label; at 122 the label's
  # ascenders sat on top of the '.5' tick of the middle panels.
  PL, PR, PT, PB = 152, 44, 104, 168
  # Those paddings are a small fraction of a six-panel canvas and half of a
  # one-panel one: reusing them for --only-all left the x label floating a
  # panel-width below the axes. Every font is a fraction of W, so shrinking the
  # margins also shrinks the type; these are the smallest that still clear the
  # tick labels at the resulting size.
  if only_all:
    PL, PR, PT, PB = 58, 20, 40, 56
  W = PL + n_panels * P + (n_panels - 1) * GAP + PR
  H = PT + P + PB
  # Printed width the fonts are sized for. Every font below is a fraction of
  # the SVG width, so f(pt) prints at pt points only if the figure is included
  # at TARGET. Pass --target-pt when including at less than \textwidth.
  TARGET = target_pt

  def f(pt_):
    return pt_ * W / TARGET

  # The rotated y label has to fit inside the PANEL HEIGHT, not the figure
  # width -- at f(8.4) 'goal-state similarity (cosine_max)' was half again
  # longer than the panel is tall and ran off both ends of the canvas.
  F_TITLE, F_TICK, F_AXIS = f(7.4), f(6.4), f(8.0)
  F_YAX = f(6.0)
  FRAME, GRIDC = '#333333', '#d5d5d2'

  # Short labels: a panel this wide fits about twelve characters, and
  # "Cartpole Swingup sparse" overran into its neighbour.
  SHORT = {'Cartpole Swingup': 'Cartpole', 'Hopper Stand': 'Hopper Stand',
           'Cartpole Swingup sparse': 'Cartpole Sparse', 'Cheetah Run': 'Cheetah',
           'Hopper Hop': 'Hopper Hop'}
  if xlabel is None:
    # 9 bins is the block count, 57 is the summed index distance; label the
    # axis for whichever measurement produced the curves rather than assuming.
    nb0 = len(next(iter(data.values()))[0])
    xlabel = ('goal-code similarity (fraction of blocks matching)' if nb0 <= 9
              else 'goal-code similarity (1 - index distance / 56)')
  present = [t for t, _, _ in TASKS if t in data]
  per_env = (np.stack([np.stack(data[t]).mean(0) for t in present])
             if present else None)
  quant = quant or {}
  panels = [] if only_all else [
      (SHORT.get(lab, lab), col, np.stack(data[t]), quant.get(t))
      for t, lab, col in TASKS if t in data]
  if per_env is not None and len(present) > 1:
    qs = [quant[t] for t in present if t in quant]
    panels.append(('All Environments' if only_all else 'All Envs', INK,
                   per_env, np.mean(qs, 0) if qs else None))

  o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
       f'<rect width="{W}" height="{H}" fill="white"/>']

  # Same positions on both axes. Five of them fitted while this was five
  # panels wide; at six, the '1' of one panel and the '0' of the next collided
  # into '1 0', so three is what a 300-unit panel holds side by side. With a
  # single panel there is no neighbour to collide with and five fit again.
  YT = XT = (0.0, 0.25, 0.5, 0.75, 1.0) if only_all else (0.0, 0.5, 1.0)

  def tl(v):
    if v in (0.0, 1.0):
      return '%g' % v
    return ('%g' % v).lstrip('0')
  for k, (label, color, arr, q) in enumerate(panels):
    x0 = PL + k * (P + GAP)
    m, sd = arr.mean(0), arr.std(0)
    nb = len(m)

    def sx(i, x0=x0):
      # i indexes the bins in order of INCREASING distance, so the axis is
      # code similarity 1 - i/(nb-1): works for the 9 block bins and for the
      # 57 index-distance bins alike.
      return x0 + (1.0 - i / (nb - 1)) * P

    def sy(v):
      # y spans [0, 1]. cosine_max itself reaches -1, and a mean over disjoint
      # pairs really can get there, but -1 needs every partner to decode to the
      # exact OPPOSITE goal, which is not the property being asked about. The
      # target for maximally dissimilar codes is UNRELATED goals -- independent
      # directions in this 1024-dim space measure 0.0002 +/- 0.031 -- so 0 is
      # the meaningful floor and the axis ends there.
      return PT + (1.0 - v) * P

    for v in YT:
      o.append(f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x0 + P}" '
               f'y2="{sy(v):.1f}" stroke="{GRIDC}" stroke-width="1.7"/>')
    for v in XT:
      gx = sx((1.0 - v) * (nb - 1))
      o.append(f'<line x1="{gx:.1f}" y1="{PT}" x2="{gx:.1f}" '
               f'y2="{PT + P}" stroke="{GRIDC}" stroke-width="1.7"/>')

    # Two nested bands from the PAIR percentiles: p5-p95 is the full range of
    # goal similarity a fixed code distance produces, p25-p75 the middle half.
    # Falls back to the +/-1 std across seeds when no percentiles were stored.
    if q is not None:
      for lo_i, hi_i, op in ((0, 4, 0.16), (1, 3, 0.30)):
        poly = ([(sx(i), sy(min(q[i, hi_i], 1.0))) for i in range(nb)] +
                [(sx(i), sy(max(q[i, lo_i], 0.0))) for i in range(nb)][::-1])
        o.append('<polygon points="%s" fill="%s" fill-opacity="%s"/>' %
                 (' '.join(f'{x:.1f},{y:.1f}' for x, y in poly), color, op))
    else:
      band = ([(sx(i), sy(min(m[i] + sd[i], 1.0))) for i in range(nb)] +
              [(sx(i), sy(max(m[i] - sd[i], 0.0))) for i in range(nb)][::-1])
      o.append('<polygon points="%s" fill="%s" fill-opacity="0.28"/>' %
               (' '.join(f'{x:.1f},{y:.1f}' for x, y in band), color))
    o.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="5.4" '
             'stroke-linejoin="round" stroke-linecap="round"/>' %
             (' '.join(f'{sx(i):.1f},{sy(m[i]):.1f}' for i in range(nb)), color))
    # perfect alignment: goal similarity == code similarity, (0,0) to (1,1)
    o.append(f'<line x1="{sx(nb - 1):.1f}" y1="{sy(0):.1f}" '
             f'x2="{sx(0):.1f}" y2="{sy(1):.1f}" stroke="#8a8880" '
             f'stroke-width="2.2" stroke-dasharray="8,6"/>')

    o.append(f'<rect x="{x0}" y="{PT}" width="{P}" height="{P}" fill="none" '
             f'stroke="{FRAME}" stroke-width="2.2"/>')
    for v in YT:
      o.append(f'<line x1="{x0 - 8}" y1="{sy(v):.1f}" x2="{x0}" '
               f'y2="{sy(v):.1f}" stroke="{FRAME}" stroke-width="2.2"/>')
      if k == 0:
        o.append(f'<text x="{x0 - 15}" y="{sy(v) + F_TICK * 0.36:.1f}" '
                 f'font-size="{F_TICK:.1f}" fill="{INK_2}" '
                 f'text-anchor="end">{tl(v)}</text>')
    for v in XT:
      gx = sx((1.0 - v) * (nb - 1))
      o.append(f'<line x1="{gx:.1f}" y1="{PT + P}" x2="{gx:.1f}" '
               f'y2="{PT + P + 8}" stroke="{FRAME}" stroke-width="2.2"/>')
      o.append(f'<text x="{gx:.1f}" y="{PT + P + F_TICK * 1.9:.1f}" '
               f'font-size="{F_TICK:.1f}" fill="{INK_2}" '
               f'text-anchor="middle">{tl(v)}</text>')
    o.append(f'<text x="{x0 + P / 2:.1f}" y="{PT - 26:.1f}" '
             f'font-size="{F_TITLE:.1f}" fill="{INK}" '
             f'text-anchor="middle">{esc(label)}</text>')

  o.append(f'<text x="{PL + (W - PL - PR) / 2:.1f}" y="{H - 22:.1f}" '
           f'font-size="{F_AXIS:.1f}" fill="{INK}" text-anchor="middle">'
           f'{esc(xlabel)}</text>')
  cy = PT + P / 2
  yx = 26.0
  o.append(f'<text x="{yx:.1f}" y="{cy:.1f}" '
           f'font-size="{F_YAX:.1f}" fill="{INK}" text-anchor="middle" '
           f'transform="rotate(-90 {yx:.1f} {cy:.1f})">'
           f'goal-state similarity</text>')
  o.append('</svg>')
  return '\n'.join(o), W, H


if __name__ == '__main__':
  main()
