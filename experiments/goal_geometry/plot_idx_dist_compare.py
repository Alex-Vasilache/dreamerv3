#!/usr/bin/env python3
"""Is the class INDEX ordered? Director against SOM-line+LiP, one panel per env.

The block-count axis asks "how many blocks differ". This asks a stricter
question: how FAR do they differ. Distance is the summed index gap
D = sum_l |c'_l - c_l|, from 0 to L*(C-1) = 56, plotted as the similarity
1 - D/56 so both axes are similarities.

If the class index is an arbitrary label, as it is in Director, moving a block
from class 2 to class 3 is no different from moving it to class 7, and the
curve carries little structure along this axis. If the codebook is a SOM
trained on a line, the index is a coordinate and goal similarity should fall
steadily as D grows.

  python3 plot_idx_dist_compare.py --out idx_dist_compare
"""
import argparse
import glob
import os
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

TASKS = [
    ('dmc_cartpole_swingup', 'Cartpole'),
    ('dmc_hopper_stand', 'Hopper Stand'),
    ('dmc_cartpole_swingup_sparse', 'Cartpole Sparse'),
    ('dmc_cheetah_run', 'Cheetah'),
    ('dmc_hopper_hop', 'Hopper Hop'),
]
ARMS = [('Director', '#5c5b56', 'idx_dist_director'),
        ('SOM-line + LiP', '#2a78d6', 'idx_dist_somlip')]
INK, INK_2, FRAME, GRIDC, BASE = '#000000', '#1a1a1a', '#333333', '#d5d5d2', '#8a8880'


def esc(s):
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def load(d):
  out = {}
  for f in sorted(glob.glob(os.path.join(HERE, d, '*.npz'))):
    z = np.load(f, allow_pickle=True)
    out.setdefault(str(z['task']), []).append(np.asarray(z['curve'], float))
  return {t: np.stack(v) for t, v in out.items()}


def build(data, only_all=False, target_pt=0.98 * 397.0):
  P, GAP = 300, 56
  PL, PR, PT, PB = 152, 44, 104, 220
  present = [(t, lab) for t, lab in TASKS
             if any(t in data[a[0]] for a in ARMS)]
  n = 1 if only_all else len(present) + 1
  # Margins sized for a six-panel strip leave a panel-width of dead space
  # around a single panel, and every font is a fraction of W so they shrink
  # with it. PB is set below, once the font sizes are known, from what the
  # tick row, the x label and the two-entry legend actually need.
  if only_all:
    PL, PR, PT, PB = 58, 20, 40, 0
  W = PL + n * P + (n - 1) * GAP + PR
  f = lambda pt: pt * W / target_pt
  F_TITLE, F_TICK, F_AXIS, F_LEG = f(7.4), f(6.4), f(8.0), f(6.8)
  # Distance below the panel of the shared x label and of the legend row.
  XLAB_DY, LEG_DY = 112.0, 186.0
  if only_all:
    XLAB_DY = F_TICK * 1.9 + F_AXIS * 1.5
    LEG_DY = XLAB_DY + F_LEG * 2.6
    PB = int(LEG_DY + F_LEG * 1.4)
  H = PT + P + PB

  o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
       f'<rect width="{W}" height="{H}" fill="white"/>']
  # One panel has no neighbour whose '0' can collide with this one's '1', so
  # the axis holds five labels instead of three.
  T = (0.0, 0.25, 0.5, 0.75, 1.0) if only_all else (0.0, 0.5, 1.0)
  tl = lambda v: '%g' % v if v in (0.0, 1.0) else ('%g' % v).lstrip('0')

  panels = ([('All Environments', None)] if only_all else
            [(lab, t) for t, lab in present] + [('All Envs', None)])
  for k, (title, task) in enumerate(panels):
    x0 = PL + k * (P + GAP)
    sy = lambda v: PT + (1.0 - v) * P
    for v in T:
      o.append(f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x0 + P}" '
               f'y2="{sy(v):.1f}" stroke="{GRIDC}" stroke-width="1.7"/>')
      gx = x0 + v * P
      o.append(f'<line x1="{gx:.1f}" y1="{PT}" x2="{gx:.1f}" '
               f'y2="{PT + P}" stroke="{GRIDC}" stroke-width="1.7"/>')
    o.append(f'<line x1="{x0:.1f}" y1="{sy(0):.1f}" x2="{x0 + P:.1f}" '
             f'y2="{sy(1):.1f}" stroke="{BASE}" stroke-width="2.2" '
             f'stroke-dasharray="8,6"/>')

    for label, col, _ in ARMS:
      d = data[label]
      if task is None:
        curves = [v.mean(0) for v in d.values()]
        if not curves:
          continue
        arr = np.stack(curves)
      else:
        if task not in d:
          continue
        arr = d[task]
      m, sd = arr.mean(0), arr.std(0)
      nb = len(m)
      sx = lambda i: x0 + (1.0 - i / (nb - 1)) * P
      band = ([(sx(i), sy(min(m[i] + sd[i], 1.0))) for i in range(nb)] +
              [(sx(i), sy(max(m[i] - sd[i], 0.0))) for i in range(nb)][::-1])
      o.append('<polygon points="%s" fill="%s" fill-opacity="0.22"/>' %
               (' '.join(f'{x:.1f},{y:.1f}' for x, y in band), col))
      o.append('<polyline points="%s" fill="none" stroke="%s" '
               'stroke-width="5.0" stroke-linejoin="round" '
               'stroke-linecap="round"/>' %
               (' '.join(f'{sx(i):.1f},{sy(m[i]):.1f}' for i in range(nb)), col))

    o.append(f'<rect x="{x0}" y="{PT}" width="{P}" height="{P}" fill="none" '
             f'stroke="{FRAME}" stroke-width="2.2"/>')
    for v in T:
      o.append(f'<line x1="{x0 - 8}" y1="{sy(v):.1f}" x2="{x0}" '
               f'y2="{sy(v):.1f}" stroke="{FRAME}" stroke-width="2.2"/>')
      if k == 0:
        o.append(f'<text x="{x0 - 15}" y="{sy(v) + F_TICK * 0.36:.1f}" '
                 f'font-size="{F_TICK:.1f}" fill="{INK_2}" '
                 f'text-anchor="end">{tl(v)}</text>')
      gx = x0 + v * P
      o.append(f'<line x1="{gx:.1f}" y1="{PT + P}" x2="{gx:.1f}" '
               f'y2="{PT + P + 8}" stroke="{FRAME}" stroke-width="2.2"/>')
      o.append(f'<text x="{gx:.1f}" y="{PT + P + F_TICK * 1.9:.1f}" '
               f'font-size="{F_TICK:.1f}" fill="{INK_2}" '
               f'text-anchor="middle">{tl(v)}</text>')
    o.append(f'<text x="{x0 + P / 2:.1f}" y="{PT - 26:.1f}" '
             f'font-size="{F_TITLE:.1f}" fill="{INK}" text-anchor="middle">'
             f'{esc(title)}</text>')

  bottom = PT + P
  o.append(f'<text x="{PL + (W - PL - PR) / 2:.1f}" y="{bottom + XLAB_DY:.1f}" '
           f'font-size="{F_AXIS:.1f}" fill="{INK}" text-anchor="middle">'
           f'goal-code similarity (1 - index distance / 56)</text>')
  cy = PT + P / 2
  yx = 26.0
  o.append(f'<text x="{yx:.1f}" y="{cy:.1f}" '
           f'font-size="{F_AXIS * 0.75:.1f}" fill="{INK}" text-anchor="middle" '
           f'transform="rotate(-90 {yx:.1f} {cy:.1f})">'
           f'goal-state similarity</text>')

  # Swatch and text offsets are absolute units, so they have to shrink with the
  # canvas or the legend swamps a single panel.
  sw, tx, dy, lw = ((52.0, 64.0, 8.0, 6.0) if not only_all else
                    (F_LEG * 3.0, F_LEG * 3.8, F_LEG * 0.34, F_LEG * 0.5))
  span = (W - PL - PR) / 2
  for i, (label, col, _) in enumerate(ARMS):
    x = PL + i * span + span * (0.22 if not only_all else 0.10)
    y = bottom + LEG_DY
    o.append(f'<line x1="{x:.1f}" y1="{y - dy:.1f}" '
             f'x2="{x + sw:.1f}" y2="{y - dy:.1f}" '
             f'stroke="{col}" stroke-width="{lw:.1f}"/>')
    o.append(f'<text x="{x + tx:.1f}" y="{y:.1f}" font-size="{F_LEG:.1f}" '
             f'fill="{INK_2}">{esc(label)}</text>')
  o.append('</svg>')
  return '\n'.join(o), W, H


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=str(HERE / 'idx_dist_compare'))
  ap.add_argument('--only-all', action='store_true',
                  help='draw only the All Envs panel, with ticks every 0.25.')
  ap.add_argument('--target-pt', type=float, default=0.98 * 397.0,
                  help='printed width in points the fonts are sized for; '
                       'match the \\includegraphics width (textwidth=397pt).')
  a = ap.parse_args()
  data = {label: load(d) for label, _, d in ARMS}
  for label, _, d in ARMS:
    if not data[label]:
      print('no results in', d)
      return
  print('%-30s %10s %10s   %s' % ('task', 'Director', 'SOM+LiP', 'floor at D=56'))
  for t, lab in TASKS:
    row = []
    for label, _, _ in ARMS:
      row.append(data[label][t].mean(0)[-1] if t in data[label] else np.nan)
    print('%-30s %10.3f %10.3f' % (t, row[0], row[1]))
  svg, W, H = build(data, a.only_all, a.target_pt)
  pathlib.Path(a.out + '.svg').write_text(svg)
  # rsvg-convert 2.42 ignores -d/-p for a pixel-sized SVG, so the six-panel
  # strip lands at ~420 dpi only because it is 2276 units wide. One panel is
  # ~380 and needs the pixel width asked for explicitly.
  cmd = ['rsvg-convert', '-f', 'png', '-d', '600', '-p', '600']
  if a.only_all:
    cmd += ['-w', str(int(round(a.target_pt / 72.0 * 600)))]
  subprocess.run(cmd + ['-o', a.out + '.png', a.out + '.svg'], check=True)
  print('layout %dx%d svg units' % (W, H))
  print('wrote', a.out + '.png')


if __name__ == '__main__':
  main()
