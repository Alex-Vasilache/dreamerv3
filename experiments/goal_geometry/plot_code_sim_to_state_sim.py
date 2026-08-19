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

TASKS = [
    ('dmc_cartpole_swingup', 'Cartpole Swingup', '#2a78d6'),
    ('dmc_hopper_stand', 'Hopper Stand', '#eb6834'),
    ('dmc_cartpole_swingup_sparse', 'Cartpole Swingup sparse', '#1baf7a'),
    ('dmc_cheetah_run', 'Cheetah Run', '#eda100'),
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
  ap.add_argument('--copy-to', default=None)
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
  svg = build(data, 'Goal-code similarity vs goal-state similarity (Director)',
              f'four tasks, {n} seeds each, band = +/-1 std across seeds')
  pathlib.Path(a.out + '.svg').write_text(svg)
  subprocess.run(['rsvg-convert', '-f', 'png', '-d', '300', '-p', '300',
                  '-o', a.out + '.png', a.out + '.svg'], check=True)
  print('wrote', a.out + '.png')
  if a.copy_to:
    for ext in ('.png', '.svg'):
      subprocess.run(['cp', a.out + ext,
                      os.path.join(a.copy_to, os.path.basename(a.out) + ext)],
                     check=True)
    print('copied to', a.copy_to)


if __name__ == '__main__':
  main()
