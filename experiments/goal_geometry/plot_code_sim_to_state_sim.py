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
PAD_L, PAD_R, PAD_T, PAD_B = 190, 210, 250, 200
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

  def sx(i):
    return PAD_L + (i / (nb - 1)) * PLOT

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
    o.append(f'<text x="{PAD_L - 18}" y="{y + 8:.1f}" font-size="21" '
             f'fill="{INK_2}" text-anchor="end">{v:.2f}</text>')
  for i in range(nb):
    o.append(f'<text x="{sx(i):.1f}" y="{sy(0) + 40:.1f}" font-size="21" '
             f'fill="{INK_2}" text-anchor="middle">{i}</text>')

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

  ends.sort()
  for k in range(1, len(ends)):
    if ends[k][0] - ends[k - 1][0] < 30:
      ends[k][0] = ends[k - 1][0] + 30
  for y, color, label, val, n in ends:
    o.append(f'<text x="{PAD_L + PLOT + 16}" y="{y + 8:.1f}" font-size="20" '
             f'font-weight="bold" fill="{color}">{val:.2f}</text>')

  lx, ly = PAD_L, S - 58
  o.append(f'<text x="{PAD_L + PLOT / 2:.1f}" y="{sy(0) + 88:.1f}" '
           f'font-size="23" font-weight="bold" fill="{INK}" '
           f'text-anchor="middle">code blocks different</text>')
  cy = PAD_T + plot_h / 2
  o.append(f'<text x="46" y="{cy:.1f}" font-size="23" font-weight="bold" '
           f'fill="{INK}" text-anchor="middle" '
           f'transform="rotate(-90 46 {cy:.1f})">goal-state similarity '
           f'(cosine_max)</text>')
  for i, (task, label, color) in enumerate(TASKS):
    if task not in data:
      continue
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
  svg = build(data, 'Codes this different -> goal-state similarity (Director)',
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
