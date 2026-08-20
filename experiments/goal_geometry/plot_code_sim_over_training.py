#!/usr/bin/env python3
"""Does the code -> goal geometry change over training? One panel per env.

Same measurement as ``code_sim_row``, repeated against ckpt_milestones at 1M,
2M, 3M and 4M steps. One line per checkpoint in a light-to-dark ramp, mean over
the four seeds. If the curves lie on top of each other the compression is a
property of the architecture, not a transient of early training.

  python3 plot_code_sim_over_training.py --results code_sim_ckpt \
      --out code_sim_over_training
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
# light -> dark, so "later in training" reads as "darker" without a legend
RAMP = [('1M', '#bcd4ef'), ('2M', '#7aa9dd'), ('3M', '#3c7dc4'),
        ('4M', '#12457e')]
INK, INK_2, FRAME, GRIDC = '#000000', '#1a1a1a', '#333333', '#d5d5d2'
BASELINE = '#8a8880'


def esc(s):
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def load(base):
  """{step_label: {task: mean curve over seeds}}."""
  out = {}
  for lab, _ in RAMP:
    step = '%012d' % (int(lab[:-1]) * 1_000_000)
    d = os.path.join(base, step)
    by = {}
    for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
      z = np.load(f, allow_pickle=True)
      by.setdefault(str(z['task']), []).append(np.asarray(z['curve'], float))
    if by:
      out[lab] = {t: np.stack(v).mean(0) for t, v in by.items()}
  return out


def build(data, sub):
  P, GAP = 300, 56
  PL, PR, PT, PB = 152, 44, 104, 200
  present = [(t, lab) for t, lab in TASKS
             if any(t in v for v in data.values())]
  n = len(present) + 1
  W = PL + n * P + (n - 1) * GAP + PR
  H = PT + P + PB
  f = lambda pt: pt * W / (0.98 * 397.0)
  F_TITLE, F_TICK, F_AXIS, F_LEG = f(7.4), f(6.4), f(8.0), f(6.8)
  SHORT = {'Cartpole Swingup': 'Cartpole'}

  o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
       f'<rect width="{W}" height="{H}" fill="white"/>']
  T = (0.0, 0.5, 1.0)
  tl = lambda v: '%g' % v if v in (0.0, 1.0) else ('%g' % v).lstrip('0')

  panels = [(lab, t) for t, lab in present] + [('All Envs', None)]
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
             f'y2="{sy(1):.1f}" stroke="{BASELINE}" stroke-width="2.2" '
             f'stroke-dasharray="8,6"/>')

    for lab, col in RAMP:
      if lab not in data:
        continue
      if task is None:
        curves = [c for t, c in data[lab].items()]
        if not curves:
          continue
        m = np.stack(curves).mean(0)
      else:
        if task not in data[lab]:
          continue
        m = data[lab][task]
      nb = len(m)
      sx = lambda i: x0 + (1.0 - i / (nb - 1)) * P
      o.append('<polyline points="%s" fill="none" stroke="%s" '
               'stroke-width="4.6" stroke-linejoin="round" '
               'stroke-linecap="round"/>' %
               (' '.join(f'{sx(i):.1f},{sy(m[i]):.1f}' for i in range(nb)),
                col))

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
             f'{esc(SHORT.get(title, title))}</text>')

  o.append(f'<text x="{PL + (W - PL - PR) / 2:.1f}" y="{H - 96:.1f}" '
           f'font-size="{F_AXIS:.1f}" fill="{INK}" text-anchor="middle">'
           f'goal-code similarity (fraction of blocks matching)</text>')
  cy = PT + P / 2
  o.append(f'<text x="26" y="{cy:.1f}" font-size="{F_AXIS * 0.75:.1f}" '
           f'fill="{INK}" text-anchor="middle" '
           f'transform="rotate(-90 26 {cy:.1f})">goal-state similarity</text>')

  span = (W - PL - PR) / len(RAMP)
  for i, (lab, col) in enumerate(RAMP):
    if lab not in data:
      continue
    x = PL + i * span + span * 0.25
    y = H - 30
    o.append(f'<line x1="{x:.1f}" y1="{y - 8:.1f}" x2="{x + 52:.1f}" '
             f'y2="{y - 8:.1f}" stroke="{col}" stroke-width="6"/>')
    o.append(f'<text x="{x + 64:.1f}" y="{y:.1f}" font-size="{F_LEG:.1f}" '
             f'fill="{INK_2}">{esc(lab)} steps</text>')
  o.append('</svg>')
  return '\n'.join(o), W, H


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--results', default=str(HERE / 'code_sim_ckpt'))
  ap.add_argument('--out', default=str(HERE / 'code_sim_over_training'))
  a = ap.parse_args()
  data = load(a.results)
  if not data:
    print('no checkpoint results under', a.results)
    return
  print('%-30s %s' % ('task', '  '.join('%8s' % l for l, _ in RAMP)))
  tasks = sorted({t for v in data.values() for t in v})
  for t in tasks:
    cells = []
    for lab, _ in RAMP:
      c = data.get(lab, {}).get(t)
      cells.append('%8.3f' % c[-1] if c is not None else '       -')
    print('%-30s %s' % (t, '  '.join(cells)))
  print('(value shown is the floor: similarity at all 8 blocks different)')
  svg, W, H = build(data, '')
  pathlib.Path(a.out + '.svg').write_text(svg)
  subprocess.run(['rsvg-convert', '-f', 'png', '-d', '600', '-p', '600',
                  '-o', a.out + '.png', a.out + '.svg'], check=True)
  print('layout %dx%d svg units' % (W, H))
  print('wrote', a.out + '.png')


if __name__ == '__main__':
  main()
