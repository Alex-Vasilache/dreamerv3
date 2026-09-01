#!/usr/bin/env python3
"""Training curves for the discretized-Gaussian manager, one panel per task.

Three arms: Director, the categorical SOM-line+LiP arm, and that same arm with
a ReLU trunk and a discretized Gaussian manager head and no epsilon-greedy
jump (e592-e599). Same row layout and styling as arm_curves_row.png so the two
figures sit together in the paper: boxed axes, light gridlines on both,
outward ticks, plain titles, a thick mean line and a +/-1 std band.

The band is not comparable across arms without reading the caption: Director
and SOM-line+LiP have 4 seeds each and the Gaussian arm has 2, so its band is
a spread between two runs rather than a standard deviation over four.

Endpoint numbers and the permutation tests for the matching table come from
gaussian_scores.py, which reads scores.jsonl rather than metrics.jsonl.

  python3 make_figure_curves_gaussian.py --out gaussian_curves_row
"""
import argparse
import glob
import json
import os
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
WORK = os.environ.get('WORK_DIR', '/work/DoyaU/vasilache/work')

# (task, short title, director exps, som+lip exps, gaussian exps). Cartpole
# sparse is absent because the Gaussian batch did not run it.
TASKS = [
    ('dmc_cartpole_swingup', 'Cartpole', range(502, 506), range(534, 538),
     (592, 596)),
    ('dmc_hopper_stand', 'Hopper Stand', range(506, 510), range(538, 542),
     (593, 597)),
    ('dmc_cheetah_run', 'Cheetah', range(554, 558), range(562, 566),
     (594, 598)),
    ('dmc_hopper_hop', 'Hopper Hop', range(566, 570), range(570, 574),
     (595, 599)),
]
# Vermillion for the Gaussian, the colour the head already carries in
# gaussian_head.png.
ARMS = [('Director', '#5c5b56'), ('SOM-line + LiP', '#2a78d6'),
        ('+ Gaussian head', '#d55e00')]
INK, INK_2, FRAME, GRIDC = '#000000', '#1a1a1a', '#333333', '#d5d5d2'
STEPS = 4_000_000
NB = 40                       # resample every 100k steps


def esc(s):
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def curve(run_dir):
  """(step, score) pairs from one run's metrics.jsonl."""
  p = os.path.join(run_dir, 'logdir', 'metrics.jsonl')
  out = []
  try:
    for line in open(p):
      try:
        r = json.loads(line)
      except Exception:
        continue
      if 'step' not in r:
        continue
      for k, v in r.items():
        if k.endswith('episode/score') and isinstance(v, (int, float)):
          out.append((int(r['step']), float(v)))
  except FileNotFoundError:
    return None
  return sorted(out) or None


def resample(pairs, grid, half=60_000):
  """Mean score in a window around each grid point, so the seeds share an x."""
  s = np.array([p[0] for p in pairs], float)
  v = np.array([p[1] for p in pairs], float)
  out = np.full(len(grid), np.nan)
  for i, g in enumerate(grid):
    m = (s >= g - half) & (s <= g + half)
    if m.any():
      out[i] = v[m].mean()
  # carry the last observation forward so a sparse tail does not break the line
  last = np.nan
  for i in range(len(out)):
    if np.isnan(out[i]):
      out[i] = last
    else:
      last = out[i]
  return out


def collect(exps, task):
  grid = np.linspace(0, STEPS, NB)
  rows = []
  for e in exps:
    d = glob.glob(os.path.join(WORK, f'e{e}_{task}_*_BIG_j*'))
    if not d:
      continue
    c = curve(sorted(d)[0])
    if c:
      rows.append(resample(c, grid))
  return grid, (np.stack(rows) if rows else None)


def build(ymaxes):
  # Unlike the geometry row, EVERY panel carries its own y-tick labels
  # (the return scale differs per environment), so the gap has to hold a
  # 4-digit label -- at GAP=56 the '1000' of Hopper Stand overlapped the
  # Cartpole panel's frame.
  P, GAP = 300, 120
  PL, PR, PT, PB = 170, 44, 104, 200
  n = len(TASKS)
  W = PL + n * P + (n - 1) * GAP + PR
  H = PT + P + PB
  f = lambda pt: pt * W / (0.98 * 397.0)
  F_TITLE, F_TICK, F_AXIS, F_LEG = f(7.4), f(6.4), f(8.0), f(7.0)

  o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
       f'<rect width="{W}" height="{H}" fill="white"/>']

  for k, (task, title, dexp, mexp, gexp) in enumerate(TASKS):
    x0 = PL + k * (P + GAP)
    ymax = ymaxes[k]
    sx = lambda v, x0=x0: x0 + (v / STEPS) * P
    sy = lambda v, ymax=ymax: PT + (1.0 - v / ymax) * P

    yt = [0, ymax / 2, ymax]
    xt = [0, STEPS / 2, STEPS]
    for v in yt:
      o.append(f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x0 + P}" '
               f'y2="{sy(v):.1f}" stroke="{GRIDC}" stroke-width="1.7"/>')
    for v in xt:
      o.append(f'<line x1="{sx(v):.1f}" y1="{PT}" x2="{sx(v):.1f}" '
               f'y2="{PT + P}" stroke="{GRIDC}" stroke-width="1.7"/>')

    for (label, color), exps in zip(ARMS, (dexp, mexp, gexp)):
      grid, arr = collect(exps, task)
      if arr is None:
        continue
      m = np.nanmean(arr, 0)
      sd = np.nanstd(arr, 0)
      ok = ~np.isnan(m)
      gx, gm, gs = grid[ok], m[ok], sd[ok]
      poly = ([(sx(x), sy(min(a + b, ymax))) for x, a, b in zip(gx, gm, gs)] +
              [(sx(x), sy(max(a - b, 0.0)))
               for x, a, b in zip(gx, gm, gs)][::-1])
      o.append('<polygon points="%s" fill="%s" fill-opacity="0.22"/>' %
               (' '.join(f'{x:.1f},{y:.1f}' for x, y in poly), color))
      o.append('<polyline points="%s" fill="none" stroke="%s" '
               'stroke-width="5.4" stroke-linejoin="round" '
               'stroke-linecap="round"/>' %
               (' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in zip(gx, gm)),
                color))

    o.append(f'<rect x="{x0}" y="{PT}" width="{P}" height="{P}" fill="none" '
             f'stroke="{FRAME}" stroke-width="2.2"/>')
    for v in yt:
      o.append(f'<line x1="{x0 - 8}" y1="{sy(v):.1f}" x2="{x0}" '
               f'y2="{sy(v):.1f}" stroke="{FRAME}" stroke-width="2.2"/>')
      o.append(f'<text x="{x0 - 15}" y="{sy(v) + F_TICK * 0.36:.1f}" '
               f'font-size="{F_TICK:.1f}" fill="{INK_2}" '
               f'text-anchor="end">{v:.0f}</text>')
    for v in xt:
      o.append(f'<line x1="{sx(v):.1f}" y1="{PT + P}" x2="{sx(v):.1f}" '
               f'y2="{PT + P + 8}" stroke="{FRAME}" stroke-width="2.2"/>')
      o.append(f'<text x="{sx(v):.1f}" y="{PT + P + F_TICK * 1.9:.1f}" '
               f'font-size="{F_TICK:.1f}" fill="{INK_2}" '
               f'text-anchor="middle">{v / 1e6:.0f}M</text>')
    o.append(f'<text x="{x0 + P / 2:.1f}" y="{PT - 26:.1f}" '
             f'font-size="{F_TITLE:.1f}" fill="{INK}" '
             f'text-anchor="middle">{esc(title)}</text>')

  o.append(f'<text x="{PL + (W - PL - PR) / 2:.1f}" y="{H - 78:.1f}" '
           f'font-size="{F_AXIS:.1f}" fill="{INK}" text-anchor="middle">'
           f'environment steps</text>')
  cy = PT + P / 2
  o.append(f'<text x="34" y="{cy:.1f}" font-size="{F_AXIS * 0.8:.1f}" '
           f'fill="{INK}" text-anchor="middle" '
           f'transform="rotate(-90 34 {cy:.1f})">episode return</text>')

  # Three entries, so the legend is laid out from its own width rather than
  # the two-entry offsets the five-panel figure could hard-code.
  span = 330
  lx = PL + (W - PL - PR) / 2 - (len(ARMS) - 1) * span / 2 - 40
  for i, (label, color) in enumerate(ARMS):
    x = lx + i * span
    y = H - 26
    o.append(f'<rect x="{x:.1f}" y="{y - F_LEG * 0.8:.1f}" width="46" '
             f'height="9" fill="{color}"/>')
    o.append(f'<text x="{x + 58:.1f}" y="{y:.1f}" font-size="{F_LEG:.1f}" '
             f'fill="{INK_2}">{esc(label)}</text>')
  o.append('</svg>')
  return '\n'.join(o), W, H


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=str(HERE / 'gaussian_curves_row'))
  ap.add_argument('--copy-to', default=None,
                  help='also copy the .png here (the paper figures dir)')
  a = ap.parse_args()

  ymaxes = []
  for task, title, dexp, mexp, gexp in TASKS:
    hi, ns = 0.0, []
    for exps in (dexp, mexp, gexp):
      _, arr = collect(exps, task)
      ns.append(0 if arr is None else len(arr))
      if arr is not None:
        hi = max(hi, float(np.nanmax(np.nanmean(arr, 0) + np.nanstd(arr, 0))))
    # round up to a clean tick so the two labelled values read well
    step = 100 if hi > 250 else 25
    ymaxes.append(max(step, float(np.ceil(hi / step) * step)))
    print('%-30s n_dir=%d n_lip=%d n_gauss=%d  ymax=%.0f'
          % (task, ns[0], ns[1], ns[2], ymaxes[-1]))
    if ns[2] == 0:
      print('  WARNING: no Gaussian runs found for %s' % task)

  svg, W, H = build(ymaxes)
  pathlib.Path(a.out + '.svg').write_text(svg)
  subprocess.run(['rsvg-convert', '-f', 'png', '-d', '600', '-p', '600',
                  '-o', a.out + '.png', a.out + '.svg'], check=True)
  print('row layout %dx%d svg units' % (W, H))
  print('wrote', a.out + '.png')
  if a.copy_to:
    dest = pathlib.Path(a.copy_to) / (pathlib.Path(a.out).name + '.png')
    dest.write_bytes(pathlib.Path(a.out + '.png').read_bytes())
    print('copied to', dest)


if __name__ == '__main__':
  main()
