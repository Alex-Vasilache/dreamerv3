#!/usr/bin/env python3
"""Three panels answering the motivation todo, in the paper's row style.

  SIZE       effective dimension of the decoded code grid next to the states
             the policy actually visits (participation ratio of the covariance
             spectrum), plus the radius ratio printed under each pair.
  OVERLAP    nearest-neighbour cosine_max, real->goal (can the manager ask for
             the states we visit?) and goal->real (is what it asks for a state
             at all?), against the real->real nearest-neighbour distance, which
             is the only honest scale for both.
  SMOOTHNESS how far one block-flip moves the decoded goal, as a fraction of
             the RMS distance between two random goals. Whisker is p5-p95 over
             the sampled flips.

  python3 plot_code_space_extent.py --results extent --out code_space_extent
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
    ('dmc_hopper_stand', 'Hop. Stand'),
    ('dmc_cartpole_swingup_sparse', 'CP Sparse'),
    ('dmc_cheetah_run', 'Cheetah'),
    ('dmc_hopper_hop', 'Hopper Hop'),
]
INK, INK_2, FRAME, GRIDC = '#000000', '#1a1a1a', '#333333', '#d5d5d2'
C_GOAL, C_REAL, C_REF = '#2a78d6', '#5c5b56', '#eb6834'


def esc(s):
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def load(results_dir):
  out = {}
  for f in sorted(glob.glob(os.path.join(results_dir, '*.npz'))):
    d = np.load(f, allow_pickle=True)
    out.setdefault(str(d['task']), []).append({
        'eff_goal': float(d['eff_dim_goal']),
        'eff_real': float(d['eff_dim_real']),
        'rad_goal': float(d['radius_goal']),
        'rad_real': float(d['radius_real']),
        'cover': float(np.mean(d['cover'])),
        'realiz': float(np.mean(d['realiz'])),
        'ref': float(np.mean(d['ref_nn'])),
        'step_rel': float(np.mean(d['step_rel'])),
        'step_p5': float(np.percentile(d['step_rel'], 5)),
        'step_p95': float(np.percentile(d['step_rel'], 95)),
    })
  return out


def agg(data, key):
  return {t: (float(np.mean([r[key] for r in v])),
              float(np.std([r[key] for r in v]))) for t, v in data.items()}


def build(data):
  P, GAP = 400, 150
  PL, PR, PT, PB = 150, 110, 110, 230
  n = 3
  W = PL + n * P + (n - 1) * GAP + PR
  H = PT + P + PB
  f = lambda pt: pt * W / (0.98 * 397.0)
  F_TITLE, F_TICK, F_LAB, F_LEG = f(8.2), f(6.2), f(6.0), f(6.6)

  present = [(t, lab) for t, lab in TASKS if t in data]
  nt = len(present)
  o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
       f'<rect width="{W}" height="{H}" fill="white"/>']

  def panel(k, title, ymax, series, yfmt='%.1f', ref=None, whisk=None):
    """series: list of (label, colour, {task: (mean, sd)})."""
    x0 = PL + k * (P + GAP)
    sy = lambda v: PT + (1.0 - min(v, ymax) / ymax) * P
    for v in np.linspace(0, ymax, 5):
      o.append(f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x0 + P}" '
               f'y2="{sy(v):.1f}" stroke="{GRIDC}" stroke-width="1.7"/>')
      o.append(f'<text x="{x0 - 14}" y="{sy(v) + F_TICK * 0.36:.1f}" '
               f'font-size="{F_TICK:.1f}" fill="{INK_2}" text-anchor="end">'
               f'{yfmt % v}</text>')
    slot = P / nt
    nb = len(series)
    bw = slot * 0.72 / nb
    for i, (t, lab) in enumerate(present):
      cx = x0 + (i + 0.5) * slot
      for j, (_, col, vals) in enumerate(series):
        if t not in vals:
          continue
        m, sd = vals[t]
        bx = cx - (nb * bw) / 2 + j * bw
        o.append(f'<rect x="{bx:.1f}" y="{sy(m):.1f}" width="{bw - 4:.1f}" '
                 f'height="{max(sy(0) - sy(m), 0.5):.1f}" fill="{col}" '
                 f'fill-opacity="0.85"/>')
        if sd > 0:
          mx = bx + (bw - 4) / 2
          o.append(f'<line x1="{mx:.1f}" y1="{sy(max(m - sd, 0)):.1f}" '
                   f'x2="{mx:.1f}" y2="{sy(min(m + sd, ymax)):.1f}" '
                   f'stroke="{INK}" stroke-width="2.4" stroke-opacity="0.55"/>')
      if whisk and t in whisk:
        lo, hi = whisk[t]
        o.append(f'<line x1="{cx:.1f}" y1="{sy(lo):.1f}" x2="{cx:.1f}" '
                 f'y2="{sy(min(hi, ymax)):.1f}" stroke="{INK}" '
                 f'stroke-width="2.6" stroke-opacity="0.5"/>')
      if ref and t in ref:
        rv = ref[t][0]
        o.append(f'<line x1="{cx - slot * 0.38:.1f}" y1="{sy(rv):.1f}" '
                 f'x2="{cx + slot * 0.38:.1f}" y2="{sy(rv):.1f}" '
                 f'stroke="{C_REF}" stroke-width="4" stroke-dasharray="9,6"/>')
      o.append(f'<text x="{cx:.1f}" y="{sy(0) + F_LAB * 2.0:.1f}" '
               f'font-size="{F_LAB:.1f}" fill="{INK_2}" text-anchor="middle" '
               f'transform="rotate(-32 {cx:.1f} {sy(0) + F_LAB * 2.0:.1f})">'
               f'{esc(lab)}</text>')
    o.append(f'<rect x="{x0}" y="{PT}" width="{P}" height="{P}" fill="none" '
             f'stroke="{FRAME}" stroke-width="2.2"/>')
    o.append(f'<text x="{x0 + P / 2:.1f}" y="{PT - 30:.1f}" '
             f'font-size="{F_TITLE:.1f}" fill="{INK}" text-anchor="middle">'
             f'{esc(title)}</text>')
    return x0

  eg, er = agg(data, 'eff_goal'), agg(data, 'eff_real')
  ymax1 = max(max(v[0] + v[1] for v in eg.values()),
              max(v[0] + v[1] for v in er.values()))
  panel(0, 'Effective dimension', np.ceil(ymax1 / 2) * 2,
        [('addressable goals', C_GOAL, eg), ('visited states', C_REAL, er)],
        yfmt='%.0f')

  cov, rea, ref = agg(data, 'cover'), agg(data, 'realiz'), agg(data, 'ref')
  panel(1, 'Nearest-neighbour similarity', 1.0,
        [('real -> goal (coverage)', C_GOAL, cov),
         ('goal -> real (realizable)', C_REAL, rea)],
        yfmt='%.2f', ref=ref)

  sr = agg(data, 'step_rel')
  wh = {t: (float(np.mean([r['step_p5'] for r in v])),
            float(np.mean([r['step_p95'] for r in v])))
        for t, v in data.items()}
  ymax3 = max(v[1] for v in wh.values())
  panel(2, 'One block-flip step', np.ceil(ymax3 * 10) / 10,
        [('step / RMS pair distance', C_GOAL, sr)], yfmt='%.1f', whisk=wh)

  # legend
  items = [('addressable goal codes', C_GOAL), ('visited states', C_REAL),
           ('real to nearest real (scale)', C_REF)]
  span = (W - PL - PR) / len(items)
  for i, (label, col) in enumerate(items):
    x = PL + i * span
    y = H - 30
    if col == C_REF:
      o.append(f'<line x1="{x:.1f}" y1="{y - 8:.1f}" x2="{x + 46:.1f}" '
               f'y2="{y - 8:.1f}" stroke="{col}" stroke-width="4" '
               f'stroke-dasharray="9,6"/>')
    else:
      o.append(f'<rect x="{x:.1f}" y="{y - F_LEG:.1f}" width="46" height="11" '
               f'fill="{col}"/>')
    o.append(f'<text x="{x + 58:.1f}" y="{y:.1f}" font-size="{F_LEG:.1f}" '
             f'fill="{INK_2}">{esc(label)}</text>')
  o.append('</svg>')
  return '\n'.join(o), W, H


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--results', default=str(HERE / 'extent'))
  ap.add_argument('--out', default=str(HERE / 'code_space_extent'))
  a = ap.parse_args()
  data = load(a.results)
  if not data:
    print('no npz in', a.results)
    return
  print('%-30s %5s %8s %8s %9s %9s %8s %8s' %
        ('task', 'n', 'effGoal', 'effReal', 'coverage', 'realiz', 'ref', 'step'))
  for t, v in data.items():
    g = lambda k: np.mean([r[k] for r in v])
    print('%-30s %5d %8.2f %8.2f %9.3f %9.3f %8.3f %8.3f' %
          (t, len(v), g('eff_goal'), g('eff_real'), g('cover'), g('realiz'),
           g('ref'), g('step_rel')))
  svg, W, H = build(data)
  pathlib.Path(a.out + '.svg').write_text(svg)
  subprocess.run(['rsvg-convert', '-f', 'png', '-d', '600', '-p', '600',
                  '-o', a.out + '.png', a.out + '.svg'], check=True)
  print('layout %dx%d svg units' % (W, H))
  print('wrote', a.out + '.png')


if __name__ == '__main__':
  main()
