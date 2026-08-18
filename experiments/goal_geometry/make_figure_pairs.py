#!/usr/bin/env python3
"""Goal-code vs goal-space geometry, per ARM instead of per environment.

The four figures in motivation.tex that plot pairwise goal-code similarity
against pairwise goal-space similarity are measured on pure-Director baselines
across four environments. They answer "how much structure does Director's code
carry", not "does any of these autoencoders carry more". This draws the same
four views with the panels being the arms:

    --code hard --metric cosmax    the code the manager samples, worker metric
    --code soft --metric cosmax    before the categorical sample discards
    --code hard --metric mse       the same under the reconstruction metric
    --code soft --metric mse

One row per task, one column per arm. Under cosmax both axes are similarities
and a perfect code lies on y=x, which is drawn. Under MSE both axes are
distances in different units, so the reference is the least-squares
proportionality through the origin instead.

Reads ``results_pairs/*.npz`` (diag_goal_geometry.py --pairs-only).

  python3 make_figure_pairs.py --code hard --metric cosmax \
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

ARMS = [('director', 'Director'), ('som_line', 'SOM-line'),
        ('lipvq_prod', 'LiP'), ('som_lipvq_line_prod', 'SOM-line + LiP')]
COLORS = {'director': '#52514e', 'som_line': '#2a78d6',
          'lipvq_prod': '#1baf7a', 'som_lipvq_line_prod': '#eda100'}
TASKS = [('dmc_cartpole_swingup', 'Cartpole Swingup'),
         ('dmc_hopper_stand', 'Hopper Stand')]

TEXTWIDTH_PT = 397.0
TARGET_PT = 0.95 * TEXTWIDTH_PT
RES = 6
PANEL = 150 * RES
GAP_X, GAP_Y = 26 * RES, 40 * RES
# Two rotated labels live here (task, then quantity) plus the tick numbers;
# at 54*RES the single combined label ran off the left edge and was longer
# than the row is tall.
PAD_L, PAD_R = 84 * RES, 10 * RES
CELL = PANEL + PAD_L + PAD_R
NCOL, NROW = len(ARMS), len(TASKS)
FIG_W = NCOL * CELL + (NCOL - 1) * GAP_X
RATIO = FIG_W / TARGET_PT
NBINS = 20


def pt(v):
  return v * RATIO


F_TITLE, F_SUB, F_TICK, F_AXIS = pt(8.0), pt(6.4), pt(6.4), pt(6.6)
TICK_LEN = pt(3.5)
PAD_T = F_TITLE * 1.3 + F_SUB * 1.8
PAD_B = TICK_LEN + F_TICK * 1.5 + F_AXIS * 2.0
ROW_H = PANEL + PAD_T + PAD_B
FIG_H = NROW * ROW_H + (NROW - 1) * GAP_Y


def esc(s):
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def ticks(hi, lo=0.0, n=4):
  span = hi - lo
  if span <= 0:
    return [lo]
  mag = 10.0 ** np.floor(np.log10(span / n))
  step = min([m * mag for m in (1, 2, 2.5, 5, 10)],
             key=lambda s: abs(s - span / n))
  first = np.ceil(lo / step) * step
  return [v for v in np.arange(first, hi * 1.0001, step)]


def fmt(v):
  if abs(v) < 1e-12:
    return '0'
  return ('%g' % v) if abs(v - round(v)) < 1e-9 else ('%.3g' % v)


def load(results_dir, code, metric):
  gk = 'sim_goal' if metric == 'cosmax' else 'mse_goal'
  out = {}
  for f in sorted(glob.glob(os.path.join(results_dir, '*.npz'))):
    d = np.load(f, allow_pickle=True)
    meta = ast.literal_eval(str(d['meta']))
    name = os.path.basename(f)
    arm = next((k for k, _ in sorted(ARMS, key=lambda a: -len(a[0]))
                if f'_{k}_s' in name), None)
    if arm is None and meta.get('impl') == 'director':
      arm = 'director'
    if arm is None:
      continue
    # Plain Hamming equality ("hard": same id per block or not) is the wrong
    # metric for a codebook trained with the SOM neighborhood loss -- a
    # 1-apart neighbor decodes to a nearby goal but counts as a full miss.
    # Where the codebook has a real topology (meta['topology'] != 'none'),
    # swap in the already-computed topology-aware graded distance instead:
    # sim_index under cosmax (mean per-block |id_i - id_j|, ring-wrapped or
    # not per the codebook's own topology, normalized to a similarity) or
    # mse_index under mse (the same distance, squared and averaged per block
    # -- diag_goal_geometry.correlations' sim_code['index'] / mse_code['index']
    # -- the mse_code['hard'] analog, since mse_hard is proportional to plain
    # Hamming by construction and can't tell a near miss from a far one).
    graded = code == 'hard' and meta.get('topology', 'none') != 'none'
    ck = (('sim_index' if metric == 'cosmax' else 'mse_index') if graded
          else (('sim_' if metric == 'cosmax' else 'mse_') + code))
    rk = (f'corr/{metric}/index/pearson/mean' if graded else
          f'corr/{metric}/{code}/pearson/mean')
    if gk not in d or ck not in d:
      continue
    g, c = np.asarray(d[gk], float), np.asarray(d[ck], float)
    iu = np.triu_indices(g.shape[0], k=1)
    r = float(d[rk]) if rk in d else float('nan')
    out.setdefault((meta['task'], arm), []).append((g[iu], c[iu], r))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--results', default=str(HERE / 'results_pairs'))
  ap.add_argument('--code', choices=['hard', 'soft'], default='hard')
  ap.add_argument('--metric', choices=['cosmax', 'mse'], default='cosmax')
  ap.add_argument('--out', default=None)
  ap.add_argument('--copy-to', default=None)
  a = ap.parse_args()

  data = load(a.results, a.code, a.metric)
  if not data:
    print('no pair pools in', a.results)
    return
  print('cells:', ', '.join(f'{t.replace("dmc_", "")}/{arm}={len(v)}'
                            for (t, arm), v in sorted(data.items())))

  rng = np.random.RandomState(0)
  svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{FIG_W:.0f}" '
         f'height="{FIG_H:.0f}" viewBox="0 0 {FIG_W:.0f} {FIG_H:.0f}" '
         f'font-family="Helvetica,Arial,sans-serif">',
         f'<rect x="0" y="0" width="{FIG_W:.0f}" height="{FIG_H:.0f}" '
         f'fill="white"/>']
  COS = a.metric == 'cosmax'

  for ti, (task, tlabel) in enumerate(TASKS):
    y0 = ti * (ROW_H + GAP_Y)
    for ai, (arm, alabel) in enumerate(ARMS):
      x0 = ai * (CELL + GAP_X)
      seeds = data.get((task, arm))

      # Axis ranges. cosmax's x range (goal-space similarity) stays fixed --
      # goal states are whatever a live rollout visits, not something this
      # figure controls. The y range (code similarity) is rescaled per panel
      # to the sample's own worst pair instead of a generic -0.05 floor: the
      # theoretical floor (all L blocks maximally apart) is rarely realized
      # by codes an actual policy emits, and for the graded/topology arms in
      # particular the observed pairs can cluster well above 0 (see
      # make_goal_image_pairs.py's e538 hard panel -- no pair near 0.2
      # similarity turned up in a 512-state pool), so a fixed floor wastes
      # most of the panel on a similarity band nothing ever reaches. MSE
      # code-distance and goal-distance live in unrelated units (L*C-dim
      # one-hot vs. 1024-dim deter), so a range shared ACROSS PANELS there
      # either flattens arms with small code MSE or clips the ones with large
      # goal MSE -- scale each panel to its own data instead. Deriving yhi
      # from the panel's own least-squares slope (rather than its own y
      # percentile) makes the reference line land exactly on the top-right
      # corner, so it reads as a real diagonal here too.
      if COS:
        xlo, xhi, yhi = -0.2, 1.05, 1.05
        if seeds:
          worst = float(min(c.min() for _, c, _ in seeds))  # max dissimilarity observed
          ylo = worst - 0.05 * (yhi - worst)
        else:
          ylo = -0.05
      elif seeds:
        xf = np.concatenate([g for g, _, _ in seeds])
        yf = np.concatenate([c for _, c, _ in seeds])
        xlo, ylo = 0.0, 0.0
        xhi = float(np.percentile(xf, 99.0)) * 1.04
        slope = float((xf * yf).sum() / max((xf * xf).sum(), 1e-30))
        yhi = slope * xhi if slope > 0 else float(np.percentile(yf, 99.9)) * 1.04
      else:
        xlo, xhi, ylo, yhi = 0.0, 1.0, 0.0, 1.0  # unused: panel has no data

      def sx(v, x0=x0, xlo=xlo, xhi=xhi):
        return x0 + PAD_L + np.clip((v - xlo) / (xhi - xlo), -0.02, 1.02) * PANEL

      def sy(v, y0=y0, ylo=ylo, yhi=yhi):
        return y0 + PAD_T + (1 - np.clip((v - ylo) / (yhi - ylo), -0.02, 1.02)) * PANEL

      ax0, ay0, ax1, ay1 = sx(xlo), sy(ylo), sx(xhi), sy(yhi)
      color = COLORS[arm]
      svg.append(f'<text x="{x0 + CELL / 2:.1f}" y="{y0 + F_TITLE * 1.1:.1f}" '
                 f'font-size="{F_TITLE:.1f}" font-weight="bold" '
                 f'text-anchor="middle" fill="black">{esc(alabel)}</text>')
      if not seeds:
        continue
      rs = np.array([r for _, _, r in seeds])
      svg.append(f'<text x="{x0 + CELL / 2:.1f}" '
                 f'y="{y0 + F_TITLE * 1.1 + F_SUB * 1.6:.1f}" '
                 f'font-size="{F_SUB:.1f}" text-anchor="middle" fill="#333">'
                 f'r = {np.nanmean(rs):.2f} &#177; {np.nanstd(rs):.2f} '
                 f'(n={len(seeds)})</text>')

      # reference: y=x under cosmax (both axes are the same similarity), the
      # least-squares proportionality through zero under MSE (different units)
      if COS:
        lo, hi = max(xlo, ylo), min(xhi, yhi)
        rx0, ry0, rx1, ry1 = sx(lo), sy(lo), sx(hi), sy(hi)
      else:
        xf = np.concatenate([g for g, _, _ in seeds])
        yf = np.concatenate([c for _, c, _ in seeds])
        slope = float((xf * yf).sum() / max((xf * xf).sum(), 1e-30))
        xe = min(xhi, yhi / slope) if slope > 0 else xhi
        rx0, ry0, rx1, ry1 = sx(0), sy(0), sx(xe), sy(slope * xe)
      svg.append(f'<line x1="{rx0:.1f}" y1="{ry0:.1f}" x2="{rx1:.1f}" '
                 f'y2="{ry1:.1f}" stroke="#888" stroke-width="{pt(0.8):.1f}" '
                 f'stroke-dasharray="{pt(3.0):.1f},{pt(2.2):.1f}"/>')

      per = max(1, 4000 // len(seeds))
      dots = []
      for g, c, _ in seeds:
        sel = rng.choice(len(g), min(per, len(g)), replace=False)
        for xv, yv in zip(g[sel], c[sel]):
          dots.append(f'<circle cx="{sx(xv):.2f}" cy="{sy(yv):.2f}" '
                      f'r="{pt(0.7):.2f}" fill="{color}" '
                      f'fill-opacity="0.13"/>')
      svg.append(''.join(dots))

      # trend: a genuinely discrete code (Hamming's L+1 exact levels) bins on
      # ITS OWN axis; a continuous one -- the soft code, or "hard" once it is
      # the graded topology-aware distance for a som-topology arm, which
      # takes far more than a handful of realized values -- bins on x like
      # any scatter. Detected from what actually got loaded for this panel,
      # not from --code, since one cosmax-hard figure can mix discrete
      # (director, lipvq_prod) and graded-continuous (som_line,
      # som_lipvq_line_prod) panels.
      all_c = np.concatenate([c for _, c, _ in seeds])
      discrete = len(np.unique(np.round(all_c, 6))) <= 2 * NBINS
      if discrete:
        lv = np.unique(np.concatenate([np.unique(c) for _, c, _ in seeds]))
        rows = [[g[np.isclose(c, v)].mean() if np.isclose(c, v).sum() >= 10
                 else np.nan for v in lv] for g, c, _ in seeds]
        tv = lv
      else:
        edges = np.linspace(xlo, xhi, NBINS + 1)
        tv = 0.5 * (edges[:-1] + edges[1:])
        rows = []
        for g, c, _ in seeds:
          idx = np.clip(np.digitize(g, edges) - 1, 0, NBINS - 1)
          rows.append([c[idx == b].mean() if (idx == b).sum() >= 10
                       else np.nan for b in range(NBINS)])
      arr = np.array(rows, float)
      with np.errstate(invalid='ignore'):
        tm, ts = np.nanmean(arr, 0), np.nanstd(arr, 0)
      ok = ~np.isnan(tm)
      if ok.sum() >= 2:
        vv, vm, vs = tv[ok], tm[ok], ts[ok]
        if discrete:
          band = ([(m + s, y) for m, s, y in zip(vm, vs, vv)] +
                  [(m - s, y) for m, s, y in zip(vm, vs, vv)][::-1])
          line = list(zip(vm, vv))
        else:
          band = ([(x, m + s) for x, m, s in zip(vv, vm, vs)] +
                  [(x, m - s) for x, m, s in zip(vv, vm, vs)][::-1])
          line = list(zip(vv, vm))
        svg.append('<polygon points="%s" fill="%s" fill-opacity="0.22" '
                   'stroke="none"/>' % (' '.join(
                       f'{sx(p):.2f},{sy(q):.2f}' for p, q in band), color))
        svg.append('<polyline points="%s" fill="none" stroke="#1A1A1A" '
                   'stroke-width="%.2f" stroke-linejoin="round"/>' % (
                       ' '.join(f'{sx(p):.2f},{sy(q):.2f}' for p, q in line),
                       pt(1.3)))

      svg.append(f'<line x1="{ax0:.1f}" y1="{ay0:.1f}" x2="{ax1:.1f}" '
                 f'y2="{ay0:.1f}" stroke="#8a8880" '
                 f'stroke-width="{pt(0.6):.2f}"/>')
      svg.append(f'<line x1="{ax0:.1f}" y1="{ay0:.1f}" x2="{ax0:.1f}" '
                 f'y2="{ay1:.1f}" stroke="#8a8880" '
                 f'stroke-width="{pt(0.6):.2f}"/>')
      tty = ay0 + TICK_LEN + F_TICK * 1.2
      for v in ([0.0, 1.0] if COS else ticks(xhi, xlo)):
        svg.append(f'<line x1="{sx(v):.1f}" y1="{ay0:.1f}" x2="{sx(v):.1f}" '
                   f'y2="{ay0 + TICK_LEN:.1f}" stroke="black" '
                   f'stroke-width="{pt(1.2):.1f}"/>')
        svg.append(f'<text x="{sx(v):.1f}" y="{tty:.1f}" '
                   f'font-size="{F_TICK:.1f}" text-anchor="middle" '
                   f'fill="black">{fmt(v)}</text>')
      for v in ticks(yhi, ylo):
        svg.append(f'<line x1="{ax0 - TICK_LEN:.1f}" y1="{sy(v):.1f}" '
                   f'x2="{ax0:.1f}" y2="{sy(v):.1f}" stroke="black" '
                   f'stroke-width="{pt(1.2):.1f}"/>')
        # Every panel scales its own y range now (cosmax to its own worst
        # observed pair, MSE to its own slope), so every column needs its
        # own tick labels -- a shared range that only the leftmost column
        # labeled no longer exists on either metric.
        svg.append(f'<text x="{ax0 - TICK_LEN - F_TICK * 0.4:.1f}" '
                   f'y="{sy(v) + F_TICK * 0.35:.1f}" '
                   f'font-size="{F_TICK:.1f}" text-anchor="end" '
                   f'fill="black">{fmt(v)}</text>')
      svg.append(f'<text x="{x0 + CELL / 2:.1f}" '
                 f'y="{tty + F_AXIS * 1.5:.1f}" font-size="{F_AXIS:.1f}" '
                 f'text-anchor="middle" fill="#333">goal-space '
                 f'{"similarity" if COS else "MSE"}</text>')
    # Two rotated labels per row: the task, then the quantity. One combined
    # label was longer than the row is tall and clipped off the left edge.
    cy = y0 + PAD_T + PANEL / 2
    cx = F_AXIS * 1.0
    svg.append(f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="{F_AXIS:.1f}" '
               f'font-weight="bold" text-anchor="middle" fill="black" '
               f'transform="rotate(-90 {cx:.1f} {cy:.1f})">'
               f'{esc(tlabel)}</text>')
    qx = F_AXIS * 2.4
    qlab = ('goal-code sim. (%s)' if COS else 'goal-code MSE (%s)') % a.code
    svg.append(f'<text x="{qx:.1f}" y="{cy:.1f}" '
               f'font-size="{F_AXIS * 0.85:.1f}" text-anchor="middle" '
               f'fill="#333" transform="rotate(-90 {qx:.1f} {cy:.1f})">'
               f'{qlab}</text>')
  svg.append('</svg>')

  base = a.out or str(HERE / f'goal_pairs_{a.metric}_{a.code}')
  pathlib.Path(base + '.svg').write_text('\n'.join(svg))
  subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', base + '.pdf',
                  base + '.svg'], check=True)
  print(f'wrote {base}.pdf  ({FIG_W:.0f} x {FIG_H:.0f})')
  for (t, arm) in sorted(data):
    rs = np.array([r for _, _, r in data[(t, arm)]])
    print('  %-22s %-22s n=%d  r=%.3f+/-%.3f'
          % (t.replace('dmc_', ''), arm, len(rs), np.nanmean(rs),
             np.nanstd(rs)))
  if a.copy_to:
    os.makedirs(a.copy_to, exist_ok=True)
    subprocess.run(['cp', base + '.pdf', os.path.join(
        a.copy_to, os.path.basename(base) + '.pdf')], check=True)
    print('copied to', a.copy_to)


if __name__ == '__main__':
  main()
