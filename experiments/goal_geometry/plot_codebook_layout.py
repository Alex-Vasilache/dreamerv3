#!/usr/bin/env python3
"""How the C classes of one block sit in space, on each arm.

One panel per arm -- by default Director against SOM-line+LiP, ``--arms`` for any
subset of the five. Each draws ONE block's C = 8 class vectors in that block's
own first ``--ndim`` principal components, joined in class index order, with the
index printed on the point and repeated by the colour ramp.

``--ndim 2`` frames a plane. ``--ndim 3`` draws a wireframe box seen from a
fixed camera, with a stem from every point to the floor -- an orthographic view
gives no other cue to where a point sits on the third axis. The printed index
is dropped automatically whenever the markers get too small to hold it, which
is what five panels of eight points does.

Every block is centered and scaled to unit RMS radius WITHIN the components
drawn, so the panels compare SHAPE and ORDER, not size -- how much of the goal
space a code can reach is a different measurement (``code_space_extent.py``),
and how much spread falls outside those components is the ``on a line`` number
above the panel. The PC signs are fixed so the class index increases to the
right where it varies with PC1 at all; that resolves PCA's arbitrary sign and
cannot manufacture an order that is not there, since a flip reverses a
scrambled sequence into another scrambled one.

The frame and, in 3-D, the camera are shared by every panel, so a direction
means the same thing across the row. The statistics printed above each panel
are over ALL blocks and seeds, not just the block drawn.

Reads the .npz written by ``codebook_layout.py``.

  python3 plot_codebook_layout.py --results layout --out codebook_layout
"""
import argparse
import os
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

ARMS = ['director', 'somlip']
ALL_ARMS = ['director', 'som_line', 'lipvq', 'somlip', 'somlip_relu']
INK, INK_2, FRAME, GRIDC = '#000000', '#1a1a1a', '#333333', '#d5d5d2'
AXIS = '#b9b8b2'
# Class index 0 -> C-1 as light -> dark, the ramp the other figures use for
# "later"/"further along". Without an ordering to read off the colour, a
# categorical palette would say the indices are unrelated labels -- which is
# true of Director and is exactly what the figure is asking about.
RAMP = ['#dbe8f7', '#bcd4ef', '#96bbe4', '#6ea3d9', '#4a86c9', '#2f6bb0',
        '#1c5090', '#0b3260']


def esc(s):
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


NDIM = 2


def project(pts, ndim=NDIM):
  """One block's (C, D) class vectors -> (C, ndim) aligned, unit-RMS coordinates.

  Returns the coordinates and the fraction of the block's variance the first
  principal component holds, i.e. how close to a straight line it is.
  """
  x = np.asarray(pts, float)
  x = x - x.mean(0, keepdims=True)
  u, s, vt = np.linalg.svd(x, full_matrices=False)
  ev = s ** 2
  pca1 = float(ev[0] / max(ev.sum(), 1e-12))
  y = x @ vt[:ndim].T
  # Scale by the spread WITHIN the subspace being drawn, not the full-space
  # spread. Under the latter a block that is genuinely 7-dimensional shows only
  # a fraction of its size in any 3-D view, so Director's panel collapsed into
  # an unreadable pile while the one-dimensional arms filled their frames. How
  # much spread lies outside the subspace is exactly what ``pca1`` reports, so
  # nothing is lost by taking it out of the picture.
  rms = np.sqrt((y ** 2).sum(1).mean())
  y = y / max(rms, 1e-12)
  idx = np.arange(len(y), dtype=float)
  # PCA fixes each axis only up to sign; orient them so a run reads the same
  # way round as its neighbours.
  if np.corrcoef(idx, y[:, 0])[0, 1] < 0:
    y[:, 0] *= -1
  for j in range(1, y.shape[1]):
    if y[:, j].sum() < 0:
      y[:, j] *= -1
  return y, pca1


def camera(y, az=-76.0, el=17.0):
  # The azimuth is close to -90 on purpose: the first principal component is
  # the axis the whole figure is about, so it has to run across the screen. At
  # -58 it pointed a third of the way into it, and eight markers strung along
  # it overlapped into an unreadable pile.
  """Orthographic 3-D -> 2-D, plus a depth key for back-to-front drawing.

  Returns screen ``(C, 2)`` with y already pointing down for SVG, and the
  distance from the camera of each point.
  """
  a, e = np.radians(az), np.radians(el)
  x, w, z = y[:, 0], y[:, 1], y[:, 2]
  sx = -np.sin(a) * x + np.cos(a) * w
  sy = -(-np.cos(a) * np.sin(e) * x - np.sin(a) * np.sin(e) * w +
         np.cos(e) * z)
  depth = (np.cos(a) * np.cos(e) * x + np.sin(a) * np.cos(e) * w +
           np.sin(e) * z)
  return np.stack([sx, sy], 1), depth


def stats(dec, ndim=NDIM):
  """(C,) index-order and line-ness statistics over every block and seed."""
  p1, mono = [], []
  for s in range(dec.shape[0]):
    for l in range(dec.shape[1]):
      y, f = project(dec[s, l], ndim)
      p1.append(f)
      t = y[:, 0]
      r = np.argsort(np.argsort(t)).astype(float)
      mono.append(abs(np.corrcoef(np.arange(len(t), dtype=float), r)[0, 1]))
  return float(np.mean(p1)), float(np.mean(mono))


def build(data, seed, block, target_pt, ndim=NDIM):
  P = 300
  n = len(data)
  # A panel title is centred on its panel and can be wider than one --
  # 'SOM-line + LiP + ReLU' is -- so the gap between panels and the outer
  # margins both have to hold that overhang or titles collide and the outer
  # ones are clipped. Sizing them from the titles rather than fixing them at
  # the widest case keeps a two-panel figure from sitting inset in the column.
  # W feeds the font scale which feeds the overhang, so estimate once and
  # settle: one pass is enough because the correction is small.
  longest = max(len(str(m[0])) for m, _ in data)
  W = 78 + n * P + (n - 1) * 84 + 78
  for _ in range(2):
    title_w = longest * (7.6 * W / target_pt) * 0.55
    # Clamped, and the clamp is not cosmetic: widening the gap shrinks each
    # panel in print, which makes the title relatively wider again. Left to
    # run, the loop diverges for any title wider than a panel -- with five
    # panels and 'SOM-line + LiP + ReLU' it has no solution at all, because
    # that title is wider than a fifth of the page at any margin.
    over = min(max(0.0, (title_w - P) / 2), P * 0.25)
    GAP = int(max(56, 2 * over + 20))
    PL = PR = int(max(24, over + 10))
    W = PL + n * P + (n - 1) * GAP + PR
  f = lambda pt: pt * W / target_pt
  # Whatever the clamp could not buy comes out of the title instead, so a long
  # label shrinks rather than colliding with its neighbour.
  fit_title = (P + 0.9 * GAP) / (longest * 0.55)
  F_TITLE = min(f(7.6), fit_title)
  F_SUB, F_AXIS, F_LEG = f(6.2), f(7.2), f(6.2)
  # Title and the two statistic lines stack above the panel. Their baselines
  # are derived from the font sizes rather than fixed offsets: the fonts scale
  # with W, so a fixed 24-unit gap that looked right at one panel count had the
  # lines overlapping at another.
  SUB2_DY = f(6.0)
  SUB1_DY = SUB2_DY + F_SUB * 1.25
  TITLE_DY = SUB1_DY + F_SUB * 0.30 + F_TITLE * 0.95
  PT = int(TITLE_DY + F_TITLE * 0.35 + f(6.0))
  # The bottom margin carries the two shared caption lines; sized from the
  # fonts rather than fixed, or they fall off whenever the figure is rendered
  # narrower than it was designed at.
  PB = int(f(52.0))
  H = PT + P + PB

  o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
       f'<rect width="{W}" height="{H}" fill="white"/>']

  # One frame for every panel: the blocks are already unit-RMS, so a shared
  # range is what makes "a line" and "a cloud" comparable across arms. In 3-D
  # the camera is shared too, so a direction means the same thing anywhere in
  # the row.
  lim = 0.0
  for _, dec in data:
    lim = max(lim, np.abs(project(dec[seed, block], ndim)[0]).max())
  lim *= 1.16
  if ndim == 3:
    # The cube's own corners set the drawing scale, so a panel is filled by the
    # box rather than by whichever arm happens to spread furthest on screen.
    corners = np.array([[i, j, k] for i in (-lim, lim) for j in (-lim, lim)
                        for k in (-lim, lim)], float)
    span = max(np.abs(camera(corners)[0]).max(), 1e-9)
  else:
    span = lim

  # Two panels leave ~139pt each in print, so eight markers strung along a line
  # get ~12pt apiece and can carry a legible digit. At five panels they could
  # not, and the index had to be left to the colour ramp alone.
  r = min(f(5.8), P * 0.88 / (2 * len(RAMP)))

  for k, (meta, dec) in enumerate(data):
    label, p1, mono, n = meta
    x0 = PL + k * (P + GAP)
    pad = P * 0.5 / span * 0.92
    to_screen = lambda s, x0=x0: (x0 + P / 2 + s[..., 0] * pad,
                                  PT + P / 2 + s[..., 1] * pad)

    def plane(v):
      """(n, ndim) -> (n, 2) screen offsets and a depth key.

      In 2-D the second component is negated because SVG's y grows downward,
      so a positive second principal component reads as up, matching the sign
      convention ``project`` applies.
      """
      v = np.atleast_2d(np.asarray(v, float))
      if ndim == 3:
        return camera(v)
      return v * np.array([1.0, -1.0]), np.zeros(len(v))

    def draw3(a, b, stroke, width, opacity=1.0, o=o, x0=x0):
      s, _ = plane(np.stack([a, b]))
      xs, ys = to_screen(s, x0)
      o.append(f'<line x1="{xs[0]:.1f}" y1="{ys[0]:.1f}" x2="{xs[1]:.1f}" '
               f'y2="{ys[1]:.1f}" stroke="{stroke}" stroke-width="{width:.1f}" '
               f'stroke-opacity="{opacity}"/>')

    if ndim == 3:
      # Wireframe cube: the 12 edges, drawn light. Without a box there is no
      # cue that the third axis exists and the panel reads as a flat scatter.
      for i in range(3):
        for sa in (-lim, lim):
          for sb in (-lim, lim):
            a, b = np.zeros(3), np.zeros(3)
            a[i], b[i] = -lim, lim
            a[(i + 1) % 3] = b[(i + 1) % 3] = sa
            a[(i + 2) % 3] = b[(i + 2) % 3] = sb
            draw3(a, b, GRIDC, 1.5)
    else:
      for v in (-1.0, 0.0, 1.0):
        draw3(np.array([v, -lim]), np.array([v, lim]), GRIDC, 1.6)
        draw3(np.array([-lim, v]), np.array([lim, v]), GRIDC, 1.6)
      o.append(f'<rect x="{x0}" y="{PT}" width="{P}" height="{P}" fill="none" '
               f'stroke="{FRAME}" stroke-width="2.2"/>')
    # The first principal axis through the middle, so "along the line" has a
    # visible reference.
    zero = np.zeros(ndim - 1)
    draw3(np.concatenate([[-lim], zero]), np.concatenate([[lim], zero]),
          AXIS, 2.0)

    y, _ = project(dec[seed, block], ndim)
    s3, depth = plane(y)
    xs, ys = to_screen(s3)

    o.append('<polyline points="%s" fill="none" stroke="%s" '
             'stroke-width="2.6" stroke-opacity="0.5" '
             'stroke-linejoin="round"/>' %
             (' '.join('%.1f,%.1f' % (a, b) for a, b in zip(xs, ys)),
              '#6d6c66'))
    if ndim == 3:
      # Stems to the floor of the box: with an orthographic camera these are
      # the only thing that says where a point sits on the third axis.
      for c in range(len(y)):
        draw3(y[c], np.array([y[c, 0], y[c, 1], -lim]), '#9c9a94', 1.4, 0.75)

    for c in np.argsort(depth):
      o.append(f'<circle cx="{xs[c]:.1f}" cy="{ys[c]:.1f}" r="{r:.1f}" '
               f'fill="{RAMP[c % len(RAMP)]}" stroke="#43413c" '
               f'stroke-width="1.1"/>')
      # Only when the marker is big enough to hold it. Five panels of eight
      # points squeeze each marker to ~3pt across, where a digit inside would
      # be illegible and the index is left to the colour ramp alone.
      if r >= f(4.0):
        # The ramp runs light to dark, so the digit has to flip with it.
        o.append(f'<text x="{xs[c]:.1f}" y="{ys[c] + r * 0.36:.1f}" '
                 f'font-size="{r * 1.05:.1f}" text-anchor="middle" '
                 f'fill="{"#ffffff" if c >= 4 else "#25405f"}">{c}</text>')

    # No square frame around the panel: the wireframe cube already bounds the
    # data, and a flat rectangle over a 3-D box reads as a second, conflicting
    # depth cue.
    # Five panels leave about 79pt each in print, so the two statistics go on
    # their own lines; on one line they ran into the neighbouring panel.
    o.append(f'<text x="{x0 + P / 2:.1f}" y="{PT - TITLE_DY:.1f}" '
             f'font-size="{F_TITLE:.1f}" fill="{INK}" text-anchor="middle">'
             f'{esc(label)}</text>')
    o.append(f'<text x="{x0 + P / 2:.1f}" y="{PT - SUB1_DY:.1f}" '
             f'font-size="{F_SUB:.1f}" fill="{INK_2}" text-anchor="middle">'
             f'on a line {p1:.2f}</text>')
    o.append(f'<text x="{x0 + P / 2:.1f}" y="{PT - SUB2_DY:.1f}" '
             f'font-size="{F_SUB:.1f}" fill="{INK_2}" text-anchor="middle">'
             f'in index order {mono:.2f}</text>')

  # Seed counts are not equal across arms -- the ReLU arm has fewer finished
  # runs -- so the line says which, rather than implying a single n.
  counts = [m[3] for m, _ in data]
  common = max(set(counts), key=counts.count)
  odd = ['%s %d' % (m[0], m[3]) for m, _ in data if m[3] != common]
  seeds = '%d seeds' % common
  if odd:
    seeds += ', ' + ' and '.join(odd)
  # Kept short deliberately: at five panels the canvas is ~2000 units and a
  # line much longer than this runs off both ends. The rest belongs in the
  # caption.
  # No separate colour key: the digit is on the point, so a row of swatches
  # saying which colour is which index would only repeat it.
  o.append(f'<text x="{W / 2:.1f}" y="{PT + P + f(15.0):.1f}" '
           f'font-size="{F_AXIS:.1f}" fill="{INK}" text-anchor="middle">'
           f'one block’s {len(RAMP)} classes, in its first {ndim} principal '
           f'components' + ('; the number on a point is its class index'
                            if r >= f(4.0) else '') + '</text>')
  o.append(f'<text x="{W / 2:.1f}" y="{PT + P + f(28.0):.1f}" '
           f'font-size="{F_LEG:.1f}" fill="{INK_2}" text-anchor="middle">'
           f'statistics are means over {len(RAMP)} blocks and seeds '
           f'({esc(seeds)})</text>')
  o.append('</svg>')
  return '\n'.join(o), W, H


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--results', default=str(HERE / 'layout'))
  ap.add_argument('--out', default=str(HERE / 'codebook_layout'))
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--block', type=int, default=0,
                  help='which of the L blocks to draw. The statistics above '
                       'each panel are over all of them either way.')
  ap.add_argument('--target-pt', type=float, default=0.98 * 397.0)
  ap.add_argument('--ndim', type=int, choices=[2, 3], default=NDIM,
                  help='2 draws a framed plane, 3 a wireframe box seen from a '
                       'fixed camera with a stem under every point.')
  ap.add_argument('--arms', default=','.join(ARMS),
                  help='comma-separated arms to draw, in order. Available: ' +
                       ', '.join(ALL_ARMS))
  a = ap.parse_args()

  data = []
  for arm in a.arms.split(','):
    p = os.path.join(a.results, arm + '.npz')
    if not os.path.exists(p):
      print('missing', p)
      return
    z = np.load(p, allow_pickle=True)
    dec = np.asarray(z['dec'], float)
    p1, mono = stats(dec, a.ndim)
    print('%-22s pca1 %.3f  mono %.3f   %d seeds x %d blocks x %d classes' %
          (str(z['label']), p1, mono, dec.shape[0], dec.shape[1], dec.shape[2]))
    data.append(((str(z['label']), p1, mono, dec.shape[0]), dec))

  svg, W, H = build(data, a.seed, a.block, a.target_pt, a.ndim)
  pathlib.Path(a.out + '.svg').write_text(svg)
  subprocess.run(['rsvg-convert', '-f', 'png',
                  '-w', str(int(round(a.target_pt / 72.0 * 600))),
                  '-o', a.out + '.png', a.out + '.svg'], check=True)
  print('layout %dx%d svg units' % (W, H))
  print('wrote', a.out + '.png')


if __name__ == '__main__':
  main()
