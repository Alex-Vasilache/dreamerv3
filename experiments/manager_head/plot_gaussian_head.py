#!/usr/bin/env python3
"""How the discretized Gaussian works inside one block's line of classes.

Four panels, all showing the same block: the C=8 ordered classes on the x axis
and their probabilities as bars. The continuous Gaussian is drawn through them,
and because the bars ARE that curve read off at the integer indices and
normalized, the curve passes exactly through the top of every bar. That is the
whole construction in one picture.

The panels then vary one thing each: the location, the scale, and the case
where the location sits on an end of the line.

  python3 plot_gaussian_head.py --out gaussian_head
"""
import argparse
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

C = 8
YMAX = 0.5
INK, INK_2, FRAME, GRIDC = '#000000', '#1a1a1a', '#333333', '#d5d5d2'
BAR, CURVE = '#2a78d6', '#5c5b56'

# (mu, sigma, title)
PANELS = [
    (3.5, 1.5, 'the curve, read off'),
    (5.5, 1.5, r'$\mu$ moves the choice'),
    (3.5, 0.6, r'$\sigma$ sets the commitment'),
    (0.0, 1.5, 'at the end of the line'),
]
TITLES = ['read off the curve', 'mu moves the choice',
          'sigma sets the commitment', 'at the end of the line']


def pmf(mu, sigma):
    z = (np.arange(C) - mu) / sigma
    e = np.exp(-0.5 * z ** 2)
    return e / e.sum()


def esc(s):
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def build():
    P, GAP = 330, 96
    # PB holds three stacked rows below the frame: the mu marker, the class
    # indices, and the shared axis label. At 150 the label sat on the indices.
    PL, PR, PT, PB = 132, 40, 108, 214
    n = len(PANELS)
    W = PL + n * P + (n - 1) * GAP + PR
    H = PT + P + PB
    f = lambda pt: pt * W / (0.98 * 397.0)
    F_TITLE, F_TICK, F_AXIS = f(7.6), f(6.4), f(7.8)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>']

    for k, (mu, sigma, _) in enumerate(PANELS):
        x0 = PL + k * (P + GAP)
        p = pmf(mu, sigma)
        # class j sits at the centre of its slot, so the end bars have margin
        sx = lambda v: x0 + (v + 0.5) * (P / C)
        sy = lambda v: PT + (1.0 - min(v, YMAX) / YMAX) * P

        for v in (0.0, 0.25, 0.5):
            o.append(f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x0 + P}" '
                     f'y2="{sy(v):.1f}" stroke="{GRIDC}" stroke-width="1.7"/>')
            if k == 0:
                o.append(f'<text x="{x0 - 14}" y="{sy(v) + F_TICK * 0.36:.1f}" '
                         f'font-size="{F_TICK:.1f}" fill="{INK_2}" '
                         f'text-anchor="end">{("%g" % v).lstrip("0") or "0"}</text>')

        # bars: the normalized probabilities
        bw = (P / C) * 0.56
        for j in range(C):
            o.append(f'<rect x="{sx(j) - bw / 2:.1f}" y="{sy(p[j]):.1f}" '
                     f'width="{bw:.1f}" height="{max(sy(0) - sy(p[j]), 0.6):.1f}" '
                     f'fill="{BAR}" fill-opacity="0.85"/>')

        # the continuous Gaussian, scaled by the SAME normalizer as the bars, so
        # it necessarily passes through the top of each one
        Z = np.exp(-0.5 * ((np.arange(C) - mu) / sigma) ** 2).sum()
        xs = np.linspace(-0.5, C - 0.5, 400)
        ys = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / Z
        pts = ' '.join('%.1f,%.1f' % (sx(x), sy(y)) for x, y in zip(xs, ys)
                       if y <= YMAX)
        o.append(f'<polyline points="{pts}" fill="none" stroke="{CURVE}" '
                 f'stroke-width="3.4" stroke-dasharray="9,6"/>')
        for j in range(C):
            if p[j] <= YMAX:
                o.append(f'<circle cx="{sx(j):.1f}" cy="{sy(p[j]):.1f}" r="6.5" '
                         f'fill="{CURVE}" stroke="white" stroke-width="2.4"/>')

        # mu marker on the axis
        if -0.5 <= mu <= C - 0.5:
            o.append(f'<line x1="{sx(mu):.1f}" y1="{sy(0):.1f}" '
                     f'x2="{sx(mu):.1f}" y2="{sy(0) + 20:.1f}" '
                     f'stroke="{CURVE}" stroke-width="3.4"/>')
            o.append(f'<text x="{sx(mu):.1f}" y="{sy(0) + 20 + F_TICK * 1.5:.1f}" '
                     f'font-size="{F_TICK:.1f}" fill="{CURVE}" '
                     f'text-anchor="middle">&#956;</text>')

        o.append(f'<rect x="{x0}" y="{PT}" width="{P}" height="{P}" fill="none" '
                 f'stroke="{FRAME}" stroke-width="2.2"/>')
        for j in range(C):
            o.append(f'<text x="{sx(j):.1f}" y="{sy(0) + F_TICK * 3.4:.1f}" '
                     f'font-size="{F_TICK:.1f}" fill="{INK_2}" '
                     f'text-anchor="middle">{j}</text>')

        lab = ('&#956; = %g,  &#963; = %g' % (mu, sigma))
        o.append(f'<text x="{x0 + P / 2:.1f}" y="{PT - 30:.1f}" '
                 f'font-size="{F_TITLE:.1f}" fill="{INK}" '
                 f'text-anchor="middle">{lab}</text>')

    o.append(f'<text x="{PL + (W - PL - PR) / 2:.1f}" y="{H - 34:.1f}" '
             f'font-size="{F_AXIS:.1f}" fill="{INK}" text-anchor="middle">'
             f'class index within one block</text>')
    cy = PT + P / 2
    o.append(f'<text x="30" y="{cy:.1f}" font-size="{F_AXIS * 0.85:.1f}" '
             f'fill="{INK}" text-anchor="middle" '
             f'transform="rotate(-90 30 {cy:.1f})">probability</text>')
    o.append('</svg>')
    return '\n'.join(o), W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'gaussian_head'))
    a = ap.parse_args()
    for mu, sigma, _ in PANELS:
        p = pmf(mu, sigma)
        print('mu=%4.1f sigma=%.1f -> %s  (mode %d)'
              % (mu, sigma, ' '.join('%.3f' % v for v in p), p.argmax()))
    svg, W, H = build()
    pathlib.Path(a.out + '.svg').write_text(svg)
    subprocess.run(['rsvg-convert', '-f', 'png', '-d', '600', '-p', '600',
                    '-o', a.out + '.png', a.out + '.svg'], check=True)
    print('layout %dx%d svg units' % (W, H))
    print('wrote', a.out + '.png')


if __name__ == '__main__':
    main()
