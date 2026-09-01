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
# 0.55, not 0.5: at the regulated entropy the Cauchy's mode reaches 0.525, and
# at 0.5 the peak and its dot were clipped off the top of the frame.
YMAX = 0.55
NU = 1.0                           # Student-t tail weight; 1 = Cauchy
INK, INK_2, FRAME, GRIDC = '#000000', '#1a1a1a', '#333333', '#d5d5d2'
BAR, CURVE, CAU = '#2a78d6', '#5c5b56', '#c8471f'

# (mu, sigma_gauss, sigma_cauchy, label). The first four panels give BOTH heads
# the same mu and sigma, so the only difference on show is the tail weight. The
# last one instead gives each head the sigma its entropy regularizer actually
# settles at, which is the operating point the agent runs in.
PANELS = [
    (3.5, 1.5, 1.5, None),
    (5.5, 1.5, 1.5, None),
    (3.5, 0.6, 0.6, None),
    (0.0, 1.5, 1.5, None),
    (3.0, None, None, 'at the regulated entropy 0.7'),
]


def pmf(mu, sigma):
    z = (np.arange(C) - mu) / sigma
    e = np.exp(-0.5 * z ** 2)
    return e / e.sum()


def pmf_t(mu, sigma, nu=NU):
    """Student-t read off the same way; nu -> inf gives ``pmf`` exactly."""
    z = (np.arange(C) - mu) / sigma
    lg = -0.5 * (nu + 1.0) * np.log1p(z * z / nu)
    e = np.exp(lg - lg.max())
    return e / e.sum()


def norm_ent(p):
    return float(-(p * np.log(np.clip(p, 1e-300, 1))).sum() / np.log(C))


def scale_for(fn, mu, target):
    """Entropy is monotone in the scale, so bisect."""
    lo, hi = 1e-4, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_ent(fn(mu, mid)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def panel_scales(mu, sg, sc):
    if sg is None:
        sg = scale_for(pmf, mu, 0.7)
    if sc is None:
        sc = scale_for(pmf_t, mu, 0.7)
    return sg, sc


def esc(s):
  return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def build():
    # GAP 150, not 96: the last panel's title is the longest and at 96 it ran
    # into the one to its left.
    P, GAP = 330, 150
    # PB holds three stacked rows below the frame: the mu marker, the class
    # indices, and the shared axis label. At 150 the label sat on the indices.
    # PR matches PL so the last panel's centred title has room; PB holds four
    # stacked rows below the frame: the mu marker, the class indices, the
    # legend, and the shared axis label.
    PL, PR, PT, PB = 132, 132, 158, 360
    n = len(PANELS)
    W = PL + n * P + (n - 1) * GAP + PR
    H = PT + P + PB
    f = lambda pt: pt * W / (0.98 * 397.0)
    F_TITLE, F_TICK, F_AXIS = f(7.6), f(6.4), f(7.8)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>']

    for k, (mu, sg0, sc0, note) in enumerate(PANELS):
        sigma, sigma_c = panel_scales(mu, sg0, sc0)
        x0 = PL + k * (P + GAP)
        p = pmf(mu, sigma)
        pc = pmf_t(mu, sigma_c)
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

        # The Cauchy, read off exactly the same way. Same normalizer trick, so
        # this curve also passes through its own dots.
        zc = (np.arange(C) - mu) / sigma_c
        Zc = np.exp(-0.5 * (NU + 1.0) * np.log1p(zc * zc / NU)).sum()
        zs = (xs - mu) / sigma_c
        yc = np.exp(-0.5 * (NU + 1.0) * np.log1p(zs * zs / NU)) / Zc
        ptc = ' '.join('%.1f,%.1f' % (sx(x), sy(y)) for x, y in zip(xs, yc)
                       if y <= YMAX)
        o.append(f'<polyline points="{ptc}" fill="none" stroke="{CAU}" '
                 f'stroke-width="3.4"/>')
        for j in range(C):
            if pc[j] <= YMAX:
                o.append(f'<circle cx="{sx(j):.1f}" cy="{sy(pc[j]):.1f}" r="6.5" '
                         f'fill="{CAU}" stroke="white" stroke-width="2.4"/>')

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

        if note is None:
            lines = [('&#956; = %g,  &#963; = %g' % (mu, sigma))]
        else:
            lines = ['at entropy 0.7',
                     '&#963; = %.2f vs %.2f' % (sigma, sigma_c)]
        # line spacing has to exceed F_TITLE (~49 units here) or the two rows
        # of the last panel's title collide
        for i, lab in enumerate(reversed(lines)):
            o.append(f'<text x="{x0 + P / 2:.1f}" y="{PT - 30 - i * 56:.1f}" '
                     f'font-size="{F_TITLE:.1f}" fill="{INK}" '
                     f'text-anchor="middle">{lab}</text>')

    # legend, on its own row between the class indices and the axis label
    ybase = PT + P
    ly = ybase + F_TICK * 6.0
    lx = PL + (W - PL - PR) / 2 - 520
    for dx, col, dash, txt in [
            (0, CURVE, '9,6', 'discretized Gaussian'),
            (600, CAU, None, f'discretized Cauchy (ν = {NU:g})')]:
        da = f' stroke-dasharray="{dash}"' if dash else ''
        o.append(f'<line x1="{lx + dx}" y1="{ly}" x2="{lx + dx + 52}" y2="{ly}" '
                 f'stroke="{col}" stroke-width="3.4"{da}/>')
        o.append(f'<circle cx="{lx + dx + 26}" cy="{ly}" r="6.5" fill="{col}" '
                 f'stroke="white" stroke-width="2.4"/>')
        o.append(f'<text x="{lx + dx + 66}" y="{ly + F_TICK * 0.36:.1f}" '
                 f'font-size="{F_TICK:.1f}" fill="{INK_2}">{esc(txt)}</text>')

    o.append(f'<text x="{PL + (W - PL - PR) / 2:.1f}" '
             f'y="{ybase + F_TICK * 8.6:.1f}" '
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
    for mu, sg0, sc0, note in PANELS:
        sigma, sigma_c = panel_scales(mu, sg0, sc0)
        p, pc = pmf(mu, sigma), pmf_t(mu, sigma_c)
        print('mu=%4.1f  %s' % (mu, note or ''))
        print('  gauss sigma=%.3f H=%.3f -> %s' % (
            sigma, norm_ent(p), ' '.join('%.4f' % v for v in p)))
        print('  cauchy sigma=%.3f H=%.3f -> %s' % (
            sigma_c, norm_ent(pc), ' '.join('%.4f' % v for v in pc)))
    svg, W, H = build()
    pathlib.Path(a.out + '.svg').write_text(svg)
    subprocess.run(['rsvg-convert', '-f', 'png', '-d', '600', '-p', '600',
                    '-o', a.out + '.png', a.out + '.svg'], check=True)
    print('layout %dx%d svg units' % (W, H))
    print('wrote', a.out + '.png')


if __name__ == '__main__':
    main()
