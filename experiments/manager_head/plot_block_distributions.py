#!/usr/bin/env python3
"""What the manager's distribution over one goal code looks like, per block.

A goal code is L=8 blocks, and the manager emits one distribution over the C=8
classes of each block. This draws all L of them for both policy classes:

  Director   an independent categorical per block, i.e. C free logits. Nothing
             ties one class to the next, so the row is jagged and the classes
             it favours are scattered along the block.
  Ours       a discretized Gaussian per block, i.e. one location and one scale.
             The row is a single bump, and where it sits is the choice.

Both are drawn at the SAME entropy, so the comparison is about the shape of the
distributions and not about how confident the two heads happen to be. Without
that control a jagged row could simply be a sharper one.

  python3 plot_block_distributions.py --out block_distributions
"""
import argparse
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

L, C = 8, 8
TARGET_H = 0.5                     # normalized entropy, = manager_actent_target
INK, INK_2, FRAME, GRIDC = '#000000', '#1a1a1a', '#333333', '#d5d5d2'
CAT, GAU = '#5c5b56', '#2a78d6'


def norm_ent(p):
    return float(-(p * np.log(np.clip(p, 1e-300, 1))).sum() / np.log(C))


def categorical(rng, scale):
    z = rng.normal(0, scale, C)
    e = np.exp(z - z.max())
    return e / e.sum()


def gaussian(mu, sigma):
    e = np.exp(-0.5 * ((np.arange(C) - mu) / sigma) ** 2)
    return e / e.sum()


def solve(fn, lo, hi, iters=80):
    """Bisect for fn(x) == TARGET_H. Direction-agnostic.

    A categorical's entropy falls as its logit scale grows while a Gaussian's
    rises with sigma, so the bracket is oriented from the endpoint values rather
    than assumed. (The first version negated one function to force a single
    direction, which made the comparison against a positive target always false
    and silently returned the bracket edge.)
    """
    flo, fhi = fn(lo), fn(hi)
    assert (flo - TARGET_H) * (fhi - TARGET_H) < 0, (
        'target %.3f not bracketed by [%.3f, %.3f]' % (TARGET_H, flo, fhi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if (fm - TARGET_H) * (flo - TARGET_H) > 0:
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return 0.5 * (lo + hi)


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def build(rows_cat, rows_gau):
    """A tight L x C grid per policy, one cell per (block, class).

    No gaps: the goal code IS an L x C matrix, so the figure is drawn as one.
    Every cell is ruled, and the probability fills its cell from the bottom.
    The fill scale is shared by every cell in BOTH panels, so cell heights are
    comparable across rows and across the two policies rather than each row
    being normalized to its own maximum.
    """
    # Square cells, so each panel is itself a square L x C grid.
    CELL = 48
    CW = RH = CELL
    PW, PH = C * CW, L * RH
    GAP = 96
    PL, PR, PT, PB = 118, 40, 76, 96
    W = PL + 2 * PW + GAP + PR
    H = PT + PH + PB
    f = lambda pt: pt * W / (0.98 * 397.0)
    F_TITLE, F_LAB, F_AXIS = f(8.0), f(6.0), f(7.4)
    hi = max(max(r.max() for r in rows_cat), max(r.max() for r in rows_gau))

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>']

    for k, (rows, col, title) in enumerate(
            [(rows_cat, CAT, 'Director: a categorical per block'),
             (rows_gau, GAU, 'Ours: a discretized Gaussian per block')]):
        x0 = PL + k * (PW + GAP)
        o.append(f'<text x="{x0 + PW / 2:.1f}" y="{PT - 30:.1f}" '
                 f'font-size="{F_TITLE:.1f}" fill="{INK}" '
                 f'text-anchor="middle">{esc(title)}</text>')

        # fills first, so the rules sit on top of them
        for r, p in enumerate(rows):
            for j in range(C):
                cx, cy = x0 + j * CW, PT + r * RH
                bh = (p[j] / hi) * RH
                if bh > 0.4:
                    o.append(f'<rect x="{cx:.1f}" y="{cy + RH - bh:.1f}" '
                             f'width="{CW}" height="{bh:.1f}" fill="{col}" '
                             f'fill-opacity="0.85"/>')
            o.append(f'<text x="{x0 - 14}" y="{PT + r * RH + RH / 2 + F_LAB * 0.36:.1f}" '
                     f'font-size="{F_LAB:.1f}" fill="{INK_2}" '
                     f'text-anchor="end">block {r + 1}</text>')

        for r in range(L + 1):
            y = PT + r * RH
            o.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + PW}" y2="{y}" '
                     f'stroke="{GRIDC}" stroke-width="1.6"/>')
        for j in range(C + 1):
            x = x0 + j * CW
            o.append(f'<line x1="{x}" y1="{PT}" x2="{x}" y2="{PT + PH}" '
                     f'stroke="{GRIDC}" stroke-width="1.6"/>')
        o.append(f'<rect x="{x0}" y="{PT}" width="{PW}" height="{PH}" '
                 f'fill="none" stroke="{FRAME}" stroke-width="2.4"/>')

        for j in range(C):
            o.append(f'<text x="{x0 + (j + 0.5) * CW:.1f}" '
                     f'y="{PT + PH + F_LAB * 2.0:.1f}" font-size="{F_LAB:.1f}" '
                     f'fill="{INK_2}" text-anchor="middle">{j}</text>')
        o.append(f'<text x="{x0 + PW / 2:.1f}" y="{PT + PH + F_AXIS * 3.0:.1f}" '
                 f'font-size="{F_AXIS:.1f}" fill="{INK}" '
                 f'text-anchor="middle">class index</text>')
    o.append('</svg>')
    return '\n'.join(o), W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'block_distributions'))
    ap.add_argument('--seed', type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    # a categorical's entropy FALLS as the logit scale grows
    scale = solve(lambda s: float(np.mean(
        [norm_ent(categorical(np.random.default_rng(i), s)) for i in range(400)])),
        0.01, 12.0)
    # (entropy falls with the logit scale, rises with sigma; solve() handles both)
    mus = rng.uniform(0, C - 1, L)
    sigma = solve(lambda g: float(np.mean([norm_ent(gaussian(m, g)) for m in mus])),
                  0.05, 8.0)

    rows_cat = [categorical(rng, scale) for _ in range(L)]
    rows_gau = [gaussian(m, sigma) for m in mus]
    print('matched at normalized entropy %.2f' % TARGET_H)
    print('  categorical logit scale %.3f -> mean entropy %.3f'
          % (scale, np.mean([norm_ent(p) for p in rows_cat])))
    print('  gaussian sigma          %.3f -> mean entropy %.3f'
          % (sigma, np.mean([norm_ent(p) for p in rows_gau])))
    print('  gaussian locations:', ' '.join('%.2f' % m for m in mus))

    svg, W, H = build(rows_cat, rows_gau)
    pathlib.Path(a.out + '.svg').write_text(svg)
    subprocess.run(['rsvg-convert', '-f', 'png', '-d', '600', '-p', '600',
                    '-o', a.out + '.png', a.out + '.svg'], check=True)
    print('layout %dx%d svg units' % (W, H))
    print('wrote', a.out + '.png')


if __name__ == '__main__':
    main()
