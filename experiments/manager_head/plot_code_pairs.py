#!/usr/bin/env python3
"""Compose render_code_pairs.py output into a labelled panel.

One row per base code: its decoded goal, then partners at 1/4, 1/2, 3/4 and the
full dmax of that code. Under each partner, the realized index distance and the
cosine_max of its goal against the base -- the quantity the worker is rewarded
on, so it is what the distance actually costs the low-level policy.

  python3 plot_code_pairs.py --npz pairs/e592.npz --out pairs/e592_sheet
"""
import argparse
import pathlib

import numpy as np
from PIL import Image, ImageDraw

SCALE = 3
PAD, TOP, LEFT, LAB = 10, 40, 96, 30


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--npz', required=True)
  ap.add_argument('--out', required=True)
  a = ap.parse_args()
  d = np.load(a.npz, allow_pickle=True)
  frames, sim, dists = d['frames'], d['sim'], d['dists']
  fracs, dmax = d['fracs'], d['dmax']
  ncol, nrow = frames.shape[0], frames.shape[1]
  h = w = frames.shape[2] * SCALE

  W = LEFT + ncol * (w + PAD) + PAD
  H = TOP + nrow * (h + LAB + PAD) + PAD
  im = Image.new('RGB', (W, H), 'white')
  dr = ImageDraw.Draw(im)

  heads = ['base code'] + ['%.0f%% of dmax' % (100 * f) for f in fracs]
  for c, t in enumerate(heads):
    dr.text((LEFT + c * (w + PAD) + 4, 14), t, fill='black')

  for r in range(nrow):
    y = TOP + r * (h + LAB + PAD)
    dr.text((6, y + h // 2), 'code %d\ndmax %d' % (r + 1, int(dmax[r])),
            fill='black')
    for c in range(ncol):
      x = LEFT + c * (w + PAD)
      img = Image.fromarray(frames[c, r]).resize((w, h), Image.NEAREST)
      im.paste(img, (x, y))
      dr.rectangle([x, y, x + w - 1, y + h - 1], outline=(60, 60, 60))
      if c > 0:
        dr.text((x + 4, y + h + 6),
                'd=%d  cos=%.2f' % (int(dists[c - 1, r]), sim[c - 1, r]),
                fill=(30, 30, 30))

  out = pathlib.Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  im.save(str(out) + '.png')
  print('wrote %s.png  (%dx%d)  task=%s' % (out, W, H, d['task']))
  for i, f in enumerate(fracs):
    print('  %3.0f%% of dmax: mean d %5.1f  mean cos_max %.3f'
          % (100 * f, dists[i].mean(), sim[i].mean()))


if __name__ == '__main__':
  main()
