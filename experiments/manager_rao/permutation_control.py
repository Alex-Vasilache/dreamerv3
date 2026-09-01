"""Is the measured Rao explained by the index ORDER, or just by the entropy?

Rao with an ordered distance only means something if the class index carries
order. Director's codebook has no topology, so its indices are arbitrary labels
and its Rao should land wherever random labels put it. SOM-line + LiP is
supposed to be different.

Control: recompute Rao after randomly permuting the class labels inside each
block, which destroys the order while leaving the distribution's shape (and
therefore its entropy) exactly intact. Ratio actual / permuted:

  ~1.0  the index order is irrelevant -- the number is pure entropy
  <1.0  mass sits on ADJACENT classes: the manager hedges locally, which is
        what an ordered codebook is for
  >1.0  mass sits on FAR-APART classes
"""
import glob
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

RESULTS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                       str(pathlib.Path(__file__).resolve().parent / 'results'))
NPERM = 100

C = 8
x = np.arange(C, dtype=np.float64) / (C - 1)
D = (x[:, None] - x[None, :]) ** 2
QN = D.max() / 2  # self-Rao maximum


def rao(p, q=None):
  q = p if q is None else q
  return np.einsum('...a,ab,...b->...', p, D, q)


rows = []
for f in sorted(RESULTS.glob('*.npz')):
  d = np.load(f)
  probs = d['probs'].astype(np.float64)          # (N, L, C)
  N, L, _ = probs.shape
  actual = rao(probs).mean() / QN

  rng = np.random.default_rng(0)
  perm_vals = []
  for _ in range(NPERM):
    # One fresh permutation per block, applied to every decision, so the
    # relabelling is a property of the codebook (as it would be for a run with
    # no topology) and not extra per-decision noise.
    q = np.stack([probs[:, l, rng.permutation(C)] for l in range(L)], axis=1)
    perm_vals.append(rao(q).mean() / QN)
  perm = float(np.mean(perm_vals))

  # Same control for the consecutive-decision (cross) Rao.
  actual_x = rao(probs[:-1], probs[1:]).mean() / QN
  rng = np.random.default_rng(1)
  px = []
  for _ in range(NPERM):
    idx = [rng.permutation(C) for _ in range(L)]
    q = np.stack([probs[:, l, idx[l]] for l in range(L)], axis=1)
    px.append(rao(q[:-1], q[1:]).mean() / QN)
  perm_x = float(np.mean(px))

  arm = f.stem.split('_')[0]
  rows.append((arm, f.stem, actual, perm, actual / perm,
               actual_x, perm_x, actual_x / perm_x))

hdr = ('run', 'Rao', 'Rao shuffled', 'ratio', 'cross', 'cross shuf', 'ratio')
print(f'{hdr[0]:<30}{hdr[1]:>10}{hdr[2]:>14}{hdr[3]:>8}'
      f'{hdr[4]:>10}{hdr[5]:>12}{hdr[6]:>8}')
print('-' * 92)
for arm, name, a, p, r, ax, px_, rx in rows:
  print(f'{name:<30}{a:>10.3f}{p:>14.3f}{r:>8.2f}{ax:>10.3f}{px_:>12.3f}{rx:>8.2f}')

print()
by = defaultdict(list)
for arm, name, a, p, r, ax, px_, rx in rows:
  by[arm].append((a, p, r, ax, px_, rx))
print(f'{"arm":<30}{"Rao":>10}{"Rao shuffled":>14}{"ratio":>8}'
      f'{"cross":>10}{"cross shuf":>12}{"ratio":>8}')
print('-' * 92)
for arm, vs in sorted(by.items()):
  m = np.mean(vs, axis=0)
  print(f'{arm + f" (n={len(vs)})":<30}{m[0]:>10.3f}{m[1]:>14.3f}{m[2]:>8.2f}'
        f'{m[3]:>10.3f}{m[4]:>12.3f}{m[5]:>8.2f}')
