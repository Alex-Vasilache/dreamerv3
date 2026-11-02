"""Offline check: one 16-bit deter SimHash vs K independent 12-bit SimHashes
(per-state normalised deter, as in tce_hash). For the K-hash version a state's
count is the average over K tables, so what matters is the share of the K
cells a state keeps: under the same checkpoint replayed again (noise) and 0.5M
later (drift). Resolution: pin pad prefix purity of the joint key.

  python wm_multihash_check.py <study_r.npz> ...
"""
import collections
import sys

import numpy as np


def keys(deter, bits, K, seed=0):
  d = deter.astype(np.float32).reshape(-1, deter.shape[-1])
  d = (d - d.mean(1, keepdims=True)) / (d.std(1, keepdims=True) + 1e-6)
  rng = np.random.default_rng(seed)
  out = []
  for _ in range(K):
    p = rng.normal(size=(d.shape[1], bits))
    out.append(((d @ p) > 0) @ (2 ** np.arange(bits)))
  return np.stack(out, -1)            # (N, K)


def purity(key, pre):
  by = collections.defaultdict(collections.Counter)
  for k, p in zip(map(tuple, key), pre):
    by[k][p] += 1
  return sum(c.most_common(1)[0][1] for c in by.values()) / len(pre)


for path in sys.argv[1:]:
  f = np.load(path)
  cks = sorted({k.split('/')[0] for k in f.files if '/' in k} - {'repeat'})
  pre = f['prefix2'].tolist() if 'prefix2' in f.files else None
  print('==', path.split('/')[-1])
  for bits, K in [(16, 1), (12, 1), (12, 4), (10, 4), (12, 8)]:
    last = keys(f[f'{cks[-1]}/deter'], bits, K)
    rep = keys(f['repeat/deter'], bits, K)
    noise = (last == rep).mean()
    drift = np.mean([(keys(f[f'{a}/deter'], bits, K) == keys(f[f'{b}/deter'], bits, K)).mean()
                     for a, b in zip(cks[-4:-1], cks[-3:])])
    pur = f' purity(joint) {purity(last, pre):.3f} purity(one) {purity(last[:, :1], pre):.3f}' if pre else ''
    print(f'  {K}x{bits} bits: same-ckpt keep {noise:.3f}  0.5M keep (last 1.5M) {drift:.3f}{pur}')
