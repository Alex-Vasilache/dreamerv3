#!/usr/bin/env python3
"""Compare start-cell hashes on the features saved by wm_hash_study.py.

For each candidate and checkpoint: effective cells exp(H), temporal coherence
(share of consecutive steps in the same cell), pin pad prefix purity; and the
drift that matters for counting: the share of states whose cell under the
checkpoint 0.5M later is the same ("kept 0.5M"), and 1.0M later.

  python wm_hash_analyze.py <study.npz> [...]
"""
import collections
import sys

import numpy as np

CLASSES = 8


def simhash(x, proj):
  return ((x @ proj) > 0) @ (2 ** np.arange(proj.shape[1]))


def candidates(f, ck, bits, rng_seed=0):
  deter = f[f'{ck}/deter'].astype(np.float32)
  stoch = f[f'{ck}/stoch'].astype(np.int64)
  code = f[f'{ck}/code'].astype(np.int64)
  B, T, D = deter.shape
  d = deter.reshape(B * T, D)
  rng = np.random.default_rng(rng_seed)
  pd = rng.normal(size=(D, bits))
  s1h = np.eye(CLASSES)[stoch.reshape(B * T, -1)].reshape(B * T, -1)
  ps = rng.normal(size=(s1h.shape[1], bits))
  dn = (d - d.mean(1, keepdims=True)) / (d.std(1, keepdims=True) + 1e-6)
  out = {
      'code4 (today)': (np.minimum(code.reshape(B * T, -1) * 4 // CLASSES, 3)
                        * 4 ** np.arange(code.shape[-1])).sum(-1),
      'deter, data-centred': simhash(d - d.mean(0, keepdims=True), pd),
      'deter, per-sample norm': simhash(dn, pd),
  }
  if 'img16' in f.files:
    im = f['img16'].astype(np.float32).reshape(B * T, -1)
    pi = rng.normal(size=(im.shape[1], bits))
    out['image 16x16 (no drift)'] = simhash(im - im.mean(0, keepdims=True), pi)
  return {k: v.reshape(B, T) for k, v in out.items()}


def stats(k, pre):
  flat = k.reshape(-1)
  cnt = collections.Counter(flat.tolist())
  p = np.array(list(cnt.values()), np.float64) / len(flat)
  r = {'eff': float(np.exp(-(p * np.log(p)).sum())), 'coh': float((k[:, 1:] == k[:, :-1]).mean())}
  if pre is not None:
    by = collections.defaultdict(collections.Counter)
    for kk, pp in zip(flat.tolist(), pre):
      by[kk][pp] += 1
    r['purity'] = sum(c.most_common(1)[0][1] for c in by.values()) / len(flat)
  return r


def main():
  for path in sys.argv[1:]:
    f = np.load(path)
    cks = sorted({k.split('/')[0] for k in f.files if '/' in k} - {'repeat'})
    pre = f['prefix2'].tolist() if 'prefix2' in f.files else None
    for bits in (12, 16):
      print(f'\n== {path.split("/")[-1]}  {bits} bits  (code4 is always 16)')
      per = {ck: candidates(f, ck, bits) for ck in cks}
      for name in per[cks[0]]:
        line = []
        for i, ck in enumerate(cks):
          s = stats(per[ck][name], pre)
          k05 = (per[ck][name] == per[cks[i + 1]][name]).mean() if i + 1 < len(cks) else np.nan
          k10 = (per[ck][name] == per[cks[i + 2]][name]).mean() if i + 2 < len(cks) else np.nan
          line.append(f"{int(ck) / 1e6:.1f}M eff {s['eff']:6.0f} coh {s['coh']:.2f}"
                      + (f" pur {s['purity']:.2f}" if 'purity' in s else '')
                      + f' kept.5 {k05:.2f} kept1 {k10:.2f}')
        if 'repeat/deter' in f.files:
          rep = candidates(f, 'repeat', bits)[name]
          line.append(f'noise floor: same checkpoint replayed again keeps {(rep == per[cks[-1]][name]).mean():.2f}')
        print(f'{name}:')
        for l in line:
          print('   ', l)


if __name__ == '__main__':
  main()
