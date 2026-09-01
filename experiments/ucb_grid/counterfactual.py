"""What WOULD anneal?  Sweep the grid resolution b and the kernel width h.

The smeared table is a separable convolution of the exact-code histogram, so we
build H over C^L = 8^8 cells once (67 MB) and contract it with the per-block
kernel along each of the L axes.  Exact, not sampled.

Decisions come from the real measured per-state manager policies (manager_rao),
sampling a state uniformly and drawing one code from its own block-wise
categorical -- the same generator the frozen-policy check uses."""
import numpy as np, json, sys

L, C, K, CU = 8, 8, 8, 10.0
RAO, OUT = sys.argv[1], sys.argv[2]
N_DEC = 500_000                      # decisions in a 4M-step run at K=8
CHECKS = [62_500, 125_000, 250_000, 375_000, 500_000]
GRIDS = [(3, 0.0), (3, 1.0), (4, 0.0), (4, 1.0), (5, 1.0), (6, 1.0)]
TASKS = {'cheetah': 'somlip_cheetah_s0', 'cartpole': 'somlip_cartpole_s1',
         'hopper': 'somlip_hopperhop_s0'}

def kernel(bins, h):
  """(C, bins): how one class deposits across the bins.  h=0 is the hard grid."""
  if h <= 0:
    kb = np.zeros((C, bins))
    kb[np.arange(C), np.minimum((np.arange(C) * bins) // C, bins - 1)] = 1.0
    return kb
  centers = (np.arange(bins) + 0.5) * (C / bins) - 0.5
  kb = np.exp(-(np.arange(C)[:, None] - centers[None, :]) ** 2 / (2 * h * h))
  return kb / kb.sum(1, keepdims=True)

def smear(H, kb, bins):
  """Contract the exact-code histogram with kb along every block axis."""
  a = H
  for ax in range(L):
    a = np.moveaxis(np.tensordot(np.moveaxis(a, ax, 0), kb, axes=([0], [0])), -1, ax)
  return a.reshape(-1)

def draw(probs, n, rng, states):
  u = rng.random((n, L, 1))
  cdf = np.cumsum(probs[states], -1)
  return (u > cdf).sum(-1).clip(0, C - 1)

rows = []
for task, f in TASKS.items():
  P = np.load(f'{RAO}/{f}.npz')['probs'].astype(np.float64)
  P /= P.sum(-1, keepdims=True)
  logP = np.log(np.maximum(P, 1e-12))
  rng = np.random.default_rng(0)
  st = rng.integers(0, len(P), N_DEC)
  codes = draw(P, N_DEC, rng, st)                          # (N, L) class ids
  flat = (codes * (C ** np.arange(L))).sum(-1)             # exact-code index
  # fixed candidate sets, drawn once from the real policy (same states)
  cst = rng.integers(0, len(P), 4000)
  cu = rng.random((4000, K, L, 1))
  cand = (cu > np.cumsum(P[cst], -1)[:, None]).sum(-1).clip(0, C - 1)
  ar = np.arange(4000)[:, None, None]; bl = np.arange(L)[None, None, :]
  lp = logP[cst[:, None, None], bl, cand].sum(-1)
  for bins, h in GRIDS:
    kb = kernel(bins, h)
    place = bins ** np.arange(L)
    cq = np.minimum((cand * bins) // C, bins - 1)
    cid = (cq * place).sum(-1)                             # (4000, K)
    H = np.zeros(C ** L, np.float32)
    prev = 0
    for n in CHECKS:
      np.add.at(H, flat[prev:n], 1.0); prev = n
      tab = smear(H.reshape((C,) * L), kb, bins)
      b = 1.0 / np.sqrt(tab[cid] + 1.0)
      w = np.exp(CU * (b - b.max(1, keepdims=True))); w /= w.sum(1, keepdims=True)
      pick = (np.cumsum(w, 1) < rng.random((len(b), 1))).sum(1).clip(0, K - 1)
      i = np.arange(len(b))
      rows.append(dict(
          task=task, bins=bins, h=h, dec=n, cells=int(bins ** L),
          nz=float((tab > 0).mean()), gt1=float((tab > 1).mean()),
          eff=float(tab.sum() ** 2 / (tab ** 2).sum()),
          mean_count=float(tab.sum() / len(tab)),
          gain=float(b[i, pick].mean() / b.mean()),
          bonus=float(b.mean()), sd=float(b.std(1).mean()),
          logp_shift=float(lp[i, pick].mean() - lp.mean())))
      print(f"{task:9s} b={bins} h={h:.0f} dec={n:7d} cells={bins**L:6d} "
            f"nz={rows[-1]['nz']:5.3f} n/cell={rows[-1]['mean_count']:7.2f} "
            f"bonus={rows[-1]['bonus']:.3f} sd={rows[-1]['sd']:.3f} "
            f"gain={rows[-1]['gain']:.3f}", flush=True)
json.dump(rows, open(OUT, 'w'), indent=1)
