"""Why does the candidate gap grow?  Hold the manager's POLICY fixed at a real,
per-state, measured distribution and advance only the count table.

For each milestone table we redraw K=8 candidates per real state from that
state's own block-wise categorical, look their bonus up in the real table, and
apply the shipped selection rule P(k) proportional to exp(c*bonus_k).
If the gain still rises with the policy frozen, the table alone explains it."""
import numpy as np, json, sys

L, C, BINS, K, CU = 8, 8, 4, 8, 10.0
NPZ, RAO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PAIR = {  # run -> rao policy file (same arm, same task, same seed)
    'e713_cartpole_s1': 'somlip_cartpole_s1',
    'e714_cheetah_s0':  'somlip_cheetah_s0',
    'e715_cheetah_s1':  'somlip_cheetah_s1',
    'e716_hopper_s0':   'somlip_hopperhop_s0',
    'e717_hopper_s1':   'somlip_hopperhop_s1',
}
PLACE = BINS ** np.arange(L)

def cell(ids):
  q = np.minimum((ids.astype(np.int64) * BINS) // C, BINS - 1)
  return (q * PLACE).sum(-1)

def draw(probs, k, rng):
  """k candidate codes per state from each state's own block-wise categorical."""
  n = probs.shape[0]
  u = rng.random((n, k, L, 1))
  cdf = np.cumsum(probs, -1)[:, None, :, :]
  return (u > cdf).sum(-1).clip(0, C - 1).astype(np.int8)   # (n, k, L)

d = np.load(NPZ)
rows = []
for run, rao in PAIR.items():
  P = np.load(f'{RAO}/{rao}.npz')['probs'].astype(np.float64)
  P = P / P.sum(-1, keepdims=True)
  logP = np.log(np.maximum(P, 1e-12))
  rng = np.random.default_rng(0)
  cand = draw(P, K, rng)                                  # (n, K, L) fixed policy
  cid = cell(cand)                                        # (n, K)
  # log pi of each candidate under its own state's policy
  ar = np.arange(P.shape[0])[:, None, None]
  bl = np.arange(L)[None, None, :]
  lp = logP[ar, bl, cand].sum(-1)                         # (n, K)
  for s in sorted({int(k_.split('|')[1]) for k_ in d.files}):
    key = f'{run}|{s}|select_tab'
    if key not in d.files: continue
    tab = d[key].astype(np.float64)
    b = 1.0 / np.sqrt(tab[cid] + 1.0)                     # (n, K)
    w = np.exp(CU * (b - b.max(1, keepdims=True))); w /= w.sum(1, keepdims=True)
    pick = (np.cumsum(w, 1) < rng.random((len(b), 1))).sum(1).clip(0, K - 1)
    i = np.arange(len(b))
    rows.append(dict(
        run=run, step=s,
        gain=float(b[i, pick].mean() / b.mean()),
        bonus=float(b.mean()),
        spread_sd=float(b.std(1).mean()),
        spread_rng=float((b.max(1) - b.min(1)).mean()),
        rel_spread=float((b.std(1) / b.mean(1)).mean()),
        logp_shift=float(lp[i, pick].mean() - lp.mean()),
        flip=float((pick != np.argmax(w * 0 + lp, 1)).mean())))
json.dump(rows, open(OUT, 'w'), indent=1)
print(f"{'run':18s}{'step':>9s}{'gain':>7s}{'bonus':>7s}{'sd':>7s}{'range':>7s}{'sd/mean':>8s}{'logpsh':>8s}")
for x in rows:
  print(f"{x['run']:18s}{x['step']:>9d}{x['gain']:>7.3f}{x['bonus']:>7.3f}"
        f"{x['spread_sd']:>7.3f}{x['spread_rng']:>7.3f}{x['rel_spread']:>8.3f}{x['logp_shift']:>8.3f}")
