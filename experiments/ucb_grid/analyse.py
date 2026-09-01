"""Coverage / spread statistics of the real selection tables, plus a simulation
of the candidate gap that the measured `ucb_novelty_gain` should reproduce."""
import numpy as np, json, sys, collections

NPZ = sys.argv[1]
C_UCB, K = 10.0, 8
d = np.load(NPZ)
runs = sorted({k.split('|')[0] for k in d.files})
steps = sorted({int(k.split('|')[1]) for k in d.files})

def gini(x):
  x = np.sort(x.astype(np.float64)); n = len(x); cx = np.cumsum(x)
  return (n + 1 - 2 * (cx / cx[-1]).sum()) / n

def sim_gain(tab, rng, n=200_000):
  p = tab.astype(np.float64); p = p / p.sum()
  bonus = 1.0 / np.sqrt(tab + 1.0)
  idx = rng.choice(len(p), size=(n, K), p=p)
  b = bonus[idx]                                    # (n, K)
  w = np.exp(C_UCB * (b - b.max(1, keepdims=True))) # stable softmax
  w /= w.sum(1, keepdims=True)
  cum = np.cumsum(w, 1)
  pick = (cum < rng.random((n, 1))).sum(1).clip(0, K - 1)
  chosen = b[np.arange(n), pick]
  return chosen.mean() / b.mean(1).mean(), b.mean(), b.max(1).mean(), b.min(1).mean()

rng = np.random.default_rng(0)
rows = []
for r in runs:
  for s in steps:
    key = f'{r}|{s}|select_tab'
    if key not in d.files: continue
    t = d[key].astype(np.float64)
    bonus = 1.0 / np.sqrt(t + 1.0)
    p = t / t.sum()
    g, bmean, bmax, bmin = sim_gain(t, rng)
    rows.append(dict(
        run=r, step=s, mass=t.sum(),
        nz=int((t > 0).sum()), gt1=int((t > 1).sum()), gt10=int((t > 10).sum()),
        eff=float(t.sum() ** 2 / (t ** 2).sum()),
        gini=float(gini(t)),
        top1pct=float(np.sort(t)[::-1][:655].sum() / t.sum()),
        bonus_cellmean=float(bonus.mean()),          # unweighted over cells
        bonus_propmean=float((bonus * p).sum()),     # weighted by proposal prob
        sim_gain=float(g), sim_bmean=float(bmean),
        sim_bmax=float(bmax), sim_bmin=float(bmin)))
json.dump(rows, open(sys.argv[2], 'w'), indent=1)

hdr = f"{'run':18s}{'step':>9s}{'nz':>7s}{'>1':>7s}{'>10':>7s}{'eff':>8s}{'gini':>6s}{'top1%':>7s}{'b_cell':>7s}{'b_prop':>7s}{'simgain':>8s}"
for r in runs:
  print('\n'+hdr)
  for x in [z for z in rows if z['run'] == r]:
    print(f"{x['run']:18s}{x['step']:>9d}{x['nz']:>7d}{x['gt1']:>7d}{x['gt10']:>7d}"
          f"{x['eff']:>8.0f}{x['gini']:>6.2f}{x['top1pct']:>7.2f}"
          f"{x['bonus_cellmean']:>7.3f}{x['bonus_propmean']:>7.3f}{x['sim_gain']:>8.3f}")
