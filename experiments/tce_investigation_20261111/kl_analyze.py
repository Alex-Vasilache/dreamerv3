import glob, numpy as np
from analyze import G, PADS, prefix
for f in sorted(glob.glob('kl/*.npz')):
  z = np.load(f, allow_pickle=True); kl = z['kl']; seq = list(z['seq']); pos = [tuple(p) for p in z['pos']]; n = len(kl)
  dt = z['deter'].astype(np.float32); chg = np.r_[0, np.linalg.norm(np.diff(dt, axis=0), axis=1)]
  first = np.zeros(n, bool); first[0] = first[2000 % n] = True
  # event at step t: the obs at t+1 shows the result of the move made at t, so the WM sees it at index t+1
  enter = np.zeros(n, bool); adv = np.zeros(n, bool); brk = np.zeros(n, bool); rew = np.zeros(n, bool)
  for t in range(n - 1):
    if seq[t + 1] != seq[t] and seq[t + 1]:
      enter[t + 1] = True
      a, b = prefix(seq[t]), prefix(seq[t + 1]); adv[t + 1] = b > a; brk[t + 1] = b < a
    if z['rew'][t] > 0: rew[t + 1] = True
  onpad = np.array([G.lay[p] in PADS for p in pos]); move = np.r_[False, [pos[t] != pos[t - 1] for t in range(1, n)]]
  plain = ~enter & ~rew & ~first
  plain_move = plain & move & ~onpad
  def s(m, x): return f'{x[m].mean():.2f} (n={m.sum()})' if m.sum() else '-'
  zk = (kl - kl[plain].mean()) / kl[plain].std(); zc = (chg - chg[plain].mean()) / chg[plain].std()
  print(f'\n== {f[3:-4]}  score {z["rew"].sum():.0f}')
  print(f'WM KL (nats): ordinary floor move {s(plain_move, kl)} | pad entered {s(enter, kl)} | advances seq {s(adv, kl)} | breaks seq {s(brk, kl)} | reward step {s(rew, kl)}')
  print(f'  z-score vs ordinary steps: pad entered {zk[enter].mean():.1f}, reward {zk[rew].mean() if rew.any() else float("nan"):.1f}; top-1% KL steps that are pad entries: {enter[kl >= np.quantile(kl, 0.99)].mean():.0%} (pad entries are {enter.mean():.1%} of steps)')
  print(f'deter change |dh|: ordinary move {s(plain_move, chg)} | pad entered {s(enter, chg)} | reward {s(rew, chg)}; z pad {zc[enter].mean():.1f}; top-1% change steps that are pad entries {enter[chg >= np.quantile(chg, 0.99)].mean():.0%}')
