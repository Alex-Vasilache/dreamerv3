import glob, numpy as np
from analyze import G, PADS, goal_pos, prefix, dist
for f in sorted(glob.glob('probes/*.npz')):
  z = np.load(f, allow_pickle=True); n = len(z['pos']); pos = [tuple(p) for p in z['pos']]; seq = list(z['seq'])
  sw = np.where(z['switch'] > 0.5)[0]; goal = {s: goal_pos(z['goalimg'][s])[0] for s in sw}
  exact = near = padhit = padn = 0
  for i, s in enumerate(sw):
    e = sw[i + 1] if i + 1 < len(sw) else n; g = goal[s]; gp = G.lay[g]
    exact += any(pos[t] == g for t in range(s, e)); near += any(dist(pos[t], g) <= 1 for t in range(s, e))
    if gp in PADS:
      padn += 1; padhit += any(G.lay[pos[t]] == gp for t in range(s, e))
  good = bad = neutral = 0
  for t in range(1, n):
    if seq[t] != seq[t - 1] and seq[t]:
      a, b = prefix(seq[t - 1]), prefix(seq[t])
      if b > a: good += 1
      elif b < a: bad += 1
      else: neutral += 1
  tot = good + bad + neutral
  print(f'{f[7:-4]:12s} holds {len(sw)}: exact cell {exact/len(sw):.0%}, within 1 cell {near/len(sw):.0%}, pad goals reached (any cell of that pad) {padhit/max(padn,1):.0%} of {padn} | presses {tot}: advance seq {good/max(tot,1):.0%}, break {bad/max(tot,1):.0%}, neutral {neutral/max(tot,1):.0%}')
