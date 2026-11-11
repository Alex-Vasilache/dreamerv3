"""Per HOLD (not per step): did 'reached' fire at any time in the hold, split by how close the player
actually got; and did it fire before or after the true arrival. tau = frac * ceiling."""
import sys, numpy as np
sys.path.insert(0, '.')
from analyze import G, goal_pos, dist
def cm(g, f):
  g = g.astype(np.float32); f = f.astype(np.float32)
  return (g * f).sum(-1) / np.maximum(np.linalg.norm(g, axis=-1), np.linalg.norm(f, axis=-1)) ** 2
pc = lambda a: f'{np.mean(a):.0%}' if len(a) else 'n/a'
for f in sys.argv[1:]:
  z = np.load(f, allow_pickle=True); K = int(z['K']); n = len(z['t']); name = f.split('/')[-1][:-4]
  c = cm(z['goal'], z['deter']); ceil = cm(z['self'], z['deter']).mean()
  pos = [tuple(p) for p in z['pos']]; H = []
  for s in range(0, n - K + 1, K):
    g = goal_pos(z['goalimg'][s])[0]; H.append((np.array([dist(pos[i], g) for i in range(s, s + K)]), c[s:s + K]))
  hc = cm(z['hand_goal'], z['hand_deter']); pd = z['hand_paddist']
  key = np.array([f'{p}_{k}' for p, k in zip(z['hand_pad'], z['hand_trial'])]); T = [(pd[key == k], hc[key == k]) for k in dict.fromkeys(key)]
  nh = [sum(d.min() == 0 for d, _ in X) for X in (H, T)]
  print(f'{name}: ceiling {ceil:.2f}; play holds {len(H)} (player got onto the goal cell in {nh[0]}); hand trials {len(T)} (got onto the pad in {nh[1]})')
  for fr in (1.0, 0.9, 0.8, 0.7):
    t = fr * ceil
    for lab, X in (('play 8-step holds', H), ('hand 48-step trials', T)):
      fire = lambda sel: [bool((cc >= t).any()) for d, cc in X if sel(d)]
      early, same, late, never, gap, dfire = 0, 0, 0, 0, [], []
      for d, cc in X:
        w = np.where(cc >= t)[0]
        if len(w): dfire.append(d[w[0]])
        if d.min() > 0: continue
        a = np.where(d == 0)[0][0]
        if not len(w): never += 1
        elif w[0] < a: early += 1; gap.append(a - w[0])
        elif w[0] == a: same += 1
        else: late += 1; gap.append(a - w[0])
      tot = max(early + same + late + never, 1); dfire = np.array(dfire)
      print(f'  {fr:.0%} {lab:19s}: fired in hold when player got ON goal {pc(fire(lambda d: d.min() == 0))}, got within 1 cell only {pc(fire(lambda d: d.min() == 1))}, never within 2 {pc(fire(lambda d: d.min() >= 2))}'
            f' | of arrivals: fired before arriving {early / tot:.0%}, same step {same / tot:.0%}, after {late / tot:.0%}, never {never / tot:.0%}'
            f' | distance at first fire: on goal {pc(dfire == 0)}, 1 cell {pc(dfire == 1)}, 2+ {pc(dfire >= 2)}', flush=True)
