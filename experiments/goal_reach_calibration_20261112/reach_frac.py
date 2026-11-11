"""Reached = cos_max >= frac * ceiling (ceiling = mean cos_max of a state with its own reconstruction).
Per frac: how often it fires when truly there (hit rate) and when not there (false alarm)."""
import sys, numpy as np
sys.path.insert(0, '.')
from analyze import G, goal_pos, dist
def cm(g, f):
  g = g.astype(np.float32); f = f.astype(np.float32)
  return (g * f).sum(-1) / np.maximum(np.linalg.norm(g, axis=-1), np.linalg.norm(f, axis=-1)) ** 2
def auc(pos, neg):
  x = np.concatenate([pos, neg]); r = x.argsort().argsort() + 1
  return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
for f in sys.argv[1:]:
  z = np.load(f, allow_pickle=True); K = int(z['K']); n = len(z['t']); name = f.split('/')[-1][:-4]
  c = cm(z['goal'], z['deter']); ceil = cm(z['self'], z['deter']).mean(); st = z['t'] % K == 0
  pin = 'pos' in z.files
  if pin:
    pos = [tuple(p) for p in z['pos']]; d = np.full(n, -1.0)
    for s in range(0, n - K + 1, K):
      g = goal_pos(z['goalimg'][s])[0]
      for i in range(s, s + K): d[i] = dist(pos[i], g)
    hc = cm(z['hand_goal'], z['hand_deter']); on = z['hand_onpad'].astype(bool); pd = z['hand_paddist']
    print(f'{name}: ceiling {ceil:.2f} | AUC play cell vs 2+: {auc(c[d == 0], c[d >= 2]):.2f}, hand pad vs 2+: {auc(hc[on], hc[pd >= 2]):.2f}')
  else:
    print(f'{name}: ceiling {ceil:.2f}')
  for fr in (1.0, 0.9, 0.8, 0.7):
    t = fr * ceil; line = f'  {fr:.0%} (tau {t:.2f}): '
    if pin:
      line += f'play: on cell {np.mean(c[d == 0] >= t):.0%}, 1 cell {np.mean(c[d == 1] >= t):.0%}, 2+ cells {np.mean(c[d >= 2] >= t):.1%} | hand: on pad {np.mean(hc[on] >= t):.0%}, 1 cell {np.mean(hc[pd == 1] >= t):.0%}, 2+ cells {np.mean(hc[pd >= 2] >= t):.1%} | '
    hold = c[:n // K * K].reshape(-1, K)
    line += f'fires at hold start {np.mean(c[st] >= t):.0%}, any time in 8 steps {np.mean(hold.max(1) >= t):.0%}'
    print(line, flush=True)
