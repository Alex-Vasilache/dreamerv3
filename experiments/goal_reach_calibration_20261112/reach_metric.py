"""cos_max vs plain cosine vs MSE as the 'reached' signal: how well each separates near from far (AUC, 1 = perfect)."""
import sys, numpy as np
sys.path.insert(0, '.')
from analyze import G, goal_pos, dist
def auc(pos, neg):
  x = np.concatenate([pos, neg]); r = x.argsort().argsort() + 1
  return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def scores(g, f):
  g = g.astype(np.float32); f = f.astype(np.float32); gn = np.linalg.norm(g, axis=-1); fn = np.linalg.norm(f, axis=-1)
  dot = (g * f).sum(-1)
  return {'cos_max': dot / np.maximum(gn, fn) ** 2, 'cosine': dot / (gn * fn + 1e-12), '-MSE': -((g - f) ** 2).mean(-1)}
for f in sys.argv[1:]:
  z = np.load(f, allow_pickle=True); K = int(z['K']); n = len(z['t']); name = f.split('/')[-1][:-4]
  S = scores(z['goal'], z['deter']); perm = np.random.default_rng(0).permutation(n); C = scores(z['goal'], z['deter'][perm])
  self_ = scores(z['self'], z['deter'])
  line = f'{name:22s}'
  if 'pos' in z.files:
    pos = [tuple(p) for p in z['pos']]; d = np.full(n, -1.0)
    for s in range(0, n - K + 1, K):
      g = goal_pos(z['goalimg'][s])[0]
      for i in range(s, s + K): d[i] = dist(pos[i], g)
    on = z['hand_onpad'].astype(bool); pd = z['hand_paddist']; H = scores(z['hand_goal'], z['hand_deter'])
    for k in S:
      line += f' | {k}: cell vs 2+ {auc(S[k][d == 0], S[k][d >= 2]):.2f}, cell vs 1 {auc(S[k][d == 0], S[k][d == 1]):.2f}, hand pad vs 1 {auc(H[k][on], H[k][pd == 1]):.2f}, pad vs 2+ {auc(H[k][on], H[k][pd >= 2]):.2f}'
  for k in S:
    # threshold = level only 5% of the state's own reconstructions fall below; how often is a fresh goal already there?
    thr = np.percentile(self_[k], 5); st = S[k][z['t'] % K == 0]
    line += f' | {k}: goal vs chance AUC {auc(S[k], C[k]):.2f}, already reached at hold start {np.mean(st >= thr):.0%}, any step {np.mean(S[k] >= thr):.0%}'
  print(line, flush=True)
