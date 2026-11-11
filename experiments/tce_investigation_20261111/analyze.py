import sys, glob, collections, numpy as np
sys.path.insert(0, '.')
from pinsim import Grid
G = Grid('six'); PADS = G.pads; TGT = ''.join(PADS)
COL = {'1': (255, 0, 0), '2': (0, 255, 0), '3': (0, 0, 255), '4': (255, 255, 0), '5': (255, 0, 255), '6': (0, 255, 255)}
proj = np.random.default_rng(0).normal(size=(1024, 16)).astype(np.float32)
def H(labels):
  c = np.array(list(collections.Counter(labels).values()), float); p = c / c.sum(); return -(p * np.log2(p)).sum()
def condH(y, x):
  tot = 0; by = collections.defaultdict(list)
  for a, b in zip(x, y): by[a].append(b)
  return sum(len(v) / len(y) * H(v) for v in by.values())
def goal_pos(img):
  g = img.astype(float).reshape(16, 4, 16, 4, 3).mean((1, 3, 4))   # [y, x]
  best = None
  for (x, y) in G.free:
    v = g[y, x]
    if best is None or v < best[0]: best = (v, (x, y))
  return best[1], best[0]
def bar(img):
  s = ''
  for i in range(6):
    c = img[14 * 4:14 * 4 + 4, (2 * i + 1) * 4:(2 * i + 2) * 4].reshape(-1, 3).mean(0)
    d = {p: np.abs(c - np.array(v)).sum() for p, v in COL.items()}
    p = min(d, key=d.get)
    if d[p] < 150: s += p
    else: break
  return s
def prefix(w):  # longest k so that the window ends with the first k target pads
  for k in range(min(len(w), 6), 0, -1):
    if w[-k:] == TGT[:k]: return k
  return 0
def dist(a, b): return G.D[G.idx[a], G.idx[b]]

if __name__ == '__main__':
  for f in sorted(glob.glob('probes/*.npz')):
    z = np.load(f, allow_pickle=True); n = len(z['pos'])
    pos = [tuple(p) for p in z['pos']]; seq = list(z['seq']); sw = z['switch'] > 0.5
    gp = [goal_pos(im) for im in z['goalimg']]
    goal = [g[0] for g in gp]; dark = np.array([g[1] for g in gp])
    sw_idx = np.where(sw)[0]
    # hold-level stats
    reached, enddist, onpad, idle, padgoals = [], [], [], 0, []
    for i, s in enumerate(sw_idx):
      e = sw_idx[i + 1] if i + 1 < len(sw_idx) else n
      g = goal[s]; tl = G.lay[g]; onpad.append(tl in PADS)
      if tl in PADS: padgoals.append(tl)
      hit = [t for t in range(s, e) if pos[t] == g]
      reached.append(bool(hit)); enddist.append(dist(pos[e - 1], g))
    # presses: new pad appended to the window
    intended = acc = 0
    for t in range(1, n):
      if seq[t] != seq[t - 1] and len(seq[t]) and (len(seq[t]) > len(seq[t - 1]) or seq[t][-1] != seq[t - 1][-1:]):
        p = seq[t][-1]; gpad = G.lay[goal[t - 1]]
        if gpad == p: intended += 1
        else: acc += 1
    presses = intended + acc
    # hash
    dt = z['deter'].astype(np.float32); x = (dt - dt.mean(1, keepdims=True)) / (dt.std(1, keepdims=True) + 1e-6)
    cell = ((x @ proj) > 0).astype(int) @ (2 ** np.arange(16))
    win = [s[-2:] for s in seq]; last = [s[-1:] for s in seq]
    pre = [prefix(s) for s in seq]
    gbar = [bar(z['goalimg'][s]) for s in sw_idx]
    print(f'\n== {f.split("/")[-1][:-4]}: score {z["rew"].sum():.0f} over {n // 2000} ep')
    print(f'goals: {np.mean(onpad):.0%} on a pad; pad goal counts ' + str(dict(sorted(collections.Counter(padgoals).items()))) +
          f'; goal position entropy {H(goal[s] for s in sw_idx):.2f} bits (uniform over {len(G.free)} cells = {np.log2(len(G.free)):.2f}); '
          f'goal image clarity (darkest cell mean) {np.median(dark[sw_idx]):.0f}/255')
    print(f'worker: reaches goal within the hold {np.mean(reached):.0%}; median end distance {np.median(enddist):.0f}; '
          f'steps standing on goal {np.mean([pos[t] == goal[t] for t in range(n)]):.0%}')
    print(f'presses: {presses} ({presses / n * 1000:.0f} per 1000 steps); accidental (not the goal pad) {acc / max(presses, 1):.0%}')
    print(f'progress: max prefix reached {max(pre)}; time at prefix>=k: ' + ' '.join(f'{k}:{np.mean(np.array(pre) >= k):.1%}' for k in range(1, 7)) +
          f'; distinct pad windows seen {len(set(seq))}')
    print(f'goal bar (sequence drawn in the goal image): length dist {dict(sorted(collections.Counter(len(b) for b in gbar).items()))}; '
          f'goal bar == current bar {np.mean([b == seq[s] for b, s in zip(gbar, sw_idx)]):.0%}; goal bar extends current by one pad {np.mean([len(b) == len(seq[s]) + 1 and b.startswith(seq[s]) or (len(seq[s]) == 6 and b[:-1] == seq[s][1:]) for b, s in zip(gbar, sw_idx)]):.0%}')
    print(f'hash: {len(set(cell))} cells visited; H(pos)={H(pos):.2f} H(pos|cell)={condH(pos, cell):.2f}; '
          f'H(last pad)={H(last):.2f} H(last|cell)={condH(last, cell):.2f} H(last|pos)={condH(last, pos):.2f}; '
          f'H(prefix)={H(pre):.2f} H(prefix|cell)={condH(pre, cell):.2f}; H(window2|cell)={condH(win, cell):.2f} vs H(window2)={H(win):.2f}')
    # hash as a lookup table: label each cell by its majority in episode 0, predict episode 1
    half = n // 2
    def cellacc(lab):
      tab = collections.defaultdict(collections.Counter)
      for c, l in zip(cell[:half], lab[:half]): tab[c][l] += 1
      hit = [(tab[c].most_common(1)[0][0] == l) for c, l in zip(cell[half:], lab[half:]) if c in tab]
      return np.mean(hit), len(hit) / (n - half)
    for name, lab in [('last pad', last), ('pos', pos)]:
      a, cov = cellacc(lab)
      print(f'hash cell -> {name}: ep1 accuracy {a:.0%} on the {cov:.0%} of ep1 steps whose cell was seen in ep0')
    # linear probe deter -> last pad / prefix, split by episode
    half = n // 2
    for name, lab in [('last pad', last), ('prefix', pre), ('window length', [len(s) for s in seq])]:
      cls = sorted(set(lab)); Y = np.array([[l == c for c in cls] for l in lab], float)
      Xb = np.c_[x, np.ones(n)]; A = Xb[:half]; Wt = np.linalg.solve(A.T @ A + 30.0 * np.eye(A.shape[1]), A.T @ Y[:half])
      acc_ = (np.argmax(Xb[half:] @ Wt, 1) == np.argmax(Y[half:], 1)).mean()
      base = Y[half:].mean(0).max()
      print(f'linear probe deter->{name}: test acc {acc_:.0%} (majority {base:.0%})')
