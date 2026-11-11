"""Offline pin pad simulation (no learning): timing cost of K-step goal holds, accidental
presses on shortest paths, and how long random/structured goal policies take to find the
sequence. Uses the real env (embodied/envs/pinpad.py) dynamics, re-implemented on the grid."""
import sys, collections, itertools, numpy as np
sys.path.insert(0, '/apps/unit/DoyaU/vasilache/apps/code/dreamerv3')
import importlib.util
spec = importlib.util.spec_from_file_location('pp', '/apps/unit/DoyaU/vasilache/apps/code/dreamerv3/embodied/envs/pinpad.py')
src = open(spec.origin).read().split('class PinPad')[1]
LAY = {k: v for k, v in [(n, eval('LAYOUT_' + n.upper(), {}, {l.split(' = ')[0]: None for l in []})) for n in []]}
ns = {}; exec('import numpy as np\n' + 'LAYOUT' + src.split('\nLAYOUT', 1)[1], ns)
MOVES = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]

class Grid:
  def __init__(s, task):
    s.lay = np.array([list(l) for l in ns['LAYOUT_' + task.upper()].split('\n')]).T
    s.pads = sorted(set(s.lay.flatten()) - set('* #\n')); s.N = len(s.pads)
    s.free = [(x, y) for (x, y), c in np.ndenumerate(s.lay) if c != '#']
    s.idx = {c: i for i, c in enumerate(s.free)}
    s.padcells = {p: [c for c in s.free if s.lay[c] == p] for p in s.pads}
    # all-pairs BFS distances
    n = len(s.free); s.D = np.full((n, n), 999, int)
    for i, c in enumerate(s.free):
      q = collections.deque([c]); s.D[i, i] = 0
      while q:
        u = q.popleft()
        for m in MOVES[1:]:
          v = (u[0] + m[0], u[1] + m[1])
          if v in s.idx and s.D[i, s.idx[v]] == 999:
            s.D[i, s.idx[v]] = s.D[i, s.idx[u]] + 1; q.append(v)
  def step_toward(s, pos, goal, rng, avoid=None):
    """One greedy shortest-path move (random tie-break). avoid: set of pad chars to avoid if an
    equally short alternative exists."""
    if pos == goal: return pos
    g = s.idx[goal]; best = []
    for m in MOVES[1:]:
      v = (pos[0] + m[0], pos[1] + m[1])
      if v in s.idx and s.D[g, s.idx[v]] == s.D[g, s.idx[pos]] - 1: best.append(v)
    if avoid:
      ok = [v for v in best if s.lay[v] not in avoid]
      best = ok or best
    return best[rng.randint(len(best))]

def run_episode(G, rng, policy, K, length=2000, stop_at_first=False):
  pos = G.free[rng.randint(len(G.free))]; seq = collections.deque(maxlen=G.N); target = tuple(G.pads)
  countdown = 0; score = 0; goal = pos; t_first = None; acc = 0; pressed_goal = 0; idle = 0
  mem = {'tried': set()}
  for t in range(length):
    if countdown:
      countdown -= 1
      if countdown == 0:
        pos = G.free[rng.randint(len(G.free))]; seq.clear()
    if t % K == 0:
      goal = policy(G, pos, seq, rng, mem)
    tgt_pad = G.lay[goal] if G.lay[goal] in G.pads else None
    if pos == goal: idle += 1
    pos = G.step_toward(pos, goal, rng, avoid=(set(G.pads) - {tgt_pad}) if policy.__name__ == 'oracle' else None)
    tile = G.lay[pos]
    if tile in G.pads and (not seq or seq[-1] != tile):
      seq.append(tile)
      if tile == tgt_pad: pressed_goal += 1
      else: acc += 1
    if tuple(seq) == target and not countdown:
      score += 10; countdown = 10
      if t_first is None: t_first = t
      if stop_at_first: break
  return score, t_first, acc, pressed_goal, idle

def nearest_cell(G, pos, pad):
  i = G.idx[pos]; cells = G.padcells[pad]
  return min(cells, key=lambda c: G.D[i, G.idx[c]])

def oracle(G, pos, seq, rng, mem):
  # next pad of the target sequence given the current window
  s = list(seq); tgt = G.pads
  for k in range(min(len(s), G.N), 0, -1):
    if tuple(s[-k:]) == tuple(tgt[:k]): nxt = tgt[k] if k < G.N else tgt[0]; break
  else: nxt = tgt[0]
  return nearest_cell(G, pos, nxt)

def rand_cell(G, pos, seq, rng, mem): return G.free[rng.randint(len(G.free))]
def rand_pad(G, pos, seq, rng, mem):
  last = seq[-1] if seq else None
  p = [q for q in G.pads if q != last][rng.randint(G.N - (last is not None))] if True else None
  return nearest_cell(G, pos, p)
def episodic_pad(G, pos, seq, rng, mem):
  """'Human' policy: pick a pad that is not in the current window (the window is drawn on
  screen), so every window of N presses is a permutation; falls back to any pad."""
  cand = [q for q in G.pads if q not in seq] or [q for q in G.pads if not seq or q != seq[-1]]
  return nearest_cell(G, pos, cand[rng.randint(len(cand))])

def flat(G, pos, seq, rng, mem): return oracle(G, pos, seq, rng, mem)

if __name__ == '__main__':
  rng = np.random.RandomState(0)
  MAX = {'four': 411, 'five': 352, 'six': 292}
  print('== timing: oracle (knows the sequence), shortest paths, avoids other pads; score per 2000-step episode')
  for task in ['four', 'five', 'six']:
    G = Grid(task)
    row = []
    for K in [1, 2, 4, 8, 16]:
      sc = [run_episode(G, rng, oracle, K)[0] for _ in range(30)]
      row.append(f'K={K}: {np.mean(sc):.0f}')
    print(f'{task:5s} max {MAX[task]}  ' + '  '.join(row))
  print('\n== accidental presses: oracle at K=8 WITHOUT pad avoidance (random shortest path)')
  def oracle_noavoid(*a): return oracle(*a)
  for task in ['four', 'five', 'six']:
    G = Grid(task); r = [run_episode(G, rng, oracle_noavoid, 8) for _ in range(30)]
    print(f'{task:5s} score {np.mean([x[0] for x in r]):.0f}  accidental/goal presses {np.sum([x[2] for x in r]) / max(1, np.sum([x[3] for x in r])):.2f}')
  print('\n== exploration: env steps until the first reward (no learning; 40 runs, cap 4M)')
  for task in ['four', 'five', 'six']:
    G = Grid(task); row = []
    for pol in [rand_cell, rand_pad, episodic_pad]:
      ts = []
      for _ in range(40 if task != 'six' else 20):
        steps = 0
        while steps < 4e6:
          sc, tf, *_ = run_episode(G, rng, pol, 8, stop_at_first=True)
          if tf is not None: steps += tf; break
          steps += 2000
        ts.append(steps)
      row.append(f'{pol.__name__}: median {np.median(ts) / 1e3:.0f}k')
    print(f'{task:5s} ' + '  '.join(row))
