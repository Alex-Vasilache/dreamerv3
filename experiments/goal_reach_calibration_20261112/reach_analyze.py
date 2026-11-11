"""Step 1 analysis: cos_max(goal, deter) vs ground truth, and what a threshold tau would do.
usage: reach_analyze.py reach/<name>.npz ..."""
import sys, numpy as np
sys.path.insert(0, '/work/DoyaU/vasilache/work/meetings/tce_investigation')
TAUS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

def cosmax(g, f):
  g = g.astype(np.float32); f = f.astype(np.float32)
  n = np.maximum(np.linalg.norm(g, axis=-1), np.linalg.norm(f, axis=-1)) + 1e-12
  return (g * f).sum(-1) / (n * n)
def q(x, ps=(5, 25, 50, 75, 95)):
  x = np.asarray(x, float)
  return ' '.join(f'{v:.2f}' for v in np.percentile(x, ps)) if len(x) else 'n/a'

def main(f):
  z = np.load(f, allow_pickle=True); K = int(z['K']); name = f.split('/')[-1][:-4]
  det, goal, slf, ep, t = z['deter'], z['goal'], z['self'], z['ep'], z['t']
  c = cosmax(goal, det); cs = cosmax(slf, det); n = len(c)
  print(f'\n=== {name}: {n} steps, K={K}')
  print(f'  self cos (decode(encode(state)) vs state), p5/25/50/75/95: {q(cs)}')
  print(f'  goal cos, all steps:                                      {q(c)}')
  off = t % K
  print('  goal cos median by step within the hold: ' + ' '.join(f'{np.median(c[off == k]):.2f}' for k in range(K)))
  perm = np.random.default_rng(0).permutation(n)
  print(f'  goal vs a random other state (chance level):              {q(cosmax(goal, det[perm]))}')
  # holds
  starts = np.where(off == 0)[0]
  holds = [(s, min(s + K, n)) for s in starts if s + K <= n and ep[s] == ep[s + K - 1]]
  pin = 'pos' in z.files
  if pin:
    from analyze import G, goal_pos, dist
    pos = [tuple(p) for p in z['pos']]
    gp = {s: goal_pos(z['goalimg'][s]) for s, _ in holds}
    d = np.full(n, -1.0); clear = np.zeros(n, bool)
    for s, e in holds:
      g, dark = gp[s]
      for i in range(s, e): d[i] = dist(pos[i], g); clear[i] = dark < 60
    for lab, m in [('on the goal cell', d == 0), ('1 cell away', d == 1), ('2-3 cells', (d >= 2) & (d <= 3)), ('4+ cells', d >= 4)]:
      print(f'  goal cos, player {lab:17s} n={m.sum():5d}: {q(c[m])}   (sharp goals only, n={(m & clear).sum()}: {q(c[m & clear])})')
  if not pin and 'obsvec' in z.files:
    # no grid here: take the visited state most similar to the goal as "where the goal is", and
    # measure how far the body is from it (obs units, each dim scaled by its std; chance = random pairs)
    ob = z['obsvec'].astype(np.float32); ob = (ob - ob.mean(0)) / (ob.std(0) + 1e-6)
    dn = det.astype(np.float32); dnn = np.linalg.norm(dn, axis=-1)
    dphys = np.full(n, np.nan)
    for s, e in holds:
      g = goal[s].astype(np.float32); cc = (dn @ g) / np.maximum(dnn, np.linalg.norm(g)) ** 2
      j = int(np.argmax(cc)); dphys[s:e] = np.linalg.norm(ob[s:e] - ob[j], axis=-1)
    print(f'  body distance to the goal state, random pairs (chance): {q(np.linalg.norm(ob - ob[perm], axis=-1))}')
    for lo, hi in [(-1, .3), (.3, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
      m = (c >= lo) & (c < hi) & ~np.isnan(dphys)
      print(f'  body distance when goal cos in [{lo:.1f},{hi:.1f}) n={m.sum():5d}: {q(dphys[m])}')
  print('  tau | holds triggered | median trigger step | triggered at hold step 0 |' + (' player <=1 cell at trigger | on-cell holds detected' if pin else ''))
  for tau in TAUS:
    trig, step0, prec, rec_hit, rec_n = [], 0, [], 0, 0
    for s, e in holds:
      w = np.where(c[s:e] >= tau)[0]
      if pin:
        truth = [i for i in range(s, e) if d[i] == 0]
        if truth:
          rec_n += 1; rec_hit += len(w) > 0
      if len(w):
        trig.append(w[0]); step0 += w[0] == 0
        if pin: prec.append(d[s + w[0]] <= 1)
    line = f'  {tau:.2f} | {len(trig) / len(holds):5.0%} | {np.median(trig) if trig else float("nan"):4.0f} | {step0 / len(holds):5.0%} |'
    if pin: line += f' {np.mean(prec) if prec else float("nan"):5.0%} | {rec_hit / max(rec_n, 1):5.0%} of {rec_n}'
    print(line)
  if 'hand_deter' in z.files:
    hc = cosmax(z['hand_goal'], z['hand_deter']); on = z['hand_onpad'].astype(bool); pd = z['hand_paddist']; ht = z['hand_t']
    print(f'  HAND (goal fixed to a pad, 48 steps): cos on the pad n={on.sum()}: {q(hc[on])}')
    for lab, m in [('1 cell from pad', (pd == 1)), ('2-3 cells', (pd >= 2) & (pd <= 3)), ('4+ cells', pd >= 4)]:
      print(f'    {lab:16s} n={m.sum():5d}: {q(hc[m])}')
    tr = z['hand_trial'].astype(int); pad = z['hand_pad']
    key = np.array([f'{p}_{k}' for p, k in zip(pad, tr)]); keys = list(dict.fromkeys(key))
    print('    tau | trials triggered | of those, on pad at trigger | <=1 cell | trials that reach the pad detected | median steps: trigger / first on pad')
    for tau in TAUS:
      trig = prec = prec1 = rec = recn = 0; ts, tp = [], []
      for k in keys:
        m = np.where(key == k)[0]; w = np.where(hc[m] >= tau)[0]; o = np.where(on[m])[0]
        if len(o):
          recn += 1; tp.append(o[0] + 1); rec += len(w) > 0
        if len(w):
          trig += 1; prec += on[m][w[0]]; prec1 += pd[m][w[0]] <= 1; ts.append(w[0] + 1)
      print(f'    {tau:.2f} | {trig / len(keys):5.0%} | {prec / max(trig, 1):5.0%} | {prec1 / max(trig, 1):5.0%} | {rec / max(recn, 1):5.0%} of {recn} | {np.median(ts) if ts else float("nan"):.0f} / {np.median(tp) if tp else float("nan"):.0f}')

if __name__ == '__main__':
  for f in sys.argv[1:]: main(f)
