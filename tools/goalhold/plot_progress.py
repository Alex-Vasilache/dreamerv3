"""Progress figures for the goal-hold work (pin pad four / five).
Run: /work/DoyaU/vasilache/work/plotenv/bin/python tools/goalhold/plot_progress.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOTS = ['/work/DoyaU/vasilache/work', '/bucket/DoyaU/vasilache/bucket/results/dreamerv3']
OUT = '/work/DoyaU/vasilache/work/meetings/goalhold'
BINS = np.arange(0, 4.0001, 0.125)

def find(tag):
  for r in ROOTS:
    g = sorted(glob.glob('%s/%s_*' % (r, tag)))
    g = [x for x in g if os.path.exists(x + '/logdir/scores.jsonl')]
    if g: return g[0]
  return None

def binned(path, key, fname):
  xs, ys = [], []
  if path is None or not os.path.exists(path + '/logdir/' + fname): return None
  for l in open(path + '/logdir/' + fname):
    try: r = json.loads(l)
    except Exception: continue
    v = [vv for kk, vv in r.items() if kk.endswith(key) and isinstance(vv, (int, float))]
    if v and 'step' in r: xs.append(r['step'] / 1e6); ys.append(v[0])
  if not xs: return None
  xs, ys = np.array(xs), np.array(ys)
  out = np.full(len(BINS) - 1, np.nan)
  for i in range(len(BINS) - 1):
    m = (xs >= BINS[i]) & (xs < BINS[i + 1])
    if m.any(): out[i] = ys[m].mean()
  return out

def curve(tags, key='episode/score', fname='scores.jsonl'):
  cs = [binned(find(t), key, fname) for t in tags]
  cs = [c for c in cs if c is not None]
  return np.array(cs) if cs else None

X = (BINS[:-1] + BINS[1:]) / 2
# name -> (pin pad four tags, pin pad five tags, colour, marker)
ARMS = [
  ('V, fixed 8 steps (5 seeds, old)', ['e1916', 'e1928', 'e1940', 'e1952', 'e1964'], ['e1917', 'e1929', 'e1941', 'e1953', 'e1965'], '#000000', 'o'),
  ('hold code, reached test off (8 steps)', ['h0408'], [], '#0072B2', 's'),
  ('reach or 8 steps, before fix', ['h0377'], ['h0381'], '#E69F00', '^'),
  ('reach or 32 steps, before fix', ['h0374'], ['h0375'], '#D55E00', 'D'),
  ('reach or 8 steps, FIXED', ['h0410', 'h0417'], ['h0414'], '#56B4E9', 'v'),
  ('reach or 32 steps, FIXED', ['h0411', 'h0416'], ['h0412'], '#009E73', 'P'),
]
fig, ax = plt.subplots(2, 2, figsize=(13, 8.6))
panels = [(ax[0, 0], 0, 'episode/score', 'scores.jsonl', 'Pin pad four: score'),
          (ax[0, 1], 1, 'episode/score', 'scores.jsonl', 'Pin pad five: score'),
          (ax[1, 0], 0, 'mgr/range_extr', 'metrics.jsonl', 'Pin pad four: manager return range'),
          (ax[1, 1], 0, 'goal/hold_reach_frac', 'metrics.jsonl', 'Pin pad four: share of switches by reach')]
handles = {}
for a, col, key, fname, title in panels:
  for name, t4, t5, c, mk in ARMS:
    cs = curve(t5 if col else t4, key, fname)
    if cs is None: continue
    m = np.nanmean(cs, 0)
    h, = a.plot(X, m, color=c, lw=2.4, marker=mk, markevery=4, ms=7, label=name)
    if len(cs) > 1:
      s = np.nanstd(cs, 0); a.fill_between(X, m - s, m + s, color=c, alpha=0.15, lw=0)
    handles[name] = h
  a.set_title(title, fontsize=17); a.tick_params(labelsize=13)
  a.grid(color='#d5d5d2', lw=0.7); a.set_axisbelow(True); a.set_ylim(bottom=0); a.set_xlim(0, 4)
ax[0, 0].set_ylabel('episode score', fontsize=15); ax[1, 0].set_ylabel('5-95% range of return', fontsize=15)
fig.text(0.5, 0.135, 'environment steps (millions)', ha='center', fontsize=15)
fig.legend(list(handles.values()), list(handles.keys()), loc='lower center', ncol=2, frameon=False, fontsize=14, handlelength=3.5)
fig.tight_layout(rect=(0, 0.15, 1, 1), h_pad=2.0)
fig.savefig(OUT + '/progress.png', dpi=110, facecolor='white'); fig.savefig(OUT + '/progress.pdf')
print('saved', OUT + '/progress.png')
