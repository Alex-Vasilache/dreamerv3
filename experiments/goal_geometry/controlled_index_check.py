"""Index distance with block count held constant.

At m=8 (every block changed) the total index distance D still ranges from 8
(each block moved one class) to 56 (each moved seven). Comparing goal MSE
across D at fixed m isolates the index axis from the block-count axis, which
is the one comparison neither figure makes on its own.
"""
import ast, glob, os, sys
import numpy as np
sys.path.insert(0, 'experiments/goal_geometry')
from make_figure_index_total import arm_of

rows = {}
for f in sorted(glob.glob('experiments/goal_geometry/results/*.npz')):
    d = np.load(f, allow_pickle=True)
    if 'index_total/rec_d' not in d:
        continue
    meta = ast.literal_eval(str(d['meta']))
    arm = arm_of(os.path.basename(f), meta)
    if arm is None:
        continue
    dd = np.asarray(d['index_total/rec_d'], int)
    mm = np.asarray(d['index_total/rec_blocks'], int)
    gm = np.asarray(d['index_total/rec_goal_mse'], float)
    L = int(meta['blocks'])
    full = mm == L                       # every block changed
    if full.sum() < 100:
        continue
    dq = dd[full]
    lo = dq <= np.percentile(dq, 25)
    hi = dq >= np.percentile(dq, 75)
    rows.setdefault((meta['task'], arm), []).append(
        (gm[full][lo].mean(), gm[full][hi].mean(),
         np.median(dq[lo]), np.median(dq[hi])))

print('%-22s %-26s %2s %10s %10s %8s   (m=%s, all blocks changed)'
      % ('task', 'arm', 'n', 'MSE lo-D', 'MSE hi-D', 'hi/lo', 'L'))
for k in sorted(rows):
    a = np.array(rows[k])
    ratio = a[:, 1] / np.maximum(a[:, 0], 1e-30)
    print('%-22s %-26s %2d %10.4g %10.4g %5.2f+/-%-5.2f  D: %d vs %d'
          % (k[0].replace('dmc_', ''), k[1], len(a), a[:, 0].mean(),
             a[:, 1].mean(), ratio.mean(), ratio.std(),
             a[:, 2].mean(), a[:, 3].mean()))

# exact two-sided permutation test: straight-through SOM arms vs the rest
from itertools import combinations
STE = ('som_line', 'som_lipvq_line_prod')
for task in ('dmc_cartpole_swingup', 'dmc_hopper_stand'):
    a, b = [], []
    for k, v in rows.items():
        if k[0] != task:
            continue
        r = [x[1] / max(x[0], 1e-30) for x in v]
        (a if k[1] in STE else b).extend(r)
    a, b = np.array(a), np.array(b)
    pool = np.concatenate([a, b]); n = len(a)
    obs = abs(a.mean() - b.mean()); cnt = tot = 0
    for idx in combinations(range(len(pool)), n):
        m = np.zeros(len(pool), bool); m[list(idx)] = True
        tot += 1; cnt += abs(pool[m].mean() - pool[~m].mean()) >= obs - 1e-12
    print('%-22s STE-SOM %.2f (n=%d) vs rest %.2f (n=%d)  p=%.3g'
          % (task.replace('dmc_', ''), a.mean(), len(a), b.mean(), len(b),
             cnt / tot))
    print('%-22s   ranges: STE %.2f-%.2f | rest %.2f-%.2f'
          % ('', a.min(), a.max(), b.min(), b.max()))
