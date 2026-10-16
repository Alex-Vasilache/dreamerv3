import glob, json, os, pickle, collections, sys
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
B = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
if os.path.exists('ts.pkl'):
    data = pickle.load(open('ts.pkl', 'rb'))
else:
    runs = sorted(glob.glob(f'{B}/e9[1-5][0-9]_pinpad_*_director-director_og*'))
    data = {}
    for d in runs:
        n = os.path.basename(d)[:4]
        M = collections.defaultdict(list)
        with open(f'{d}/logdir/metrics.jsonl') as f:
            for l in f:
                r = json.loads(l); s = r['step']
                for k, v in r.items():
                    if k.startswith(('timer/', 'usage/')): continue
                    if isinstance(v, (int, float)): M[k].append((s, v))
        S = [json.loads(l) for l in open(f'{d}/logdir/scores.jsonl')]
        data[n] = dict(name=os.path.basename(d), M={k: np.array(v) for k, v in M.items()},
                       S=np.array([(s['step'], s['episode/score']) for s in S]))
    pickle.dump(data, open('ts.pkl', 'wb'))

KEYS = {
 'score': None,
 'rrate': 'epstats/reward_rate',
 'x_rew': 'train/mgr_extr_rew', 'x_ret': 'train/mgr_extr_ret', 'x_val': 'train/mgr_extr_val',
 'x_adv': 'train/mgr_extr_adv', 'x_rsc': 'train/mgr/rscale_extr', 'x_rng': 'train/mgr/range_extr',
 'e_rew': 'train/mgr_expl_rew', 'e_adv': 'train/mgr_expl_adv', 'e_rsc': 'train/mgr/rscale_expl',
 'm_advmag': 'train/mgr_adv_mag', 'm_ent': 'train/mgr_ent/skill', 'm_entn': 'train/mgr_ent_norm_skill_mean',
 'm_aesc': 'train/mgr_actent_skill_scale_mean',
 'w_ent': 'train/wkr_ent/action', 'w_rand': 'train/wkr_rand/action', 'w_aesc': 'train/wkr_actent_action_scale_mean',
 'w_grew': 'train/wkr_goal_rew', 'w_advmag': 'train/wkr_goal_adv_mag', 'w_ret': 'train/wkr_goal_ret',
 'g_rec': 'train/goal/rec_mean', 'g_kl': 'train/goal/kl_mean', 'g_klsc': 'train/goal/kl_adapt_scale_mean',
 'g_ent': 'train/goal/entropy_mean',
 'l_rew': 'train/loss/rew', 'l_img': 'train/loss/image', 'l_dyn': 'train/loss/dyn', 'l_con': 'train/loss/con',
 'dyn_ent': 'train/dyn_ent', 'rep_ent': 'train/rep_ent',
 'gn_ac': 'train/opt/ac_grad_norm', 'gn_mod': 'train/opt/model_grad_norm', 'gn_goal': 'train/opt/goal_grad_norm',
 'pr_ac': 'train/opt/ac_param_rms',
 'rep_items': 'replay/items',
}
BW = 1e5
NB = 40
def binned(arr):
    if arr is None or len(arr) == 0: return np.full(NB, np.nan)
    idx = (arr[:, 0] // BW).astype(int)
    out = np.full(NB, np.nan)
    for i in range(NB):
        m = idx == i
        if m.any(): out[i] = arr[m, 1].mean()
    return out
T = {}
for n, D in data.items():
    t = {}
    for k, key in KEYS.items():
        t[k] = binned(D['S']) if key is None else binned(D['M'].get(key))
    T[n] = t
pickle.dump(T, open('binned.pkl', 'wb'))

def fmt(x):
    if np.isnan(x): return '   nan'
    a = abs(x)
    if a == 0: return '     0'
    if a >= 1000 or a < 0.01: return f'{x:6.0e}'
    if a >= 10: return f'{x:6.1f}'
    return f'{x:6.3f}'

print('=== per-run time series, 100k bins, 0..3.9M (first 20 bins then every 4th) ===')
cols = list(range(20)) + list(range(20, NB, 4))
for n in sorted(T):
    print(f'\n##### {data[n]["name"]}')
    print('step(100k) ' + ' '.join(f'{c:6d}' for c in cols))
    for k in KEYS:
        print(f'{k:10s} ' + ' '.join(fmt(T[n][k][c]) for c in cols))

print('\n=== pooled Spearman corr of score with each metric (runs that ever scored >50 in a bin), per-run then averaged ===')
from itertools import chain
def rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r
res = collections.defaultdict(list)
for n, t in T.items():
    if np.nanmax(t['score']) < 50: continue
    for k in KEYS:
        if k == 'score': continue
        a, b = t['score'], t[k]
        m = ~np.isnan(a) & ~np.isnan(b)
        if m.sum() < 8 or np.std(b[m]) == 0: continue
        res[k].append(np.corrcoef(rank(a[m]), rank(b[m]))[0, 1])
for k, v in sorted(res.items(), key=lambda kv: -abs(np.mean(kv[1]))):
    print(f'{k:10s} mean {np.mean(v):+.2f}  min {np.min(v):+.2f} max {np.max(v):+.2f} n={len(v)}')

print('\n=== event-aligned: metric at [pre-find, peak, collapse onset, collapsed, late] ===')
# find = first bin score>=50 ; peak = argmax over 0..20 ; collapse = first bin after peak with score < 0.25*peak
ev = collections.defaultdict(lambda: collections.defaultdict(list))
for n, t in sorted(T.items()):
    s = t['score']
    if np.nanmax(s) < 50: continue
    f = int(np.nanargmax(s >= 50))
    p = int(np.nanargmax(np.where(np.arange(NB) < 25, s, -1)))
    after = [i for i in range(p, NB) if s[i] < 0.25 * s[p]]
    c = after[0] if after else None
    print(n, 'find', f, 'peak', p, f'{s[p]:.0f}', 'collapse', c)
    if c is None: continue
    pts = dict(pre=max(f - 1, 0) if f > 0 else 0, find=f, peak=p, pre_col=max(c - 1, p), col=c,
               col2=min(c + 2, NB - 1), late=NB - 2)
    for k in KEYS:
        for pn, i in pts.items():
            ev[k][pn].append(t[k][i])
print('metric     ' + ' '.join(f'{p:>8s}' for p in ['pre', 'find', 'peak', 'pre_col', 'col', 'col2', 'late']))
for k in KEYS:
    print(f'{k:10s} ' + ' '.join(f'{np.nanmedian(ev[k][p]):8.3g}' for p in ['pre', 'find', 'peak', 'pre_col', 'col', 'col2', 'late']))
