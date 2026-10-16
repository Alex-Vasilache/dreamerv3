"""Final per-arm summary for the pinpad collapse causal study (e1031-e1053)."""
import glob, json, os, collections
import numpy as np
W = '/work/DoyaU/vasilache/work'
B = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'

def load(d):
    S = np.array([(json.loads(l)['step'], json.loads(l)['episode/score'])
                  for l in open(f'{d}/logdir/scores.jsonl')])
    M = collections.defaultdict(list)
    for l in open(f'{d}/logdir/metrics.jsonl'):
        r = json.loads(l)
        for k in ('train/wkr_goal_rew', 'train/mgr_extr_rew', 'epstats/reward_rate',
                  'train/goal/rec_mean'):
            if k in r: M[k].append((r['step'], r[k]))
    return S, {k: np.array(v) for k, v in M.items()}

def win(a, lo, hi):
    if a is None or len(a) == 0: return np.nan
    m = (a[:, 0] >= lo) & (a[:, 0] < hi)
    return a[m, 1].mean() if m.any() else np.nan

runs = collections.defaultdict(list)
for d in sorted(glob.glob(f'{W}/e10[3-5][0-9]_pinpad_four_director-director_og-collapse_*')):
    n = os.path.basename(d)
    arm = n.split('collapse_')[1].rsplit('_s', 1)[0]
    runs[arm].append(d)
# historical controls (identical config, 4M, archived)
for e in ('e924', 'e930', 'e936'):
    runs['ctrl_bench(e924/30/36)'] += glob.glob(f'{B}/{e}_pinpad_four_director-director_og_*')

W1, W2, W3 = (1e5, 3e5), (6e5, 9e5), (9e5, 1.2e6)
order = ['ctrl_bench(e924/30/36)', 'ctrl', 'freeze_worker', 'freeze_manager',
         'freeze_manager_model', 'freeze_worker_goal', 'freeze_goal',
         'freeze_goal_model', 'freeze_model']
print(f"{'arm':24s} {'run':6s} {'maxstep':>8s} | score 0.1-0.3M  0.6-0.9M  0.9-1.2M | "
      f"goalrew 0.1-0.3M 0.6-0.9M | imag_rew/real_rew 0.6-0.9M")
for arm in order:
    agg = collections.defaultdict(list)
    for d in runs.get(arm, []):
        S, M = load(d)
        s1, s2, s3 = win(S, *W1), win(S, *W2), win(S, *W3)
        g1, g2 = win(M.get('train/wkr_goal_rew'), *W1), win(M.get('train/wkr_goal_rew'), *W2)
        xi, rr = win(M.get('train/mgr_extr_rew'), *W2), win(M.get('epstats/reward_rate'), *W2)
        for k, v in zip('s1 s2 s3 g1 g2'.split(), (s1, s2, s3, g1, g2)): agg[k].append(v)
        print(f"{arm:24s} {os.path.basename(d)[:5]:6s} {S[:,0].max():8.0f} | "
              f"{s1:14.0f} {s2:8.0f} {s3:9.0f} | {g1:15.2f} {g2:8.2f} | "
              f"{xi:.3f} / {rr:.4f}")
    if agg:
        f = lambda k: np.nanmean(agg[k])
        print(f"{'  -> MEAN':24s} {'':6s} {'':8s} | {f('s1'):14.0f} {f('s2'):8.0f} {f('s3'):9.0f} | "
              f"{f('g1'):15.2f} {f('g2'):8.2f} |")
