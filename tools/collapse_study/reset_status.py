"""Reset study status: scores and key metrics AFTER each run resumed."""
import glob, json, os
import numpy as np

W = '/work/DoyaU/vasilache/work'
BW = 5e4


def after_resume(path):
    rows = [json.loads(l) for l in open(path)]
    cut = max([i for i in range(1, len(rows)) if rows[i]['step'] < rows[i - 1]['step']] or [0])
    return rows[:cut], rows[cut:]


def bins(pairs, start):
    a = np.array(pairs, float)
    if not len(a):
        return []
    idx = ((a[:, 0] - start) // BW).astype(int)
    return [a[idx == i, 1].mean() if (idx == i).any() else np.nan for i in range(idx.max() + 1)]


for e in ['e1060', 'e1061', 'e1062', 'e1063', 'e1067', 'e1064', 'e1065', 'e1066']:
    d = glob.glob(f'{W}/{e}_*')[0]
    arm = d.split('director_og-')[1].split('_s')[0]
    _, S = after_resume(f'{d}/logdir/scores.jsonl')
    _, M = after_resume(f'{d}/logdir/metrics.jsonl')
    start = S[0]['step'] if S else 0
    last = M[-1]['step'] if M else 0
    sc = bins([(x['step'], x['episode/score']) for x in S], start)
    keys = [('train/wkr_goal_rew', 'goal'), ('train/mgr_ent_norm_skill_mean', 'mgr_ent'),
            ('train/mgr_extr_ret', 'mgr_ret')]
    mets = {n: bins([(x['step'], x[k]) for x in M if k in x], start) for k, n in keys}
    print(f'{e} {arm:11s} step {last:>8} (+{(last - start) / 1e3:.0f}k)  max ep {max([x["episode/score"] for x in S] or [0]):.0f}')
    print('   score   ' + ' '.join(f'{v:5.0f}' for v in sc))
    for n, v in mets.items():
        print(f'   {n:8s}' + ' '.join(f'{x:5.2f}' for x in v))
