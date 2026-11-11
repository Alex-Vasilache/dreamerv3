import json, glob, sys, numpy as np
B='/bucket/DoyaU/vasilache/bucket/results/dreamerv3'; W='/work/DoyaU/vasilache/work'
seeds=json.load(open('seeds.json'))
keys=['train/fromto/novelty','train/fromto/novelty_uniform','train/fromto/lean','train/fromto/weight','train/fromto/mass','train/fromto/row_mass_mean','train/fromto/row_mass_median','train/goal/perplexity','train/goal/used_frac','train/wkr_goal_rew','train/mgr_ent/skill','train/mgr_expl_rew']
for arm in ['TCEsqrt']:
  for r in sorted([s for s in seeds if s['arm']==arm and s['task']=='pinpad_six' and s['maxstep']>=3.98], key=lambda s:s['seed']):
    d=(glob.glob(f"{B}/{r['run']}")+glob.glob(f"{W}/{r['run']}"))[0]
    rows={k:[] for k in keys}
    for l in open(d+'/logdir/metrics.jsonl'):
      j=json.loads(l)
      for k in keys:
        if k in j: rows[k].append((j['step'],j[k]))
    def at(k,s):
      a=np.array(rows[k]);
      if not len(a): return np.nan
      m=(a[:,0]>=s-1e5)&(a[:,0]<s+1e5); return a[m,1].mean() if m.any() else np.nan
    print(f"s{r['seed']:2d} solved={r['late']>50!s:5} first={r['first']:.1f}", ' | '.join(f"{k.split('/')[-1][:10]} "+' '.join(f'{at(k,s):.3g}' for s in (0.5e6,1e6,2e6,3e6)) for k in keys))
