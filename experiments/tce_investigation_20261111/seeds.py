import sys, json, re, numpy as np
sys.path.insert(0, '/apps/unit/DoyaU/vasilache/apps/code/dreamerv3/tools/collapse_study')
from compare_late import runs, scores, cross50
ARMS = [
 ('OG', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
 ('DV3', 'e12[3-8]?_{t}_dreamerv3-pubmatch_s*'),
 ('ent', 'e1[89][089]?_{t}_director-director_og-wkr_ent_fixed-pubmatch_s*'),
 ('V', 'e19[1-6]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-pubmatch_s*'),
 ('USSOM', '[eh]????_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-pubmatch_s*'),
 ('TCEsqrt', '[eh]????_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_hash-pubmatch_s*'),
 ('TCEkl10f', 'h0???_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir-tce_kl-tce_kl_c10-tce_kl_floor-pubmatch_s*'),
 ('TCEcrit', 'h03??_{t}_director-director_og-wkr_ent_fixed-wkr_v3-fromto_dir-tce_hash-tce_critic-pubmatch_s*'),
]
out = []
for t in ['pinpad_four', 'pinpad_five', 'pinpad_six']:
  print(f'\n## {t}   (first>0 = first rewarded episode; c50 = sustained-50 crossing; late = mean 3-4M; max step)')
  for arm, p in ARMS:
    cells = []
    for d in runs(p.format(t=t)):
      a = scores(d)
      if not len(a): continue
      seed = re.search(r'_s(\d+)_', d).group(1)
      f = a[a[:, 1] > 0, 0]; first = f[0] / 1e6 if len(f) else np.nan
      m = (a[:, 0] >= 3e6) & (a[:, 0] < 4e6)
      late = a[m, 1].mean() if m.sum() > 3 else np.nan
      c = cross50(a)
      cells.append((int(seed), first, c / 1e6 if c is not None and c == c else np.nan, late, a[:, 0].max() / 1e6))
      out.append(dict(task=t, arm=arm, seed=int(seed), first=first, c50=cells[-1][2], late=late, maxstep=cells[-1][4], run=d.split('/')[-1]))
    cells.sort()
    print(f'{arm:9s} ' + '  '.join(f's{s}:{fi:.1f}/{c:.1f}/{l:.0f}' + ('' if ms >= 3.98 else f'@{ms:.1f}') for s, fi, c, l, ms in cells))
json.dump(out, open('/work/DoyaU/vasilache/work/meetings/tce_investigation/seeds.json', 'w'), indent=0, default=float)
