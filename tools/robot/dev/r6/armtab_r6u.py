"""Rounds 6u-6w per arm: each run's hard-set stats averaged over its swept checkpoints, then mean +- std over runs.  python armtab_r6u.py"""
import numpy as np
from hardstats import load, stats
ARMS = [('6u A: eq +-3, slope +-5', 'fin6u', range(10020, 10023)), ('6u B: A + p_zero .5', 'fin6u', range(10023, 10025)),
        ('6u C: A + tilt penalty 0', 'fin6u', [10025, 10026, 10040, 10041, 10042]), ('6u D: eq +-3 only', 'fin6u', range(10027, 10029)),
        ('6u E: eq +-3, slope +-3', 'fin6u', range(10029, 10031)), ('6u F: A + eq every 60 s', 'fin6u', range(10031, 10033)),
        ('6v H: C + fine .15, warm e10025', 'fin6v', [10053, 10054, 10055, 10060, 10063, 10064, 10065]),
        ('6v J: C + fine .15, warm e10028', 'fin6v', [10058, 10059]),
        ('6w K: H continued', 'fin6w', [10070, 10071]), ('6w L/M: H continued, eq every 30 s', 'fin6w', range(10072, 10076)),
        ('6w N: L + fine weight .8', 'fin6w', [10076, 10077])]
D = {p: {fl: (load(f'{p}_{fl}'), load(f'{p}_z{fl}')) for fl in 'ct'} for p in ('fin6u', 'fin6v', 'fin6w')}
print('| arm | n | carpet corr | carpet falls/min | table corr | table falls/min | table zero-stick falls/min | table zero-stick roll (mm/s) |')
print('|---|---|---|---|---|---|---|---|')
for name, p, exps in ARMS:
  row = {}
  for fl in 'ct':
    st, zt = D[p][fl]
    per = []
    for e in exps:
      ks = [k for k in st if k[0] == f'e{e}']
      if ks: per.append([stats(st[k], zt.get(k)) for k in ks])
    for m in ('corr', 'falls', 'zfalls', 'drift'):
      v = [np.mean([s[m] for s in r]) for r in per]
      row[fl + m] = f'{np.mean(v):.2f} ± {np.std(v):.2f}' if m == 'corr' else f'{np.mean(v):.0f} ± {np.std(v):.0f}' if m == 'drift' else f'{np.mean(v):.1f} ± {np.std(v):.1f}'
    row['n'] = len(per)
  print(f"| {name} | {row['n']} | {row['ccorr']} | {row['cfalls']} | {row['tcorr']} | {row['tfalls']} | {row['tzfalls']} | {row['tdrift']} |")
