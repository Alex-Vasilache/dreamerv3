"""Mean ± std table rows for arms: python armtab.py 'label|tag|e1,e2' ...  (fit corr, rate SD, falls/min, worst-variant corr)."""
import glob, json, sys
import numpy as np
EV = '/work/DoyaU/vasilache/work/sysid/r6/ev'


def stats(tag, exp):
  d = {}
  for f in glob.glob(f'{EV}/{tag}/{exp}_*.json'):
    d.update(json.load(open(f)))
  f = d['fit']
  return [f.get('corr_f', np.nan), f['rate_std'], f['dips_per_min'], np.nanmin([v.get('corr_f', np.nan) for v in d.values()])]


def cell(v, fmt):
  return fmt.format(v[0]) if len(v) == 1 else (fmt + ' ± ' + fmt).format(np.mean(v), np.std(v))


for spec in sys.argv[1:]:
  label, tag, exps = spec.split('|')
  s = np.array([stats(tag, e) for e in exps.split(',')]).T
  n = s.shape[1]
  print(f"| {label}{'' if n == 2 else f' (n={n})'} | " + ' | '.join(cell(v, f) for v, f in zip(s, ['{:.2f}', '{:.0f}', '{:.1f}', '{:.2f}'])) + ' |')
