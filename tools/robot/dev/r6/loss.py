"""Score closed-loop sweep variants against the robot recordings.
  python loss.py NAME [NAME...]   -> table of per-policy and total loss per variant."""
import glob, json, math, sys
import numpy as np
sys.path.insert(0, '/work/DoyaU/vasilache/work/sysid/r6')
from sweep import POL
real = json.load(open('/work/DoyaU/vasilache/work/sysid/r6/real_stats.json'))
# (key, kind, scale): 'log' compares log ratios, 'abs' differences.
TERMS = [('rate_std', 'log', 0.25), ('corr_f', 'abs', 0.15), ('dips_per_min', 'log1', 0.7),
         ('sat_frac', 'abs', 0.1)]
for d in ('back', 'zero', 'fwd'):
  TERMS += [(f'{d}_tilt_med', 'abs', 1.0), (f'{d}_tilt_p5', 'abs', 1.5), (f'{d}_tilt_p95', 'abs', 1.5),
            (f'{d}_fwd_med', 'abs', 250.0), (f'{d}_rate_std', 'log', 0.25)]


def term(s, r, kind, sc):
  if kind == 'log': return (math.log(max(s, 1e-3) / max(r, 1e-3)) / sc) ** 2
  if kind == 'log1': return (math.log((s + 1) / (r + 1)) / sc) ** 2
  return ((s - r) / sc) ** 2


def score(s, r):
  ts = [term(s[k], r[k], kind, sc) for k, kind, sc in TERMS if k in r and k in s and not math.isnan(r[k])]
  return float(np.mean(ts)) if ts else float('nan')


def load(names):
  res = {}
  for n in names:
    for f in glob.glob(f'/work/DoyaU/vasilache/work/sysid/r6/res/{n}/*.json'):
      p = f.split('/')[-1].split('_')[0]
      for v, s in json.load(open(f)).items():
        res.setdefault(v, {})[p] = s
  return res


if __name__ == '__main__':
  res = load(sys.argv[1:]); pols = list(POL)
  rows = []
  for v, ps in res.items():
    l = {p: score(ps[p], real[POL[p][1].split('/')[-1]]) for p in pols if p in ps}
    rows.append((np.mean(list(l.values())) if len(l) == len(pols) else float('inf'), v, l))
  print(f"{'variant':16s} {'total':>6s} " + ' '.join(f'{p:>6s}' for p in pols))
  for tot, v, l in sorted(rows):
    print(f'{v:16s} {tot:6.2f} ' + ' '.join(f"{l.get(p, float('nan')):6.2f}" for p in pols))
