"""Summarise evalpol results: per policy, fitted-sim stats and robustness over the set.
  python evsum.py TAG [TAG...]"""
import glob, json, sys
import numpy as np
rows = {}
for tag in sys.argv[1:]:
  for f in glob.glob(f'/work/DoyaU/vasilache/work/sysid/r6/ev/{tag}/*.json'):
    p = f.split('/')[-1].rsplit('_', 1)[0]
    rows.setdefault(p, {}).update(json.load(open(f)))
print(f"{'policy':8s} {'n':>2s} | fit: {'corr':>5s} {'rate':>5s} {'dips':>5s} {'fwdP95':>6s} {'fwdV':>5s} {'backV':>6s} {'sat':>5s} | set: {'corr':>5s} {'minC':>5s} {'rate':>5s} {'maxR':>5s} {'dips':>5s}")
for p, vs in sorted(rows.items()):
  f = vs.get('fit', {})
  g = lambda k: f.get(k, float('nan'))
  c = [v.get('corr_f', np.nan) for v in vs.values()]; r = [v['rate_std'] for v in vs.values()]; d = [v.get('dips_per_min', np.nan) for v in vs.values()]
  print(f"{p:8s} {len(vs):2d} | fit: {g('corr_f'):5.2f} {g('rate_std'):5.1f} {g('dips_per_min'):5.1f} {g('fwd_tilt_p95'):6.2f} {g('fwd_fwd_med'):5.0f} {g('back_fwd_med'):6.0f} {g('sat_frac'):5.2f} | set: {np.nanmean(c):5.2f} {np.nanmin(c):5.2f} {np.mean(r):5.1f} {np.max(r):5.1f} {np.nanmean(d):5.1f}")
