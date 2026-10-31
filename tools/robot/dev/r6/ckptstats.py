"""Per run: fit corr at each swept checkpoint, its mean and SD (checkpoint oscillation), rate, falls, worst variant.
  python ckptstats.py TAG [TAG...]"""
import glob, json, re, sys
import numpy as np
for tag in sys.argv[1:]:
  runs = {}
  for f in glob.glob(f'/work/DoyaU/vasilache/work/sysid/r6/ev/{tag}/*.json'):
    e, s = re.search(r'/(e\d+)@(\d+)_', f).groups()
    runs.setdefault(e, {}).setdefault(int(s), {}).update(json.load(open(f)))
  print(tag)
  for e in sorted(runs):
    pts = runs[e]; st = sorted(pts)
    fit = [pts[s].get('fit', {}) for s in st]
    c = [f.get('corr_f', np.nan) for f in fit]; r = [f.get('rate_std', np.nan) for f in fit]; d = [f.get('dips_per_min', np.nan) for f in fit]
    mc = [np.nanmin([v.get('corr_f', np.nan) for v in pts[s].values()]) for s in st]
    print(f"  {e} n={len(st)} corr {' '.join(f'{x:.2f}' for x in c)} | mean {np.nanmean(c):.2f} sd {np.nanstd(c):.2f} | rate {np.nanmean(r):.0f} falls {np.nanmean(d):.1f} max {np.nanmax(d):.1f} minC {np.nanmean(mc):.2f}")
