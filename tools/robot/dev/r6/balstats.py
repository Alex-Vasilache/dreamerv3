"""Balance policies: per run, wobble (tilt-rate SD) at each swept checkpoint, its mean and SD, tilt SD, falls,
drift (|mean wheel speed|), worst variant's wobble and bumper time.  python balstats.py TAG [TAG...]"""
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
    r = [f.get('rate_std', np.nan) for f in fit]; t = [f.get('tilt_std', np.nan) for f in fit]
    d = [f.get('dips_per_min', np.nan) for f in fit]; dr = [abs(f.get('fwd_mean', np.nan)) for f in fit]
    wr = [np.nanmax([v.get('rate_std', np.nan) for v in pts[s].values()]) for s in st]
    wb = [np.nanmax([v.get('bumper_frac', np.nan) for v in pts[s].values()]) for s in st]
    print(f"  {e} n={len(st)} rate {' '.join(f'{x:.1f}' for x in r)} | mean {np.nanmean(r):.1f} sd {np.nanstd(r):.1f} | "
          f"tilt {np.nanmean(t):.2f} falls {np.nanmean(d):.1f} drift {np.nanmean(dr):.0f} | worst rate {np.nanmean(wr):.1f} bumper {np.nanmean(wb):.2f}")
