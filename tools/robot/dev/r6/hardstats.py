"""Round 6u: per run and checkpoint, on the hard set (balance-point offsets eq*, slopes up*/dn*/ramp5):
stick: tracking corr (mean over variants, worst), falls/min; zero stick: drift |mean speed| (mm/s), falls/min, bumper time.
  python hardstats.py STICK_TAG ZERO_TAG   (e.g. mid6u_t mid6u_zt)"""
import glob, json, re, sys
import numpy as np


def load(tag):
  runs = {}
  for f in glob.glob(f'/work/DoyaU/vasilache/work/sysid/r6/ev/{tag}/*.json'):
    e, s = re.search(r'/(e\d+)@(\d+)_', f).groups()
    runs.setdefault((e, int(s)), {}).update(json.load(open(f)))
  return runs


def nz(x):
  return 0.0 if x is None or not np.isfinite(x) else x


def stats(stick, zero):
  out = {}
  if stick:
    c = {k: nz(v.get('corr_f')) for k, v in stick.items()}
    out.update(corr=np.mean(list(c.values())), corr_min=min(c.values()), corr_fit=c.get('fit', np.nan),
               falls=np.mean([v['dips_per_min'] for v in stick.values()]))
  if zero:
    off = [v for k, v in zero.items() if k != 'fit']
    out.update(drift=np.mean([nz(abs(v['fwd_mean'])) if np.isfinite(v['fwd_mean']) else 2000 for v in off]),
               zfalls=np.mean([v['dips_per_min'] for v in off]), bump=np.mean([v['bumper_frac'] for v in off]),
               zrate=np.nanmean([v['rate_std'] for v in off]))
  return out


if __name__ == '__main__':
  st, zt = load(sys.argv[1]), load(sys.argv[2])
  print(f'{sys.argv[1]} / {sys.argv[2]}: stick corr mean/min/fit, falls | zero stick drift mm/s, falls, bumper, wobble')
  for key in sorted(set(st) | set(zt)):
    s = stats(st.get(key), zt.get(key))
    print(f"  {key[0]}@{key[1]:<7d} corr {s.get('corr', np.nan):.2f}/{s.get('corr_min', np.nan):.2f}/{s.get('corr_fit', np.nan):.2f} "
          f"falls {s.get('falls', np.nan):5.1f} | drift {s.get('drift', np.nan):5.0f} falls {s.get('zfalls', np.nan):5.1f} "
          f"bump {s.get('bump', np.nan):.2f} wobble {s.get('zrate', np.nan):4.1f}")
