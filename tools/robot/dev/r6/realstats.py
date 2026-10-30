"""Robot recordings in sim_closed_loop's summary terms (per stick direction, dips/min)."""
import sys, json, numpy as np, pandas as pd
out = {}
for p in sys.argv[1:]:
  d = pd.read_csv(p); th = np.degrees(d.theta.values); hz = 1 / np.median(np.diff(d.t_obs.values))
  fwd = -0.5 * (d.wheel_speed_l + d.wheel_speed_r).values; cf = d.cmd_f.values
  s = dict(rate_std=float(np.degrees(d.angular_velocity).std()), tilt_mean=float(th.mean()))
  mv = np.abs(cf) > 0.05
  if mv.sum() > 50: s['corr_f'] = float(np.corrcoef(fwd[mv], cf[mv])[0, 1])
  for key, m in (('back', cf < -0.05), ('zero', np.abs(cf) <= 0.05), ('fwd', cf > 0.05)):
    if m.sum() > 20:
      s.update({f'{key}_tilt_med': float(np.median(th[m])), f'{key}_tilt_p5': float(np.percentile(th[m], 5)),
                f'{key}_tilt_p95': float(np.percentile(th[m], 95)), f'{key}_fwd_med': float(np.median(fwd[m])),
                f'{key}_rate_std': float(np.degrees(d.angular_velocity.values[m]).std())})
  below = th < -6.0
  a = np.stack([d.act_l.values, d.act_r.values], 1)
  s['sat_frac'] = float((np.abs(a) > 0.95).mean())
  s['dips_per_min'] = float((below[1:] & ~below[:-1]).sum() / (len(th) / hz / 60))
  out[p.split('/')[-1][:-4]] = s
  print(p.split('/')[-1][:-4], {k: round(v, 2) for k, v in s.items()})
json.dump(out, open('real_stats.json', 'w'), indent=1)
