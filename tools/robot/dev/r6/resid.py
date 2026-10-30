import sys, json
sys.path.insert(0, '.')
from loss import load, real, POL
res = load(sys.argv[1].split(','))
K = ['rate_std','corr_f','sat_frac','dips_per_min','zero_tilt_med','zero_tilt_p5','zero_tilt_p95','back_tilt_med','back_fwd_med','fwd_tilt_med','fwd_tilt_p95','fwd_fwd_med']
print(f"{'':16s}" + ''.join(f'{k[:9]:>10s}' for k in K))
for p in POL:
  r = real[POL[p][1].split('/')[-1]]
  print(f'{p+" REAL":16s}' + ''.join(f'{r.get(k, float("nan")):10.2f}' for k in K))
  for v in sys.argv[2:]:
    s = res[v][p]
    print(f'{"  "+v:16s}' + ''.join(f'{s.get(k, float("nan")):10.2f}' for k in K))
