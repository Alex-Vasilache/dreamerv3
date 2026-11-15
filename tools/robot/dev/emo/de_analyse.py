"""Round DE: does each control signal go LOW inside control cuts?
AUC (low value marks 'inside'), bumper contact +-0.25 s excluded, minutes 5-30,
plus z time-course around cut onsets. Usage: python de.py RUNS_DIR OUT_PREFIX"""
import sys, glob, json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
try:
  import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
except Exception: plt = None
R, OUT = sys.argv[1], sys.argv[2]
def auc(i, o): return mannwhitneyu(o, i).statistic / (len(i) * len(o))
SIG = [('dominance', 'dominance (empowerment)', '#000000', 'o', '-'),
       ('dominance_ev', 'dominance (evidence)', '#D55E00', 's', '--'),
       ('slope', 'command→tilt slope (model-free)', '#0072B2', 'D', ':'),
       ('arousal_ph_now', 'arousal (reference)', '#56B4E9', '^', '-.')]
res, curves = {}, {}
for n in range(44990, 44998):
  g = 0.5 if n % 2 == 0 else 0.35
  f = glob.glob(f'{R}/e{n}_*/logdir/emotion.csv')[0]
  d = pd.read_csv(f, usecols=['t','theta','act0','pain','control_cut','dominance',
                              'dominance_ev','ctl_ev','arousal_ph_now'])
  t0 = d.t.iloc[0]; d = d[(d.t > t0 + 300)].reset_index(drop=True)
  th, a = d.theta.values, d.act0.values
  y = np.r_[th[2:] - 2*th[1:-1] + th[:-2], np.nan, np.nan]
  W = 130; k = lambda x: pd.Series(x).rolling(W, min_periods=W//2).mean().values
  ma, my = k(a), k(y); d['slope'] = -(k(a*y) - ma*my) / np.maximum(k(a*a) - ma**2, 1e-9)
  cut = d.control_cut.values != 0
  near = np.convolve((d.pain.fillna(0).values > 0).astype(float), np.ones(33), 'same') > 0
  r = dict(gain=g, cut_share=float(cut.mean()), contact_in=float(near[cut].mean()))
  on = np.flatnonzero(np.diff(cut.astype(int)) == 1) + 1
  for c, *_ in SIG:
    v = d[c].ffill().values.astype(float); ok = np.isfinite(v) & ~near
    r[c] = auc(v[cut & ok], v[~cut & ok])
    z = (v - np.nanmean(v[~cut])) / (np.nanstd(v[~cut]) + 1e-9)
    seg = [z[i - 325:i + 650] for i in on if 325 <= i and i + 650 <= len(z)]
    if seg: curves.setdefault((g, c), []).append(np.nanmean(seg, 0))
  res[n] = r; print(n, {k: round(v, 3) for k, v in r.items()}, flush=True)
df = pd.DataFrame(res).T
summ = df.groupby('gain').agg(['mean', 'std']).round(3)
print(summ.T.to_string()); json.dump(res, open(OUT + '.json', 'w'), indent=1)
if plt is None: sys.exit(0)
fig, ax = plt.subplots(2, 1, figsize=(11.5, 7), sharex=True)
tt = (np.arange(975) - 325) / 65
for i, g in enumerate((0.5, 0.35)):
  for c, lab, col, m, ls in SIG:
    if (g, c) not in curves: continue
    C = np.array(curves[(g, c)]); mu, sd = C.mean(0), C.std(0)
    ax[i].plot(tt, mu, color=col, ls=ls, marker=m, markevery=65, label=f'{lab}: AUC {df[df.gain==g][c].mean():.2f}')
    ax[i].fill_between(tt, mu - sd, mu + sd, color=col, alpha=0.12)
  ax[i].axvspan(0, 5, color='0.88', zorder=0); ax[i].axhline(0, color='0.5', lw=0.6)
  ax[i].set_title(f'wheels x{g} for 5 s (grey); curves include bumper contact, AUC (low = inside a cut) excludes it; mean ± SD, n = 4', fontsize=10); ax[i].set_ylabel('z (per run)')
  ax[i].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
ax[1].set_xlabel('time from cut onset (s)'); fig.tight_layout(); fig.savefig(OUT + '.png', dpi=150)
