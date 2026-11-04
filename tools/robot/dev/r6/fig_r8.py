"""Round 8: command tracking across floors between the fitted table (f=0) and carpet (f=1), and beyond,
replaying Alex's stick (3.5 min per floor), plus zero-stick standing. r6_mixed (e8261@838500) against
round 8 at 702000 steps (A scratch, B warm from e8261). mean +- std over seeds.
  python fig_r8.py OUT_PREFIX"""
import glob, json, re, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'axes.titlesize': 15, 'axes.labelsize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})
EV = '/work/DoyaU/vasilache/work/sysid/r6/ev'


def load(tag):
  out = {}
  for f in glob.glob(f'{EV}/{tag}/*.json'):
    e = re.search(r'/(e\d+)@\d+_', f).group(1); out.setdefault(e, {}).update(json.load(open(f)))
  return out


T = {**load('r8f_t'), **load('fl8_t')}; C = {**load('r8f_c'), **load('fl8_c')}
ZT = {**load('r8fz_t'), **load('fl8z_t')}; ZC = load('fl8z_c')
# floor value of each variant; 'fit' = the fitted floor itself (table 0 / carpet 1, with its rolling stiction 7)
VT = [('fm025', -0.25), ('f000', 0.0), ('f025', 0.25), ('f050', 0.5)]
VC = [('f075', 0.75), ('f100', 1.0), ('f120', 1.2)]
ARMS = [('r6_mixed (e8261)', '#000000', ['e8261']), ('round 8 A: floors, from scratch', '#56B4E9', [f'e{n}' for n in range(19300, 19304)]),
        ('round 8 B: floors, warm from e8261', '#0072B2', [f'e{n}' for n in range(19304, 19308)])]
fig, ax = plt.subplots(2, 2, figsize=(20, 9))
for name, col, runs in ARMS:
  for a, key, lab in [(ax[0, 0], 'corr_f', 'forward tracking (corr)'), (ax[0, 1], 'dips_per_min', 'falls per minute')]:
    xs, ys = [], []
    for v, f in VT + VC:
      src = T if (v, f) in VT else C
      vals = [src[e][v][key] for e in runs if e in src and v in src[e]]
      if vals:
        xs.append(f); ys.append((np.mean(vals), np.std(vals)))
    y = np.array(ys); a.errorbar(xs, y[:, 0], y[:, 1], color=col, marker='o', lw=2, capsize=4, label=name)
    fit = [C[e]['fit'][key] for e in runs if e in C and 'fit' in C[e]]
    a.errorbar([1.02], [np.mean(fit)], [np.std(fit)], color=col, marker='s', ms=9, ls='none', capsize=4)
    a.set_xlabel('floor: 0 = fitted table ... 1 = fitted carpet (square: carpet with its rolling stiction)'); a.set_ylabel(lab)
  for a, src, vs, lab in [(ax[1, 0], ZT, VT, 'table-side floors'), (ax[1, 1], ZC, VC, 'carpet-side floors')]:
    xs, ys = [], []
    for v, f in vs:
      vals = [abs(src[e][v]['fwd_mean']) for e in runs if e in src and v in src[e]]
      if vals:
        xs.append(f); ys.append((np.mean(vals), np.std(vals)))
    if ys:
      y = np.array(ys); a.errorbar(xs, y[:, 0], y[:, 1], color=col, marker='o', lw=2, capsize=4, label=name)
    a.set_title(f'stick at zero, {lab}: rolling speed'); a.set_xlabel('floor'); a.set_ylabel('|mean wheel speed| (mm/s)')
ax[0, 0].set_title('tracking the recorded stick'); ax[0, 1].set_title('falls with the recorded stick')
for a in ax.flat:
  a.grid(color='#d5d5d2', lw=0.7)
ax[0, 0].legend(loc='lower left')
plt.tight_layout()
for ext in ('png', 'pdf'):
  plt.savefig(f'{sys.argv[1]}.{ext}', dpi=200, facecolor='white')
