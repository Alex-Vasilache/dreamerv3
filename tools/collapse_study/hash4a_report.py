"""Per-seed report for the 4x12 WM hash TCE arms.

  hash4a  adaptive w (e19100-e19130, array 4795008); bottom row = TCE weight
  hash4e  smoothed w + floor 0.3, US-SOM (h0001-h0036) and plain SOM (h0037-h0072);
          bottom row = lean (w sits on the floor most of the time)

Table: per run, steps reached, late 3-4M mean, first reward and sustained-50
crossing (pin pads), median TCE weight and lean before the first reward and late.
Figure: pin pad five / six per seed, score (top) and TCE weight or lean (bottom).

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/hash4a_report.py [hash4a|hash4e]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import runs, scores, binned, cross50, OUT, TASKS  # noqa: E402
from figs import figdir  # noqa: E402

B = 'director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line'
SETS = {
    'hash4a': ('TCE, 4x12 WM hash, adaptive w', 1, 'TCE weight w (log)',
               [('', '-', f'e191[0-2]?_{{t}}_{B}-goal_even-fromto_dir-tce_hash4a-pubmatch_s*')]),
    'hash4e': ('TCE, 4x12 WM hash, smoothed w, floor 0.3', 2, 'lean = policy / random novelty',
               [('US-SOM', '-', f'h00[0-3]?_{{t}}_{B}-goal_even-fromto_dir-tce_hash4e-pubmatch_s*'),
                ('plain SOM', '--', f'h00[3-7]?_{{t}}_{B}-fromto_dir-tce_hash4e-pubmatch_s*')]),
}
WHICH = sys.argv[1] if len(sys.argv) > 1 else 'hash4a'
TITLE, ROW2, YLAB, ARMS = SETS[WHICH]
COLS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#000000']
LS = ['-', '--', '-.', ':', (0, (5, 1, 1, 1)), (0, (1, 1))]


def fromto(d):
  a = []
  for line in open(f'{d}/logdir/metrics.jsonl'):
    if 'fromto/weight' in line:
      r = json.loads(line)
      a.append((r['step'], r['train/fromto/weight'], r.get('train/fromto/lean', np.nan)))
  return np.array(a) if a else np.zeros((0, 3))


def med(a, lo, hi, col):
  m = (a[:, 0] >= lo) & (a[:, 0] < hi)
  return np.nanmedian(a[m, col]) if m.any() else np.nan


def f(v, fmt='{:.2f}'):
  return '-' if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


rows = ['| run | arm | task | seed | steps (M) | late 3-4M | first reward (M) | sustained-50 (M) | w before reward | lean before reward | w late | lean late |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True)
for t, title, _ in TASKS:
 for arm, als, pat in ARMS:
  for i, d in enumerate(runs(pat.format(t=t))):
    name = os.path.basename(d)
    seed = name.split('_s')[-1].split('_')[0]
    s, ft = scores(d), fromto(d)
    reach = s[:, 0].max() if len(s) else 0
    late = s[s[:, 0] >= 3e6, 1].mean() if reach >= 3.98e6 else np.nan
    pin = t.startswith('pinpad')
    first = s[s[:, 1] > 0, 0].min() if pin and (s[:, 1] > 0).any() else None
    c50 = cross50(s) if pin else None
    cut = first if first is not None else reach
    rows.append(f'| {name[:6]} | {arm or "-"} | {title} | {seed} | {reach / 1e6:.2f} | {f(late, "{:.0f}")} | '
                f'{f(first / 1e6 if first else None)} | {f(c50 / 1e6 if c50 else None)} | '
                f'{f(med(ft, 0, cut, 1))} | {f(med(ft, 0, cut, 2), "{:.3f}")} | '
                f'{f(med(ft, 3e6, 4e6, 1))} | {f(med(ft, 3e6, 4e6, 2), "{:.3f}")} |')
    if t in ('pinpad_five', 'pinpad_six'):
      j = 0 if t == 'pinpad_five' else 1
      x = (np.arange(0, 4e6, 1e5) + 5e4) / 1e6
      ls = LS[i % 6] if len(ARMS) == 1 else als
      axes[0, j].plot(x, binned(s), color=COLS[i % 6], ls=ls, lw=2.4, label=f'{arm} seed {seed}'.strip())
      if len(ft):
        axes[1, j].plot(ft[:, 0] / 1e6, ft[:, ROW2], color=COLS[i % 6], ls=ls, lw=1.8)
      axes[0, j].set_title(title, fontsize=17)
for j in range(2):
  axes[1, j].set_xlabel('environment steps (M)', fontsize=15)
  for ax in axes[:, j]:
    ax.set_xlim(0, 4); ax.tick_params(labelsize=13); ax.grid(True, color='#d5d5d2', lw=0.7)
    for sp in ('top', 'right'):
      ax.spines[sp].set_visible(False)
axes[0, 0].set_ylabel('episode score', fontsize=15)
axes[1, 0].set_ylabel(YLAB, fontsize=15)
for ax in axes[1]:
  if ROW2 == 1:
    ax.set_yscale('log')
  else:
    ax.axhline(1.0, color='#8a8880', lw=1, ls='--')
axes[0, 1].legend(fontsize=14, frameon=False)
fig.suptitle(f'{TITLE}: per seed', fontsize=17)
fig.tight_layout()
for ext in ('png', 'pdf'):
  fig.savefig(f'{figdir(WHICH)}/{WHICH}_seeds.{ext}', dpi=200)
open(f'{figdir(WHICH)}/{WHICH}_table.md', 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
print(f'{figdir(WHICH)}/{WHICH}_seeds.png')
