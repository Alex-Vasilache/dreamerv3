"""Per-seed mechanism report for TCE-KL (h0073-h0078 c=1, h0133-h0162 c=10, h0163-h0192 c=30).

Figure 1 (tcekl_mech_score_gap): pin pad five / pin pad six / cheetah per seed,
  score (top) and gap = E_pi[b] - E_uniform[b] of the log bonus (bottom; > 0 = rare goals preferred).
Figure 2 (tcekl_mech_weight_ent): same runs, TCE weight alpha' (top, log) and manager
  entropy controller scale alpha (bottom, log).
Table (tcekl_mech_table.md): per run, steps, last-0.5M score, first reward, median gap and
  weight before the first reward (pin pads) or over 0-1M (DMC), and over the last 0.5M.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/tcekl_report.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import runs, scores, binned, OUT, TASKS  # noqa: E402
from figs import figdir  # noqa: E402

B = 'director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-fromto_dir'
ARMS = [('c=10', '#0072B2', f'h01??_{{t}}_{B}-tce_kl-tce_kl_c10-pubmatch_s*'),
        ('c=10 floor', '#D55E00', f'h02??_{{t}}_{B}-tce_kl-tce_kl_c10-tce_kl_floor-pubmatch_s*'),
        ('c=10 4x12', '#56B4E9', f'h0[12]??_{{t}}_{B}-tce_kl4-tce_kl_c10-pubmatch_s*')]
LS = ['-', '--', ':', '-.', (0, (1, 1))]
KEYS = ['train/fromto/gap', 'train/fromto/weight', 'train/mgr_actent_skill_scale_mean']
PLOT = [('pinpad_five', 'Pin Pad Five'), ('pinpad_six', 'Pin Pad Six'), ('dmc_cheetah_run', 'Cheetah Run')]


def metrics(d):
  a = []
  for line in open(f'{d}/logdir/metrics.jsonl'):
    if 'fromto/gap' in line:
      r = json.loads(line)
      a.append([r['step']] + [r.get(k, np.nan) for k in KEYS])
  return np.array(a) if a else np.zeros((0, 4))


def smooth(y, k=9):
  if len(y) < k:
    return y
  return np.convolve(np.pad(y, (k // 2, k // 2), mode='edge'), np.ones(k) / k, mode='valid')


def med(a, lo, hi, col):
  m = (a[:, 0] >= lo) & (a[:, 0] < hi)
  return np.nanmedian(a[m, col]) if m.any() else np.nan


def f(v, fmt='{:.2f}'):
  return '-' if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


rows = ['| run | arm | task | seed | steps (M) | last 0.5M score | first reward (M) | gap early | weight early | alpha early | gap late | weight late |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
f1, ax1 = plt.subplots(2, 3, figsize=(18, 9.5), sharex=True)
f2, ax2 = plt.subplots(2, 3, figsize=(18, 9.5), sharex=True)
for t, title, _ in TASKS:
  for arm, col, pat in ARMS:
    for d in runs(pat.format(t=t)):
      name = os.path.basename(d)
      seed = int(name.split('_s')[-1].split('_')[0])
      s, m = scores(d), metrics(d)
      if not len(s) or not len(m):
        continue
      reach = s[:, 0].max()
      last = s[s[:, 0] >= reach - 5e5, 1].mean()
      pin = t.startswith('pinpad')
      first = s[s[:, 1] > 0, 0].min() if pin and (s[:, 1] > 0).any() else None
      cut = (first if first is not None else reach) if pin else 1e6
      rows.append(f'| {name[:5]} | {arm} | {title} | {seed} | {reach / 1e6:.2f} | {last:.0f} | '
                  f'{f(first / 1e6 if first else None)} | {f(med(m, 0, cut, 1))} | {f(med(m, 0, cut, 2), "{:.3f}")} | '
                  f'{f(med(m, 0, cut, 3), "{:.3f}")} | {f(med(m, reach - 5e5, reach + 1, 1))} | {f(med(m, reach - 5e5, reach + 1, 2), "{:.3f}")} |')
      for j, (pt, _) in enumerate(PLOT):
        if pt != t:
          continue
        x = m[:, 0] / 1e6
        kw = dict(color=col, ls=LS[seed % 5], lw=2.2)
        xb = np.arange(0, 4e6, 1e5)
        sb = [s[(s[:, 0] >= lo) & (s[:, 0] < lo + 1e5), 1].mean() if ((s[:, 0] >= lo) & (s[:, 0] < lo + 1e5)).any() else np.nan for lo in xb]
        ax1[0, j].plot((xb + 5e4) / 1e6, sb, label=f'{arm} seed {seed}', **kw)
        ax1[1, j].plot(x, smooth(m[:, 1]), **kw)
        ax2[0, j].plot(x, smooth(m[:, 2]), label=f'{arm} seed {seed}', **kw)
        ax2[1, j].plot(x, smooth(m[:, 3]), **kw)
REFS = [('Director (OG), mean of 5', '#E69F00', ':', 'e12[3-8]?_{t}_director-director_og-pubmatch_s*'),
        ('V + US-SOM (no TCE), mean of 5', '#000000', (0, (1, 1)),
         'e20[0-5]?_{t}_director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even-pubmatch_s*')]
for j, (pt, _) in enumerate(PLOT):
  for lab, col, ls, pat in REFS:
    st = [binned(scores(d)) for d in runs(pat.format(t=pt))]
    if st:
      with np.errstate(all='ignore'):
        ax1[0, j].plot((np.arange(0, 4e6, 1e5) + 5e4) / 1e6, np.nanmean(np.stack(st), 0), color=col, ls=ls, lw=2.0, label=lab)
for fig, axes, ylab, name in ((f1, ax1, ('episode score', 'gap (policy - uniform bonus)'), 'tcekl_mech_score_gap'),
                              (f2, ax2, ("TCE weight alpha'", 'manager entropy scale alpha'), 'tcekl_mech_weight_ent')):
  for j, (_, title) in enumerate(PLOT):
    axes[0, j].set_title(title, fontsize=17)
    axes[1, j].set_xlabel('environment steps (M)', fontsize=15)
    for a in axes[:, j]:
      a.set_xlim(0, 4); a.tick_params(labelsize=13); a.grid(True, color='#d5d5d2', lw=0.7)
      for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
  axes[0, 0].set_ylabel(ylab[0], fontsize=15); axes[1, 0].set_ylabel(ylab[1], fontsize=15)
  if name.endswith('gap'):
    for a in axes[1]:
      a.axhline(0, color='#000000', lw=1)
  else:
    for a in axes.flat:
      a.set_yscale('log')
  h, l = [], []
  for a in axes[0]:
    for hh, ll in zip(*a.get_legend_handles_labels()):
      if ll not in l:
        h.append(hh); l.append(ll)
  fig.set_size_inches(18, 11.5)
  leg = fig.legend(h, l, loc='lower center', ncol=6, fontsize=12, frameon=False)
  fig.canvas.draw()
  lh = leg.get_window_extent().height / fig.bbox.height
  fig.suptitle('TCE-KL per seed (smoothed over 9 log points)', fontsize=17)
  fig.tight_layout(rect=(0, lh + 0.01, 1, 1))
  for ext in ('png', 'pdf'):
    fig.savefig(f'{figdir("tcekl")}/{name}.{ext}', dpi=200)
  print(f'{figdir("tcekl")}/{name}.png')
open(f'{figdir("tcekl")}/tcekl_mech_table.md', 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
