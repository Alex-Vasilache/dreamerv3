"""Extra-seed check (array 4807344, h0253-h0312): US-SOM vs TCE sqrt vs TCE-KL c=10 + floor, all seeds.

Pools the original 5 seeds (s0-s4) with the extra ones (s5+) per arm, OG as reference (5 seeds).
Figure (pubseeds_check): top row = mean score per arm (pin pad six, cheetah, hopper);
bottom-left = fraction of pin pad six seeds past the sustained-50 crossing by step x.
Table (pubseeds_table.md): per arm and task, n, solved (sustained-50) count, late 3.5-4M mean.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/pubseeds_report.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import runs, scores, binned, cross50, perm_p  # noqa: E402
from figs import figdir  # noqa: E402

V = 'director-director_og-wkr_ent_fixed-wkr_v3-goal_som_line-goal_even'
ARMS = [('Director (OG)', '#E69F00', ':', ['e12[3-8]?_{t}_director-director_og-pubmatch_s*']),
        ('V + US-SOM (no TCE)', '#000000', '-.', [f'e20[0-5]?_{{t}}_{V}-pubmatch_s*', f'h0???_{{t}}_{V}-pubmatch_s*']),
        ('TCE sqrt bonus (1x16)', '#009E73', '--', [f'e95[0-2]?_{{t}}_{V}-fromto_dir-tce_hash-pubmatch_s*',
                                                    f'h0???_{{t}}_{V}-fromto_dir-tce_hash-pubmatch_s*']),
        ('TCE-KL c=10 + alpha floor', '#D55E00', (0, (6, 2, 1, 2, 1, 2)),
         [f'h0???_{{t}}_{V}-fromto_dir-tce_kl-tce_kl_c10-tce_kl_floor-pubmatch_s*'])]
TASKS = [('pinpad_six', 'Pin Pad Six'), ('dmc_cheetah_run', 'Cheetah Run'), ('dmc_hopper_hop', 'Hopper Hop')]
X = (np.arange(0, 4e6, 1e5) + 5e4) / 1e6


def collect(pats, t):
  out = {}
  for p in pats:
    for d in runs(p.format(t=t)):
      s = scores(d)
      if len(s):
        seed = int(os.path.basename(d).split('_s')[-1].split('_')[0])
        out[seed] = (os.path.basename(d)[:5], s)
  return out


rows = ['| task | arm | seeds done (running) | solved (sustained 50) | late 3.5-4M mean ± sd | p vs US-SOM |', '|---|---|---|---|---|---|']
seedrows = ['| task | arm | seed | run | steps (M) | sustained-50 at (M) | late 3.5-4M |', '|---|---|---|---|---|---|---|']
fig, ax = plt.subplots(2, 3, figsize=(18, 10))
for j, (t, title) in enumerate(TASKS):
  ref = None
  for lab, col, ls, pats in ARMS:
    got = collect(pats, t)
    if not got:
      continue
    done = {k: v for k, v in got.items() if v[1][:, 0].max() >= 3.98e6}
    late = np.array([v[1][v[1][:, 0] >= 3.5e6, 1].mean() for v in done.values()])
    cr = {k: cross50(v[1]) for k, v in got.items()}
    solved = sum(cr[k] is not None for k in done)
    if lab.startswith('V +'):
      ref = late
    p = f'{perm_p(late, ref):.2f}' if ref is not None and not lab.startswith(('V +', 'Director')) and len(late) > 1 else '-'
    rows.append(f'| {title} | {lab} | {len(done)} ({len(got) - len(done)}) | {solved}/{len(done)} | '
                f'{late.mean():.0f} ± {late.std():.0f} | {p} |' if len(late) else f'| {title} | {lab} | 0 ({len(got)}) | - | - | - |')
    for k in sorted(got):
      name, s = got[k]
      lm = s[s[:, 0] >= 3.5e6, 1].mean() if k in done else np.nan
      seedrows.append(f'| {title} | {lab} | {k} | {name} | {s[:, 0].max() / 1e6:.2f} | '
                      f'{"-" if cr[k] is None else f"{cr[k] / 1e6:.2f}"} | {"running" if k not in done else f"{lm:.0f}"} |')
    with np.errstate(all='ignore'):
      m = np.nanmean(np.stack([binned(v[1]) for v in got.values()]), 0)
    ax[0, j].plot(X, m, color=col, ls=ls, lw=2.4, label=f'{lab}')
    if t == 'pinpad_six':
      xs = np.arange(0, 4.01e6, 5e4)
      frac = [np.mean([cr[k] is not None and cr[k] <= x for k in got]) for x in xs]
      ax[1, 0].plot(xs / 1e6, frac, color=col, ls=ls, lw=2.4)
      ax[1, 0].text(3.95, frac[-1] + 0.015, f'n={len(got)}', color=col, fontsize=12, ha='right')
  ax[0, j].set_title(title, fontsize=17)
  ax[0, j].set_ylabel('episode score (mean over seeds)' if j == 0 else '', fontsize=15)
ax[1, 0].set_ylabel('fraction of seeds solved\n(sustained 50)', fontsize=15)
ax[1, 0].set_title('Pin Pad Six: seeds solved by step', fontsize=17)
ax[1, 0].set_ylim(0, 1)
for a in (ax[1, 1], ax[1, 2]):
  a.axis('off')
for a in ax.flat:
  if a.axison:
    a.set_xlim(0, 4); a.tick_params(labelsize=13); a.grid(True, color='#d5d5d2', lw=0.7)
    a.set_xlabel('environment steps (M)', fontsize=15)
    for sp in ('top', 'right'):
      a.spines[sp].set_visible(False)
h, l = ax[0, 0].get_legend_handles_labels()
ax[1, 1].legend(h, l, loc='center left', fontsize=14, frameon=False)
fig.suptitle('Extra seeds: all seeds pooled (s0-s4 original + s5+ from array 4807344)', fontsize=17)
fig.tight_layout()
for ext in ('png', 'pdf'):
  fig.savefig(f'{figdir("tcekl")}/pubseeds_check.{ext}', dpi=200)
print(f'{figdir("tcekl")}/pubseeds_check.png')
open(f'{figdir("tcekl")}/pubseeds_table.md', 'w').write('\n'.join(rows + [''] + seedrows) + '\n')
print('\n'.join(rows)); print(); print('\n'.join(seedrows))
