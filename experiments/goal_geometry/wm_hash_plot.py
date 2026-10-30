"""Figure for the start-cell hash study (wm_hash_study.py output).

  /work/DoyaU/vasilache/work/plotenv/bin/python wm_hash_plot.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wm_hash_analyze import candidates, stats

D = '/work/DoyaU/vasilache/work/wm_hash_study'
RUNS = [('e6202_r', 'Pin Pad Six'), ('e6203_r', 'Cartpole Swingup'), ('e6204_r', 'Cheetah Run')]
STY = {'code4 (today)': ('#0072B2', '-', 'o', 'goal code, 4 bins/block (today)'),
       'deter, data-centred': ('#D55E00', '--', 's', 'deter SimHash, data-centred'),
       'deter, per-sample norm': ('#009E73', '-.', '^', 'deter SimHash, per-state norm'),
       'image 16x16 (no drift)': ('#000000', ':', 'D', 'image 16x16 SimHash (no drift)')}
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for j, (run, title) in enumerate(RUNS):
  f = np.load(f'{D}/{run}.npz')
  cks = sorted({k.split('/')[0] for k in f.files if '/' in k} - {'repeat'})
  per = {ck: candidates(f, ck, 16) for ck in cks}
  rep = candidates(f, 'repeat', 16)
  x = np.array([int(c) for c in cks]) / 1e6
  for name, (col, ls, mk, lab) in STY.items():
    if name not in per[cks[0]]:
      continue
    kept = [(per[a][name] == per[b][name]).mean() for a, b in zip(cks[:-1], cks[1:])]
    eff = [stats(per[c][name], None)['eff'] for c in cks]
    floor = (rep[name] == per[cks[-1]][name]).mean()
    axes[0, j].plot(x[:-1], kept, color=col, ls=ls, marker=mk, lw=2.4, ms=8, label=lab)
    axes[0, j].plot([x[-1]], [floor], color=col, marker=mk, ms=12, mfc='white', mew=2.4)
    axes[1, j].plot(x, eff, color=col, ls=ls, marker=mk, lw=2.4, ms=8)
  axes[0, j].set_title(title, fontsize=17)
  axes[0, j].set_ylim(0, 1.05)
  axes[1, j].set_yscale('log')
  for ax in axes[:, j]:
    ax.tick_params(labelsize=13); ax.grid(True, color='#d5d5d2', lw=0.7)
    for sp in ('top', 'right'):
      ax.spines[sp].set_visible(False)
  axes[1, j].set_xlabel('checkpoint (M env steps)', fontsize=15)
axes[0, 0].set_ylabel('share of states keeping\ntheir cell 0.5M later', fontsize=15)
axes[1, 0].set_ylabel('effective cells exp(H)', fontsize=15)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, fontsize=14, frameon=False)
fig.text(0.5, 0.115, 'open marker at 4M: same checkpoint replayed again (noise from the sampled posterior); 16 bits = 65,536 rows',
         ha='center', fontsize=13)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig('/work/DoyaU/vasilache/work/collapse_figs/wm_hash_study.png', dpi=200, facecolor='white')
fig.savefig('/work/DoyaU/vasilache/work/collapse_figs/wm_hash_study.pdf')
print('/work/DoyaU/vasilache/work/collapse_figs/wm_hash_study.png')
