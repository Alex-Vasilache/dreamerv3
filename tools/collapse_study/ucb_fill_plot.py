"""Figure: how the UCB-in-imagination count table fills up (from ucb_fill_extract.py).

Top row: the bonus 1/sqrt(n+1) over the whole 8^8 goal grid at four milestones of
one arm-A run. Rows = blocks 0-3, columns = blocks 4-7, each lexicographic, so a
pixel is a pooled 8x8 patch; dark = tried a lot in imagination lately (low
bonus), white = untried (bonus ~1). Bottom row: where the mass sits on two
blocks (0 and 1, the other six summed out), and the share of cells still near
untried over training for arm A and arm B.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/ucb_fill_plot.py e1821 e1853
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

D = '/work/DoyaU/vasilache/work/collapse_figs/ucb_fill'
a_exp, b_exp = (sys.argv[1:3] + ['e1821', 'e1853'])[:2] if len(sys.argv) > 1 else ('e1821', 'e1853')
A = np.load(os.path.join(D, f'{a_exp}.npz'))
Bz = np.load(os.path.join(D, f'{b_exp}.npz'))
steps = lambda z: sorted({int(k.split('_')[0]) for k in z.files})

fig = plt.figure(figsize=(17, 9.6))
outer = fig.add_gridspec(2, 1, hspace=0.32)
gs = outer[0].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.12)
gb = outer[1].subgridspec(1, 6, width_ratios=[1, 1, 1, 0.05, 0.5, 1.15], wspace=0.12)
show = [s for s in (500000, 1000000, 2000000, 4000000) if s in steps(A)]
for i, s in enumerate(show):
  ax = fig.add_subplot(gs[0, i])
  im = ax.imshow(A[f'{s}_bonus_map'], cmap='Blues_r', vmin=0, vmax=1, interpolation='nearest')
  n = A[f'{s}_bonus_map'].shape[0]
  for k in range(1, 8):          # block-0 / block-4 boundaries
    ax.axhline(k * n / 8 - 0.5, color='#d5d5d2', lw=0.4); ax.axvline(k * n / 8 - 0.5, color='#d5d5d2', lw=0.4)
  ax.set_title(f'{s / 1e6:g}M steps', fontsize=17)
  ax.set_xticks([]); ax.set_yticks([])
  if i == 0:
    ax.set_ylabel('blocks 0-3 (class order)', fontsize=15)
  ax.set_xlabel('blocks 4-7 (class order)', fontsize=15)
cax = fig.add_subplot(gs[0, 4])
cb = fig.colorbar(im, cax=cax); cb.set_label('UCB bonus  1/sqrt(n+1)', fontsize=14); cb.ax.tick_params(labelsize=12)

pair_steps = [s for s in (500000, 2000000, 4000000) if s in steps(A)]
# Shown relative to an even spread (1/64 per cell): 1 = average, >1 = favoured.
vmax = max((A[f'{s}_pair01'] * 64).max() for s in pair_steps)
for i, s in enumerate(pair_steps):
  ax = fig.add_subplot(gb[0, i])
  im2 = ax.imshow(A[f'{s}_pair01'] * 64, cmap='Blues', vmin=0, vmax=vmax, interpolation='nearest')
  ax.set_title(f'blocks 0 x 1, {s / 1e6:g}M', fontsize=17)
  ax.set_xlabel('block 1 class', fontsize=15)
  if i == 0:
    ax.set_ylabel('block 0 class', fontsize=15)
  ax.set_xticks(range(8)); ax.set_yticks(range(8)); ax.tick_params(labelsize=12)
cax2 = fig.add_subplot(gb[0, 3])
cb2 = fig.colorbar(im2, cax=cax2); cb2.set_label('count mass vs even spread', fontsize=14); cb2.ax.tick_params(labelsize=12)
ax = fig.add_subplot(gb[0, 5])
for z, lab, c in ((A, f'arm A ({a_exp}, pin pad five)', '#0072B2'), (Bz, f'arm B ({b_exp}, cartpole)', '#E69F00')):
  st = steps(z)
  x = np.array(st) / 1e6
  ax.plot(x, [z[f'{s}_frac_hi'][0] for s in st], color=c, lw=2.4, marker='o', label=lab)
ax.set_ylim(0, 1); ax.set_xlim(0, 4.1)
ax.set_title('cells still near untried\n(bonus > 0.5, n < 3)', fontsize=15)
ax.set_xlabel('environment steps (M)', fontsize=15); ax.set_ylabel('share of cells', fontsize=15)
ax.tick_params(labelsize=12); ax.grid(True, color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
ax.legend(fontsize=12, frameon=False, loc='lower left')
fig.suptitle(f'UCB in imagination (c=3): how the decaying count table fills up, {a_exp} (arm A, pin pad five)',
             fontsize=17, y=0.995)
out = os.path.join(os.path.dirname(D), 'ucb_fill')
fig.savefig(out + '.png', dpi=200, facecolor='white', bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight')
print(out + '.png')
