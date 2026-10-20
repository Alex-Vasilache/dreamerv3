"""Figure: how the UCB-in-imagination count table fills up (from ucb_fill_extract.py).

Two questions, one chart each.

  How much of the goal space has been tried?  Share of the table's cells by
  bonus band over training (stacked): well tried (bonus < 0.25, n > 15),
  partly tried, near untried (bonus > 0.5, n < 3). Counts decay (half-life
  0.5M steps), so a region left alone drifts back toward "untried".

  Where do the tries go?  One small heatmap per goal-code block: class (SOM
  order, so neighbouring rows are neighbouring goals) against training step,
  shaded by the share of tries on that class relative to an even spread.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/ucb_fill_plot.py e1821
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/work/DoyaU/vasilache/work/collapse_figs/ucb_fill'
EXP = sys.argv[1] if len(sys.argv) > 1 else 'e1821'
LABEL = sys.argv[2] if len(sys.argv) > 2 else 'arm A, Pin Pad Five'
z = np.load(os.path.join(D, f'{EXP}.npz'))
steps = sorted({int(k.split('_')[0]) for k in z.files if k.endswith('_bands')})
x = np.array(steps) / 1e6

# One hue, light -> dark = untried -> well tried.
BAND_COLORS = ['#1c5cab', '#6da7ec', '#e8eef7']
BAND_NAMES = ['well tried  (n > 15)', 'partly tried', 'near untried  (n < 3)']
INK, MUTED, GRID = '#1c1e22', '#5d636e', '#d5d5d2'


def style(ax):
  ax.tick_params(labelsize=13, direction='out', length=3, colors=MUTED)
  for sp in ax.spines.values():
    sp.set_color('#999999'); sp.set_linewidth(0.8)


fig = plt.figure(figsize=(15, 11.5))
outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.35], hspace=0.42)

# --- how much has been tried ---
ax = fig.add_subplot(outer[0])
bands = np.stack([z[f'{s}_bands'] for s in steps], 1) * 100      # (3, T) percent
ys = np.cumsum(bands, 0)
lo = np.zeros_like(x)
for i in range(3):
  ax.fill_between(x, lo, ys[i], color=BAND_COLORS[i], lw=0)
  lo = ys[i]
for xi in x:
  ax.axvline(xi, color='white', lw=0.6, alpha=0.6)
# Direct labels at the right edge, centred in each band.
mids = (np.concatenate([[0], ys[:-1, -1]]) + ys[:, -1]) / 2
for i, m in enumerate(mids):
  ax.text(x[-1] + 0.06, m, f'{BAND_NAMES[i]}\n{bands[i, -1]:.0f}%', va='center', fontsize=13,
          color=INK)
ax.text(x[0] + 0.08, 100 - bands[2, 0] / 2 + 10,
        f'at {x[0]:g}M: {bands[2, 0]:.0f}% near untried', fontsize=13, color=INK)
ax.text(x[0] + 0.08, 100 - bands[2, 0] / 2 - 2,
        'counts decay (half-life 0.5M steps), so regions left alone\nslowly turn back to untried',
        fontsize=12, color=MUTED, va='top')
ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, 100)
ax.set_ylabel('share of goal cells (%)', fontsize=15, color=INK)
ax.set_xlabel('environment steps (M)', fontsize=15, color=INK)
ax.set_title('How much of the goal space the manager has tried in imagination', fontsize=17,
             color=INK, loc='left')
style(ax)

# --- where the tries go ---
inner = outer[1].subgridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.06],
                             height_ratios=[0.16, 1, 1], wspace=0.15, hspace=0.5)
marg = np.stack([z[f'{s}_block_marg'] for s in steps], -1)        # (8, C, T)
rel = marg * marg.shape[1]                                        # 1 = even spread
vmax = float(np.ceil(rel.max() * 2) / 2)
for b in range(8):
  ax = fig.add_subplot(inner[1 + b // 4, b % 4])
  im = ax.imshow(rel[b], origin='lower', aspect='auto', cmap='Blues', vmin=0, vmax=vmax,
                 extent=(x[0] - 0.25, x[-1] + 0.25, -0.5, rel.shape[1] - 0.5),
                 interpolation='nearest')
  ax.set_title(f'block {b}', fontsize=15, color=INK)
  ax.set_yticks(range(rel.shape[1]))
  ax.set_xticks([1, 2, 3, 4])
  if b % 4:
    ax.set_yticklabels([])
  else:
    ax.set_ylabel('class (SOM order)', fontsize=13, color=INK)
  if b < 4:
    ax.set_xticklabels([])
  style(ax)
fig.text(0.47, 0.035, 'environment steps (M)', ha='center', fontsize=15, color=INK)
cax = fig.add_subplot(inner[1:, 4])
cb = fig.colorbar(im, cax=cax)
cb.set_label('share of tries vs. even spread (1 = even)', fontsize=13, color=INK)
cb.ax.tick_params(labelsize=12, colors=MUTED)
# Spread of each block's tries over its classes: normalized entropy, 1 = even.
ent = -(marg * np.log(np.maximum(marg, 1e-12))).sum(1) / np.log(marg.shape[1])   # (8, T)
tax = fig.add_subplot(inner[0, :4]); tax.axis('off')
tax.text(0, 0.5, 'Where the tries go: early on each block favours a band of neighbouring classes '
         f'(up to {rel[..., 0].max():.1f}x an even share);\nby {x[-1]:g}M the band has flattened and moved '
         f'(peak {rel[..., -1].max():.1f}x; evenness {ent[:, 0].mean():.2f} -> {ent[:, -1].mean():.2f}, 1 = even)',
         fontsize=16, color=INK, va='center')
fig.suptitle(f'UCB in imagination (c=3), {EXP} ({LABEL})', fontsize=14, color=MUTED, x=0.125,
             ha='left', y=0.985)
out = os.path.join(os.path.dirname(D), f'ucb_fill_{EXP}')
fig.savefig(out + '.png', dpi=200, facecolor='white', bbox_inches='tight')
fig.savefig(out + '.pdf', bbox_inches='tight')
print(out + '.png')
print('evenness per block, first -> last:', np.round(ent[:, 0], 2), np.round(ent[:, -1], 2))
for b in range(8):
  print(f'block {b}: mass on classes 3-4 {marg[b, 3:5, 0].sum():.2f} -> {marg[b, 3:5, -1].sum():.2f}, '
        f'on 0/7 {marg[b, [0, 7], 0].sum():.2f} -> {marg[b, [0, 7], -1].sum():.2f}')
