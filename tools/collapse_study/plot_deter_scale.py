"""Plot tools/deter_scale.py output: world-model state size vs goal-AE fit."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(f'{HERE}/deter_scale.json'))
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
BLUE, GRAY = '#2a78d6', '#8a897f'
FROZEN = 'e1048'

runs = {}
for r in rows:
    runs.setdefault(r['run'], []).append(r)

PANELS = [
    ('norm', 'The world-model state keeps growing',
     'mean size of the state vector |deter|', None),
    ('sse', 'So the logged goal-AE error rises...',
     'reconstruction error (sum of squares)', None),
    ('unexpl', '...but its fit, relative to the state, does not',
     'share of state variance the AE misses', (0, 1)),
    ('corr', 'Which dimensions are active keeps changing',
     'similarity of per-dimension spread\nto the 0.5M checkpoint', (-0.1, 1.05)),
]

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED})
fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
for ax, (key, title, ylabel, ylim) in zip(axes, PANELS):
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.axvspan(0.3, 0.6, color='#f6e7e1', lw=0)
    first = True
    for name, rs in sorted(runs.items()):
        x = np.array([r['step'] for r in rs]) / 1e6
        y = np.array([r[key] for r in rs])
        if name == FROZEN:
            # Its world model is the 0.15M state: one point at 0.15M. Panel 4 is
            # a self-similarity, trivially 1.0 for a frozen model, so skip it.
            if key != 'corr':
                ax.plot([0.15], [y[0]], color=GRAY, marker='D', ms=7, ls='none',
                        label='state at 0.15M (from the run frozen there, e1048)')
        else:
            ax.plot(x, y, color=BLUE, lw=1.6, marker='o', ms=4, alpha=0.85,
                    label='normal training (4 runs)' if first else None)
            first = False
    ax.set_title(title, fontsize=11, color=INK, loc='left', fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('env steps (M)')
    ax.set_xlim(0, 4.2)
    if ylim: ax.set_ylim(*ylim)
for ax in axes:
    lo, hi = ax.get_ylim()
    ax.text(0.45, hi - 0.02 * (hi - lo), 'score\ncollapses', ha='center', va='top',
            fontsize=8.5, color='#b4532f')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper right', ncol=2, frameon=False, fontsize=10,
           bbox_to_anchor=(0.995, 0.86))
fig.suptitle('Goal autoencoder vs the changing world-model state (pinpad_four, director_og, size6m)',
             x=0.01, ha='left', fontsize=13, color=INK, fontweight='bold')
fig.text(0.01, 0.905, 'Each point: one milestone checkpoint, measured on the same 8,192 '
         'probe states. Shaded: where the score collapses in these runs.',
         fontsize=9.5, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.80))
fig.savefig(f'{HERE}/deter_scale.png', dpi=130, facecolor='white')
print(f'{HERE}/deter_scale.png')
