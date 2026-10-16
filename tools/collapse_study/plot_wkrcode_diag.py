"""Code-input worker (e1054-56) vs matched control (e1057-59): five diagnostics."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
OUT = os.environ.get('OUT', os.path.dirname(os.path.abspath(__file__)))
BW, XMAX = 5e4, 1.2e6
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
ARMS = [('collapse_ctrl12', 'Control (decoded goal)', '#8a897f'),
        ('collapse_wkr_code', 'Worker sees goal code', '#2a78d6')]
PANELS = [
    ('score', 'Episode score', None),
    ('epstats/reward_rate', 'Real task reward per step', None),
    ('train/mgr_extr_rew', 'Imagined task reward (mgr_extr_rew)', None),
    ('train/mgr_extr_ret', 'Imagined task return (mgr_extr_ret)', None),
    ('train/wkr_goal_rew', 'Worker goal reward', None),
    ('train/mgr_ent_norm_skill_mean', 'Manager normalized entropy', None),
    ('train/goal/rec_mean', 'Goal autoencoder error', None),
]

def binned(pairs):
    a = np.array(pairs, float)
    edges = np.arange(0, XMAX + BW, BW)
    idx = np.digitize(a[:, 0], edges) - 1
    return edges[:-1] + BW / 2, np.array(
        [a[idx == i, 1].mean() if (idx == i).any() else np.nan
         for i in range(len(edges) - 1)])

data = {}
for tag, _, _ in ARMS:
    runs = []
    for d in sorted(glob.glob(f'{B}/e105[4-9]_pinpad_four_director-director_og-{tag}_s*')):
        series = {'score': [(json.loads(l)['step'], json.loads(l)['episode/score'])
                            for l in open(f'{d}/logdir/scores.jsonl')]}
        for l in open(f'{d}/logdir/metrics.jsonl'):
            r = json.loads(l)
            for key, _, _ in PANELS[1:]:
                if key in r:
                    series.setdefault(key, []).append((r['step'], r[key]))
        runs.append({k: binned(v) for k, v in series.items()})
    data[tag] = runs

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED})
fig, axes = plt.subplots(2, 4, figsize=(19, 7.4), sharex=True)
axes.flat[-1].axis('off')
for ax, (key, title, _) in zip(axes.flat, PANELS):
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for tag, label, color in ARMS:
        runs = [r for r in data[tag] if key in r]
        for r in runs:
            ax.plot(r[key][0] / 1e6, r[key][1], color=color, lw=0.9, alpha=0.35)
        st = np.stack([r[key][1] for r in runs])
        m = np.where((~np.isnan(st)).sum(0) >= 2, np.nanmean(st, 0), np.nan)
        ax.plot(runs[0][key][0] / 1e6, m, color=color, lw=2.4, label=label)
    ax.set_title(title, fontsize=11, color=INK, loc='left', fontweight='bold')
    ax.set_xlim(0, XMAX / 1e6)
for ax in list(axes[1][:3]) + [axes[0, 3]]: ax.set_xlabel('env steps (M)')
axes[0, 3].tick_params(labelbottom=True)
axes[0, 0].set_ylim(0, 310)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper right', ncol=2, frameon=False, fontsize=10,
           bbox_to_anchor=(0.995, 0.925))
fig.suptitle('Worker conditioned on the goal code vs the decoded goal',
             x=0.01, ha='left', fontsize=13, color=INK, fontweight='bold')
fig.text(0.01, 0.935, 'pinpad_four, director_og, size6m; e1054-56 (code) vs e1057-59 '
         '(control, same commit). Thick = mean of 3 seeds, thin = single seeds. 50k-step bins.',
         fontsize=9.5, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(f'{OUT}/wkrcode_diagnostics.png', dpi=130, facecolor='white')
print(f'{OUT}/wkrcode_diagnostics.png')
