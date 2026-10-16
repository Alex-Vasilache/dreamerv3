import glob, json, os, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
OUT = os.path.dirname(os.path.abspath(__file__))
BW, XMAX, FREEZE = 5e4, 1.2e6, 1.5e5
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
BLUE, GRAY = '#2a78d6', '#9a998f'

def binned(pairs):
    a = np.array(pairs)
    edges = np.arange(0, XMAX + BW, BW)
    idx = np.digitize(a[:, 0], edges) - 1
    ys = np.array([a[idx == i, 1].mean() if (idx == i).any() else np.nan
                   for i in range(len(edges) - 1)])
    return edges[:-1] + BW / 2, ys

runs = collections.defaultdict(list)
for d in sorted(glob.glob(f'{B}/e10[3-5][0-9]_pinpad_four_director-director_og-collapse_*')):
    arm = os.path.basename(d).split('collapse_')[1].rsplit('_s', 1)[0]
    S = [json.loads(l) for l in open(f'{d}/logdir/scores.jsonl')]
    G = []
    for l in open(f'{d}/logdir/metrics.jsonl'):
        r = json.loads(l)
        if 'train/wkr_goal_rew' in r: G.append((r['step'], r['train/wkr_goal_rew']))
    runs[arm].append(dict(name=os.path.basename(d)[:5],
                          score=binned([(s['step'], s['episode/score']) for s in S]),
                          goal=binned(G)))

ARMS = [('ctrl', 'Control (nothing frozen)'),
        ('freeze_worker', 'Worker frozen'),
        ('freeze_manager', 'Manager frozen'),
        ('freeze_manager_model', 'Manager + world model frozen'),
        ('freeze_worker_goal', 'Worker + goal AE frozen'),
        ('freeze_goal', 'Goal AE frozen'),
        ('freeze_goal_model', 'Goal AE + world model frozen'),
        ('freeze_model', 'World model frozen')]

def ctrl_mean(key):
    xs = runs['ctrl'][0][key][0]
    st = np.stack([r[key][1] for r in runs['ctrl']])
    return xs, np.where((~np.isnan(st)).sum(0) >= 2, np.nanmean(st, 0), np.nan)

def figure(key, ylabel, ylim, fname, title):
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharex=True, sharey=True)
    cx, cy = ctrl_mean(key)
    for ax, (arm, label) in zip(axes.flat, ARMS):
        ax.set_facecolor('white')
        ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        ax.axvspan(0, FREEZE / 1e6, color='#f3f2ec', lw=0)
        ax.axvline(FREEZE / 1e6, color=MUTED, lw=1, ls=':')
        if arm != 'ctrl':
            ax.plot(cx / 1e6, cy, color=GRAY, lw=2, ls='--', label='control mean')
        rs = runs[arm]
        for r in rs:
            ax.plot(r[key][0] / 1e6, r[key][1], color=BLUE, lw=1, alpha=0.45)
        st = np.stack([r[key][1] for r in rs])
        m = np.where((~np.isnan(st)).sum(0) >= 2, np.nanmean(st, 0), np.nan)
        ax.plot(rs[0][key][0] / 1e6, m, color=BLUE, lw=2.4, label='mean of seeds (where ≥2 have data)')
        ax.set_title(label, fontsize=11, color=INK, loc='left', fontweight='bold')
        ax.set_xlim(0, XMAX / 1e6); ax.set_ylim(*ylim)
    for ax in axes[1]: ax.set_xlabel('env steps (M)')
    for ax in axes[:, 0]: ax.set_ylabel(ylabel)
    h, l = axes[0, 1].get_legend_handles_labels()
    h2, l2 = axes[0, 1].lines[2:3], ['single seed']
    fig.legend(h + h2, l + l2 + [], loc='upper right', ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(title, x=0.01, ha='left', fontsize=13, color=INK, fontweight='bold')
    fig.text(0.01, 0.935, 'pinpad_four, director_og, size6m. Shaded = before the freeze; '
             'dotted line = freeze at ~150k steps. 50k-step bins. Some frozen-arm seeds were stopped early once their outcome was clear.',
             fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f'{OUT}/{fname}', dpi=130, facecolor='white')
    print(f'{OUT}/{fname}')

figure('score', 'episode score', (0, 300), 'freeze_scores.png',
       'Pinpad score under each freeze')
figure('goal', 'worker goal reward', (-0.05, 0.8), 'freeze_goalrew.png',
       "Worker goal reward under each freeze (the 'jump' that coincided with collapse)")
