"""Reset study figure: score after re-initialising one component of a collapsed run."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = '/work/DoyaU/vasilache/work'
HERE = os.path.dirname(os.path.abspath(__file__))
BW = 2.5e4
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
BLUE, GRAY = '#2a78d6', '#8a897f'
ARMS = [('reset_none', 'Nothing reset (control)', ['e1060', 'e1061']),
        ('reset_mgr', 'Manager reset', ['e1062', 'e1063', 'e1067']),
        ('reset_goal', 'Goal autoencoder reset', ['e1064', 'e1065']),
        ('reset_wkr', 'Worker reset', ['e1066'])]
SRC = {'e1060': 'e1057', 'e1062': 'e1057', 'e1064': 'e1057', 'e1066': 'e1057',
       'e1061': 'e1058', 'e1063': 'e1058', 'e1065': 'e1058', 'e1067': 'e1059'}


def resumed_scores(e):
    d = glob.glob(f'{W}/{e}_*')[0]
    start = int(open(f'{d}/logdir/reset_done').read().split('actions')[1].split(',')[0].strip(" ':")) * 8 \
        if os.path.exists(f'{d}/logdir/reset_done') and 'actions' in open(f'{d}/logdir/reset_done').read() else None
    rows = [json.loads(l) for l in open(f'{d}/logdir/scores.jsonl')]
    if start is None:   # control: resume step = checkpoint step of its source
        cut = max([i for i in range(1, len(rows)) if rows[i]['step'] < rows[i - 1]['step']] or [0])
        start = rows[cut]['step']
    new = [(r['step'] - start, r['episode/score']) for r in rows if r['step'] >= start]
    # drop copied pre-resume episodes that happen to lie beyond the resume step
    seen, out = set(), []
    for i, r in enumerate(rows):
        pass
    return start, new


def source_curve(src):
    d = glob.glob(f'/bucket/DoyaU/vasilache/bucket/results/dreamerv3/{src}_*')[0]
    rows = [json.loads(l) for l in open(f'{d}/logdir/scores.jsonl')]
    return [(r['step'], r['episode/score']) for r in rows]


def binned(pairs, lo, hi):
    a = np.array(pairs, float)
    edges = np.arange(lo, hi + BW, BW)
    idx = np.digitize(a[:, 0], edges) - 1
    return edges[:-1] + BW / 2, np.array(
        [a[idx == i, 1].mean() if (idx == i).any() else np.nan for i in range(len(edges) - 1)])


plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED})
fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), sharey=True)
for ax, (arm, title, exps) in zip(axes, ARMS):
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for i, e in enumerate(exps):
        d = glob.glob(f'{W}/{e}_*')[0]
        rd = f'{d}/logdir/reset_done'
        # resume step: first metrics row written after the restart
        rows = [json.loads(l) for l in open(f'{d}/logdir/metrics.jsonl')]
        cut = max([j for j in range(1, len(rows)) if rows[j]['step'] < rows[j - 1]['step']] or [0])
        start = rows[cut]['step']
        sc = [json.loads(l) for l in open(f'{d}/logdir/scores.jsonl')]
        scut = max([j for j in range(1, len(sc)) if sc[j]['step'] < sc[j - 1]['step']] or [0])
        new = [(r['step'], r['episode/score']) for r in sc[scut:] if r['step'] >= start]
        old = source_curve(SRC[e])
        xo, yo = binned(old, 0, start)
        xn, yn = binned(new, start, 1.62e6)
        ax.plot(xo / 1e6, yo, color=GRAY, lw=1.2, alpha=0.7,
                label='before reset (source run)' if i == 0 else None)
        ax.plot(xn / 1e6, yn, color=BLUE, lw=2, alpha=0.9 - 0.2 * i,
                label='after reset' if i == 0 else None)
        ax.axvline(start / 1e6, color=MUTED, ls=':', lw=1)
    ax.set_title(f'{title} ({len(exps)} seed{"s" if len(exps) > 1 else ""})',
                 fontsize=11, color=INK, loc='left', fontweight='bold')
    ax.set_xlabel('env steps (M)'); ax.set_xlim(0, 1.62); ax.set_ylim(0, 300)
axes[0].set_ylabel('episode score (25k-step bins)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper right', ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.995, 0.86))
fig.suptitle('Resetting one component of a collapsed run (pinpad_four, director_og, size6m)',
             x=0.01, ha='left', fontsize=13, color=INK, fontweight='bold')
fig.text(0.01, 0.905, 'Gray: the original run (found, then lost the sequence). Dotted line: '
         'restart at ~1.17M with the component re-initialised. Blue: after the restart.',
         fontsize=9.5, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.80))
fig.savefig(f'{HERE}/reset_scores.png', dpi=130, facecolor='white')
print(f'{HERE}/reset_scores.png')
