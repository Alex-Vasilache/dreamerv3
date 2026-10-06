"""Batch length 64 vs 32 with the world-model fix, against the director_og baselines.

All director_og, size6m, 4M steps, 3 seeds:
  baseline  e919-e936            (old code: critic gradient into the WM)
  fix, b32  e1071-73, e1080-94   (hrl_repval_grad False)
  fix, b64  e1095-e1112          (same + batch length 64, lr unchanged)
Reads /work first, then the bucket, so it works while runs are still going.
Writes collapse_figs/b64_vs_b32.png and prints a per-task table.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = '/work/DoyaU/vasilache/work'
B = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
OUT = '/work/DoyaU/vasilache/work/collapse_figs/b64_vs_b32.png'
TASKS = [('pinpad_four', 'Pin Pad Four', [1071, 1072, 1073], 1101),
         ('pinpad_five', 'Pin Pad Five', [1080, 1081, 1082], 1098),
         ('pinpad_six', 'Pin Pad Six', [1083, 1084, 1085], 1095),
         ('dmc_cartpole_swingup', 'Cartpole swingup', [1086, 1087, 1088], 1104),
         ('dmc_cheetah_run', 'Cheetah run', [1089, 1090, 1091], 1107),
         ('dmc_hopper_hop', 'Hopper hop', [1092, 1093, 1094], 1110)]
GRAY, BLUE, ORANGE = '#9a998f', '#2a78d6', '#d9711c'
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
BW, XMAX = 1e5, 4e6


def find(e):
  for root in (W, B):
    ds = sorted(glob.glob(f'{root}/{e}_*'))
    if ds:
      return ds[0]


def scores(d):
  f = f'{d}/logdir/scores.jsonl'
  return np.array([(r['step'], r['episode/score']) for r in map(json.loads, open(f))]) if os.path.exists(f) else np.zeros((0, 2))


def binned(a):
  edges = np.arange(0, XMAX + BW, BW)
  idx = np.digitize(a[:, 0], edges) - 1 if len(a) else np.array([], int)
  return (edges[:-1] + BW / 2) / 1e6, np.array(
      [a[idx == i, 1].mean() if (idx == i).any() else np.nan for i in range(len(edges) - 1)])


def runs(task, exps):
  if exps == 'base':
    return sorted(glob.glob(f'{B}/e9[1-3][0-9]_{task}_director-director_og_s*'))
  return [d for d in (find(f'e{n}') for n in exps) if d]


def late(a, lo=3e6, hi=4e6):
  w = a[(a[:, 0] >= lo) & (a[:, 0] < hi), 1]
  return w.mean() if len(w) > 20 else np.nan


plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED})
fig, axes = plt.subplots(1, 6, figsize=(26, 4.4))
rows = []
for ax, (task, title, b32, b64_0) in zip(axes, TASKS):
  ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
  for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
  row = [title]
  for label, exps, color in [('director_og baseline', 'base', GRAY),
                             ('fix, batch length 32', b32, BLUE),
                             ('fix, batch length 64', [b64_0 + k for k in range(3)], ORANGE)]:
    ds = runs(task, exps)
    curves, lates, reach = [], [], []
    for d in ds:
      a = scores(d)
      if not len(a):
        continue
      x, y = binned(a)
      curves.append(y)
      ax.plot(x, y, color=color, lw=0.7, alpha=0.35)
      lates.append(late(a)); reach.append(a[:, 0].max() / 1e6)
    if curves:
      st = np.stack(curves)
      with np.errstate(all='ignore'):
        m = np.where((~np.isnan(st)).sum(0) >= 2, np.nanmean(st, 0), np.nan)
      ax.plot(x, m, color=color, lw=2.2, label=label)
    done = [v for v in lates if not np.isnan(v)]
    row.append(f'{np.mean(done):.0f} ± {np.std(done):.0f} (n={len(done)})' if done else
               f'running, at {min(reach):.1f}-{max(reach):.1f}M')
  rows.append(row)
  ax.set_title(title, fontsize=11, color=INK, loc='left', fontweight='bold')
  ax.set_xlabel('env steps (M)'); ax.set_xlim(0, 4); ax.set_ylim(bottom=0)
axes[0].set_ylabel('episode score (100k-step bins)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper right', ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.995, 1.0))
fig.suptitle('Batch length 64 vs 32 with the world-model fix (director_og, size6m, 3 seeds, 4M)',
             x=0.01, ha='left', fontsize=13, color=INK, fontweight='bold')
fig.text(0.01, 0.905, 'Thick: mean where at least 2 seeds have data. Thin: single seeds. '
         'Batch-64 runs still in progress end early.', fontsize=9.5, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(OUT, dpi=120, facecolor='white')
print(OUT)
print(f'{"task":18s} | {"baseline 3-4M":20s} | {"fix b32 3-4M":20s} | {"fix b64 3-4M":20s}')
for r in rows:
  print(f'{r[0]:18s} | {r[1]:20s} | {r[2]:20s} | {r[3]:20s}')
