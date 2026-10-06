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
OUT = '/work/DoyaU/vasilache/work/collapse_figs/b64_vs_b32.png'  # + .pdf
TASKS = [('pinpad_four', 'Pin Pad Four', [1071, 1072, 1073], 1101),
         ('pinpad_five', 'Pin Pad Five', [1080, 1081, 1082], 1098),
         ('pinpad_six', 'Pin Pad Six', [1083, 1084, 1085], 1095),
         ('dmc_cartpole_swingup', 'Cartpole swingup', [1086, 1087, 1088], 1104),
         ('dmc_cheetah_run', 'Cheetah run', [1089, 1090, 1091], 1107),
         ('dmc_hopper_hop', 'Hopper hop', [1092, 1093, 1094], 1110)]
# Okabe-Ito, as in the paper's size6m_vs_published figure: orange = before the
# WM fix, green = fix at 4x32, blue = fix at 4x64. Minimum pairwise separation
# under severity-1.0 deuteranopia (Machado 2009) is 18.1 OKLab dE*100.
OLD_C, FIX_C, B64_C = '#E69F00', '#009E73', '#0072B2'
GRID = '#d5d5d2'
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


fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
axes = axes.ravel()
rows = []
for ax, (task, title, b32, b64_0) in zip(axes, TASKS):
  row = [title]
  for label, exps, color in [('Director, before the WM fix', 'base', OLD_C),
                             ('Director, WM fix, batch 4x32', b32, FIX_C),
                             ('Director, WM fix, batch 4x64', [b64_0 + k for k in range(3)], B64_C)]:
    curves, lates, reach = [], [], []
    for d in runs(task, exps):
      a = scores(d)
      if not len(a):
        continue
      x, y = binned(a)
      curves.append(y)
      lates.append(late(a)); reach.append(a[:, 0].max() / 1e6)
    if curves:
      st = np.stack(curves)
      with np.errstate(all='ignore'):
        n_ok = (~np.isnan(st)).sum(0)
        m = np.where(n_ok >= 2, np.nanmean(st, 0), np.nan)
        sd = np.where(n_ok >= 2, np.nanstd(st, 0, ddof=1), np.nan)
      ax.fill_between(x, m - sd, m + sd, color=color, lw=0, alpha=0.15)
      ax.plot(x, m, color=color, lw=2.4, label=label)
    done = [v for v in lates if not np.isnan(v)]
    row.append(f'{np.mean(done):.0f} ± {np.std(done):.0f} (n={len(done)})' if done else
               f'running, at {min(reach):.1f}-{max(reach):.1f}M')
  rows.append(row)
  ax.set_title(title, fontsize=17)
  ax.set_xlim(0, 4); ax.set_ylim(bottom=0)
  ax.tick_params(labelsize=13, direction='out', length=3)
  ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
  for sp in ax.spines.values():
    sp.set_color('#333333'); sp.set_linewidth(0.8)
for ax in (axes[0], axes[3]):
  ax.set_ylabel('episode return', fontsize=15)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.text(0.5, 0.075, 'environment steps (M)', ha='center', fontsize=15)
fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=2.0)
fig.savefig(OUT, dpi=200, facecolor='white')
fig.savefig(OUT.replace('.png', '.pdf'))
print(OUT)
print(f'{"task":18s} | {"baseline 3-4M":20s} | {"fix b32 3-4M":20s} | {"fix b64 3-4M":20s}')
for r in rows:
  print(f'{r[0]:18s} | {r[1]:20s} | {r[2]:20s} | {r[3]:20s}')
