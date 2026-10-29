"""Figure: which goal code suits the UCB count kernel, OG vs OG+SOM vs OG+SOM+LiP.

Reads experiments/goal_geometry/ucb_kernel_geometry.py output: per run, M[l, c, c']
= mean goal dissimilarity (1 - cosine_max) between classes c and c' of block l,
other blocks held at random values. Per run:

  step profile   mean M at class step |c - c'| = d, divided by the mean over
                 all class pairs (1 = no closer than a random other class).
  order          Spearman between |c - c'| and M, per block and class, mean
                 (1 = farther in index is always farther in goal).
  evenness       coefficient of variation of the one-step distances M[l,c,c+1]
                 over classes and blocks (0 = every step the same size, so one
                 kernel width fits everywhere).
  bin purity     mean M between classes in the SAME bin / between DIFFERENT
                 bins, for 4 and 2 contiguous bins per block (UCB B's goal and
                 start axes); lower = a bin groups similar goals.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/ucb_kernel_plot.py
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = '/work/DoyaU/vasilache/work/paper_geom/ucb_kernel'
OUT = '/work/DoyaU/vasilache/work/collapse_figs/ucb_kernel_geometry'
ARMS = [('og', 'OG', '#E69F00'), ('som', 'OG + SOM', '#CC79A7'), ('somlip', 'OG + SOM + LiP', '#0072B2')]
# `ucb_kernel_plot.py someven`: V + SOM against V + SOM + evenness loss (e2000-e2059).
if len(sys.argv) > 1 and sys.argv[1] == 'someven':
  ARMS = [('vsom', 'V + SOM', '#CC79A7'), ('veven', 'V + US-SOM', '#0072B2')]
  OUT += '_someven'
TASKS = [('pinpad_four', 'Pin Pad 4'), ('pinpad_five', 'Pin Pad 5'), ('pinpad_six', 'Pin Pad 6'),
         ('dmc_cartpole_swingup', 'Cartpole'), ('dmc_cheetah_run', 'Cheetah'), ('dmc_hopper_hop', 'Hopper')]
INK, MUTED = '#1c1e22', '#5d636e'


def spearman(x, y):
  rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
  # average ranks for ties in x (class gaps repeat)
  for v in np.unique(x):
    rx[x == v] = rx[x == v].mean()
  return np.corrcoef(rx, ry)[0, 1]


def measures(M):
  L, C, _ = M.shape
  off = ~np.eye(C, dtype=bool)
  ref = M[:, off].mean()
  prof = np.array([np.mean([M[:, c, c + d] for c in range(C - d)]) for d in range(1, C)]) / ref
  gap = np.abs(np.arange(C)[:, None] - np.arange(C)[None])
  rho = [spearman(gap[c][off[c]], M[l, c][off[c]]) for l in range(L) for c in range(C)]
  adj = np.array([[M[l, c, c + 1] for c in range(C - 1)] for l in range(L)])
  even = adj.std() / adj.mean()
  purity = {}
  for bins in (4, 2):
    lab = np.arange(C) * bins // C
    same = (lab[:, None] == lab[None]) & off
    purity[bins] = M[:, same].mean() / M[:, lab[:, None] != lab[None]].mean()
  return prof, np.nanmean(rho), even, purity


def load():
  res = {}
  for arm, *_ in ARMS:
    for f in sorted(glob.glob(f'{R}/{arm}/*.npz')):
      z = np.load(f)
      res.setdefault((arm, str(z['task'])), []).append(measures(z['M']))
  return res


def style(ax, title):
  ax.set_title(title, fontsize=16, color=INK, loc='left')
  ax.tick_params(labelsize=13, colors=MUTED)
  ax.grid(True, color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)
  for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)


def main():
  res = load()
  fig, axes = plt.subplots(2, 3, figsize=(19, 11))
  # (a) step profile, mean over tasks (each the mean over seeds), band = sd over tasks.
  ax = axes[0, 0]
  d = np.arange(1, 8)
  for arm, label, color in ARMS:
    per_task = [np.mean([r[0] for r in res[arm, t]], 0) for t, _ in TASKS if (arm, t) in res]
    if not per_task:
      continue
    p = np.array(per_task)
    ax.fill_between(d, p.mean(0) - p.std(0), p.mean(0) + p.std(0), color=color, alpha=0.15, lw=0)
    ax.plot(d, p.mean(0), color=color, lw=2.6, marker='o', ms=6, label=label)
  ax.axhline(1, color='#8a8880', ls='--', lw=1.2)
  ax.set_xlabel('class step within a block |c - c\'|', fontsize=14, color=INK)
  ax.set_ylabel('goal distance / mean over all class pairs', fontsize=14, color=INK)
  style(ax, '(a) Do neighbouring classes decode\nto nearby goals?')
  ax.legend(fontsize=13, frameon=False, loc='lower right')
  # (b)-(e) per-task dot plots.
  panels = [(axes[0, 1], lambda r: r[1], 'Spearman(class gap, goal distance)',
             '(b) Order: farther index = farther goal?', 1.0),
            (axes[0, 2], lambda r: r[2], 'CV of one-step distances',
             '(c) Evenness: is every class step\nthe same size? (0 = even)', 0.0),
            (axes[1, 0], lambda r: r[3][4], 'within-bin / between-bin distance',
             '(d) Bin purity, 4 bins per block\n(UCB B goal axis; lower = better)', None),
            (axes[1, 1], lambda r: r[3][2], 'within-bin / between-bin distance',
             '(e) Bin purity, 2 bins per block\n(UCB B start axis; lower = better)', None)]
  table = {}
  for ax, get, ylab, title, ideal in panels:
    for i, (arm, label, color) in enumerate(ARMS):
      xs, ys = [], []
      for j, (t, _) in enumerate(TASKS):
        if (arm, t) not in res:
          continue
        v = [get(r) for r in res[arm, t]]
        xs.append(j + (i - (len(ARMS) - 1) / 2) * 0.22); ys.append(np.mean(v))
        ax.scatter(np.full(len(v), j + (i - (len(ARMS) - 1) / 2) * 0.22), v, s=12, color=color, alpha=0.35, lw=0)
      ax.scatter(xs, ys, s=70, color=color, edgecolor='white', lw=1.2, zorder=3, label=label)
      table.setdefault(title.split(')')[0] + ')', {})[arm] = np.mean(ys) if ys else np.nan
    if ideal is not None:
      ax.axhline(ideal, color='#8a8880', ls='--', lw=1.2)
    ax.set_xticks(range(len(TASKS))); ax.set_xticklabels([n for _, n in TASKS], fontsize=12, rotation=20)
    ax.set_ylabel(ylab, fontsize=14, color=INK)
    style(ax, title)
  # (f) where the unevenness is: one-step size by position, and block strength.
  ax = axes[1, 2]
  pos = np.arange(7)
  for arm, label, color in ARMS:
    steps, strength = [], []
    for f in glob.glob(f'{R}/{arm}/*.npz'):
      M = np.load(f)['M']
      adj = np.array([[M[l, c, c + 1] for c in range(7)] for l in range(8)])
      steps.append(adj.mean(0) / adj.mean())
      s = adj.sum(1); strength.append(np.sort(s / s.mean()))
    ax.plot(pos, np.mean(steps, 0), color=color, lw=2.6, marker='o', ms=6, label=label)
    print(f'{label:16s} step by position ' + ' '.join(f'{v:.2f}' for v in np.mean(steps, 0)) +
          '   block length weakest->strongest ' + ' '.join(f'{v:.2f}' for v in np.mean(strength, 0)))
  ax.axhline(1, color='#8a8880', ls='--', lw=1.2)
  ax.set_xticks(pos); ax.set_xticklabels([f'{c}-{c + 1}' for c in pos], fontsize=12)
  ax.set_xlabel('class step (position along the block)', fontsize=14, color=INK)
  ax.set_ylabel('one-step goal distance / its mean', fontsize=14, color=INK)
  style(ax, '(f) Where steps are uneven: the end\nclasses sit close together')
  fig.tight_layout(h_pad=3)
  fig.savefig(OUT + '.png', dpi=200, facecolor='white'); fig.savefig(OUT + '.pdf')
  print(OUT + '.png')
  for arm, label, _ in ARMS:
    per_task = [np.mean([r[0] for r in res[arm, t]], 0) for t, _ in TASKS if (arm, t) in res]
    if per_task:
      print(f'{label:16s} profile ' + ' '.join(f'{v:.2f}' for v in np.mean(per_task, 0)) +
            f'   runs {sum(len(res.get((arm, t), [])) for t, _ in TASKS)}')
  for k, v in table.items():
    print(k, '  '.join(f'{a} {x:.3f}' for a, x in v.items()))


if __name__ == '__main__':
  main()
