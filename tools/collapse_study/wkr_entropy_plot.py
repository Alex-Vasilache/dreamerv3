"""Worker entropy (nats, train/wkr_ent/action) per task over training, for OG, OG +
fixed worker entropy, and the worker-v3 arms R and V. Mean over the seeds present.

  /work/DoyaU/vasilache/work/plotenv/bin/python tools/collapse_study/wkr_entropy_plot.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_late import SETS, TASKS, runs, metric_curve, style, finish, OUT

arms = SETS['wkrv3']
fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), sharex=True)
xd = (np.arange(0, 4e6, 2e5) + 1e5) / 1e6
for ax, (task, title, _) in zip(axes.ravel(), TASKS):
  for key, label, color, pat in arms:
    cs = [c for c in (metric_curve(d, 'train/wkr_ent/action') for d in runs(pat.format(t=task))) if c is not None]
    if not cs:
      continue
    with np.errstate(all='ignore'):
      m = np.nanmean(np.stack(cs), 0)
    ax.plot(xd, m, color=color, lw=2.4, ls='-' if len(cs) >= 2 else '--',
            label=label if len(cs) >= 5 else f'{label} (n={len(cs)})')
  ax.set_xlim(0, 4)
  style(ax, title)
finish(fig, axes, 'environment steps (M)', 'worker entropy (nats)', 2, f'{OUT}/wkrv3_wkrent')
