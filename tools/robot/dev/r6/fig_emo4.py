"""Round EA/EB figure: per minute, mean +- SD over seeds, per arm.

  python fig_emo4.py STATS.json OUT.png [arm ...]   (STATS from emostats.py)
"""

import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Okabe-Ito
COLORS = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2',
          '#D55E00', '#CC79A7']
ROWS = [('up', 'time upright'), ('cross', 'bumper crossings / min'),
        ('fall_per_push', 'falls per upright push')]


def main():
  stats = json.load(open(sys.argv[1]))
  arms = sys.argv[3:] or list(stats)
  # Without pushes there are no falls per push: show the wobble instead.
  pushed = any(m['pushes'] for a in arms for r in stats[a].values() for m in r)
  rows = ROWS if pushed else ROWS[:2] + [('wobble', 'wobble (tilt-rate SD, deg/s)')]
  fig, axes = plt.subplots(len(rows), 1, figsize=(7.5, 6.5), sharex=True)
  for arm, color in zip(arms, COLORS):
    runs = list(stats[arm].values())
    M = min(len(r) for r in runs)
    x = np.arange(1, M + 1)
    for ax, (key, label) in zip(axes, rows):
      Y = np.array([[r[i][key] for i in range(M)] for r in runs], float)
      m, s = np.nanmean(Y, 0), np.nanstd(Y, 0)
      ax.plot(x, m, color=color, label=f'{arm} (n={len(runs)})', lw=1.6)
      lo = np.maximum(m - s, 0)
      hi = np.minimum(m + s, 1) if key in ('up', 'fall_per_push') else m + s
      ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
      ax.set_ylabel(label)
  axes[0].legend(frameon=False, fontsize=8, ncol=2)
  axes[1].set_ylim(0, 80)
  axes[0].set_ylim(0, 1.02)
  axes[2].set_ylim(0, 1.02 if pushed else None)
  axes[-1].set_xlabel('minute of training (sim table, '
                      + ('pushes every ~5 s, walls at 0.5 m)' if pushed else 'no pushes or walls)'))
  for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
  plt.tight_layout()
  plt.savefig(sys.argv[2], dpi=130)


if __name__ == '__main__':
  main()
