"""Rounds RF/RF2 figure: catch reflex variants against b (no reflex) and c
(the first reflex), per minute, mean +- SD over seeds.

  python fig_reflex.py OUT_STEM STATS.json [STATS.json ...]

STATS from reflexstats.py (several, e.g. Saion and IAS, are merged by arm).
Rows: bumper crossings / min and impact pain / min (top), share of steps the
reflex changed the command and peak true bumper force per contact (bottom).
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Colour + marker + dash per arm (deuteranopia: hues apart in lightness,
# shape and dash carry identity too).
STY = {
    'b': dict(color='#000000', marker='o', ls='-', mfc='#000000', label='b: no reflex'),
    'c': dict(color='#888888', marker='v', ls='--', mfc='#888888', label='c: first reflex (tmax 0.2)'),
    's4z2': dict(color='#0072B2', marker='D', ls='-', mfc='#0072B2', label='search 4, log-z 3, tmin 0.02'),
    'g4z2': dict(color='#E69F00', marker='s', ls='-.', mfc='#E69F00', label='slope 4, cap 0.5, log-z 3'),
    's4on': dict(color='#56B4E9', marker='^', ls=':', mfc='white', label='search 4, always on'),
    's16z': dict(color='#009E73', marker='P', ls=(0, (5, 1.5, 1, 1.5, 1, 1.5)), mfc='white',
                 label='search 16 (every 4th), log-z 3'),
}
plt.rcParams.update({'axes.titlesize': 17, 'axes.labelsize': 15, 'xtick.labelsize': 13,
                     'ytick.labelsize': 13, 'legend.fontsize': 14})
PANELS = [('cross', 'bumper crossings / min'), ('pain', 'impact pain / min'),
          ('fire', 'reflex: share of steps changed'), ('force', 'peak contact force (N)')]


def main(out, paths):
  res = {}
  for p in paths:
    for arm, runs in json.load(open(p)).items():
      res.setdefault(arm, {}).update(runs)
  fig, axes = plt.subplots(2, 2, figsize=(14, 9.4), sharex=True)
  axes = axes.ravel()
  for arm, st in STY.items():
    runs = res.get(arm)
    if not runs:
      continue
    M = min(len(r['minutes']) for r in runs.values())
    x = np.arange(1, M + 1)
    for ax, (key, _) in zip(axes, PANELS):
      Y = np.array([[m[key] for m in r['minutes'][:M]] for r in runs.values()], float)
      m, s = np.nanmean(Y, 0), np.nanstd(Y, 0)
      ax.plot(x, m, color=st['color'], ls=st['ls'], lw=2.4, marker=st['marker'],
              mfc=st['mfc'], ms=6, markevery=5, label=f"{st['label']} (n={len(runs)})")
      ax.fill_between(x, np.maximum(m - s, 0), m + s, color=st['color'], alpha=0.15, lw=0)
  for ax, (key, lab) in zip(axes, PANELS):
    ax.set_title(lab)
    ax.set_ylim(bottom=0)
    ax.grid(color='#d5d5d2', lw=0.7)
    ax.set_axisbelow(True)
  axes[0].set_ylim(0, 45)   # minutes 1-2 (100-200) cut: the late rates matter
  axes[1].set_ylim(0, 6)
  axes[2].set_yscale('symlog', linthresh=1e-3)
  axes[2].set_ylim(0, 1)
  axes[0].set_ylabel('per minute')
  axes[2].set_ylabel('share; N')
  fig.text(0.5, 0.125, 'minute of robot time (65 Hz steps)', ha='center', fontsize=15)
  h, l = axes[0].get_legend_handles_labels()
  fig.legend(h, l, loc='lower center', ncol=3, frameon=False)
  fig.tight_layout(rect=(0, 0.15, 1, 1))
  for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=200, facecolor='white')


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2:])
