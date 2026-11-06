"""Figures for docs/robot/LIVE_20261106.md.

  python tools/robot/dev/fig_live_20261106.py MINUTES.txt OUTDIR

MINUTES.txt is tools/robot/dev/demo_stats.py's per-minute table for each run,
each preceded by a '## <run dir name>' line (run demo_stats.py on the trainer
host and concatenate). Writes live_20261106_minutes.png and
live_20261106_link.png.
"""

import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
# Categorical slots 1-3 of the dataviz reference palette, fixed order.
RUNS = [
    ('e40006', 'e40006: ratio schedule', '#2a78d6'),
    ('e41041', 'e41041: control, no schedule', '#eb6834'),
    ('e41045', 'e41045: schedule, emotion on, no drive home from min 4', '#1baf7a'),
]


def read_minutes(path):
  runs, cur = {}, None
  for line in pathlib.Path(path).read_text().splitlines():
    if line.startswith('## '):
      cur = line[3:].split('_')[0]
      runs[cur] = []
    elif cur and line.strip():
      f = line.split()
      runs[cur].append(dict(min=int(f[0]), upright=float(f[1]),
                            cross=float(f[2]), wobble=float(f[3])))
  return runs


def style(ax):
  ax.set_facecolor(SURF)
  for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
  for s in ('left', 'bottom'):
    ax.spines[s].set_color(GRID)
  ax.tick_params(colors=INK2, labelsize=9)
  ax.grid(axis='y', color=GRID, lw=0.8)
  ax.set_axisbelow(True)


def minutes_figure(runs, out):
  fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True,
                           facecolor=SURF, constrained_layout=True)
  panels = [('cross', 'Bumper crossings per minute', (0, 130)),
            ('upright', 'Fraction of time upright (|tilt| < 3 deg)', (0, 1.02))]
  for ax, (key, title, ylim) in zip(axes, panels):
    style(ax)
    ax.axvspan(6.5, 20.5, color='#f0efec', zorder=0)
    ax.axvline(5.5, color=INK2, lw=1, ls=(0, (3, 3)))
    for rid, label, color in RUNS:
      rows = runs.get(rid, [])
      if not rows:
        continue
      x = [r['min'] for r in rows]
      y = [r[key] for r in rows]
      ax.plot(x, y, color=color, lw=2, marker='o', ms=4, label=label,
              zorder=3)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc='left', fontsize=10.5, color=INK, pad=6)
  axes[0].text(5.6, 122, 'ratio -> 128\n(schedule runs)', fontsize=8,
               color=INK2, va='top')
  axes[0].text(13.5, 122, 'minutes 7-20: where earlier runs relapsed',
               fontsize=8, color=INK2, ha='center', va='top')
  axes[1].set_xlabel('Live minute (drives home not counted)', color=INK2,
                     fontsize=9)
  axes[1].set_xlim(0.5, 25.5)
  axes[0].legend(loc='upper right', bbox_to_anchor=(1, 0.86), fontsize=8.5,
                 frameon=False, labelcolor=INK)
  fig.suptitle('Live balancing on the robot, 2026-11-06 (learner on IAS)',
               x=0.01, ha='left', fontsize=12, color=INK)
  fig.savefig(out, dpi=150, facecolor=SURF)


def link_figure(out):
  # Policy age when a new policy is installed on the phone (s): p10, p50,
  # p90 from the trainer's push log. Saion: e10101 (2026-11-03), only its
  # reported range (3-6 s) is known, so no median. float16: the link test
  # with the phone (no base); its bar ends at p75 (4.3 s): its p90 is 61 s,
  # from dropped weight streams.
  rows = [
      ('Saion (OIST), int8 deltas', None, 3.0, 6.0, '#eb6834', '3-6 s'),
      ('IAS, float16 whole (link test)', 3.8, 2.7, 4.3, '#2a78d6',
       '3.8 s (bar to p75; p90 61 s)'),
      ('IAS, int8 deltas + emotion heads (e41045)', 1.7, 1.4, 2.4, '#2a78d6',
       '1.7 s'),
      ('IAS, int8 deltas (e40006)', 1.0, 0.8, 1.4, '#2a78d6', '1.0 s'),
  ]
  fig, ax = plt.subplots(figsize=(8, 2.8), facecolor=SURF,
                         constrained_layout=True)
  style(ax)
  ax.grid(axis='x', color=GRID, lw=0.8)
  ax.grid(axis='y', visible=False)
  for i, (label, med, lo, hi, color, note) in enumerate(rows):
    ax.plot([lo, hi], [i, i], color=color, lw=6, alpha=0.35,
            solid_capstyle='round')
    if med is not None:
      ax.plot([med], [i], 'o', color=color, ms=9, zorder=3,
              markeredgecolor=SURF, markeredgewidth=2)
    ax.text(hi + 0.12, i, note, va='center', fontsize=9, color=INK)
  ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9,
                color=INK)
  ax.set_xlim(0, 8.5)
  ax.set_xlabel('Age of the policy when it reaches the phone (s): median, p10-p90',
                color=INK2, fontsize=9)
  ax.set_title('Weight pushes from Darmstadt: OIST (orange) vs IAS (blue)',
               loc='left', fontsize=10.5, color=INK)
  fig.savefig(out, dpi=150, facecolor=SURF)


def main():
  src, outdir = sys.argv[1], pathlib.Path(sys.argv[2])
  outdir.mkdir(parents=True, exist_ok=True)
  minutes_figure(read_minutes(src), outdir / 'live_20261106_minutes.png')
  link_figure(outdir / 'live_20261106_link.png')


if __name__ == '__main__':
  main()
