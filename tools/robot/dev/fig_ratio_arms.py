"""Per-minute crossings and time upright per arm (mean ± SD over seeds).

  python fig_ratio_arms.py OUT_STEM 'label=SCAN.json:e19490-e19493' ...
SCAN.json from a per-run scan (run, cr, up per minute).
"""
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#000000', '#0072B2', '#E69F00', '#56B4E9', '#D55E00', '#009E73', '#CC79A7']
plt.rcParams.update({'axes.titlesize': 17, 'axes.labelsize': 15, 'xtick.labelsize': 13,
                     'ytick.labelsize': 13, 'legend.fontsize': 14})


def main(out, specs):
  rows = [('cr', 'bumper crossings / min', 60), ('up', 'time upright', 1.02)]
  if all('speed' in r for spec in specs for r in json.load(open(spec.split('=')[1].split(':')[0]))):
    rows.append(('speed', '|wheel speed| (mm/s)', None))
  fig, ax = plt.subplots(len(rows), 1, figsize=(10, 4.2 * len(rows) + 0.6), sharex=True)
  for spec, c in zip(specs, COLORS):
    label, rest = spec.split('=')
    path, rng = rest.split(':')
    a, b = [int(x.strip('e')) for x in rng.split('-')]
    runs = [r for r in json.load(open(path)) if a <= int(r['run'].split('_')[0][1:]) <= b]
    M = min(len(r['cr']) for r in runs); x = np.arange(1, M + 1)
    for k, (key, _, _) in enumerate(rows):
      Y = np.array([r[key][:M] for r in runs], float); m, s = Y.mean(0), Y.std(0)
      ax[k].plot(x, m, color=c, lw=2.4, label=f'{label} (n={len(runs)})')
      ax[k].fill_between(x, np.maximum(m - s, 0), np.minimum(m + s, 1) if key == 'up' else m + s,
                         color=c, alpha=0.15, lw=0)
  for a_, (_, lab, top) in zip(ax, rows):
    a_.set_ylabel(lab); a_.set_ylim(0, top)
  ax[-1].set_xlabel('minute of robot time')
  ax[0].set_title('Sim, Germany floor (table), no pushes, seeds 791-794')
  for a_ in ax: a_.grid(color='#d5d5d2', lw=0.7); a_.set_axisbelow(True)
  h, l = ax[0].get_legend_handles_labels()
  fig.legend(h, l, loc='lower center', ncol=2, frameon=False)
  fig.tight_layout(rect=(0, 0.08, 1, 1))
  for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=200, facecolor='white')


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2:])
