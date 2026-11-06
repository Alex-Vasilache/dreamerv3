"""Late collapse vs the learner's replay ratio: live demos and sim rounds EA-EC.

  python fig_collapse_ratio.py LIVE.json SIM.json OUT_STEM
LIVE/SIM json: lists of {run, cr, up, fps, ratio} per run (per-minute bumper
crossings and time upright; ratio = learner steps / robot steps).
"""
import json, re, sys
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLK, BLU, ORA = '#000000', '#0072B2', '#E69F00'
plt.rcParams.update({'axes.titlesize': 17, 'axes.labelsize': 15, 'xtick.labelsize': 13,
                     'ytick.labelsize': 13, 'legend.fontsize': 14})
PARAMS = 'sbatch/sim_bal_emo4.txt sbatch/sim_bal_emo5.txt sbatch/sim_bal_emo6.txt sbatch/sim_bal_e30114.txt'.split()


def grid(ax):
  ax.grid(color='#d5d5d2', lw=0.7); ax.set_axisbelow(True)


def main(live_path, sim_path, out, root='.'):
  live = [r for r in json.load(open(live_path)) if 'err' not in r and len(r['cr']) >= 7]
  fig, ax = plt.subplots(2, 2, figsize=(13, 9))
  for r in live:
    hi = r['ratio'] > 380
    x = np.arange(1, len(r['cr']) + 1)
    c = ORA if hi else BLU
    ax[0, 0].plot(x, r['cr'], color=c, lw=2.4, alpha=0.75)
    ax[0, 1].plot(x, r['up'], color=c, lw=2.4, alpha=0.75)
  ax[0, 0].set_title('Live demos, Nov 2-6 (one line per run)')
  ax[0, 1].set_title('Live demos: time upright')
  ax[0, 0].set_ylabel('bumper crossings / min'); ax[0, 1].set_ylabel('time upright')
  ax[0, 0].set_ylim(0, 140); ax[0, 1].set_ylim(0, 1.02)
  # sim: arm -> gpu -> runs
  arm, gpu = {}, {}
  for f in PARAMS:
    for l in open(f'{root}/{f}'):
      if l.startswith('e'):
        w = l.split(); gpu[w[0]] = w[4]; arm[w[0]] = f + re.sub(r'--seed \d+', '', ' '.join(w[5:]))
  A = defaultdict(lambda: defaultdict(list)); rat = defaultdict(list)
  for r in json.load(open(sim_path)):
    e = r['run'].split('_')[0]
    c = np.full(27, np.nan); c[:min(27, len(r['cr']))] = r['cr'][:27]; A[arm[e]][gpu[e]].append(c); rat[gpu[e]].append(r['ratio'])
  V, P, D = [], [], []
  for a, g in A.items():
    if 'v100' in g and 'p100' in g:
      v, p = np.nanmean(g['v100'], 0), np.nanmean(g['p100'], 0)
      V.append(v); P.append(p); D.append(np.nanmean(v[14:]) - np.nanmean(p[14:]))
  V, P, D = np.array(V), np.array(P), np.array(D)
  x = np.arange(1, V.shape[1] + 1)
  for Y, c, lab in [(P, BLU, f'P100 learner, ratio {np.median(rat["p100"]):.0f}'),
                    (V, ORA, f'V100 learner, ratio {np.median(rat["v100"]):.0f}')]:
    m, s = np.nanmean(Y, 0), np.nanstd(Y, 0)
    ax[1, 0].plot(x, m, color=c, lw=2.4, label=lab)
    ax[1, 0].fill_between(x, np.maximum(m - s, 0), m + s, color=c, alpha=0.15, lw=0)
  ax[1, 0].set_ylim(0, 60); ax[1, 0].set_ylabel('bumper crossings / min')
  ax[1, 0].set_title(f'Sim rounds EA-EC ({len(D)} arms, mean ± SD)')
  ax[1, 0].axvspan(15, 27, color='#d5d5d2', alpha=0.3, lw=0)
  o = np.argsort(D)
  ax[1, 1].bar(np.arange(len(D)), D[o], color=[ORA if d > 0 else BLU for d in D[o]])
  ax[1, 1].axhline(0, color=BLK, lw=1)
  ax[1, 1].set_title('Sim: V100 − P100, crossings/min after min 15')
  ax[1, 1].set_ylabel('difference (crossings / min)'); ax[1, 1].set_xlabel('arm (sorted)')
  ax[1, 1].text(0.03, 0.95, f'mean {D.mean():+.1f} ± {D.std() / np.sqrt(len(D)):.1f} (SEM)\nV100 worse in {(D > 0).sum()}/{len(D)}',
                transform=ax[1, 1].transAxes, va='top', fontsize=13)
  for a in ax.flat: grid(a)
  for a in [ax[0, 0], ax[0, 1], ax[1, 0]]: a.set_xlabel('minute of robot time')
  h = [plt.Line2D([], [], color=BLU, lw=2.4), plt.Line2D([], [], color=ORA, lw=2.4)]
  fig.legend(h, ['slower learner (ratio 220-320)', 'faster learner (ratio 340-560)'],
             loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
  fig.tight_layout(rect=(0, 0.04, 1, 1))
  for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=200, facecolor='white')


if __name__ == '__main__':
  main(*sys.argv[1:4], root=sys.argv[4] if len(sys.argv) > 4 else '.')
