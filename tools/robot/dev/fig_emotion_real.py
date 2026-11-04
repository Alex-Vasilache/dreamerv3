"""Figures for emotion traces of real runs (tools/robot/dev/emotion_trace.py).

  python tools/robot/dev/fig_emotion_real.py OUT_PREFIX NAME=TRACE.npz [NAME=TRACE.npz ...]

OUT_PREFIX_session.png: the first trace over its session (tilt, mood/valence,
arousal and its parts), 1 s means. OUT_PREFIX_events.png: every trace's
signals around bumper hits (tilt crossing +6.5 / -5 deg), mean +- SE.
"""

import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({'axes.titlesize': 15, 'axes.labelsize': 13, 'xtick.labelsize': 11,
                     'ytick.labelsize': 11, 'legend.fontsize': 11})
OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
HZ = 65


def smooth(x, n=HZ):
  k = np.ones(n) / n
  return np.convolve(x, k, mode='same')


def load(path):
  d = dict(np.load(path))
  d['tilt'] = np.degrees(d['theta'])
  live = np.abs(d['drive']).sum(1) > 0
  return d, live


def zs(x):
  return (x - x.mean()) / (x.std() + 1e-8)


def session(prefix, name, path):
  d, live = load(path)
  t = (d['t'] - d['t'][0]) / 60
  fig, ax = plt.subplots(2, 2, figsize=(20, 8.5), sharex=True)
  a = ax[0, 0]
  a.plot(t, d['tilt'], color='#999999', lw=0.4, label='tilt')
  a.plot(t, smooth(d['tilt']), color=OI[0], lw=1.2, label='tilt, 1 s mean')
  a.axhline(6.5, color=OI[6], ls=':', lw=1); a.axhline(-5, color=OI[6], ls=':', lw=1)
  a.set_ylim(-15, 15); a.set_ylabel('tilt (deg)'); a.set_title(f'{name}: the robot'); a.legend(loc='upper right')
  a = ax[0, 1]
  a.plot(t, smooth(d['reward']), color=OI[3], lw=1.2, label='reward')
  a.plot(t, smooth(d['reward_hat']), color=OI[3], ls='--', lw=1, label='predicted reward')
  a.set_ylabel('reward per step'); a.set_title('reward and its prediction'); a.legend(loc='lower right')
  a = ax[1, 0]
  a.plot(t, smooth(d['mood']), color=OI[1], lw=1.4, label='mood (value)')
  a.plot(t, smooth(d['valence']), color=OI[5], lw=1.0, label='valence (TD error)')
  a.plot(t, smooth(d['arousal']) * 2 - 1, color=OI[6], lw=1.2, label='arousal (scaled to -1..1)')
  a.set_ylim(-1.05, 1.05); a.set_xlabel('minute of the session'); a.set_ylabel('signal')
  a.set_title('emotion shown on the face'); a.legend(loc='lower right')
  a = ax[1, 1]
  for k, c in (('unexpected', OI[6]), ('novelty', OI[2]), ('reward_err', OI[3]), ('empowerment', OI[5])):
    a.plot(t, smooth(zs(d[k]), 4 * HZ), color=c, lw=1.2, label=k)
  a.set_xlabel('minute of the session'); a.set_ylabel('z-score, 4 s mean')
  a.set_title('parts of arousal'); a.legend(loc='upper right', ncol=2)
  for a in ax.flat:
    a.grid(color='#d5d5d2', lw=0.7)
  plt.tight_layout()
  for ext in ('png', 'pdf'):
    plt.savefig(f'{prefix}_session.{ext}', dpi=200, facecolor='white')
  plt.close(fig)


def events(prefix, traces):
  keys = [('arousal', 'arousal'), ('face_valence', 'face valence'), ('unexpected', 'unexpected surprise (nats)'),
          ('empowerment', 'empowerment (nats)')]
  fig, ax = plt.subplots(2, 2, figsize=(20, 8.5))
  win = np.arange(-2 * HZ, 3 * HZ)
  for i, (name, path) in enumerate(traces):
    d, live = load(path)
    th = d['tilt']
    hit = np.flatnonzero(((th[1:] > 6.5) & (th[:-1] <= 6.5)) | ((th[1:] < -5) & (th[:-1] >= -5))) + 1
    hit = hit[(hit > 3 * 60 * HZ) & (hit < len(th) - 3 * HZ) & (hit > 2 * HZ)]
    hit = hit[live[hit]]
    for a, (k, lab) in zip(ax.flat, keys):
      x = d[k] - np.median(d[k])
      if not len(hit):
        continue
      m = np.stack([x[h + win] for h in hit])
      mu, se = m.mean(0), m.std(0) / np.sqrt(len(hit))
      a.plot(win / HZ, mu, color=OI[(i + 1) % 8], lw=1.6, label=f'{name} (n={len(hit)})')
      a.fill_between(win / HZ, mu - se, mu + se, color=OI[(i + 1) % 8], alpha=0.15)
      a.set_title(lab + ' around a bumper hit'); a.set_xlabel('s from the hit'); a.set_ylabel('minus session median')
  for a in ax.flat:
    a.axvline(0, color='#999999', lw=1); a.grid(color='#d5d5d2', lw=0.7)
  ax[0, 0].legend(loc='upper left')
  plt.tight_layout()
  for ext in ('png', 'pdf'):
    plt.savefig(f'{prefix}_events.{ext}', dpi=200, facecolor='white')
  plt.close(fig)


if __name__ == '__main__':
  prefix = sys.argv[1]
  traces = [a.split('=', 1) for a in sys.argv[2:]]
  session(prefix, *traces[0])
  events(prefix, traces)
