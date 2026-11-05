"""Per-minute stats of the emotion rounds (sim_phone runs with pushes).

  python emostats.py OUT.json ARMS.txt   (ARMS: lines "<exp> <arm> <seed>")

Per minute of a run's recorded experience: time upright (|tilt| < 3 deg),
bumper crossings (tilt crossing +6.5 or -5 deg, as demo_stats.py), wobble
(tilt-rate SD, deg/s), mean |wheel speed| (mm/s), reward. From the run's
emotion.csv (sim_phone): pushes and the share of them followed within 1 s
by a bumper crossing (falls per push, pushes on an upright body only), arousal events, mean arousal and
pain.
"""

import csv
import glob
import json
import sys

import numpy as np

HZ = 65
W = '/work/DoyaU/vasilache/work'


def run(n):
  dirs = glob.glob(f'{W}/e{n}_*')
  if not dirs:
    return None
  d = dirs[0] + '/logdir'
  L = [np.load(f) for f in sorted(glob.glob(d + '/online_shared/experience/*.npz'))]
  L = [x for x in L if 'drive' in x.files]
  if not L:
    return None
  ori = np.concatenate([x['orientation'] for x in L])
  wheels = np.concatenate([x['wheels'] for x in L])
  rew = np.concatenate([x['reward'] for x in L])
  tilt = np.degrees(ori[:, 0] * 0.18)
  rate = np.degrees(ori[:, 1] * 3.0)
  speed = np.abs(wheels.mean(1)) / 3.3e-4
  cross = np.zeros(len(tilt), bool)
  cross[1:] = ((tilt[1:] > 6.5) & (tilt[:-1] <= 6.5)) | (
      (tilt[1:] < -5) & (tilt[:-1] >= -5))
  pushes, events, arousal, pain = [], [], [], []
  try:
    rows = list(csv.DictReader(open(d + '/emotion.csv')))
  except OSError:
    rows = []
  last = -10**9
  for r in rows:
    k = int(r['step'])
    if float(r['push'] or 0) and k - last > 10:
      pushes.append(k)
    if float(r['push'] or 0):
      last = k
    if float(r['event'] or 0):
      events.append(k)
    arousal.append((k, float(r['arousal'])))
    pain.append((k, float(r['pain'])))
  out = []
  for i in range(len(tilt) // (60 * HZ)):
    a, b = i * 60 * HZ, (i + 1) * 60 * HZ
    ps = [p for p in pushes if a <= p < b]
    # Only pushes that met an upright body: one lying on a bumper cannot fall.
    fell = [bool(cross[p:p + HZ].any()) for p in ps
            if p < len(tilt) and abs(tilt[p]) < 3]
    ar = [v for k, v in arousal if a <= k < b]
    pa = [v for k, v in pain if a <= k < b]
    out.append(dict(
        up=float(np.mean(np.abs(tilt[a:b]) < 3)), cross=int(cross[a:b].sum()),
        wobble=float(rate[a:b].std()), speed=float(speed[a:b].mean()),
        reward=float(rew[a:b].mean()), pushes=len(ps),
        fall_per_push=float(np.mean(fell)) if fell else float('nan'),
        events=sum(1 for k in events if a <= k < b),
        arousal=float(np.mean(ar)) if ar else float('nan'),
        pain=float(np.mean(pa)) if pa else float('nan')))
  return out


if __name__ == '__main__':
  arms = [l.split() for l in open(sys.argv[2]) if l.strip()]
  res = {}
  for n, arm, seed in arms:
    r = run(int(n))
    if r:
      res.setdefault(arm, {})[n] = r
  json.dump(res, open(sys.argv[1], 'w'))
  for arm, runs in res.items():
    M = min(len(v) for v in runs.values())
    lo, hi = 9, M
    f = lambda key: [np.nanmean([m[key] for m in v[lo:hi]]) for v in runs.values()]
    first = [next((i + 1 for i, m in enumerate(v) if m['up'] > 0.8), None) for v in runs.values()]
    print(f'{arm:8s} n={len(runs)} mins={M} first>0.8 {first} | min {lo + 1}-{hi}: '
          f'up {np.mean(f("up")):.2f} cross {np.mean(f("cross")):.1f}+-{np.std(f("cross")):.1f} '
          f'wobble {np.mean(f("wobble")):.1f} fall/push {np.nanmean(f("fall_per_push")):.2f} '
          f'speed {np.mean(f("speed")):.0f} events/min {np.mean(f("events")):.1f}')
