"""How a live session went, minute by minute, from its recorded experience.

  python tools/robot/dev/demo_stats.py RUN_DIR [RUN_DIR ...]   # a demo dir (s01, s02, ...) or a logdir's parent

Per minute of robot time (paused steps left out): time upright (|tilt| < 3
deg), bumper crossings (tilt crossing +6.5 or -5 deg, the measured bumpers
less ~2.5 deg), tilt-rate SD (wobble, deg/s), mean |wheel speed| (mm/s),
mean reward. For reference (2026-11-05): the Oct 23-28 OIST demos settled
to 1-8 crossings/min and 7-24 deg/s after minute 3; the sim's live recipe on
the table reaches ~12-20 crossings/min at minutes 10-20.
"""

import glob
import os
import sys

import numpy as np

HZ = 65


def session(path):
  files = sorted(glob.glob(f'{path}/logdir/online_shared/experience/*.npz'))
  data = [np.load(f) for f in files]
  data = [d for d in data if 'drive' in d.files]
  if not data:
    return
  act = np.concatenate([d['drive'] for d in data])
  ori = np.concatenate([d['orientation'] for d in data])
  wheels = np.concatenate([d['wheels'] for d in data])
  reward = np.concatenate([d['reward'] for d in data])
  live = ~(np.abs(act).sum(1) == 0)
  tilt = np.degrees(ori[live, 0] * 0.18)
  rate = np.degrees(ori[live, 1] * 3.0)
  speed = np.abs(wheels[live].mean(1)) / 3.3e-4
  reward = reward[live]
  print(f'{path}: {len(tilt) / HZ / 60:.1f} live min')
  print('  min  upright  crossings/min  wobble deg/s  |speed| mm/s  reward')
  for i in range(len(tilt) // (60 * HZ)):
    s = slice(i * 60 * HZ, (i + 1) * 60 * HZ)
    t = tilt[s]
    hits = np.sum((t[1:] > 6.5) & (t[:-1] <= 6.5)) + np.sum((t[1:] < -5) & (t[:-1] >= -5))
    print(f'  {i + 1:3d}  {np.mean(np.abs(t) < 3):7.2f}  {hits:13d}  {rate[s].std():12.1f}'
          f'  {speed[s].mean():11.0f}  {reward[s].mean():6.2f}')


if __name__ == '__main__':
  for arg in sys.argv[1:]:
    subs = sorted(glob.glob(f'{arg}/s0*')) or [arg]
    for sub in subs:
      if os.path.isdir(f'{sub}/logdir'):
        session(sub)
