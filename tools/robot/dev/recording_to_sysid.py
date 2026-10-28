"""A saved policy's run on the robot (the phone's recordings/<name>_<time>.csv,
written while the Dreamer Player drives) as a sysid_fit.py recording.

  python tools/robot/dev/recording_to_sysid.py REC.csv OUT_DIR

OUT_DIR must be named *_<surface> like the probe's recordings (e.g.
~/logdir/robot_recordings/sysid/sim4-table-balance_table: sysid_fit.py takes
the floor from after the first underscore). Appends to
OUT_DIR/balance.csv as replay_to_sysid.py does, so several recordings on one
floor go in one directory; each stretch lands as its own test,
`<recording>:<segment>`.

Unlike a replay this needs no reconstruction: the phone wrote its own clock
(t_obs when the sensors were read, t_drive when the output went to the
serial writer) and what the base was sent after limit_current and zero_fix.
Yaw rate comes from the wheel speeds, as in replay_to_sysid.py. Rows where
the base was sent nothing (the robot held, or stopped) and stalls longer
than 0.1 s cut the run.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..')))
from embodied.envs import robot_mujoco as rm  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('csv')
  ap.add_argument('out')
  args = ap.parse_args()
  name = os.path.splitext(os.path.basename(args.csv))[0]
  d = pd.read_csv(args.csv)
  n = len(d)
  if n < 2:
    raise SystemExit(f'{name}: only {n} rows')
  wl, wr = d.wheel_speed_l.values, d.wheel_speed_r.values
  held = (d.sent_l.values == 0) & (d.sent_r.values == 0)
  stall = np.zeros(n, bool)
  stall[1:] = np.diff(d.t_obs.values) > 0.1
  cut = np.zeros(n, bool)
  cut[1:] = held[:-1] & ~held[1:]
  segment = np.cumsum(stall | cut)
  # speed_l > speed_r is a right turn (clockwise from above, negative yaw).
  yaw = -(wl - wr) / 548.0 * rm.WHEEL_R / (2 * rm.HALF_TRACK)
  t0 = d.t_obs.values[0]
  rows = pd.DataFrame(dict(
      phase='balance', test=[f'{name}:{s}' for s in segment],
      theta=d.theta.values, angular_velocity=d.angular_velocity.values,
      wheel_speed_l=wl, wheel_speed_r=wr, gyro_y=yaw,
      # A gap at every cut, so no window spans one.
      t_obs=d.t_obs.values - t0 + 1.0 * segment,
      t_drive=d.t_drive.values - t0 + 1.0 * segment,
      sent_l=d.sent_l.values, sent_r=d.sent_r.values))
  rows = rows[~held]
  os.makedirs(args.out, exist_ok=True)
  path = os.path.join(args.out, 'balance.csv')
  if os.path.exists(path):
    before = pd.read_csv(path)
    if before.test.str.startswith(name + ':').any():
      print(f'{name}: already in {path}')
      return
    # Later recordings continue the clock, so windows never mix two of them.
    offset = before.t_obs.max() + 10.0
    rows['t_obs'] += offset
    rows['t_drive'] += offset
    rows = pd.concat([before, rows])
  rows.to_csv(path, index=False)
  span = d.t_obs.values[-1] - t0
  print(f'{name}: {n} steps over {span:.0f} s ({(n - 1) / span:.1f} Hz), '
        f'{segment[-1] + 1} segments, {held.sum()} held rows dropped -> {path}')


if __name__ == '__main__':
  main()
