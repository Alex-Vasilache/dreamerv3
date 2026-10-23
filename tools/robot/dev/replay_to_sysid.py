"""A real training run's experience as a sysid_fit.py recording.

sysid_probe.py records scripted wheel tests; a training run records minutes
of the robot balancing under its own policy, which is the regime the sim has
to get right. This writes that experience in the probe's CSV layout, as phase
`balance`, so sysid_fit.py can score (and fit) the sim on it:

  python tools/robot/dev/replay_to_sysid.py RUN_LOGDIR OUT_DIR [--name s02]

OUT_DIR must be named *_<surface> like the probe's recordings (e.g.
$SYSID/data/e2060_carpet). Appends to OUT_DIR/balance.csv when several
sessions go into one surface; each lands as its own test, `<name>:<episode>`.

What the replay does not hold is reconstructed:

* Time. Steps are spaced at the run's measured rate (env0/timing.csv rows
  over their span; the phone free-runs its clock), and a stall the trainer
  saw (> 0.1 s between two rows) cuts the run there.
* The output the base got. The replay holds the policy's action (with
  onboard, the one the phone applied); the phone's limit_current and zero_fix
  (robot_mujoco.phone_output) give `sent`. It is driven LAG seconds after its
  observation: the onboard policy and the serial write, measured on the probe
  as t_drive - t_obs of the reply that prompted it.
* Yaw rate, from the wheel speeds (no gyro in the observation).
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..')))
from embodied.envs import robot_mujoco as rm  # noqa: E402

LAG = 0.012  # s from an observation to its action reaching the base


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('logdir')
  ap.add_argument('out')
  ap.add_argument('--name', default=None)
  ap.add_argument('--lag', type=float, default=LAG)
  args = ap.parse_args()
  name = args.name or os.path.basename(os.path.dirname(
      os.path.normpath(args.logdir)))
  cfg = yaml.safe_load(open(os.path.join(args.logdir, 'config.yaml')))
  r = cfg['env']['robot']
  keys = ('wheels', 'orientation', 'drive', 'is_first')
  parts = {k: [] for k in keys}
  for f in sorted(glob.glob(os.path.join(
      args.logdir, 'online_shared/experience/*.npz'))):
    try:
      chunk = np.load(f)
      got = [chunk[k] for k in keys]
    except Exception:
      continue
    for k, v in zip(keys, got):
      parts[k].append(v)
  d = {k: np.concatenate(v) for k, v in parts.items()}
  n = len(d['drive'])
  timing = pd.read_csv(os.path.join(args.logdir, 'env0/timing.csv'))
  t_rx = timing.t.values
  hz = (len(t_rx) - 1) / (t_rx[-1] - t_rx[0])
  # Rows of timing.csv are the trainer's receive times of the same steps,
  # aligned at the end (the first few predate the replay's first chunk).
  rx = np.full(n, np.nan)
  m = min(n, len(t_rx))
  rx[n - m:] = t_rx[len(t_rx) - m:]
  stall = np.zeros(n, bool)
  stall[1:] = np.diff(rx) > 0.1
  t = np.arange(n) / hz
  if r.get('friction_comp'):
    raise SystemExit('friction_comp runs are not reconstructed here')
  if r.get('command_bias', 0.0):
    raise SystemExit('command_bias runs are not reconstructed here')

  wheels = d['wheels'] / r['obs_wheel_scale']
  theta = d['orientation'][:, 0] * r['obs_theta_scale'] + r.get('theta_zero', 0.0)
  rate = d['orientation'][:, 1] * r['obs_rate_scale']
  act = np.clip(d['drive'] * r.get('command_scale', 1.0), -1, 1)
  sent = np.zeros_like(act)
  requested = np.zeros(2)
  for i in range(n):
    for w in range(2):
      sent[i, w], requested[w] = rm.phone_output(act[i, w], wheels[i, w],
                                                requested[w])
  # speed_l > speed_r is a right turn (clockwise from above, negative yaw).
  yaw = -(wheels[:, 0] - wheels[:, 1]) / 548.0 * rm.WHEEL_R / (2 * rm.HALF_TRACK)

  episode = np.cumsum(d['is_first'] | stall)
  # A gap in time at every episode start or stall, so no window spans one.
  t = t + 1.0 * episode
  rows = pd.DataFrame(dict(
      phase='balance', test=[f'{name}:{e}' for e in episode],
      theta=theta, angular_velocity=rate,
      wheel_speed_l=wheels[:, 0], wheel_speed_r=wheels[:, 1], gyro_y=yaw,
      t_obs=t, t_drive=t + args.lag, sent_l=sent[:, 0], sent_r=sent[:, 1]))
  clipped = (np.abs(d['orientation']) >= r['obs_clip'] - 1e-6).any(1)
  rows = rows[~clipped]
  os.makedirs(args.out, exist_ok=True)
  path = os.path.join(args.out, 'balance.csv')
  if os.path.exists(path):
    # Later sessions continue the clock, so windows never mix two of them.
    before = pd.read_csv(path)
    offset = before.t_obs.max() + 10.0
    rows['t_obs'] += offset
    rows['t_drive'] += offset
    rows = pd.concat([before, rows])
  rows.to_csv(path, index=False)
  print(f'{name}: {n} steps at {hz:.1f} Hz, {episode[-1]} segments, '
        f'{clipped.sum()} clipped rows dropped -> {path}')


if __name__ == '__main__':
  main()
