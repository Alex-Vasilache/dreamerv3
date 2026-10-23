"""Fit the MuJoCo robot (embodied/envs/robot_mujoco.py) to sysid_probe.py data.

Each recording is cut into short windows. A window starts the model from the
measured state (tilt, tilt rate, wheel speeds, yaw rate), feeds it the outputs
the base was actually sent (sent_l/r, at the phone's t_drive plus a delay) and
compares what it does with what the robot did at each t_obs. The motors are
fitted on the stand (the air phase, the body held still), then each floor on
its bumper phase with the motors fixed.

  .venv/bin/python tools/robot/dev/sysid_fit.py ~/logdir/robot_sysid/*_carpet ~/logdir/robot_sysid/*_table
  .venv/bin/python tools/robot/dev/sysid_fit.py ... --stage motor        # one stage
  .venv/bin/python tools/robot/dev/sysid_fit.py ... --show               # score the current defaults

Results print as the values to put in robot_mujoco.py (Physics defaults and
SURFACES) and a per-test error table.
"""

import argparse
import json
import math
import multiprocessing
import os
import sys
import time

import mujoco
import numpy as np
import pandas as pd
from scipy import optimize

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..')))
from embodied.envs import robot_mujoco as rm  # noqa: E402

DT = 0.002
WINDOW = 1.0
BUMPER_LO, BUMPER_HI = -0.120, 0.161   # env.robot defaults, from theta_zero 0
# Error scales: a window's error is the mean square of these units.
SCALE = dict(theta=math.radians(2.0), rate=math.radians(30.0), wheel=300.0)

MOTOR = dict(wheel_omega=11.4, stall=0.06, stick=0.27, slide=0.09,
             armature=5e-4, delay=0.01)
FLOOR = dict(rolling=0.002, contact_tc=0.02, bumper_friction=0.5, drag=0.0,
             lean=0.0)
# Fitted in log space, so each stays positive.
LOG = {'wheel_omega', 'stall', 'stick', 'slide', 'armature', 'rolling',
       'contact_tc', 'bumper_friction'}
LIMITS = dict(delay=(0.0, 0.1), lean=(-0.05, 0.05), slide=(1e-3, 0.6),
              stick=(1e-3, 0.9), rolling=(1e-6, 0.03), contact_tc=(0.002, 0.2),
              bumper_friction=(0.05, 3.0), drag=(0.0, 30.0),
              wheel_omega=(5.0, 25.0), stall=(0.01, 0.5),
              armature=(1e-5, 5e-3))


def load(path, phase, skip=()):
  df = pd.read_csv(os.path.join(path, phase + '.csv'))
  df = df[~df.test.isin(skip)].sort_values('t_obs').reset_index(drop=True)
  return df


COLUMNS = ('theta', 'angular_velocity', 'wheel_speed_l', 'wheel_speed_r',
           'gyro_y', 't_obs', 't_drive', 'sent_l', 'sent_r')


def arrays(df, a, b):
  """Rows a..b as plain arrays: cheap to hand to a worker."""
  return {c: df[c].values[a:b + 1].astype(np.float64) for c in COLUMNS}


def windows(df, every=0.5):
  """(start, end) row ranges of WINDOW seconds, every `every` seconds, that do
  not span a gap in the recording or two tests of different kinds."""
  t = df.t_obs.values
  out, i = [], 0
  while i < len(df):
    j = np.searchsorted(t, t[i] + WINDOW)
    if j >= len(df):
      break
    if np.diff(t[i:j + 1]).max() < 0.3:
      out.append((i, j))
    i = np.searchsorted(t, t[i] + every)
  return out


def simulate(params, floor, rows, fixed):
  """Run one window; returns the model's [theta, rate, speed_l, speed_r] at
  each row's t_obs after the first."""
  p = dict(MOTOR, **params)
  f = dict(FLOOR, **(floor or {}))
  phys = rm.Physics(
      -BUMPER_LO, BUMPER_HI, DT, wheel_omega=p['wheel_omega'],
      stall=p['stall'], stick=p['stick'], slide=p['slide'],
      armature=p['armature'], rolling=f['rolling'],
      contact_tc=f['contact_tc'], bumper_friction=f['bumper_friction'],
      drag=f['drag'], fixed=fixed)
  m, d = phys.model, phys.data
  r0 = {c: v[0] for c, v in rows.items()}
  wheels = np.array([r0['wheel_speed_l'], r0['wheel_speed_r']]) / phys.unit
  for dof, w in zip(phys.dofs, wheels):
    d.qvel[dof] = w
  if not fixed:
    body = m.body('robot').id
    com = m.body_ipos[body].copy()
    m.body_ipos[body] = (0.0, -com[2] * math.tan(f['lean']), com[2])
    mujoco.mj_setConst(m, d)
    pitch = -float(r0['theta'])
    d.qpos[2] = rm.WHEEL_R
    d.qpos[3:7] = (math.cos(pitch / 2), math.sin(pitch / 2), 0.0, 0.0)
    wx = -float(r0['angular_velocity'])
    d.qvel[3] = wx
    d.qvel[5] = float(r0['gyro_y'])
    d.qvel[1] = -rm.WHEEL_R * (wx + wheels.mean())
    mujoco.mj_forward(m, d)
  t0 = float(r0['t_obs'])
  drive = rows['t_drive'] - t0 + p['delay']
  sent = np.stack([rows['sent_l'], rows['sent_r']], 1)
  obs_t = rows['t_obs'][1:] - t0
  out = np.zeros((len(obs_t), 4))
  now, k, current = 0.0, 0, sent[0]
  # Before the first drive of the window, the previous row's output applies.
  for i, target in enumerate(obs_t):
    while now < target - 1e-9:
      while k < len(drive) and drive[k] <= now:
        current = sent[k]
        k += 1
      phys.step(current, 1)
      now += DT
    if fixed:
      theta = rate = 0.0
    else:
      w, x, y, z = d.qpos[3:7]
      theta = -math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
      rate = -float(d.qvel[3])
    out[i] = (theta, rate, *phys.speed)
  return out


def window_error(rows, sim, fixed):
  real = np.stack([rows[c] for c in (
      'theta', 'angular_velocity', 'wheel_speed_l', 'wheel_speed_r')], 1)[1:]
  err = (sim - real) / np.array(
      [SCALE['theta'], SCALE['rate'], SCALE['wheel'], SCALE['wheel']])
  if fixed:
    err = err[:, 2:]
  return float(np.mean(err ** 2))


def _one(job):
  params, floor, rows, fixed = job
  try:
    error = window_error(rows, simulate(params, floor, rows, fixed), fixed)
  except Exception:  # noqa: BLE001  an unstable trial
    return 1e3
  return error if np.isfinite(error) else 1e3


class Problem:

  def __init__(self, sets, pool):
    self.windows = []   # (rows, fixed, surface, test)
    for df, wins, fixed, surface in sets:
      for a, b in wins:
        self.windows.append(
            (arrays(df, a, b), fixed, surface, df.test.iloc[a]))
    self.pool = pool

  def errors(self, motor, floors):
    jobs = [(motor, None if fixed else floors[surface], rows, fixed)
            for rows, fixed, surface, _ in self.windows]
    keys = [(surface, test) for _, _, surface, test in self.windows]
    chunk = max(1, len(jobs) // (4 * os.cpu_count()))
    return keys, self.pool.map(_one, jobs, chunksize=chunk)

  def loss(self, motor, floors):
    _, errs = self.errors(motor, floors)
    return float(np.mean(errs))


def fit(problem, motor, floors, free_motor, free_floor, surfaces, iters,
        method='nm'):
  names = list(free_motor) + [f'{s}.{n}' for s in surfaces for n in free_floor]
  start = {n: motor[n] for n in free_motor}
  start.update({f'{s}.{n}': floors[s][n] for s in surfaces for n in free_floor})
  log = {n for n in names if n.split('.')[-1] in LOG}
  x0 = np.array([math.log(start[n]) if n in log else start[n] for n in names])
  best = [np.inf, None]
  calls = [0, time.time()]

  def split(x):
    vals = {}
    for v, n in zip(x, names):
      v = math.exp(v) if n in log else float(v)
      lo, hi = LIMITS.get(n.split('.')[-1], (-np.inf, np.inf))
      vals[n] = min(max(v, lo), hi)
    m = dict(motor, **{n: vals[n] for n in free_motor})
    f = {s: dict(floors[s], **{n: vals[f'{s}.{n}'] for n in free_floor})
         for s in floors}
    return m, f

  def objective(x):
    m, f = split(x)
    value = problem.loss(m, f)
    calls[0] += 1
    if calls[0] % 50 == 0:
      print(f'  [{calls[0]} evaluations, {time.time() - calls[1]:.0f} s, '
            f'best {best[0]:.4f}]', flush=True)
    if value < best[0]:
      best[:] = [value, (m, f)]
      print(f'  {value:.4f}  ' + '  '.join(
          f'{n}={v:.4g}' for n, v in zip(names, [
              (math.exp(a) if n in log else a) for a, n in zip(x, names)])),
          flush=True)
    return value

  if method == 'de':
    # Global: the whole box of LIMITS, the start as one member. Each
    # objective call is already spread over the pool's workers.
    bounds = []
    for n in names:
      lo, hi = LIMITS[n.split('.')[-1]]
      bounds.append((math.log(lo), math.log(hi)) if n in log else (lo, hi))
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    optimize.differential_evolution(
        objective, bounds, x0=x0, maxiter=iters, popsize=8, tol=1e-4,
        polish=False, init='sobol', seed=0)
  optimize.minimize(objective, x0 if method == 'nm' else pack_best(best, names, log),
                    method='Nelder-Mead',
                    options=dict(maxfev=iters if method == 'nm' else iters * 2,
                                 xatol=1e-3, fatol=1e-4, adaptive=True))
  return best[1]


def pack_best(best, names, log):
  """The best (motor, floors) so far as an optimiser vector."""
  m, f = best[1]
  out = []
  for n in names:
    s, _, k = n.rpartition('.')
    v = f[s][k] if s else m[k]
    out.append(math.log(v) if n in log else v)
  return np.array(out)


def report(problem, motor, floors):
  keys, errs = problem.errors(motor, floors)
  table = {}
  for (surface, test), e in zip(keys, errs):
    table.setdefault(surface, {}).setdefault(test.split(':')[0], []).append(e)
  for surface, tests in table.items():
    allv = [e for v in tests.values() for e in v]
    print(f'{surface}: mean window error {np.mean(allv):.3f} '
          f'over {len(allv)} windows')
    print('   ' + '  '.join(f'{t} {np.mean(v):.2f}' for t, v in tests.items()))


def main():
  ap = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('dirs', nargs='+',
                  help='sysid_probe recordings, named *_<surface>')
  ap.add_argument('--stage', choices=['motor', 'floor', 'all'], default='all')
  ap.add_argument('--show', action='store_true')
  ap.add_argument('--iters', type=int, default=300,
                  help='Nelder-Mead evaluations, or DE generations')
  ap.add_argument('--method', choices=['nm', 'de'], default='nm',
                  help='de: a global search (Saion), then Nelder-Mead from its best')
  ap.add_argument('--joint', action='store_true',
                  help='one fit of motor and floors together, on all data')
  ap.add_argument('--every', type=float, default=0.5)
  ap.add_argument('--skip', nargs='*', default=[],
                  help='surface:test to leave out, e.g. table:prbs0_0.6')
  ap.add_argument('--start', default=None, help='JSON of a previous result')
  ap.add_argument('--out', default=None, help='write the result as JSON')
  ap.add_argument('--workers', type=int, default=os.cpu_count())
  args = ap.parse_args()

  motor = dict(MOTOR)
  floors = {}
  air, ground = [], []
  for path in args.dirs:
    surface = os.path.basename(os.path.normpath(path)).split('_', 1)[1]
    floors[surface] = dict(FLOOR, **rm.SURFACES.get(surface, {}))
    skip = [s.split(':', 1)[1] for s in args.skip
            if s.split(':', 1)[0] == surface]
    if os.path.exists(os.path.join(path, 'air.csv')):
      df = load(path, 'air', ['wait'] + skip)
      air.append((df, windows(df, args.every), True, 'air'))
    if os.path.exists(os.path.join(path, 'bumper.csv')):
      df = load(path, 'bumper', ['wait'] + skip)
      ground.append((df, windows(df, args.every), False, surface))
  if args.start:
    with open(args.start) as f:
      saved = json.load(f)
    motor.update(saved['motor'])
    for s, v in saved['floors'].items():
      floors.setdefault(s, dict(FLOOR)).update(v)

  with multiprocessing.Pool(args.workers) as pool:
    if args.show:
      report(Problem(air + ground, pool), motor, floors)
      return
    if args.joint:
      surfaces = sorted({s for *_, s in ground})
      print('== motor and floors together: ' + ', '.join(surfaces))
      motor, floors = fit(
          Problem(air + ground, pool), motor, floors,
          ['wheel_omega', 'stall', 'stick', 'slide', 'armature', 'delay'],
          ['rolling', 'contact_tc', 'bumper_friction', 'drag', 'lean'],
          surfaces, args.iters, args.method)
    if args.stage in ('motor', 'all') and air and not args.joint:
      print('== motor, on the stand')
      motor, _ = fit(Problem(air, pool), motor, floors,
                     ['wheel_omega', 'stall', 'stick', 'slide', 'armature',
                      'delay'], [], [], args.iters, args.method)
    if args.stage in ('floor', 'all') and ground and not args.joint:
      surfaces = sorted({s for *_, s in ground})
      print('== floors: ' + ', '.join(surfaces))
      motor, floors = fit(Problem(ground, pool), motor, floors, [],
                          ['rolling', 'contact_tc', 'bumper_friction', 'drag',
                           'lean'], surfaces, args.iters, args.method)
    report(Problem(air + ground, pool), motor, floors)
  result = dict(motor=motor, floors=floors)
  print(json.dumps(result, indent=1))
  if args.out:
    with open(args.out, 'w') as f:
      json.dump(result, f, indent=1)


if __name__ == '__main__':
  main()
