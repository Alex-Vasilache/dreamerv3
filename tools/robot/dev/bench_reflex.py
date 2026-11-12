"""Time the catch reflex variants on a real policy, in numpy as the sim's
phone loop runs them (int8 push, emotion heads).

  python tools/robot/dev/bench_reflex.py <run logdir> [n steps]

Takes the run's newest published policy (online_shared/policy), steps the
PolicyRunner on synthetic sensors (the recorded sensors are not
stored, so the observations are small random tilts: the timing is what
matters), and reports per-step ms of act() alone and of the reflex per variant
(modulate(): the reflex and the efference copy it triggers).
"""

import glob
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'dreamerv3', 'deploy'))

import elements  # noqa: E402
from dreamerv3.deploy import export as exportlib  # noqa: E402
from embodied.envs.robot import _load_policy_pickle  # noqa: E402
from policy_runner import PolicyRunner  # noqa: E402

VARIANTS = {
    'slope4': dict(mode='slope', steps=4),
    'slope16': dict(mode='slope', steps=16),
    'search16': dict(mode='search', steps=16),
    'search16_5': dict(mode='search', steps=16, offsets='0.5,1'),
    'search16_e2': dict(mode='search', steps=16, every=2),
    'search16_e4': dict(mode='search', steps=16, every=4),
    'search8': dict(mode='search', steps=8),
}


def main():
  logdir = sys.argv[1]
  n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
  config = elements.Config.load(os.path.join(logdir, 'config.yaml'))
  pkl = sorted(glob.glob(os.path.join(logdir, 'online_shared', 'policy', 'policy_*.pkl')))[-1]
  data = _load_policy_pickle(open(pkl, 'rb').read())
  params = data['params'] if isinstance(data, dict) and 'params' in data else data
  arrays, meta = exportlib.pack_arrays(config, params, 'int8', True)
  blob = exportlib.encode(meta, arrays, stamp='0', compress=True)
  policy = PolicyRunner.build_blob(blob, os.path.join(tempfile.mkdtemp(), 'p.npz'))
  sensors = dict(theta=0.0, angular_velocity=0.0, wheel_speed_l=0.0, wheel_speed_r=0.0)
  for name, cfg in [('none', None)] + list(VARIANTS.items()):
    r = PolicyRunner()
    r.configure(dict(body=dict(theta_zero=0.0, lo=-0.12, hi=0.161, zone=0.035,
                               pain_mode='impact'),
                     reflex=dict(on=cfg is not None, tmax=-1e9, **(cfg or {}))))
    r.install(policy, 'bench')
    rng = np.random.default_rng(0)
    act_ms, ref_ms = [], []
    for i in range(n):
      s = dict(sensors, theta=0.05 * rng.normal(), angular_velocity=0.5 * rng.normal())
      t0 = time.perf_counter()
      _, a = r.act(s)
      t1 = time.perf_counter()
      r.modulate(a)
      t2 = time.perf_counter()
      r.applied(r._chosen)
      if i >= 10:
        act_ms.append((t1 - t0) * 1e3)
        ref_ms.append((t2 - t1) * 1e3)
    print(f'{name:12s} act {np.median(act_ms):6.2f} ms  reflex mean {np.mean(ref_ms):6.2f} '
          f'median {np.median(ref_ms):6.2f} max {np.max(ref_ms):6.2f} ms')


if __name__ == '__main__':
  main()
