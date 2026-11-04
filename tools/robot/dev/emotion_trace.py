"""Valence and arousal along a recorded robot run (dreamerv3/deploy/emotion.py).

Replays a session's experience (the observations the phone saw and the
actions it applied) through the policy the learner had published at that
time, one snapshot per `--snap_s` seconds of the session, and writes every
step's emotion signals to OUT.npz, plus a 10 Hz JSON trace for the face page.

  python tools/robot/dev/emotion_trace.py SESSION_LOGDIR OUT.npz [--snap_s 60]

SESSION_LOGDIR is a live run's logdir (its online_shared/policy/ keeps the
published policy pickles) or a sim run's (logdir/ckpt_milestones/).
"""

import argparse
import datetime
import glob
import json
import os
import pathlib
import sys
import tempfile

import numpy as np
import ruamel.yaml as yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'dreamerv3' / 'deploy'))
sys.path.insert(0, str(ROOT / 'tools' / 'robot' / 'dev'))

import elements  # noqa: E402

from dreamerv3.deploy import export as exportlib  # noqa: E402
from numpy_policy import NumpyPolicy  # noqa: E402
from emotion import Emotion  # noqa: E402
import export_policy  # noqa: E402

KEYS = ('valence', 'mood', 'face_valence', 'helpless', 'arousal', 'dominance', 'td', 'value', 'reward_hat',
        'surprise', 'novelty', 'unexpected', 'reward_err', 'empowerment')


def chunk_time(path):
  # 20261104T233925F123456-... : the chunk's first step, local time
  stamp = os.path.basename(path).split('-')[0]
  return datetime.datetime.strptime(stamp[:15], '%Y%m%dT%H%M%S').timestamp()


def snapshots(logdir):
  """(unix time, params path) of every policy the learner published."""
  # Full agent params only: the published policy pickles lack the critic.
  # Milestones (sim runs, and live runs with run.save_every_steps) give the
  # model as it was during the run; else the run's latest checkpoint, read
  # as the final model's view of the whole session (time 0 = from the start).
  out = [(os.path.getmtime(p), p) for p in glob.glob(f'{logdir}/ckpt_milestones/*')]
  if not out:
    out = [(0.0, p) for p in glob.glob(f'{logdir}/ckpt/*') if not p.endswith('latest')][-1:]
  return sorted(out)


def load_policy(config, path, tmp):
  params = export_policy.load_params(path)
  arrays, meta = exportlib.pack_arrays(config, params, 'float32', emotion=True)
  blob = exportlib.encode(meta, arrays, compress=False)
  file = pathlib.Path(tmp) / 'policy.npz'
  file.write_bytes(blob)
  return NumpyPolicy(str(file), seed=0), meta


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('logdir')
  ap.add_argument('out')
  ap.add_argument('--snap_s', type=float, default=60.0)
  ap.add_argument('--every', type=int, default=4, help='empowerment every N steps')
  ap.add_argument('--max_min', type=float, default=0, help='stop after this many minutes')
  args = ap.parse_args()
  logdir = pathlib.Path(args.logdir)
  config = elements.Config(yaml.YAML(typ='safe').load((logdir / 'config.yaml').read_text()))
  hz = float(config.env.robot.hz)
  chunks = sorted(glob.glob(f'{logdir}/online_shared/experience/*.npz'))
  snaps = snapshots(logdir)
  assert chunks and snaps, (len(chunks), len(snaps))
  rows = {k: [] for k in KEYS}
  extra = {k: [] for k in ('theta', 'rate', 'speed', 'reward', 'drive', 'snap', 't')}
  tmp = tempfile.mkdtemp()
  current, policy, emo, carry = None, None, None, None
  for ci, path in enumerate(chunks):
    d = np.load(path)
    if 'drive' not in d.files:
      continue
    t0 = chunk_time(path)
    # The newest snapshot published before this chunk began, at most one per snap_s.
    usable = [s for s in snaps if s[0] <= t0 + 1]
    if not usable:
      continue
    pick = usable[-1]
    if current is None or pick[0] - current[0] >= args.snap_s:
      current = pick
      policy, meta = load_policy(config, pick[1], tmp)
      gamma = meta['emotion']['gamma']
      if emo is None:
        emo = Emotion(policy, gamma=gamma, every=args.every)
      else:
        emo.p = policy   # keep the running normalisation across snapshots
      carry = policy.initial()
      print(f'chunk {ci}/{len(chunks)}: policy {os.path.basename(pick[1])}', flush=True)
    for t in range(len(d['reward'])):
      first = bool(d['is_first'][t])
      if first:
        carry = policy.initial()
      deter = policy.precompute(carry, first)
      obs = {'orientation': d['orientation'][t], 'wheels': d['wheels'][t]}
      tokens = policy._encode(obs)
      stoch, logit = policy._obs_stoch(deter, tokens)
      sig = emo.step(deter, logit, stoch, float(d['reward'][t]), is_first=first)
      for k in KEYS:
        rows[k].append(sig[k])
      act = np.asarray(d['drive'][t], np.float32)
      carry = dict(deter=deter, stoch=stoch, prevact=act)
      extra['theta'].append(float(d['orientation'][t][0]) * float(config.env.robot.obs_theta_scale))
      extra['rate'].append(float(d['orientation'][t][1]) * float(config.env.robot.obs_rate_scale))
      extra['speed'].append(float(np.mean(d['wheels'][t])) / float(config.env.robot.obs_wheel_scale))
      extra['reward'].append(float(d['reward'][t]))
      extra['drive'].append(act)
      extra['snap'].append(current[0])
      extra['t'].append(t0 + t / hz)
    if args.max_min and len(rows['valence']) > args.max_min * 60 * hz:
      break
  out = {k: np.asarray(v, np.float32) for k, v in rows.items()}
  out.update({k: np.asarray(v) for k, v in extra.items()})
  np.savez_compressed(args.out, **out)
  # 10 Hz trace for the face page: time from the start, valence, arousal, tilt.
  step = max(1, int(round(hz / 10)))
  t = out['t'] - out['t'][0]
  trace = dict(hz=10, t=np.round(t[::step], 2).tolist(),
               valence=np.round(out['face_valence'][::step], 3).tolist(),
               arousal=np.round(out['arousal'][::step], 3).tolist(),
               dominance=np.round(out['dominance'][::step], 3).tolist(),
               theta=np.round(np.degrees(out['theta'][::step]), 2).tolist())
  pathlib.Path(args.out).with_suffix('.json').write_text(json.dumps(trace))
  print('wrote', args.out, len(t), 'steps', flush=True)


if __name__ == '__main__':
  main()
