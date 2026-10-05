"""Where the actor's step time goes, with no JAX, no learner and no cluster.

Adds the layers of the actor's step path back one at a time, against the real
robot, with a random policy standing in for the agent so a run starts in a
second rather than after a JAX compile.

What it found on 2026-09-03, when the phone ticked at 50Hz but the actor only
consumed ~33 frames a second: none of it was ours. The whole step path costs
0.5-1.1ms -- socket 0.06ms, reward 0.02ms, observation 0.06ms, replay.add
0.10ms, driver and its subprocess the rest -- against a 20ms budget. Everything
else was time parked on the socket.

The `arrival` stage then separated the two ends. The phone made its frames
20.00ms apart (p90 21.6), and we received them 20.1ms apart at the median but
45.9ms at p90 and 176ms at p99, 1.47 frames arriving per wake: evenly produced,
clumpily delivered. ICMP to the phone ran 6ms at best against a 129ms average,
which is the WiFi radio dozing. A WifiLock(WIFI_MODE_FULL_HIGH_PERF) in the app
took the average to 19ms and the clumping to 1.29 frames per wake, and what
remains is the link itself -- the reverse direction is just as ragged (phone to
Mac, 5.6ms best against a 38ms average). A cable (adb reverse) or a quieter
access point is what is left; there is nothing to optimise in this repo.

  .venv/bin/python tools/time_robot_steps.py                  # all stages
  .venv/bin/python tools/time_robot_steps.py --stages env     # just one
  .venv/bin/python tools/time_robot_steps.py --steps 1000

Read the report as: `blocked` is time parked on the socket waiting for the
phone's next tick, which is free -- the phone is busy, not us. Everything else
is ours, and the sum of ours is what has to stay under the 20ms control period.
"""

import argparse
import collections
import sys
import time

import numpy as np
import ruamel.yaml as yaml

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import embodied  # noqa: E402
from embodied.envs.robot import SmartphoneRobot  # noqa: E402

CONFIGS = 'dreamerv3/configs.yaml'


def robot_kwargs(overrides):
  """The env's configured defaults, so this measures the real thing."""
  root = __import__('pathlib').Path(__file__).resolve().parent.parent
  config = yaml.YAML(typ='safe').load((root / CONFIGS).read_text())
  kwargs = dict(config['defaults']['env']['robot'])
  # Applied by the wrappers in main.py, not by the env itself.
  for key in ('repeat', 'use_logdir'):
    kwargs.pop(key, None)
  kwargs.update(overrides)
  return kwargs


class Timer:
  """Wall-clock totals per label, plus a per-step tally."""

  def __init__(self):
    self.steps = []
    self.current = collections.defaultdict(float)

  def wrap(self, obj, name, label):
    inner = getattr(obj, name)
    def outer(*args, **kwargs):
      start = time.perf_counter()
      try:
        return inner(*args, **kwargs)
      finally:
        self.current[label] += time.perf_counter() - start
    setattr(obj, name, outer)

  def section(self, label, seconds):
    self.current[label] += seconds

  def commit(self, total):
    self.current['total'] = total
    self.steps.append(dict(self.current))
    self.current.clear()


def report(name, timer, labels):
  steps = timer.steps
  if not steps:
    print(f'{name}: no steps')
    return
  total = np.array([s['total'] for s in steps]) * 1e3
  order = sorted(total)
  rate = len(steps) / (total.sum() / 1e3)
  print(f'--- {name}: {len(steps)} steps, {rate:.1f} Hz consumed')
  print(f'    {"step total":<14} p50 {order[len(order) // 2]:7.2f}ms  '
        f'p90 {order[int(.9 * len(order))]:7.2f}ms')
  ours = np.zeros(len(steps))
  for label in labels:
    v = np.array([s.get(label, 0.0) for s in steps]) * 1e3
    if not v.any():
      continue
    o = sorted(v)
    if label != 'blocked':
      ours += v
    tag = '(free)' if label == 'blocked' else ''
    print(f'    {label:<14} p50 {o[len(o) // 2]:7.2f}ms  '
          f'p90 {o[int(.9 * len(o))]:7.2f}ms  mean {v.mean():6.2f}ms {tag}')
  rest = total - ours - np.array([s.get('blocked', 0.0) for s in steps]) * 1e3
  o = sorted(rest)
  print(f'    {"unaccounted":<14} p50 {o[len(o) // 2]:7.2f}ms  '
        f'p90 {o[int(.9 * len(o))]:7.2f}ms  mean {rest.mean():6.2f}ms')
  ours = ours + rest
  o = sorted(ours)
  print(f'    {"OURS (sum)":<14} p50 {o[len(o) // 2]:7.2f}ms  '
        f'p90 {o[int(.9 * len(o))]:7.2f}ms  mean {ours.mean():6.2f}ms  '
        f'<- must stay under 20ms')


def make_env(kwargs, timer=None):
  env = SmartphoneRobot(**kwargs)
  if timer is not None:
    # _read blocks until the phone's next frame lands, so it is the idle half;
    # everything else on this list is work we do.
    timer.wrap(env, '_send', 'send')
    timer.wrap(env, '_read', 'blocked')
    timer.wrap(env, '_read_nowait', 'drain')
    timer.wrap(env, '_evaluate', 'reward')
    timer.wrap(env, '_obs', 'obs_build')
    timer.wrap(env, '_record', 'timing_csv')
  return env


def random_action(act_space, size=None):
  act = {}
  for key, space in act_space.items():
    if key == 'reset':
      continue
    if space.discrete:
      act[key] = np.random.randint(0, space.high, size=space.shape)
    else:
      act[key] = np.random.uniform(-1, 1, size=space.shape).astype(np.float32)
  return act


def stage_env(args, kwargs):
  """The env alone: socket, reward and observation assembly."""
  timer = Timer()
  env = make_env(kwargs, timer)
  act = {**random_action(env.act_space), 'reset': True}
  for i in range(args.steps + args.warmup):
    if i == args.warmup:
      timer.steps.clear()
    start = time.perf_counter()
    obs = env.step(act)
    act = {**random_action(env.act_space), 'reset': bool(obs['is_last'])}
    timer.commit(time.perf_counter() - start)
  env.close()
  report('env only', timer,
         ['blocked', 'send', 'drain', 'reward', 'obs_build', 'timing_csv'])


def stage_driver(args, kwargs, parallel):
  """Add the Driver, optionally with the env in its own process."""
  timer = Timer()
  # In the parallel case the env lives in a subprocess, so the wrappers cannot
  # reach it; the split then shows as driver overhead against stage `env`.
  fns = [lambda: make_env(kwargs, None if parallel else timer)]
  driver = embodied.Driver(fns, parallel=parallel)
  act_space = driver.act_space

  def policy(carry, obs, **kw):
    start = time.perf_counter()
    length = len(obs['is_first'])
    acts = {k: np.stack([v] * length) for k, v in
            random_action(act_space).items()}
    timer.section('policy', time.perf_counter() - start)
    return carry, acts, {}

  seen = [0]
  def count(tran, worker):
    seen[0] += 1
  driver.on_step(count)

  for i in range(args.steps + args.warmup):
    if i == args.warmup:
      timer.steps.clear()
    start = time.perf_counter()
    driver(policy, steps=1)
    timer.commit(time.perf_counter() - start)
  driver.close()
  name = 'driver, env in a subprocess' if parallel else 'driver, env in process'
  report(name, timer,
         ['blocked', 'send', 'drain', 'reward', 'obs_build', 'timing_csv',
          'policy'])


def stage_replay(args, kwargs):
  """Add the replay append the actor does on every step."""
  timer = Timer()
  env = make_env(kwargs, timer)
  replay = embodied.replay.Replay(
      length=64, capacity=1e5, directory=None, chunksize=1024)
  act = {**random_action(env.act_space), 'reset': True}
  for i in range(args.steps + args.warmup):
    if i == args.warmup:
      timer.steps.clear()
    start = time.perf_counter()
    obs = env.step(act)
    action = random_action(env.act_space)
    tran = {**{k: v for k, v in obs.items() if not k.startswith('log/')},
            **action}
    inner = time.perf_counter()
    replay.add(tran, 0)
    timer.section('replay_add', time.perf_counter() - inner)
    act = {**action, 'reset': bool(obs['is_last'])}
    timer.commit(time.perf_counter() - start)
  env.close()
  report('env + replay.add', timer,
         ['blocked', 'send', 'drain', 'reward', 'obs_build', 'timing_csv',
          'replay_add'])


def stage_arrival(args, kwargs):
  """Compare when the phone made each frame with when it reached us.

  Every frame carries the phone's own `t`, so the two gap distributions
  separate a phone that ticks unevenly from a link that delivers evenly-spaced
  frames in bursts. Nothing is dropped here; every frame is timed.
  """
  env = make_env(kwargs, None)
  env._connect()
  env._send(0.0, 0.0, reset=True)
  sent = made = arrived = None
  gaps_made, gaps_arrived, per_wake = [], [], []
  for i in range(args.steps + args.warmup):
    # One blocking read, then everything else already buffered: the size of
    # that batch is the burst size.
    header, _ = env._read()
    now = time.perf_counter()
    batch = [header]
    while True:
      more = env._read_nowait()
      if more is None:
        break
      batch.append(more[0])
    if i >= args.warmup:
      for frame in batch:
        if made is not None:
          gaps_made.append((frame['t'] - made) * 1e3)
        made = frame['t']
      if arrived is not None:
        gaps_arrived.append((now - arrived) * 1e3)
      arrived = now
      per_wake.append(len(batch))
    else:
      made, arrived = batch[-1]['t'], now
    env._send(0.0, 0.0, reset=False)
  env.close()
  def show(label, xs, unit='ms'):
    o = sorted(xs)
    q = lambda p: o[int(p * (len(o) - 1))]
    print(f'    {label:<22} p10 {q(.1):6.2f}  p50 {q(.5):6.2f}  '
          f'p90 {q(.9):6.2f}  p99 {q(.99):6.2f}  max {o[-1]:7.2f} {unit}')
  print(f'--- frame arrival: {len(per_wake)} wakes, '
        f'{sum(per_wake)} frames')
  show('phone made them', gaps_made)
  show('we received them', gaps_arrived)
  show('frames per wake', [float(x) for x in per_wake], '')
  print(f'    mean frames per wake {np.mean(per_wake):.2f} '
        f'(1.00 means the link keeps up with the tick)')


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--stages', nargs='+', default=[
      'arrival', 'env', 'replay', 'driver', 'driver_parallel'])
  parser.add_argument('--steps', type=int, default=400)
  parser.add_argument('--warmup', type=int, default=50)
  parser.add_argument('--host', default='0.0.0.0')
  parser.add_argument('--port', type=int, default=3000)
  parser.add_argument('--task', default='balance')
  args = parser.parse_args(argv)

  kwargs = robot_kwargs(dict(
      host=args.host, port=args.port, task=args.task,
      # One long episode: a reset costs a reconnect-shaped stall that would
      # swamp the per-step numbers we are after.
      length=10 ** 9, fall_angle=0.0, recover_gain=0.0, status_every=0,
      logdir=None))

  print(f'Listening on {args.host}:{args.port}; start the phone app.')
  print(f'{args.steps} steps per stage after {args.warmup} of warmup.\n')
  for stage in args.stages:
    if stage == 'arrival':
      stage_arrival(args, kwargs)
    elif stage == 'env':
      stage_env(args, kwargs)
    elif stage == 'replay':
      stage_replay(args, kwargs)
    elif stage == 'driver':
      stage_driver(args, kwargs, parallel=False)
    elif stage == 'driver_parallel':
      stage_driver(args, kwargs, parallel=True)
    else:
      raise SystemExit(f'Unknown stage {stage!r}')
    print()
    # The phone retries every couple of seconds, so give it time to come back
    # before the next stage binds the port again.
    time.sleep(3)


if __name__ == '__main__':
  sys.exit(main())
