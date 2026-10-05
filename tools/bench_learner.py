"""Learner throughput against network shape, without a robot or a replay.

The per-shape sweep through main.py costs a process start, an XLA compile and
a replay prefill of batch*length samples before a single number comes out --
minutes per point, most of it waiting. Nothing about the learner's ceiling
needs any of that: it is one jitted function fed batches of a fixed shape.

So this builds the agent in-process and feeds it synthetic batches through the
same `agent.stream` path the learner uses, which keeps the host-side cost (the
device_put, the seed) in the measurement. One process, one JAX start, a compile
per shape.

  .venv/bin/python tools/bench_learner.py --smoke               # 15s sanity run
  .venv/bin/python tools/bench_learner.py                       # default grid
  .venv/bin/python tools/bench_learner.py --batches 16 64 256 1024
  .venv/bin/python tools/bench_learner.py --deters 256 512 --units 256
  .venv/bin/python tools/bench_learner.py --configs robot_daydreamer

`ratio at N Hz` is what train_ratio would actually achieve against an actor
stepping at N Hz, since train_ratio counts samples per agent step.

Run `--smoke` after any change to this file. It exercises the same code path at
a size that compiles on CPU in seconds, so a shape or dtype mistake surfaces
immediately rather than after a minute of Metal compile per shape -- which is
exactly how the two bugs in the first version of this script were found.
"""

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import ruamel.yaml as yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import elements  # noqa: E402
import jax  # noqa: E402


def build_config(config_names, overrides):
  import dreamerv3.main as main
  configs = yaml.YAML(typ='safe').load(
      elements.Path(main.folder / 'configs.yaml').read())
  config = elements.Config(configs['defaults'])
  for name in config_names:
    config = config.update(configs[name])
  return config.update(overrides)


def synthetic(spaces, batch_size, batch_length, seed=0):
  """Batches shaped exactly like the replay stream's, filled with noise."""
  rng = np.random.default_rng(seed)
  batch = {}
  for key, space in spaces.items():
    shape = (batch_size, batch_length) + space.shape
    if space.dtype == bool:
      # is_first only on the first step, so the sequences look like episodes
      # rather than a batch of resets.
      value = np.zeros(shape, bool)
      if key == 'is_first':
        value[:, 0] = True
    elif np.issubdtype(space.dtype, np.integer):
      low = 0 if space.low is None else int(np.max(space.low))
      high = int(np.min(space.high)) if space.high is not None else 2
      high = max(high, low + 1)
      value = rng.integers(low, high, shape).astype(space.dtype)
    else:
      value = rng.normal(size=shape).astype(space.dtype)
    batch[key] = value
  while True:
    yield batch


def bench_policy(config, deter, units, seconds, warmup):
  """Time one policy call, which is what the actor pays per control step.

  With the learner on Saion the actor's only job is this call, and it has the
  whole 20ms control period to do it in. That is a far looser budget than the
  learner's, so the two halves can be sized against different limits.
  """
  import dreamerv3.main as main
  config = _shape(config, deter, units, config.batch_size)
  agent = main.make_agent(config)
  obs = {}
  rng = np.random.default_rng(0)
  for key, space in agent.obs_space.items():
    shape = (1,) + space.shape
    if space.dtype == bool:
      obs[key] = np.zeros(shape, bool)
    else:
      obs[key] = rng.normal(size=shape).astype(space.dtype)
  obs['is_first'] = np.ones((1,), bool)
  carry = agent.init_policy(1)
  for _ in range(warmup):
    carry, acts, _ = agent.policy(carry, obs, mode='train')
  jax.block_until_ready(carry)
  start = time.perf_counter()
  n = 0
  while n < 20 or time.perf_counter() - start < seconds:
    carry, acts, _ = agent.policy(carry, obs, mode='train')
    jax.block_until_ready(acts)
    n += 1
  ms = (time.perf_counter() - start) / n * 1e3
  params = sum(int(np.prod(v.shape)) for k, v in agent.params.items()
               if not k.startswith('opt/'))
  print('  deter %-5s units %-4s  policy %6.2f ms/step  -> max %5.0f Hz  '
        '(%4.1f%% of a 20ms period, %.2fM params)'
        % (deter, units, ms, 1000 / ms, 100 * ms / 20, params / 1e6), flush=True)


def _shape(config, deter, units, batch_size):
  return config.update({
      'batch_size': batch_size,
      'agent.dyn.rssm.deter': deter,
      'agent.dyn.rssm.hidden': units,
      'agent.policy.units': units,
      'agent.value.units': units,
      'agent.rewhead.units': units,
      'agent.conhead.units': units,
      'agent.enc.simple.units': units,
      'agent.dec.simple.units': units,
  })


def bench(config, deter, units, batch_size, seconds, warmup, actor_hz):
  import dreamerv3.main as main
  config = _shape(config, deter, units, batch_size)
  agent = main.make_agent(config)
  # agent.params carries the optimizer slots too (roughly 3x the model), so
  # count what the Agent itself reports as the model.
  params = sum(
      int(np.prod(v.shape)) for k, v in agent.params.items()
      if not k.startswith('opt/'))
  # The stream hands the learner batch_length steps plus the replay context it
  # conditions on, and the compiled train function is shaped for the sum.
  length = config.batch_length + config.replay_context
  stream = iter(agent.stream(synthetic(agent.spaces, batch_size, length)))
  carry = agent.init_train(batch_size)

  for _ in range(warmup):
    carry, outs, mets = agent.train(carry, next(stream))
  # JAX dispatch is asynchronous, so the clock must start after the compile and
  # the warmup batches have actually landed.
  jax.block_until_ready(carry)
  start = time.perf_counter()
  # A time budget rather than a step count: one gradient step at batch 1024 is
  # ~64x one at batch 16, and a fixed count would make the big shapes take
  # minutes for no extra precision.
  steps = 0
  while steps < 4 or time.perf_counter() - start < seconds:
    carry, outs, mets = agent.train(carry, next(stream))
    steps += 1
  jax.block_until_ready(carry)
  elapsed = time.perf_counter() - start

  # Charged in batch_length, matching how train_ratio counts samples.
  samples = steps * batch_size * config.batch_length / elapsed
  print('  deter %-5s units %-4s batch %-5s %8.0f samples/s %7.2f grad/s  '
        'ratio at %dHz %6.0f   (%.2fM params)'
        % (deter, units, batch_size, samples, steps / elapsed, actor_hz,
           samples / actor_hz, params / 1e6), flush=True)
  return samples


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', default=['robot_daydreamer'])
  parser.add_argument('--deters', nargs='+', type=int, default=[256])
  parser.add_argument('--units', nargs='+', type=int, default=[256])
  parser.add_argument('--batches', nargs='+', type=int, default=[16, 64, 256, 1024])
  parser.add_argument('--seconds', type=float, default=20.0)
  parser.add_argument('--warmup', type=int, default=5)
  parser.add_argument('--actor_hz', type=float, default=45.0)
  parser.add_argument(
      '--mode', choices=['learner', 'policy'], default='learner',
      help='learner: training throughput. policy: the actor\'s per-step cost.')
  parser.add_argument(
      '--smoke', action='store_true',
      help='Tiny shapes on CPU: checks the harness, not the hardware.')
  args = parser.parse_args(argv)

  if args.smoke:
    os.environ['DREAMERV3_PLATFORM'] = 'cpu'
    args.deters, args.units, args.batches = [16], [16], [2]
    args.seconds, args.warmup = 1.0, 1

  overrides = {
      'logdir': '/tmp/bench_learner',
      # Nothing here talks to a robot; the env is built only for its spaces.
      'env.robot.host': '127.0.0.1',
  }
  if args.smoke:
    overrides.update({'batch_length': 8, 'agent.dyn.rssm.classes': 2})
  config = build_config(args.configs, overrides)
  print(f'{" ".join(args.configs)}, batch_length {config.batch_length}, '
        f'{args.seconds:.0f}s timed per shape after {args.warmup} warmup steps')
  for deter in args.deters:
    for units in args.units:
      for batch in args.batches:
        try:
          if args.mode == 'policy':
            bench_policy(config, deter, units, args.seconds, args.warmup)
            break
          bench(config, deter, units, batch, args.seconds, args.warmup,
                args.actor_hz)
        except Exception as e:  # noqa: BLE001 - report and keep sweeping
          print(f'  deter {deter:<5} units {units:<4} batch {batch:<5} '
                f'FAILED: {type(e).__name__}: {e}'[:2000], flush=True)


if __name__ == '__main__':
  sys.exit(main())
