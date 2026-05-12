import collections
import multiprocessing as mp
import os
import pickle
import time
from dataclasses import dataclass
from functools import partial as bind

import elements
import embodied
import numpy as np
import portal


@dataclass(frozen=True)
class OnlineSharedPaths:
  root: str

  @property
  def experience(self):
    return elements.Path(self.root) / 'experience'

  @property
  def policy(self):
    return elements.Path(self.root) / 'policy'

  @property
  def learner_ready(self):
    return elements.Path(self.root) / 'learner_ready'

  @property
  def shutdown(self):
    return elements.Path(self.root) / 'shutdown'


class MpCoordination:

  def __init__(self, shutdown=None, learner_ready=None):
    self._shutdown = shutdown
    self._learner_ready = learner_ready

  def wait_learner_ready(self):
    if self._learner_ready is not None:
      self._learner_ready.wait()

  def set_learner_ready(self):
    if self._learner_ready is not None:
      self._learner_ready.set()

  def shutdown_set(self):
    return self._shutdown is not None and self._shutdown.is_set()

  def set_shutdown(self):
    if self._shutdown is not None:
      self._shutdown.set()


class FileCoordination:

  def __init__(self, paths):
    self._paths = paths

  def wait_learner_ready(self, poll=0.05):
    while not self._paths.learner_ready.exists():
      time.sleep(poll)

  def set_learner_ready(self):
    self._paths.learner_ready.parent.mkdir(parents=True, exist_ok=True)
    self._paths.learner_ready.write('1')

  def shutdown_set(self):
    return self._paths.shutdown.exists()

  def set_shutdown(self):
    self._paths.shutdown.parent.mkdir(parents=True, exist_ok=True)
    self._paths.shutdown.write('1')


class SharedExperienceBuffer:
  """Filesystem-backed experience store shared between actor and learner."""

  def __init__(self, replay):
    self._replay = replay

  def append(self, transition, worker=0):
    self._replay.add(transition, worker)

  def flush(self):
    self._replay.save()

  def refresh(self):
    replay = self._replay
    if not replay.directory:
      replay.load(amount=replay.capacity)
      return
    replay.items.clear()
    replay.fifo.clear()
    replay.itemid = 0
    if hasattr(replay.sampler, 'indices'):
      replay.sampler.indices.clear()
      replay.sampler.keys.clear()
    replay.chunks.clear()
    replay.refs.clear()
    replay.streams.clear()
    replay.current.clear()
    replay.saved.clear()
    if replay.online:
      replay.lengths.clear()
      replay.queue.clear()
    replay.load(amount=replay.capacity)

  def ready(self, batch_size, batch_length):
    del batch_length
    return len(self._replay) >= batch_size

  def stats(self):
    return self._replay.stats()


class SharedPolicyWeights:
  """Synchronized policy weights shared between learner and actor."""

  def __init__(self, directory):
    self.directory = elements.Path(directory)
    self.directory.mkdir(parents=True, exist_ok=True)
    self.latest = self.directory / 'latest'
    self._stamp = None

  def publish(self, agent):
    data = agent.save()
    payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    stamp = str(time.time_ns())
    path = self.directory / f'policy_{stamp}.pkl'
    _atomic_write(path, payload)
    _atomic_write(self.latest, stamp.encode('utf-8'))
    self._stamp = stamp

  def load_into_agent(self, agent):
    if not self.latest.exists():
      return False
    stamp = self.latest.read_text().strip()
    if stamp == self._stamp:
      return False
    path = self.directory / f'policy_{stamp}.pkl'
    if not path.exists():
      return False
    data = pickle.loads(path.read_bytes())
    agent.load(data, regex=agent.model.policy_keys)
    self._stamp = stamp
    return True


def _shared_paths(logdir):
  paths = OnlineSharedPaths(str(elements.Path(logdir) / 'online_shared'))
  paths.experience.mkdir(parents=True, exist_ok=True)
  paths.policy.mkdir(parents=True, exist_ok=True)
  return paths


def launch(make_agent, make_env, make_logger, make_replay, make_stream, args):
  paths = _shared_paths(args.logdir)
  ctx = mp.get_context('spawn')
  coordination = MpCoordination(ctx.Event(), ctx.Event())
  processes = [
      ctx.Process(
          name='dreamerv3_learner',
          target=_learner_process,
          args=(make_agent, make_logger, make_replay, make_stream, paths,
                args, coordination),
          daemon=False),
      ctx.Process(
          name='dreamerv3_actor',
          target=_actor_process,
          args=(make_agent, make_env, make_logger, make_replay, paths, args,
                coordination),
          daemon=False),
  ]
  for process in processes:
    process.start()
  try:
    for process in processes:
      process.join()
  except KeyboardInterrupt:
    for process in processes:
      process.terminate()
    for process in processes:
      process.join()


def standalone_actor(make_agent, make_env, make_logger, make_replay, args):
  paths = _shared_paths(args.logdir)
  coordination = FileCoordination(paths)
  _setup_process('actor', args)
  try:
    run_actor(make_agent, make_env, make_logger, make_replay, paths, args,
              coordination)
  finally:
    coordination.set_shutdown()


def standalone_learner(make_agent, make_logger, make_replay, make_stream, args):
  paths = _shared_paths(args.logdir)
  coordination = FileCoordination(paths)
  _setup_process('learner', args)
  run_learner(make_agent, make_logger, make_replay, make_stream, paths, args,
              coordination)


def _actor_process(make_agent, make_env, make_logger, make_replay, paths, args,
                   coordination):
  _setup_process('actor', args)
  try:
    run_actor(make_agent, make_env, make_logger, make_replay, paths, args,
              coordination)
  finally:
    coordination.set_shutdown()


def _learner_process(make_agent, make_logger, make_replay, make_stream, paths,
                     args, coordination):
  _setup_process('learner', args)
  run_learner(make_agent, make_logger, make_replay, make_stream, paths, args,
              coordination)


def _setup_process(role, args):
  logdir = elements.Path(args.logdir)
  portal.setup(
      errfile=logdir / f'error_{role}',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      ipv6=False)


def _atomic_write(path, data):
  path = elements.Path(path)
  tmp = path.parent / f'.{path.name}.tmp'
  tmp.write(data, mode='wb')
  os.replace(tmp, path)


def run_actor(make_agent, make_env, make_logger, make_replay, paths, args,
              coordination=None):
  agent = make_agent()
  logger = make_logger()
  replay = make_replay()
  shared_buffer = SharedExperienceBuffer(replay)
  shared_policy = SharedPolicyWeights(paths.policy)

  if coordination is not None:
    coordination.wait_learner_ready()
  step = logger.step
  usage = elements.Usage(**args.usage)
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  should_log = embodied.LocalClock(args.log_every)
  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(shared_buffer.append)
  driver.on_step(logfn)

  shared_policy.load_into_agent(agent)

  print('Start actor loop')
  policy = lambda *a, **kw: agent.policy(*a, mode='train', **kw)
  driver.reset(agent.init_policy)
  while step < args.steps:
    shared_policy.load_into_agent(agent)
    driver(policy, steps=max(10, args.batch_size))
    shared_buffer.flush()
    if should_log(step):
      logger.add(epstats.result(), prefix='epstats')
      logger.add(shared_buffer.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()
  shared_buffer.flush()
  logger.close()


def run_learner(make_agent, make_logger, make_replay, make_stream, paths, args,
                coordination=None):
  agent = make_agent()
  logger = make_logger()
  replay = make_replay()
  shared_buffer = SharedExperienceBuffer(replay)
  shared_policy = SharedPolicyWeights(paths.policy)

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  train_fps = elements.FPS()
  batch_steps = args.batch_size * args.batch_length
  should_log = embodied.LocalClock(args.log_every)
  should_save = embodied.LocalClock(args.save_every)
  carry_train = agent.init_train(args.batch_size)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()
  shared_policy.publish(agent)
  if coordination is not None:
    coordination.set_learner_ready()

  print('Start learner loop')
  print('Waiting for online replay prefill...')
  while not shared_buffer.ready(args.batch_size, args.batch_length):
    shared_buffer.refresh()
    if coordination is not None and coordination.shutdown_set():
      shared_buffer.refresh()
      if shared_buffer.ready(args.batch_size, args.batch_length):
        break
      print('Actor exited before online replay prefill completed.')
      logger.close()
      return
    time.sleep(0.05)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  while step < args.steps:
    shared_buffer.refresh()
    if not shared_buffer.ready(args.batch_size, args.batch_length):
      if coordination is not None and coordination.shutdown_set():
        shared_buffer.refresh()
        if not shared_buffer.ready(args.batch_size, args.batch_length):
          break
      time.sleep(0.01)
      continue
    with elements.timer.section('stream_next'):
      batch = next(stream_train)
    carry_train, outs, mets = agent.train(carry_train, batch)
    train_fps.step(batch_steps)
    step.increment(batch_steps)
    if 'replay' in outs:
      replay.update(outs['replay'])
    train_agg.add(mets, prefix='train')
    shared_policy.publish(agent)
    if should_log(step):
      logger.add(train_agg.result())
      logger.add(shared_buffer.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()
    if should_save(step):
      cp.save()
  logger.close()
