import collections
import multiprocessing as mp
import os
import pickle
import threading
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

  @property
  def actor_step(self):
    return elements.Path(self.root) / 'actor_step'


class MpCoordination:

  def __init__(self, shutdown=None, learner_ready=None, partner_pid=None):
    self._shutdown = shutdown
    self._learner_ready = learner_ready
    self._partner_pid = partner_pid

  def partner_exited(self):
    if self._partner_pid is None:
      return False
    pid = self._partner_pid.value if hasattr(self._partner_pid, 'value') else self._partner_pid
    if not pid:
      return False
    try:
      os.kill(pid, 0)
    except OSError:
      return True
    return False

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
    self._sync_lock = threading.Lock()

  def append(self, transition, worker=0):
    self._replay.add(transition, worker)

  def flush(self):
    self._replay.save()

  def sync(self, amount=None):
    replay = self._replay
    with self._sync_lock:
      if amount is None:
        amount = replay.capacity
      replay.load(amount=amount)

  def ready(self, batch_size, batch_length):
    return len(self._replay.sampler) >= batch_size

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


def launch(
    make_agent, make_env, make_logger, make_replay, make_stream, args,
    make_actor_agent=None, make_actor_replay=None):
  paths = _shared_paths(args.logdir)
  make_actor_agent = make_actor_agent or make_agent
  make_actor_replay = make_actor_replay or make_replay
  ctx = mp.get_context('spawn')
  partner_pid = ctx.Value('i', 0)
  coordination = MpCoordination(ctx.Event(), ctx.Event(), partner_pid)
  actor_process = ctx.Process(
      name='dreamerv3_actor',
      target=_actor_process,
      args=(make_actor_agent, make_env, make_logger, make_actor_replay, paths,
            args, coordination),
      daemon=False)
  learner_process = ctx.Process(
      name='dreamerv3_learner',
      target=_learner_process,
      args=(make_agent, make_logger, make_replay, make_stream, paths,
            args, coordination),
      daemon=False)
  processes = [learner_process, actor_process]
  for process in processes:
    process.start()
  partner_pid.value = actor_process.pid
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


def _write_actor_step(path, step):
  _atomic_write(path, str(int(step)).encode('utf-8'))


def _read_actor_step(path):
  if not path.exists():
    return 0
  return int(path.read_text().strip())


def _actor_finished(paths, args, coordination):
  if coordination is not None and (
      coordination.shutdown_set() or coordination.partner_exited()):
    return True
  return _read_actor_step(paths.actor_step) >= int(args.steps)


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
  should_sync_policy = embodied.LocalClock(args.online_sync_every)
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
  flush_pending = 0
  flush_every = max(1, int(args.online_actor_flush_steps))
  while step < args.steps:
    if should_sync_policy(step):
      shared_policy.load_into_agent(agent)
    driver(policy, steps=100)
    _write_actor_step(paths.actor_step, step)
    flush_pending += 100 * args.envs
    if flush_pending >= flush_every:
      shared_buffer.flush()
      flush_pending = 0
    if should_log(step):
      logger.add(epstats.result(), prefix='epstats')
      logger.add(shared_buffer.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()
  shared_buffer.flush()
  _write_actor_step(paths.actor_step, step)
  logger.close()


def _log_learner(logdir, message):
  path = elements.Path(logdir) / 'online_learner.log'
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, 'a', encoding='utf-8') as file:
    file.write(f'{time.time():.3f} {message}\n')


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
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)
  should_sync_policy = embodied.LocalClock(args.online_sync_every)
  should_sync_replay = embodied.LocalClock(args.online_replay_sync_interval)
  carry_train = agent.init_train(args.batch_size)
  last_actor_step = -1

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
  prefill_deadline = None
  prefill_log = time.time()
  prefill_target = max(args.batch_size * args.batch_length, args.batch_size)
  while not shared_buffer.ready(args.batch_size, args.batch_length):
    shared_buffer.sync(amount=prefill_target)
    if time.time() - prefill_log >= 30:
      _log_learner(
          logdir,
          f'prefill waiting items={len(replay)} sampler={len(replay.sampler)}')
      prefill_log = time.time()
    if coordination is not None and coordination.partner_exited():
      prefill_deadline = prefill_deadline or time.time() + 30
    if prefill_deadline and time.time() > prefill_deadline:
      break
    time.sleep(0.05)
  if not shared_buffer.ready(args.batch_size, args.batch_length):
    print('Actor exited before online replay prefill completed.')
    _log_learner(logdir, 'prefill failed')
    logger.close()
    return

  _log_learner(
      logdir, f'prefill complete items={len(replay)} sampler={len(replay.sampler)}')
  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))
  carry_report = agent.init_report(args.batch_size)
  trained = False
  while not _actor_finished(paths, args, coordination):
    actor_step = _read_actor_step(paths.actor_step)
    if actor_step > last_actor_step:
      shared_buffer.sync(amount=prefill_target)
      last_actor_step = actor_step
    elif should_sync_replay(step):
      shared_buffer.sync(amount=prefill_target)
    if not shared_buffer.ready(args.batch_size, args.batch_length):
      time.sleep(0.01)
      continue
    repeats = should_train(actor_step)
    if repeats <= 0:
      time.sleep(0.001)
      continue
    for _ in range(repeats):
      if _actor_finished(paths, args, coordination):
        break
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train, outs, mets = agent.train(carry_train, batch)
      train_fps.step(batch_steps)
      step.increment(batch_steps)
      if not trained:
        trained = True
        _log_learner(logdir, f'first train step={int(step)} actor_step={actor_step}')
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
      if should_report(step) and len(replay):
        agg = elements.Agg()
        for _ in range(args.consec_report * args.report_batches):
          carry_report, mets = agent.report(carry_report, next(stream_report))
          agg.add(mets)
        logger.add(agg.result(), prefix='report')
    if should_sync_policy(step):
      shared_policy.publish(agent)
    if should_log(step):
      logger.add(train_agg.result())
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()
    if should_save(step):
      cp.save()
  _log_learner(
      logdir,
      f'learner stop step={int(step)} actor_step={_read_actor_step(paths.actor_step)}')
  logger.close()
