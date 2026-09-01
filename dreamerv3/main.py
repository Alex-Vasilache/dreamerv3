import importlib
import os
import pathlib
import sys
from functools import partial as bind

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import collections

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml


class WandBOutputWithFPS(elements.logger.WandBOutput):
  """WandBOutput that passes a configurable fps to wandb.Video for faster gifs."""

  def __init__(self, name, video_fps=4, report_video_fps=None, videos=True, **kwargs):
    super().__init__(name, **kwargs)
    self._video_fps = video_fps
    # Report videos are subsampled by report_video_time_stride, so they need a
    # proportionally lower fps to appear at the same speed as episode videos.
    self._report_video_fps = report_video_fps if report_video_fps is not None else video_fps
    self._videos = videos

  def __call__(self, summaries):
    import wandb
    bystep = collections.defaultdict(dict)
    for step, name, value in summaries:
      if not self._pattern.search(name):
        continue
      if isinstance(value, str):
        bystep[step][name] = value
      elif len(value.shape) == 0:
        bystep[step][name] = float(value)
      elif len(value.shape) == 1:
        bystep[step][name] = wandb.Histogram(value)
      elif len(value.shape) in (2, 3):
        value = value[..., None] if len(value.shape) == 2 else value
        assert value.shape[3] in [1, 3, 4], value.shape
        if value.dtype != np.uint8:
          value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
        value = np.transpose(value, [2, 0, 1])
        bystep[step][name] = wandb.Image(value)
      elif len(value.shape) == 4:
        if not self._videos:
          continue
        assert value.shape[3] in [1, 3, 4], value.shape
        value = np.transpose(value, [0, 3, 1, 2])
        if value.dtype != np.uint8:
          value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
        # Episode rollout videos (epstats/) are 1 frame/step → use full fps.
        # Report videos are stride-subsampled → use the lower report fps.
        fps = self._video_fps if name.startswith('epstats/') else self._report_video_fps
        bystep[step][name] = wandb.Video(value, fps=fps, format='gif')
    for step, metrics in bystep.items():
      self._wandb.log(metrics, step=step)


def main(argv=None):
  from .agent import Agent
  [elements.print(line) for line in Agent.banner]

  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))

  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)

  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  print('Run script:', config.script)
  if not config.script.endswith(('_env', '_replay')):
    logdir.mkdir()
    if config.script != 'online_actor':
      config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
      online_actor_flush_steps=config.online_actor_flush_steps,
      online_sync_every=config.online_sync_every,
      online_replay_sync_interval=config.online_replay_sync_interval,
  )

  if config.script == 'train':
    if config.online_learning:
      actor_agent = bind(make_agent, config)
      if config.online_actor_cpu:
        actor_agent = bind(make_agent, config.update(
            jax=config.jax.update(platform='cpu', prealloc=False)))
      actor_replay = bind(make_online_actor_replay, config, 'replay')
      embodied.run.online.launch(
          bind(make_agent, config),
          bind(make_env, config),
          bind(make_logger, config),
          bind(make_replay, config, 'replay'),
          bind(make_stream, config),
          args,
          make_actor_agent=actor_agent,
          make_actor_replay=actor_replay)
    else:
      embodied.run.train(
          bind(make_agent, config),
          bind(make_replay, config, 'replay'),
          bind(make_env, config),
          bind(make_stream, config),
          bind(make_logger, config),
          args)

  elif config.script == 'online_learner':
    embodied.run.online.standalone_learner(
        bind(make_agent, config),
        bind(make_logger, config),
        bind(make_replay, config, 'replay'),
        bind(make_stream, config),
        args)

  elif config.script == 'online_actor':
    actor_agent = bind(make_agent, config)
    if config.online_actor_cpu:
      actor_agent = bind(make_agent, config.update(
          jax=config.jax.update(platform='cpu', prealloc=False)))
    embodied.run.online.standalone_actor(
        actor_agent,
        bind(make_env, config),
        bind(make_logger, config),
        bind(make_online_actor_replay, config, 'replay'),
        args)

  elif config.script == 'train_eval':
    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'eval_only':
    embodied.run.eval_only(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel':
    embodied.run.parallel.combined(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel_env':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_env(
        bind(make_env, config), config.replica, args, is_eval)

  elif config.script == 'parallel_envs':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_envs(
        bind(make_env, config), bind(make_env, config), args)

  elif config.script == 'parallel_replay':
    embodied.run.parallel.parallel_replay(
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_stream, config),
        args)

  else:
    raise NotImplementedError(config.script)


def make_agent(config):
  from .agent import Agent
  env = make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)
  cpdir = elements.Path(config.logdir)
  cpdir = cpdir.parent if config.replicas > 1 else cpdir
  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      # The code-count memory derives its decay and insertion weight from the
      # replay window rather than hardcoding them, so it needs these two.
      train_ratio=config.run.train_ratio,
      replay_size=config.replay.size,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def make_logger(config):
  step = elements.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = []
  outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
  for output in config.logger.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
      outputs.append(elements.logger.JSONLOutput(
          logdir, 'scores.jsonl', 'episode/score'))
    elif output == 'tensorboard':
      outputs.append(elements.logger.TensorBoardOutput(
          logdir, config.logger.fps))
    elif output == 'expa':
      exp = logdir.split('/')[-4]
      run = '/'.join(logdir.split('/')[-3:])
      proj = 'embodied' if logdir.startswith(('/cns/', 'gs://')) else 'debug'
      outputs.append(elements.logger.ExpaOutput(
          exp, run, proj, config.logger.user, config.flat))
    elif output == 'wandb':
      import hashlib, os as _os
      # In online-learning mode (actor + learner in separate processes), both
      # processes call make_logger() → wandb.init().  Sharing the same run_id
      # creates two competing WandB connections: the learner's report metrics
      # and videos end up in a local run-dir that is "shadowed" by the actor's
      # resumed connection, so they never appear on the dashboard.
      #
      # Fix: give each role its own run_id within a shared WandB group so both
      # actor (epstats) and learner (train/report) metrics appear, without
      # conflicts.  Non-online runs use the original single run_id (no group).
      is_online = (
          config.online_learning or
          config.script in ('online_actor', 'online_learner'))
      is_actor = (
          _os.environ.get('DREAMERV3_ACTOR_PROCESS') or
          config.script == 'online_actor')
      name = '/'.join(logdir.split('/')[-3:])
      group_id = hashlib.md5(logdir.encode()).hexdigest()[:8]
      if is_online:
        role = 'actor' if is_actor else 'learner'
        run_id = group_id + f'_{role}'
        run_name = name + f'/{role}'
      else:
        run_id = group_id
        run_name = name
      kwargs = dict(
          mode=config.logger.wandb_mode,
          config=config.flat,
          id=run_id,
          resume='allow',
      )
      if is_online:
        kwargs['group'] = group_id
        kwargs['job_type'] = 'actor' if is_actor else 'learner'
      if config.logger.wandb_project:
        kwargs['project'] = config.logger.wandb_project
      if config.logger.wandb_entity:
        kwargs['entity'] = config.logger.wandb_entity
      wandb_fps = int(getattr(config.logger, 'wandb_fps', 4))
      report_video_fps = int(getattr(config.logger, 'report_wandb_fps', 5))
      wandb_videos = bool(getattr(config.logger, 'wandb_videos', False))
      try:
        outputs.append(WandBOutputWithFPS(
            run_name, video_fps=wandb_fps, report_video_fps=report_video_fps,
            videos=wandb_videos, **kwargs))
      except Exception as e:
        print(f'WandB init failed, skipping WandB output: {e}')
    elif output == 'scope':
      outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
    else:
      raise NotImplementedError(output)
  logger = elements.Logger(step, outputs, multiplier)
  return logger


def make_online_actor_replay(config, folder, mode='train'):
  cap = config.online_actor_replay_size
  if cap:
    config = config.update(replay=config.replay.update(size=cap))
  return make_replay(config, folder, mode)


def make_replay(config, folder, mode='train'):
  batlen = config.batch_length if mode == 'train' else config.report_length
  consec = config.consec_train if mode == 'train' else config.consec_report
  capacity = config.replay.size if mode == 'train' else config.replay.size / 10
  train_need = config.consec_train * config.batch_length + config.replay_context
  report_need = (
      config.consec_report * config.report_length + config.replay_context)
  # Train and report streams sample the same replay; chunks must cover both.
  length = max(train_need, report_need) if mode == 'train' else (
      consec * batlen + config.replay_context)
  assert config.batch_size * length <= capacity

  if folder == 'replay' and config.online_learning:
    directory = elements.Path(config.logdir) / 'online_shared' / 'experience'
  else:
    directory = elements.Path(config.logdir) / folder
  if config.replicas > 1:
    directory /= f'{config.replica:05}'
  kwargs = dict(
      length=length, capacity=int(capacity), online=config.replay.online,
      chunksize=config.replay.chunksize, directory=directory)
  if folder == 'replay' and config.online_learning:
    kwargs['save_wait'] = True

  if config.replay.fracs.uniform < 1 and mode == 'train':
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
    selectors = embodied.replay.selectors
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(),
        priority=selectors.Prioritized(**config.replay.prio),
        recency=selectors.Recency(recency),
    ), config.replay.fracs)

  return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
  suite, task = config.task.split('_', 1)
  if suite == 'memmaze':
    from embodied.envs import from_gym
    import memory_maze  # noqa
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'langroom': 'embodied.envs.langroom:LangRoom',
      'procgen': 'embodied.envs.procgen:ProcGen',
      'bsuite': 'embodied.envs.bsuite:BSuite',
      'memmaze': lambda task, **kw: from_gym.FromGym(
          f'MemoryMaze-{task}-v0', **kw),
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'
  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.NormalizeAction(env, name)
  env = embodied.wrappers.UnifyDtypes(env)
  env = embodied.wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)
  return env


def make_stream(config, replay, mode):
  train_need = config.consec_train * config.batch_length + config.replay_context
  report_need = (
      config.consec_report * config.report_length + config.replay_context)
  fn = bind(replay.sample, config.batch_size, mode)
  stream = embodied.streams.Stateless(fn)
  stream = embodied.streams.Consec(
      stream,
      length=config.batch_length if mode == 'train' else config.report_length,
      consec=config.consec_train if mode == 'train' else config.consec_report,
      prefix=config.replay_context,
      # Longer stored sequences (for report) forbid strict==True on train slices.
      strict=(mode == 'train' and report_need <= train_need),
      contiguous=True)

  return stream


if __name__ == '__main__':
  main()
