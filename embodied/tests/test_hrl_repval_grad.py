"""The HRL replay value losses must not train the world model by default.

With ``agent.hrl_repval_grad`` on, the three replay critics (manager task,
manager exploration, worker goal) backprop into enc/RSSM and the manager loses
the Pin Pad sequence (EXPERIMENTS e1068-e1094). Every loss except the replay
value losses is scaled to zero and weight decay and warmup are off, so a change
in an enc/dyn parameter after real train steps can only come from that path.
"""
import elements
import numpy as np
import pytest
import ruamel.yaml as yaml

from dreamerv3.agent import Agent
from embodied.tests.test_goal_ae_agent import CONFIGS, train_steps

ONLY_REPVAL = {'rec': 0.0, 'rew': 0.0, 'con': 0.0, 'dyn': 0.0, 'rep': 0.0,
               'policy': 0.0, 'value': 0.0, 'repval': 1.0,
               'goal_autoencoder': 0.0, 'goal_duration_prior': 0.0,
               'goal_soft_reuse': 0.0}


def make_agent(hrl_repval_grad=None):
  from embodied.envs import dummy
  raw = yaml.YAML(typ='safe').load(CONFIGS.read())
  config = elements.Config(raw['defaults']).update(raw['debug'])
  overrides = {'agent.loss_scales': ONLY_REPVAL,
               'agent.opt.wd': 0.0, 'agent.opt.warmup': 0}
  if hrl_repval_grad is not None:
    overrides['agent.hrl_repval_grad'] = hrl_repval_grad
  config = config.update(overrides)
  assert config.agent.use_hrl and config.agent.repval_loss
  config = elements.Config(
      **config.agent, logdir='/tmp/hrl_repval_grad_test', seed=0, jax=config.jax,
      batch_size=config.batch_size, batch_length=config.batch_length,
      replay_context=config.replay_context, report_length=config.report_length,
      replica=0, replicas=1)
  env = dummy.Dummy('disc', size=(64, 64), length=20)
  keep = ('image', 'vector', 'reward', 'is_first', 'is_last', 'is_terminal')
  obs_space = {k: v for k, v in env.obs_space.items() if k in keep}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  return Agent(obs_space, act_space, config)


def world_model_change(agent):
  keys = [k for k in agent.params if k.startswith(('enc/', 'dyn/'))]
  assert keys
  before = {k: np.asarray(agent.params[k]).copy() for k in keys}
  train_steps(agent, steps=3)
  return max(float(np.abs(np.asarray(agent.params[k]) - before[k]).max()) for k in keys)


@pytest.mark.parametrize('flag', [None, False])
def test_replay_value_losses_do_not_move_the_world_model(flag):
  # None = the configs.yaml default, which must be off.
  assert world_model_change(make_agent(flag)) == 0.0


def test_the_old_behaviour_is_still_reachable():
  assert world_model_change(make_agent(True)) > 0.0
