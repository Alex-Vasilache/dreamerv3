"""Agent-level integration tests for the goal-autoencoder ablation arms.

The unit tests in ``test_goal_ae.py`` check the module in isolation. These
build the *real* ``dreamerv3.agent.Agent`` from the *real* ``configs.yaml``
blocks and take actual train steps, which is the only way to catch the failure
modes that live in the wiring rather than the math: a config key that does not
reach the module, a parameter that lands outside its optimizer group, a loss
key that does not match its scale, or a call site in ``hrl/`` that still
assumes the Director head.
"""
import functools
import re

import elements
import jax
import numpy as np
import pytest
import ruamel.yaml as yaml

import embodied
from dreamerv3.agent import Agent

ARMS = ['goal_vq', 'goal_som', 'goal_lipvq', 'goal_som_lipvq']
CONFIGS = elements.Path(__file__).parent.parent.parent / 'dreamerv3/configs.yaml'


def make_config(*blocks, **overrides):
  raw = yaml.YAML(typ='safe').load(CONFIGS.read())
  config = elements.Config(raw['defaults'])
  for block in ('debug',) + blocks:
    config = config.update(raw[block])
  config = config.update(overrides) if overrides else config
  return elements.Config(
      **config.agent, logdir='/tmp/goal_ae_test', seed=0, jax=config.jax,
      batch_size=config.batch_size, batch_length=config.batch_length,
      replay_context=config.replay_context, report_length=config.report_length,
      replica=0, replicas=1)


def make_agent(*blocks, **overrides):
  from embodied.envs import dummy
  # 64x64: the simple CNN encoder downsamples 4x (mults [2,3,4,4]), so a
  # smaller image reshapes to a zero-sized spatial grid.
  env = dummy.Dummy('disc', size=(64, 64), length=20)
  # Keep only the keys a real DMC/loconav env has. The dummy env's `int2d`
  # yields a per-element (B, T, 2) reconstruction loss, which trips the agent's
  # own (B, T) loss-shape assertion regardless of the goal-AE arm.
  keep = ('image', 'vector', 'reward', 'is_first', 'is_last', 'is_terminal')
  obs_space = {k: v for k, v in env.obs_space.items() if k in keep}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  return Agent(obs_space, act_space, make_config(*blocks, **overrides))


def make_batch(agent, batch=None, seed=0):
  """A synthetic replay batch matching exactly what ``Agent.train`` expects.

  Shapes come from ``agent.spaces`` (obs + act + ext) at length
  ``batch_length + replay_context``, and ``train`` pops a ``seed`` entry that
  the real stream wrapper injects. Random rather than zeros so the goal
  autoencoder has something to reconstruct and gradients are not degenerate.
  """
  batch = batch or agent.config.batch_size
  length = agent.config.batch_length + agent.config.replay_context
  rng = np.random.default_rng(seed)
  data = {}
  for key, space in agent.spaces.items():
    shape = (batch, length, *space.shape)
    if space.dtype == bool:
      data[key] = np.zeros(shape, bool)
    elif np.issubdtype(space.dtype, np.integer):
      hi = 2 if space.classes is None else int(
          np.asarray(space.classes).flatten()[0])
      data[key] = rng.integers(0, max(hi, 2), shape).astype(space.dtype)
    else:
      data[key] = rng.normal(0, 1, shape).astype(space.dtype)
  data['is_first'] = np.zeros((batch, length), bool)
  data['is_first'][:, 0] = True
  data['is_last'] = np.zeros((batch, length), bool)
  data['is_terminal'] = np.zeros((batch, length), bool)
  return data


def train_steps(agent, steps=3):
  """Run real train steps and return the last non-empty metrics.

  ``Agent.train`` returns metrics one call behind (they are fetched
  asynchronously), so a single step yields an empty dict.
  """
  from embodied.jax import internal
  carry = agent.init_train(agent.config.batch_size)
  # embodied runs with jax_transfer_guard='disallow', so the batch has to be
  # placed on device exactly as Agent.stream's wrapper does it -- handing
  # train() raw numpy raises "Disallowed host-to-device transfer".
  base = internal.device_put(make_batch(agent), agent.train_sharded)
  last = {}
  for i in range(steps):
    data = {**base, 'seed': agent._seeds(i, agent.train_mirrored)}
    carry, outs, mets = agent.train(carry, data)
    if mets:
      last = mets
  assert last, 'no metrics after {} steps'.format(steps)
  return last


@functools.lru_cache(maxsize=None)
def trained(arm):
  """Build one agent per arm, train it, and cache the result.

  Constructing and JIT-compiling a full agent dominates this file's runtime, so
  every read-only assertion shares a single build per arm.
  """
  agent = make_agent(*([arm] if arm else []))
  before = {k: np.asarray(v).copy() for k, v in agent.params.items()}
  mets = train_steps(agent, steps=3)
  return agent, before, mets


class TestBuildsAndTrains:

  @pytest.mark.parametrize('arm', ARMS)
  def test_agent_builds(self, arm):
    agent, _, _ = trained(arm)
    assert agent.model.goal_ae_impl == 'vq'

  def test_director_arm_still_builds(self):
    agent, _, _ = trained('')
    assert agent.model.goal_ae_impl == 'director'

  @pytest.mark.parametrize('arm', ARMS)
  def test_train_step_metrics_are_finite(self, arm):
    _, _, mets = trained(arm)
    assert 'loss/goal_autoencoder' in mets
    for k, v in mets.items():
      v = np.asarray(v)
      assert np.isfinite(v).all(), (arm, k, v)

  @pytest.mark.parametrize('arm', ARMS)
  def test_vq_metrics_are_logged(self, arm):
    _, _, mets = trained(arm)
    # agent.train returns metrics unprefixed; run/train.py is what adds the
    # 'train/' prefix when it aggregates them.
    for key in ('goal/rec_q', 'goal/codebook', 'goal/commit',
                'goal/som', 'goal/quant_err',
                'goal/used_frac', 'goal/perplexity',
                'goal/lip_penalty', 'goal/lip_bound_max'):
      assert key in mets, (arm, key, sorted(k for k in mets if 'goal' in k))

  def test_som_and_lip_terms_are_active_only_in_their_arms(self):
    def m(arm, key):
      return float(trained(arm)[2]['goal/' + key])
    assert m('goal_vq', 'som') == 0.0
    assert m('goal_som', 'som') > 0.0
    assert m('goal_vq', 'lip_penalty') == 0.0
    assert m('goal_lipvq', 'lip_penalty') > 0.0
    assert m('goal_som_lipvq', 'som') > 0.0
    assert m('goal_som_lipvq', 'lip_penalty') > 0.0

  def test_director_arm_logs_no_vq_metrics(self):
    _, _, mets = trained('')
    assert 'goal/quant_err' not in mets
    assert 'goal/rec_mean' in mets


class TestParameterPlumbing:

  def _table_key(self, agent):
    # agent.params also holds optimizer slots under opt/state_*/..., which end
    # with the same suffix; the model parameter is the one without that prefix.
    keys = [k for k in agent.params
            if k.endswith('goal_dec/codebook/table') and not k.startswith('opt/')]
    assert len(keys) == 1, sorted(k for k in agent.params if 'codebook' in k)
    return keys[0]

  @pytest.mark.parametrize('arm', ARMS)
  def test_codebook_is_in_an_optimizer_group(self, arm):
    # The codebook must belong to a module group the optimizer owns, not be
    # silently left out of every group (which would freeze it without error).
    agent, _, _ = trained(arm)
    table = self._table_key(agent)
    owned = set()
    for module in agent.model.modules:
      owned |= {k for k in agent.params if k.startswith(module.path + '/')}
    assert table in owned

  @pytest.mark.parametrize('arm', ARMS)
  def test_policy_keys_ship_the_codebook_to_the_actor(self, arm):
    # In the online/parallel path the actor only receives params matching this
    # regex; a codebook outside it would decode goals from stale weights.
    agent, _, _ = trained(arm)
    table = self._table_key(agent)
    assert re.search(agent.model.policy_keys.lstrip('^'), table), table
    assert table in set(agent.policy_keys), sorted(agent.policy_keys)[:5]

  @pytest.mark.parametrize('arm', ARMS)
  def test_codebook_actually_changes_during_training(self, arm):
    agent, before, _ = trained(arm)
    table = self._table_key(agent)
    assert not np.allclose(before[table], np.asarray(agent.params[table])), arm

  @pytest.mark.parametrize('arm', ARMS)
  def test_encoder_actually_changes_during_training(self, arm):
    agent, before, _ = trained(arm)
    key = [k for k in agent.params
           if k.endswith('goal_enc/out/kernel') and not k.startswith('opt/')][0]
    assert not np.allclose(before[key], np.asarray(agent.params[key])), arm

  @pytest.mark.parametrize('arm', ['goal_vq', 'goal_som'])
  def test_no_lipschitz_bounds_without_the_lip_arms(self, arm):
    agent, _, _ = trained(arm)
    assert not [k for k in agent.params
                if k.endswith('/c') and not k.startswith('opt/')], arm

  @pytest.mark.parametrize('arm', ['goal_lipvq', 'goal_som_lipvq'])
  def test_lip_bounds_on_both_networks(self, arm):
    # lip_apply defaults to enc_dec (motivation.tex): the decoder's bound is
    # what makes a small code edit a small GOAL edit.
    agent, _, _ = trained(arm)
    cs = [k for k in agent.params
          if k.endswith('/c') and not k.startswith('opt/')]
    assert any('goal_enc' in k for k in cs), (arm, cs)
    assert any('goal_dec' in k for k in cs), (arm, cs)

  @pytest.mark.parametrize('arm', ['goal_lipvq', 'goal_som_lipvq'])
  def test_lipschitz_bounds_receive_updates(self, arm):
    # The goal optimizer warms up linearly over 1000 steps, so after 3 steps
    # the learning rate is ~1.2e-7 and the bounds move far below allclose's
    # default tolerance -- assert a nonzero change, not a visible one. That the
    # penalty can actually pull the bounds down is covered at a usable learning
    # rate in test_goal_ae.py::TestLearning.
    agent, before, _ = trained(arm)
    cs = [k for k in agent.params
          if k.endswith('/c') and not k.startswith('opt/')]
    assert cs, arm
    moved = [k for k in cs
             if float(np.abs(before[k] - np.asarray(agent.params[k])).max()) > 0]
    assert moved, (arm, cs)


class TestPolicyAndReport:

  @pytest.mark.parametrize('arm', ARMS)
  def test_policy_step_runs(self, arm):
    # The policy path decodes the manager's sampled code through the codebook;
    # a shape or ownership mistake there would only surface at rollout time.
    agent, _, _ = trained(arm)
    carry = agent.init_policy(4)
    obs = {k: np.zeros((4, *v.shape), v.dtype)
           for k, v in agent.obs_space.items()}
    obs['is_first'] = np.ones(4, bool)
    carry, act, outs = agent.policy(carry, obs)
    for key in agent.act_space:
      assert key in act, (key, sorted(act))
      assert np.isfinite(np.asarray(act[key])).all(), key


class TestSaveLoad:

  @pytest.mark.parametrize('arm', ARMS)
  def test_checkpoint_roundtrip_preserves_the_codebook(self, arm):
    # Requeue depends on this: a codebook that failed to save would silently
    # reset at every 48h segment boundary.
    agent, _, _ = trained(arm)
    table = [k for k in agent.params
             if k.endswith('goal_dec/codebook/table')
             and not k.startswith('opt/')][0]
    saved = agent.save()
    wanted = np.asarray(agent.params[table]).copy()
    fresh = make_agent(arm)
    assert not np.allclose(np.asarray(fresh.params[table]), wanted)
    fresh.load(saved)
    np.testing.assert_allclose(
        np.asarray(fresh.params[table]), wanted, rtol=1e-6)
