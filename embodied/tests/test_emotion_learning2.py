"""Emotion-inspired learning, round 2 (docs/robot/LIT_EMOTION_LEARNING.md):
M2 regret distillation (agent.regret) and M5 controllability gate
(agent.control), both built on the learner's candidate outlooks
(agent.outlook, Agent._cand_outlook)."""

import functools

import elements
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ruamel.yaml as yaml

from dreamerv3.agent import (
    Agent, control_weights, outlook_deltas, outlook_score, regret_targets,
    threat_gate)

CONFIGS = elements.Path(__file__).parent.parent.parent / 'dreamerv3/configs.yaml'


# ------------------------------------------------------------ helpers ----

def test_deltas_follow_the_reflex_order():
  d = outlook_deltas('0.25,0.5,1')
  np.testing.assert_allclose(d, [0, -0.25, 0.25, -0.5, 0.5, -1, 1])
  np.testing.assert_allclose(outlook_deltas('1,0.5'), [0, -0.5, 0.5, -1, 1])


def test_outlook_score_is_the_discounted_nstep_objective():
  rew = jnp.array([[[1.0, 1.0]]])          # J=1, M=1, n=2
  val = jnp.array([[10.0]])
  crew = jnp.array([[[-1.0, 0.0]]])
  cval = jnp.array([[-4.0]])
  O = outlook_score(rew, val, crew, cval, 0.5, 2.0, 4.0, 1.0)
  R = 1 + 0.5 + 0.25 * 10
  C = -1 + 0.25 * -4
  np.testing.assert_allclose(O, [[R / 2 + C / 4]], rtol=1e-6)
  # No fear: the threat term drops out.
  O0 = outlook_score(rew, val, crew, cval, 0.5, 2.0, 4.0, 0.0)
  np.testing.assert_allclose(O0, [[R / 2]], rtol=1e-6)


def test_regret_picks_the_best_candidate_and_weights_gated_starts():
  # J=3 candidates, M=3 starts; start 0: own is best, start 1: cand 2 best
  # by 2, start 2: cand 1 best by 4 but not gated.
  O = jnp.array([[0.0, 0.0, 0.0],
                 [-1.0, 1.0, 4.0],
                 [-2.0, 2.0, 1.0]])
  cands = jnp.stack([jnp.full((3, 2), v) for v in (0.0, -0.5, 0.5)])
  gate = jnp.array([True, True, False])
  a_star, rho, w = regret_targets(O, cands, gate)
  np.testing.assert_allclose(rho, [0, 2, 4])
  np.testing.assert_allclose(a_star[:, 0], [0.0, 0.5, -0.5])
  # Mean 1 over the gated starts (rho 0 and 2 -> mean 1): w = rho / 1.
  np.testing.assert_allclose(w, [0, 2, 0])


def test_regret_without_gated_starts_is_zero():
  O = jnp.array([[0.0, 0.0], [1.0, 1.0]])
  cands = jnp.zeros((2, 2, 2))
  _, _, w = regret_targets(O, cands, jnp.array([False, False]))
  np.testing.assert_allclose(w, 0)
  _, _, w = regret_targets(jnp.zeros((2, 2)), cands, jnp.array([True, True]))
  np.testing.assert_allclose(w, 0)


def test_threat_gate_high_rising_and_out_of_contact():
  threat = np.full((2, 6), 1e-3, np.float32)
  threat[0, 3] = 0.5                         # spike, rising
  threat[1, 2] = threat[1, 3] = 0.5          # high, flat at t=3
  threat[1, 0] = 0.5                         # high at t=0: no predecessor
  cost = np.zeros((2, 6), np.float32)
  gate = np.asarray(threat_gate(threat, cost, zmin=1.0, rising=True))
  assert gate[0, 3] and gate[1, 2]
  assert not gate[1, 3] and not gate[1, 0]
  assert gate.sum() == 2
  # Without `rising` the flat high step counts too (t=0 as well).
  gate = np.asarray(threat_gate(threat, cost, zmin=1.0, rising=False))
  assert gate[1, 3] and gate[1, 0]
  # Already in pain: not a pre-pain state.
  cost[0, 3] = 0.3
  assert not np.asarray(threat_gate(threat, cost, 1.0, True, 0.05))[0, 3]


def test_control_weights_grow_with_spread_mean_one():
  O = jnp.array([[0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.1, 1.0, 3.0],
                 [0.0, -0.1, -1.0, -3.0]])
  w, D = control_weights(O, 1.0)
  w = np.asarray(w)
  np.testing.assert_allclose(w.mean(), 1.0, rtol=1e-6)
  assert np.all(np.diff(w) > 0)
  assert np.all(np.diff(np.asarray(D)) > 0)
  # Equal spread everywhere: uniform weights.
  w, _ = control_weights(jnp.tile(O[:, :1], (1, 4)) + jnp.arange(3.0)[:, None], 1.0)
  np.testing.assert_allclose(w, 1.0, rtol=1e-5)


# -------------------------------------------------------- agent-level ----

def make_agent(**overrides):
  raw = yaml.YAML(typ='safe').load(CONFIGS.read())
  config = elements.Config(raw['defaults']).update(raw['debug'])
  config = config.update({'agent.use_hrl': False, 'agent.fear.weight': 1.0,
                          **overrides})
  config = elements.Config(
      **config.agent, logdir='/tmp/le2_test', seed=0, jax=config.jax,
      batch_size=config.batch_size, batch_length=config.batch_length,
      replay_context=config.replay_context, report_length=config.report_length,
      replica=0, replicas=1)
  obs_space = {
      'vector': elements.Space(np.float32, (4,)),
      'cost': elements.Space(np.float32, ()),
      'reward': elements.Space(np.float32, ()),
      'is_first': elements.Space(bool, ()),
      'is_last': elements.Space(bool, ()),
      'is_terminal': elements.Space(bool, ()),
  }
  act_space = {'action': elements.Space(np.float32, (2,), -1, 1)}
  return Agent(obs_space, act_space, config)


def make_batch(agent, seed=0):
  B = agent.config.batch_size
  T = agent.config.batch_length + agent.config.replay_context
  rng = np.random.default_rng(seed)
  data = {}
  for key, space in agent.spaces.items():
    shape = (B, T, *space.shape)
    if space.dtype == bool:
      data[key] = np.zeros(shape, bool)
    elif np.issubdtype(space.dtype, np.integer):
      data[key] = rng.integers(0, 2, shape).astype(space.dtype)
    else:
      data[key] = rng.uniform(-1, 1, shape).astype(space.dtype)
  data['cost'] = (rng.uniform(0, 1, (B, T)) < 0.1).astype(np.float32)
  data['is_first'][:, 0] = True
  return data


def train(agent, steps=3):
  from embodied.jax import internal
  carry = agent.init_train(agent.config.batch_size)
  base = internal.device_put(make_batch(agent), agent.train_sharded)
  last = {}
  for i in range(steps):
    data = {**base, 'seed': agent._seeds(i, agent.train_mirrored)}
    carry, _, mets = agent.train(carry, data)
    if mets:
      last = mets
  assert last
  return last


@functools.lru_cache(maxsize=None)
def trained(arm):
  over = {
      'base': {},
      'm2': {'agent.regret.beta': 0.1, 'agent.regret.gate_z': -1.0,
             'agent.regret.gate_rising': False, 'agent.regret.gate_cost': 2.0},
      'm5': {'agent.control.weight': 1.0},
  }[arm]
  agent = make_agent(**over)
  before = {k: np.asarray(v).copy() for k, v in agent.params.items()}
  return agent, before, train(agent)


def test_off_by_default():
  _, _, mets = trained('base')
  assert not any(k.startswith(('regret/', 'control/',
                               'outlook/')) for k in mets)
  assert 'loss/regret' not in mets


def test_m2_regret_trains_with_finite_metrics():
  agent, before, mets = trained('m2')
  assert 'regret' in agent.model.scales
  for k in ('loss/regret', 'regret/gate_frac', 'regret/rho_all',
            'outlook/spread'):
    v = float(mets[k])
    assert np.isfinite(v), (k, v)
  assert float(mets['regret/gate_frac']) > 0
  assert float(mets['outlook/spread']) >= 0
  changed = [k for k in agent.params if k.startswith('pol/')
             and not np.allclose(before[k], np.asarray(agent.params[k]))]
  assert changed


def test_m5_control_weights_logged_and_bounded():
  _, _, mets = trained('m5')
  lo, hi = float(mets['control/w_min']), float(mets['control/w_max'])
  assert 0 <= lo <= 1 <= hi
  assert 'loss/regret' not in mets


def test_cand_outlook_shapes_and_own_command_first():
  import ninjax as nj
  agent, _, _ = trained('m5')
  # Run the method in a pure function over the agent's params.
  model = agent.model
  M = 5
  rssm = model.dyn
  starts = dict(
      deter=jnp.zeros((M, rssm.deter), jnp.float32),
      stoch=jax.nn.one_hot(jnp.zeros((M, rssm.stoch), int), rssm.classes))
  params = {k: v for k, v in agent.params.items()}

  def fn(starts):
    return model._cand_outlook(starts)

  _, (O, cands, key) = nj.pure(fn)(params, starts, seed=0)
  assert key == 'action'
  assert O.shape == (7, M) and cands.shape == (7, M, 2)
  np.testing.assert_allclose(cands[1] - cands[0], np.clip(
      cands[0] - 0.25, -1, 1) - cands[0], atol=1e-6)
  assert np.all(np.isfinite(np.asarray(O)))
