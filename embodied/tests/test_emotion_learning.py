"""Emotion-inspired learning mechanisms, round 1 (docs/robot/LIT_EMOTION_LEARNING.md):
M1 retroactive pain tagging (replay `tag`), M3 stress -> per-start lambda
(agent.stress_lam), M4 fast fear (MultiOptimizer module_scale, fear.lr_mult)."""

import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
import pytest

import embodied
from embodied.core.replay import Replay, pain_tag
from embodied.jax import nets
from embodied.jax.opt import MultiOptimizer, scale_by_rms
from dreamerv3.agent import stress_lambda
from dreamerv3.hrl import lambda_return

f32 = jnp.float32


# ---------------------------------------------------------------- M1 ----

class TestPainTag:

  def test_calm_step_before_pain_is_tagged(self):
    cost = np.zeros(20); cost[10] = 0.5
    assert pain_tag(cost, window=5)

  def test_no_pain_no_tag(self):
    assert not pain_tag(np.zeros(20), window=5)

  def test_pain_only_in_contact_is_not_tagged(self):
    # Every step before the pain is itself in pain (already in contact).
    cost = np.full(20, 0.5)
    assert not pain_tag(cost, window=5)

  def test_pain_at_the_first_step_has_no_lead_up(self):
    cost = np.zeros(20); cost[0] = 0.5
    assert not pain_tag(cost, window=5)

  def test_window_counts_only_the_steps_just_before(self):
    # In contact for the 6 steps before the pain: a 5-step window sees only
    # contact, a 7-step window reaches a calm step.
    cost = np.zeros(20); cost[4:11] = 0.5
    cost[:4] = 0.5  # contact from the start up to the event at 10
    assert not pain_tag(cost, window=5)
    cost = np.zeros(20); cost[4:11] = 0.5
    assert pain_tag(cost, window=7)

  def test_threshold(self):
    cost = np.zeros(20); cost[10] = 0.03
    assert not pain_tag(cost, window=5, thresh=0.05)
    assert pain_tag(cost, window=5, thresh=0.01)

  def test_episode_start_breaks_the_window(self):
    cost = np.zeros(20); cost[:10] = 0.5; cost[10] = 0.5
    first = np.zeros(20, bool); first[8] = True
    # calm steps exist only before is_first... here none are calm at all
    assert not pain_tag(cost, is_first=first, window=5)
    cost = np.zeros(20); cost[10] = 0.5
    first = np.zeros(20, bool); first[10] = True
    # The pain opens a new episode: the calm steps before it belong to the
    # previous one and are not its lead-up.
    assert not pain_tag(cost, is_first=first, window=5)
    first = np.zeros(20, bool); first[7] = True
    assert pain_tag(cost, is_first=first, window=5)

  def test_reward_drop_counts_when_enabled(self):
    rew = np.ones(20); rew[12:] = -1.0
    assert not pain_tag(np.zeros(20), rew, window=5)
    assert pain_tag(np.zeros(20), rew, window=5, drop=1.0)
    assert not pain_tag(np.zeros(20), rew, window=5, drop=3.0)


def _fill(replay, costs):
  for i, c in enumerate(costs):
    replay.add({'obs': np.float32(i), 'cost': np.float32(c),
                'reward': np.float32(0), 'is_first': i == 0})


class TestReplayTag:

  def test_off_by_default(self):
    r = Replay(length=8, capacity=1000)
    _fill(r, np.zeros(50))
    assert r.tag_share == 0.0
    assert 'tagged' not in r.stats()

  def test_tagged_items_and_share(self):
    cost = np.zeros(400); cost[200] = 0.8
    r = Replay(length=8, capacity=10000, tag=dict(share=0.25, window=4))
    _fill(r, cost)
    # Items are sequences [s, s+8): tagged when they hold the event at 200 and
    # a calm step <= 4 steps before it, i.e. s in 193..199.
    starts = {int(r._getseq(*r.items[i], ['obs'])['obs'][0]) for i in r.tagged}
    assert starts == set(range(193, 200))
    n = 4000
    data = r.sample(n)
    tagged = ((data['cost'] >= 0.05) & (np.arange(8) >= 1)).any(1)
    # 0.25 drawn from tagged items + 7/393 of the uniform 0.75.
    expect = 0.25 + 0.75 * 7 / len(r.items)
    assert abs(tagged.mean() - expect) < 0.03

  def test_report_mode_ignores_the_share(self):
    cost = np.zeros(400); cost[200] = 0.8
    r = Replay(length=8, capacity=10000, tag=dict(share=0.9, window=4))
    _fill(r, cost)
    data = r.sample(2000, 'report')
    assert ((data['cost'] >= 0.05).any(1)).mean() < 0.1

  def test_tagged_items_are_never_evicted(self):
    cost = np.zeros(300); cost[20] = 0.8
    r = Replay(length=8, capacity=50, tag=dict(share=0.25, window=4))
    _fill(r, cost)
    assert len(r.tagged) == 7
    # Capacity counts unprotected items only; the tagged ones survive.
    assert len(r.items) - len(r.protected) <= 50
    assert r.protected == r.tagged
    for i in r.tagged:
      seq = r._getseq(*r.items[i], ['cost'])
      assert seq['cost'].max() >= 0.8
    data = r.sample(500)
    assert ((data['cost'] >= 0.05).any(1)).mean() > 0.15

  def test_actor_replay_never_tags(self, tmpdir):
    from dreamerv3 import main as dmain
    import elements
    import ruamel.yaml as yaml
    configs = elements.Path('dreamerv3/configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs)
    config = elements.Config(configs['defaults']).update(
        logdir=str(tmpdir), batch_size=2, batch_length=8, replay_context=0,
        online_learning=False)
    config = config.update({'replay.tag.share': 0.25, 'replay.size': 1000})
    assert dmain.make_replay(config, 'replay').tag_share == 0.25
    assert dmain.make_replay(config, 'replay', tag=False).tag_share == 0.0
    assert dmain.make_replay(config, 'eval_replay', 'eval').tag_share == 0.0


# ---------------------------------------------------------------- M3 ----

class TestStressLambda:

  def test_formula(self):
    s = jnp.asarray([0.0, 0.0, 0.0, 0.0, 10.0], f32)
    z = (s - s.mean()) / (s.std() + 1e-6)
    lam = np.asarray(stress_lambda(s, 0.95, 0.5, 2.0))
    expect = 0.95 * (1 - 0.5 * np.clip(np.asarray(z), 0, 2) / 2)
    np.testing.assert_allclose(lam, expect, rtol=1e-6)
    # The surprised start (z = 2) gets lam / 2; the others keep lam.
    assert lam[-1] == pytest.approx(0.95 * 0.5, rel=1e-5)
    np.testing.assert_allclose(lam[:4], 0.95, rtol=1e-6)

  def test_uniform_surprise_leaves_lambda(self):
    lam = np.asarray(stress_lambda(jnp.ones(8, f32), 0.95, 0.5))
    np.testing.assert_allclose(lam, 0.95, rtol=1e-6)

  def test_bounds(self):
    s = jnp.asarray(np.random.default_rng(0).exponential(1, 256), f32)
    lam = np.asarray(stress_lambda(s, 0.95, 0.5))
    assert lam.min() >= 0.95 * 0.5 - 1e-6 and lam.max() <= 0.95 + 1e-6

  def test_per_row_lambda_in_the_return(self):
    # lambda_return with a (B, 1) lambda equals one call per row with that
    # row's scalar lambda.
    rng = np.random.default_rng(1)
    B, T = 4, 6
    rew = jnp.asarray(rng.normal(size=(B, T)), f32)
    val = jnp.asarray(rng.normal(size=(B, T)), f32)
    last = jnp.zeros((B, T), f32)
    term = jnp.zeros((B, T), f32)
    lams = np.asarray([0.95, 0.7, 0.5, 0.0], np.float32)
    out = lambda_return(last, term, rew, val, val, 0.99, jnp.asarray(lams)[:, None])
    for i, l in enumerate(lams):
      ref = lambda_return(last, term, rew, val, val, 0.99, float(l))
      np.testing.assert_allclose(np.asarray(out[i]), np.asarray(ref[i]), rtol=1e-5)


# ---------------------------------------------------------------- M4 ----

@pytest.fixture
def float32_compute():
  old = nets.COMPUTE_DTYPE
  nets.COMPUTE_DTYPE = jnp.float32
  yield
  nets.COMPUTE_DTYPE = old


def _opt_step(factor):
  a = nets.Linear(3, name='a')
  b = nets.Linear(3, name='b')
  tx = optax.chain(scale_by_rms(0.999, 1e-20), optax.scale(-1e-2))
  opt = MultiOptimizer(
      {'g': ([a, b], tx)}, module_scale={'fear': ([a], factor)}, name='opt')
  x = jnp.asarray(np.random.default_rng(0).normal(size=(5, 4)), f32)

  def loss():
    return (a(x) ** 2).sum().astype(f32) + ((b(x) - 1) ** 2).sum().astype(f32)

  def step():
    return opt(loss)

  params = nj.init(step)({}, seed=0)
  new, _ = nj.pure(step)(dict(params), seed=0)
  return params, new


def test_module_scale_multiplies_only_that_modules_update(float32_compute):
  p0, p1 = _opt_step(1.0)
  q0, q3 = _opt_step(3.0)
  for k in p0:
    if not (k.startswith('a/') or k.startswith('b/')):
      continue
    np.testing.assert_allclose(np.asarray(p0[k]), np.asarray(q0[k]))
    d1 = np.asarray(p1[k]) - np.asarray(p0[k])
    d3 = np.asarray(q3[k]) - np.asarray(q0[k])
    mult = 3.0 if k.startswith('a/') else 1.0
    np.testing.assert_allclose(d3, mult * d1, rtol=1e-5, atol=1e-8)
  assert any(np.abs(np.asarray(p1[k]) - np.asarray(p0[k])).max() > 0
             for k in p0 if k.startswith('a/'))


def test_module_scale_one_is_a_noop():
  opt = MultiOptimizer({}, module_scale={'fear': ([], 1.0)}, name='opt_noop')
  assert opt.module_scale == {}
