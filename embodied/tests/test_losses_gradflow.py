"""Gradient routing and stop-gradient placement in the actor-critic losses.

The actor-critic losses are where a missing or misplaced `stop_gradient` is
silent: the run trains, the numbers look plausible, and the only symptom is that
the critic is being dragged by the policy objective (or vice versa). Nothing in
the suite asserted the routing, so this file does.

Rather than instantiate ninjax heads, it uses minimal fakes with the same API
(`pred`, `loss`, `logp`, `entropy`) whose outputs are explicit functions of a
parameter tensor. That makes "does a gradient reach X" a direct `jax.grad` call.

Covered: `imag_loss` (flat v3, previously untested), `repl_loss` (previously
untested), `imag_loss_wkr` and `imag_loss_mgr`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.hrl.losses import (
    imag_loss,
    imag_loss_mgr,
    imag_loss_wkr,
    repl_loss,
)

B, T = 2, 9


class FakeValue:
  """A critic whose prediction is `theta` and whose loss is a plain MSE."""

  def __init__(self, theta):
    self.theta = theta

  def pred(self):
    return self.theta

  def loss(self, target):
    return jnp.square(self.theta - target)


class FakePolicy:
  """A policy whose log-prob and entropy are explicit functions of `phi`."""

  minent = 0.0
  maxent = 1.0

  def __init__(self, phi):
    self.phi = phi

  def logp(self, act):
    return self.phi * act

  def entropy(self):
    return jnp.tanh(self.phi)


class FakeNorm:
  """Identity normalizer matching the `Normalize` call/stats API."""

  def __init__(self, offset=0.0, scale=1.0):
    self.offset, self.scale = offset, scale

  def stats(self):
    return jnp.float32(self.offset), jnp.float32(self.scale)

  def __call__(self, x, update, weights=None):
    return self.stats()


def flat_inputs(theta, phi):
  act = {'action': jnp.ones((B, T), jnp.float32)}
  rew = jnp.linspace(0, 1, B * T).reshape(B, T).astype(jnp.float32)
  con = jnp.full((B, T), 0.95, jnp.float32)
  return dict(
      act=act, rew=rew, con=con,
      policy={'action': FakePolicy(phi)},
      value=FakeValue(theta), slowvalue=FakeValue(jax.lax.stop_gradient(theta)),
      retnorm=FakeNorm(), valnorm=FakeNorm(), advnorm=FakeNorm(),
      update=True)


class TestImagLossGradientRouting:
  """`imag_loss`: the flat v3 actor-critic, used when `use_hrl=False`."""

  def test_policy_loss_does_not_reach_the_critic(self):
    phi = jnp.ones((B, T), jnp.float32)
    def fn(theta):
      losses, _, _ = imag_loss(**flat_inputs(theta, phi))
      return losses['policy'].sum()
    g = np.asarray(jax.grad(fn)(jnp.zeros((B, T), jnp.float32)))
    assert np.all(g == 0.0), (
        'the advantage or the weight is not stop-gradiented: the actor '
        'objective is training the critic')

  def test_value_loss_does_not_reach_the_policy(self):
    theta = jnp.zeros((B, T), jnp.float32)
    def fn(phi):
      losses, _, _ = imag_loss(**flat_inputs(theta, phi))
      return losses['value'].sum()
    g = np.asarray(jax.grad(fn)(jnp.ones((B, T), jnp.float32)))
    assert np.all(g == 0.0)

  def test_policy_loss_does_reach_the_policy(self):
    theta = jnp.zeros((B, T), jnp.float32)
    def fn(phi):
      losses, _, _ = imag_loss(**flat_inputs(theta, phi))
      return losses['policy'].sum()
    g = np.asarray(jax.grad(fn)(jnp.ones((B, T), jnp.float32)))
    assert np.any(g != 0.0), 'the actor receives no gradient at all'

  def test_value_loss_does_reach_the_critic(self):
    phi = jnp.ones((B, T), jnp.float32)
    def fn(theta):
      losses, _, _ = imag_loss(**flat_inputs(theta, phi))
      return losses['value'].sum()
    g = np.asarray(jax.grad(fn)(jnp.zeros((B, T), jnp.float32)))
    assert np.any(g != 0.0)

  def test_the_action_is_stop_gradiented_inside_logp(self):
    """`logp(sg(act[k]))` -- the sampled action must be a constant."""
    theta = jnp.zeros((B, T), jnp.float32)
    phi = jnp.ones((B, T), jnp.float32)
    def fn(act_val):
      ins = flat_inputs(theta, phi)
      ins['act'] = {'action': act_val}
      losses, _, _ = imag_loss(**ins)
      return losses['policy'].sum()
    g = np.asarray(jax.grad(fn)(jnp.ones((B, T), jnp.float32)))
    assert np.all(g == 0.0)

  def test_reward_does_not_leak_gradient_through_the_policy_loss(self):
    """The return enters the actor loss only through a stop-gradiented adv."""
    theta = jnp.zeros((B, T), jnp.float32)
    phi = jnp.ones((B, T), jnp.float32)
    def fn(rew):
      ins = flat_inputs(theta, phi)
      ins['rew'] = rew
      losses, _, _ = imag_loss(**ins)
      return losses['policy'].sum()
    g = np.asarray(jax.grad(fn)(jnp.ones((B, T), jnp.float32)))
    assert np.all(g == 0.0)

  def test_continuation_does_not_leak_through_the_stop_gradiented_weight(self):
    theta = jnp.zeros((B, T), jnp.float32)
    phi = jnp.ones((B, T), jnp.float32)
    def fn(con):
      ins = flat_inputs(theta, phi)
      ins['con'] = con
      losses, _, _ = imag_loss(**ins)
      return losses['policy'].sum() + losses['value'].sum()
    g = np.asarray(jax.grad(fn)(jnp.full((B, T), 0.95, jnp.float32)))
    assert np.all(g == 0.0), 'weight must be sg() in both loss terms'


class TestImagLossShapes:

  def test_loss_tensors_are_one_shorter_than_the_time_axis(self):
    losses, outs, mets = imag_loss(
        **flat_inputs(jnp.zeros((B, T)), jnp.ones((B, T))))
    assert losses['policy'].shape == (B, T - 1)
    assert losses['value'].shape == (B, T - 1)
    assert outs['ret'].shape == (B, T - 1)

  def test_all_metrics_are_finite_scalars(self):
    _, _, mets = imag_loss(**flat_inputs(jnp.zeros((B, T)), jnp.ones((B, T))))
    for k, v in mets.items():
      v = np.asarray(v)
      assert v.shape == (), (k, v.shape)
      assert np.isfinite(v), (k, v)

  def test_entropy_is_reported_for_every_action_key(self):
    _, _, mets = imag_loss(**flat_inputs(jnp.zeros((B, T)), jnp.ones((B, T))))
    assert 'ent/action' in mets and 'rand/action' in mets


class TestReplLoss:
  """`repl_loss`: value regression on the real replay tail."""

  @staticmethod
  def inputs(theta, T_=T):
    return dict(
        last=jnp.zeros((B, T_), bool),
        term=jnp.zeros((B, T_), jnp.float32),
        rew=jnp.ones((B, T_), jnp.float32),
        boot=jnp.zeros((B, T_), jnp.float32),
        value=FakeValue(theta),
        slowvalue=FakeValue(jax.lax.stop_gradient(theta)),
        valnorm=FakeNorm(), update=True)

  def test_shapes(self):
    losses, outs, _ = repl_loss(**self.inputs(jnp.zeros((B, T))))
    assert losses['repmgr_extr_value'].shape == (B, T - 1)
    assert outs['repmgr_extr_ret'].shape == (B, T - 1)

  def test_value_head_name_controls_the_loss_key(self):
    losses, outs, _ = repl_loss(
        **self.inputs(jnp.zeros((B, T))), value_head='val')
    assert 'repval_value' in losses and 'repval_ret' in outs

  def test_a_single_step_window_returns_zeros_instead_of_crashing(self):
    losses, outs, _ = repl_loss(**self.inputs(jnp.zeros((B, 1)), T_=1))
    assert np.all(np.asarray(losses['repmgr_extr_value']) == 0.0)
    assert outs['repmgr_extr_ret'].shape == (B, 0)

  def test_steps_after_an_episode_end_get_zero_weight(self):
    ins = self.inputs(jnp.ones((B, T)))
    last = np.zeros((B, T), bool)
    last[:, 4:] = True
    ins['last'] = jnp.asarray(last)
    losses, _, _ = repl_loss(**ins)
    got = np.asarray(losses['repmgr_extr_value'])
    assert np.all(got[:, 4:] == 0.0), got

  def test_gradient_reaches_the_critic_only(self):
    def fn(theta):
      losses, _, _ = repl_loss(**self.inputs(theta))
      return losses['repmgr_extr_value'].sum()
    g = np.asarray(jax.grad(fn)(jnp.zeros((B, T), jnp.float32)))
    assert np.any(g != 0.0)

  def test_the_bootstrap_is_treated_as_a_constant_target(self):
    theta = jnp.zeros((B, T), jnp.float32)
    def fn(boot):
      ins = self.inputs(theta)
      ins['boot'] = boot
      losses, _, _ = repl_loss(**ins)
      return losses['repmgr_extr_value'].sum()
    g = np.asarray(jax.grad(fn)(jnp.zeros((B, T), jnp.float32)))
    assert np.all(g == 0.0), 'the replay target must not backprop into boot'


def wkr_inputs(theta, phi, **over):
  ins = dict(
      act={'action': jnp.ones((B, T), jnp.float32)},
      wkr_goal_rew=jnp.linspace(0, 1, B * T).reshape(B, T).astype(jnp.float32),
      con=jnp.full((B, T), 0.95, jnp.float32),
      policy={'action': FakePolicy(phi)},
      wkr_goal_value=FakeValue(theta),
      wkr_goal_slowvalue=FakeValue(jax.lax.stop_gradient(theta)),
      wkr_goal_retnorm=FakeNorm(), wkr_goal_valnorm=FakeNorm(),
      wkr_goal_advnorm=FakeNorm(), update=True)
  ins.update(over)
  return ins


class TestWorkerLoss:

  def test_policy_loss_does_not_reach_the_critic(self):
    phi = jnp.ones((B, T), jnp.float32)
    def fn(theta):
      losses, _, _ = imag_loss_wkr(**wkr_inputs(theta, phi))
      return losses['wkr_policy'].sum()
    assert np.all(np.asarray(jax.grad(fn)(jnp.zeros((B, T), jnp.float32))) == 0.0)

  def test_value_loss_does_not_reach_the_policy(self):
    theta = jnp.zeros((B, T), jnp.float32)
    def fn(phi):
      losses, _, _ = imag_loss_wkr(**wkr_inputs(theta, phi))
      return losses['wkr_goal_value'].sum()
    assert np.all(np.asarray(jax.grad(fn)(jnp.ones((B, T), jnp.float32))) == 0.0)

  def test_goal_reward_does_not_backprop_through_the_actor_loss(self):
    """Our worker is pure REINFORCE, so the reward is a constant to the actor.

    This is a deliberate difference from TF Director, whose worker used
    `actor_grad_cont: backprop` and let the goal reward's gradient reach the
    imagined states. Recorded so the difference stays visible.
    """
    theta = jnp.zeros((B, T), jnp.float32)
    phi = jnp.ones((B, T), jnp.float32)
    def fn(rew):
      losses, _, _ = imag_loss_wkr(**wkr_inputs(theta, phi, wkr_goal_rew=rew))
      return losses['wkr_policy'].sum()
    assert np.all(np.asarray(jax.grad(fn)(jnp.ones((B, T), jnp.float32))) == 0.0)

  def test_shapes(self):
    losses, outs, _ = imag_loss_wkr(
        **wkr_inputs(jnp.zeros((B, T)), jnp.ones((B, T))))
    assert losses['wkr_policy'].shape == (B, T - 1)
    assert losses['wkr_goal_value'].shape == (B, T - 1)
    assert outs['wkr_goal_ret'].shape == (B, T - 1)

  def test_metrics_are_finite(self):
    _, _, mets = imag_loss_wkr(
        **wkr_inputs(jnp.zeros((B, T)), jnp.ones((B, T))))
    for k, v in mets.items():
      assert np.isfinite(np.asarray(v)), (k, v)

  @pytest.mark.parametrize('window', [0, 2, 4])
  def test_skill_window_resets_the_return_at_goal_boundaries(self, window):
    """`skill_window` must zero lambda exactly on the window boundaries."""
    ins = wkr_inputs(jnp.zeros((B, T)), jnp.ones((B, T)), skill_window=window)
    losses, outs, _ = imag_loss_wkr(**ins)
    assert outs['wkr_goal_ret'].shape == (B, T - 1)
    assert np.all(np.isfinite(np.asarray(outs['wkr_goal_ret'])))

  def test_a_boundary_mask_is_accepted_in_place_of_an_int_window(self):
    mask = jnp.zeros((B, T), jnp.float32).at[:, 4].set(1.0)
    ins = wkr_inputs(jnp.zeros((B, T)), jnp.ones((B, T)), skill_window=mask)
    _, outs, _ = imag_loss_wkr(**ins)
    assert outs['wkr_goal_ret'].shape == (B, T - 1)

  def test_the_step_zero_boundary_is_never_treated_as_a_reset(self):
    """A mask with a 1 at t=0 must behave like no reset there."""
    m0 = jnp.zeros((B, T), jnp.float32).at[:, 0].set(1.0)
    a, _, _ = imag_loss_wkr(
        **wkr_inputs(jnp.zeros((B, T)), jnp.ones((B, T)), skill_window=m0))
    b, _, _ = imag_loss_wkr(
        **wkr_inputs(jnp.zeros((B, T)), jnp.ones((B, T)),
                     skill_window=jnp.zeros((B, T), jnp.float32)))
    np.testing.assert_allclose(
        np.asarray(a['wkr_policy']), np.asarray(b['wkr_policy']), rtol=1e-6)


C = 4


class FakeManagerHead(FakePolicy):
  """Manager head: `pred()` is a onehot-shaped `(B, T, C)` skill distribution."""

  def pred(self):
    return jnp.zeros((B, T, C), jnp.float32)

  def logp(self, event):
    return (self.phi[..., None] * event).sum(-1)


def mgr_inputs(theta, phi, **over):
  ins = dict(
      skills={'skill': jnp.ones((B, T, C), jnp.float32)},
      mgr_extr_rew=jnp.linspace(0, 1, B * T).reshape(B, T).astype(jnp.float32),
      mgr_expl_rew=jnp.full((B, T), 0.1, jnp.float32),
      con=jnp.full((B, T), 0.95, jnp.float32),
      manager_policy={'skill': FakeManagerHead(phi)},
      mgr_extr_value=FakeValue(theta),
      mgr_extr_slowvalue=FakeValue(jax.lax.stop_gradient(theta)),
      mgr_expl_value=FakeValue(theta),
      mgr_expl_slowvalue=FakeValue(jax.lax.stop_gradient(theta)),
      mgr_extr_retnorm=FakeNorm(), mgr_expl_retnorm=FakeNorm(),
      mgr_extr_valnorm=FakeNorm(), mgr_expl_valnorm=FakeNorm(),
      mgr_advnorm=FakeNorm(), update=True)
  ins.update(over)
  return ins


class TestManagerLoss:
  """`imag_loss_mgr` with the skill head only (the production Director arm)."""

  def test_policy_loss_does_not_reach_either_critic(self):
    phi = jnp.ones((B, T), jnp.float32)
    def fn(theta):
      losses, _, _ = imag_loss_mgr(**mgr_inputs(theta, phi))
      return losses['mgr_policy'].sum()
    g = np.asarray(jax.grad(fn)(jnp.zeros((B, T), jnp.float32)))
    assert np.all(g == 0.0)

  def test_extrinsic_value_loss_does_not_reach_the_policy(self):
    theta = jnp.zeros((B, T), jnp.float32)
    def fn(phi):
      losses, _, _ = imag_loss_mgr(**mgr_inputs(theta, phi))
      return losses['mgr_extr_value'].sum()
    g = np.asarray(jax.grad(fn)(jnp.ones((B, T), jnp.float32)))
    assert np.all(g == 0.0)

  def test_pooled_rewards_do_not_backprop_through_the_policy_loss(self):
    theta = jnp.zeros((B, T), jnp.float32)
    phi = jnp.ones((B, T), jnp.float32)
    def fn(rew):
      losses, _, _ = imag_loss_mgr(**mgr_inputs(theta, phi, mgr_extr_rew=rew))
      return losses['mgr_policy'].sum()
    g = np.asarray(jax.grad(fn)(jnp.ones((B, T), jnp.float32)))
    assert np.all(g == 0.0)

  def test_loss_keys_and_shapes(self):
    losses, outs, _ = imag_loss_mgr(
        **mgr_inputs(jnp.zeros((B, T)), jnp.ones((B, T))))
    assert set(losses) == {'mgr_policy', 'mgr_extr_value', 'mgr_expl_value'}
    for k, v in losses.items():
      assert v.shape == (B, T - 1), (k, v.shape)
    assert outs['ret'].shape == (B, T - 1)

  def test_exploration_reward_enters_the_return_with_its_weight(self):
    base, _, _ = imag_loss_mgr(
        **mgr_inputs(jnp.zeros((B, T)), jnp.ones((B, T))), mgr_expl_weight=0.0)
    wet, _, _ = imag_loss_mgr(
        **mgr_inputs(jnp.zeros((B, T)), jnp.ones((B, T))), mgr_expl_weight=1.0)
    assert not np.allclose(
        np.asarray(base['mgr_policy']), np.asarray(wet['mgr_policy']))

  def test_metrics_are_finite(self):
    _, _, mets = imag_loss_mgr(
        **mgr_inputs(jnp.zeros((B, T)), jnp.ones((B, T))))
    for k, v in mets.items():
      v = np.asarray(v)
      assert v.shape == (), (k, v.shape)
      assert np.isfinite(v), (k, v)

  def test_fixed_k_decision_mean_is_a_plain_mean(self):
    """With `switch_mask=None` every column is a decision, so no rescaling."""
    _, _, mets = imag_loss_mgr(
        **mgr_inputs(jnp.zeros((B, T)), jnp.ones((B, T))))
    rew = np.linspace(0, 1, B * T).reshape(B, T).astype(np.float32)
    assert float(mets['mgr_extr_rew']) == pytest.approx(rew[:, 1:].mean(), rel=1e-5)

  def test_switch_mask_restricts_the_reported_reward_to_real_decisions(self):
    sw = np.zeros((B, T), np.float32)
    sw[:, ::4] = 1.0
    _, _, mets = imag_loss_mgr(
        **mgr_inputs(jnp.zeros((B, T)), jnp.ones((B, T))),
        switch_mask=jnp.asarray(sw))
    rew = np.linspace(0, 1, B * T).reshape(B, T).astype(np.float32)
    m = sw[:, :-1]
    want = (rew[:, 1:] * m).sum() / m.sum()
    assert float(mets['mgr_extr_rew']) == pytest.approx(want, rel=1e-4)
