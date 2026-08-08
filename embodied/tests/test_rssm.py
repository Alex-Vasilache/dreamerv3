"""RSSM world-model core: shapes, dtypes, carry threading, KL routing.

`dreamerv3/rssm.py` had no test anywhere in the suite despite being the world
model every other component reads from. The file is byte-identical to the
upstream `code/dreamerv3` tree, so these tests are about pinning behaviour the
HRL code depends on, not about hunting for local edits.

The properties that matter downstream:

* `reset` must wipe the carry, or episode boundaries leak state between
  episodes and the manager's block boundaries straddle two episodes.
* KL balancing must be routed so `dyn` trains only the prior and `rep` only the
  posterior. A swapped `sg` here trains the world model against itself.
* Everything must stay in `COMPUTE_DTYPE`; a silent float32 promotion inside the
  scan costs memory and changes numerics.
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

import elements
import embodied.jax.nets as nn
from dreamerv3.rssm import RSSM

B, T = 2, 6
DETER, STOCH, CLASSES, TOKENS = 16, 4, 4, 12


def act_space():
  return {'action': elements.Space(np.float32, (3,), -1, 1)}


def make(**kw):
  return RSSM(
      act_space(), deter=DETER, stoch=STOCH, classes=CLASSES, hidden=16,
      blocks=2, obslayers=1, imglayers=1, dynlayers=1, unroll=False,
      name='rssm', **kw)


def inputs(T_=T, reset_at=None):
  rng = np.random.default_rng(0)
  tokens = jnp.asarray(rng.normal(size=(B, T_, TOKENS)), jnp.float32)
  action = {'action': jnp.asarray(
      rng.uniform(-1, 1, size=(B, T_, 3)), jnp.float32)}
  reset = np.zeros((B, T_), bool)
  reset[:, 0] = True
  if reset_at is not None:
    reset[:, reset_at] = True
  return tokens, action, jnp.asarray(reset)


def run(fn, *args, seed=0, module=None):
  """Init then apply a pure ninjax function, returning its output.

  The module is closed over rather than passed as an argument: ninjax treats
  every positional argument as traced data and cannot accept a Module there.
  """
  rssm = module or make()
  g = lambda *a: fn(rssm, *a)
  params = nj.init(g)({}, *args, seed=seed)
  _, out = nj.pure(g)(params, *args, seed=seed)
  return out


def grads_of(fn, *args, module=None, seed=0):
  """Gradient of a scalar ninjax function w.r.t. the module's parameters.

  `nj.grad` must be called from inside an impure context (see
  `embodied/jax/opt.py`), so it is wrapped and then run through `nj.pure`.
  """
  rssm = module or make()
  plain = lambda *a: fn(rssm, *a)
  params = nj.init(plain)({}, *args, seed=seed)

  def impure(*a):
    out = nj.grad(plain, [rssm])(*a)
    return out[2]  # (loss, params, grads[, aux])

  _, grad = nj.pure(impure)(params, *args, seed=seed)
  return grad


def observe_fn(rssm, tokens, action, reset):
  carry = rssm.initial(B)
  return rssm.observe(carry, tokens, action, reset, training=True)


class TestInitialCarry:

  def test_shapes(self):
    out = run(lambda r: r.initial(B))
    assert out['deter'].shape == (B, DETER)
    assert out['stoch'].shape == (B, STOCH, CLASSES)

  def test_dtype_is_the_compute_dtype(self):
    out = run(lambda r: r.initial(B))
    assert out['deter'].dtype == nn.COMPUTE_DTYPE
    assert out['stoch'].dtype == nn.COMPUTE_DTYPE

  def test_initial_carry_is_all_zeros(self):
    out = run(lambda r: r.initial(B))
    assert np.all(np.asarray(out['deter'], np.float32) == 0.0)
    assert np.all(np.asarray(out['stoch'], np.float32) == 0.0)

  def test_batch_size_is_honoured(self):
    out = run(lambda r: r.initial(5))
    assert out['deter'].shape == (5, DETER)


class TestObserve:

  def test_output_shapes(self):
    carry, entries, feat = run(observe_fn, *inputs())
    assert carry['deter'].shape == (B, DETER)
    assert entries['deter'].shape == (B, T, DETER)
    assert entries['stoch'].shape == (B, T, STOCH, CLASSES)
    assert feat['logit'].shape == (B, T, STOCH, CLASSES)

  def test_all_outputs_are_compute_dtype(self):
    carry, entries, feat = run(observe_fn, *inputs())
    for name, v in [('deter', feat['deter']), ('stoch', feat['stoch']),
                    ('logit', feat['logit'])]:
      assert v.dtype == nn.COMPUTE_DTYPE, (name, v.dtype)

  def test_outputs_are_finite(self):
    _, _, feat = run(observe_fn, *inputs())
    for k, v in feat.items():
      assert np.all(np.isfinite(np.asarray(v, np.float32))), k

  def test_stoch_is_a_onehot_sample_over_the_class_axis(self):
    _, _, feat = run(observe_fn, *inputs())
    s = np.asarray(feat['stoch'], np.float32)
    np.testing.assert_allclose(s.sum(-1), 1.0, atol=1e-2)

  def test_single_step_matches_the_first_step_of_the_scan(self):
    tokens, action, reset = inputs()
    def both(r, tok, act, res):
      carry = r.initial(B)
      c1, e1, f1 = r.observe(
          carry, tok[:, 0], {k: v[:, 0] for k, v in act.items()},
          res[:, 0], training=True, single=True)
      c2, e2, f2 = r.observe(
          carry, tok[:, :1], {k: v[:, :1] for k, v in act.items()},
          res[:, :1], training=True)
      return f1['deter'], f2['deter'][:, 0]
    a, b = run(both, tokens, action, reset)
    np.testing.assert_allclose(
        np.asarray(a, np.float32), np.asarray(b, np.float32), rtol=1e-2, atol=1e-3)

  def test_carry_equals_the_last_entry(self):
    carry, entries, _ = run(observe_fn, *inputs())
    np.testing.assert_allclose(
        np.asarray(carry['deter'], np.float32),
        np.asarray(entries['deter'][:, -1], np.float32), rtol=1e-3)


class TestReset:
  """`reset` must wipe the incoming carry before the step is computed."""

  @staticmethod
  def _tail(tok, action, reset, module):
    """deter from the reset step onward, run in its own RNG stream."""
    def fn(r, t, a, res):
      _, _, feat = r.observe(r.initial(B), t, a, res, training=True)
      return feat['deter'][:, 3:]
    return np.asarray(
        run(fn, tok, action, reset, module=module, seed=0), np.float32)

  def test_two_sequences_agree_after_a_mid_sequence_reset(self):
    """State before a reset must not influence anything after it.

    The two rollouts are run as separate `pure` calls with the same seed: two
    `observe` calls inside one pure function would draw different `stoch`
    samples and differ for reasons that have nothing to do with the reset.
    """
    tokens, action, _ = inputs()
    reset = np.zeros((B, T), bool)
    reset[:, 0] = True
    reset[:, 3] = True
    reset = jnp.asarray(reset)
    alt = jnp.asarray(
        np.random.default_rng(9).normal(size=(B, T, TOKENS)), jnp.float32)
    alt = alt.at[:, 3:].set(tokens[:, 3:])
    module = make()
    a = self._tail(tokens, action, reset, module)
    b = self._tail(alt, action, reset, module)
    np.testing.assert_allclose(
        a, b, rtol=1e-2, atol=1e-3,
        err_msg='state leaked across a reset boundary')

  def test_without_a_reset_history_does_change_the_future(self):
    """Control for the test above: the model is not simply ignoring history."""
    tokens, action, _ = inputs()
    reset = np.zeros((B, T), bool)
    reset[:, 0] = True
    reset = jnp.asarray(reset)
    alt = jnp.asarray(
        np.random.default_rng(9).normal(size=(B, T, TOKENS)), jnp.float32)
    alt = alt.at[:, 3:].set(tokens[:, 3:])
    module = make()
    a = self._tail(tokens, action, reset, module)
    b = self._tail(alt, action, reset, module)
    assert not np.allclose(a, b, rtol=1e-2)


class TestImagine:

  def test_shapes_with_a_fixed_action_sequence(self):
    rng = np.random.default_rng(1)
    acts = {'action': jnp.asarray(rng.uniform(-1, 1, (B, 4, 3)), jnp.float32)}

    def fn(r, a):
      carry = r.initial(B)
      return r.imagine(carry, a, 4, training=True)

    carry, feat, action = run(fn, acts)
    assert feat['deter'].shape == (B, 4, DETER)
    assert feat['stoch'].shape == (B, 4, STOCH, CLASSES)
    assert carry['deter'].shape == (B, DETER)

  def test_dtypes_are_compute_dtype(self):
    rng = np.random.default_rng(1)
    acts = {'action': jnp.asarray(rng.uniform(-1, 1, (B, 3, 3)), jnp.float32)}
    def fn(r, a):
      return r.imagine(r.initial(B), a, 3, training=True)
    _, feat, _ = run(fn, acts)
    for k in ('deter', 'stoch', 'logit'):
      assert feat[k].dtype == nn.COMPUTE_DTYPE, k

  def test_a_callable_policy_is_supported(self):
    def fn(r):
      pol = lambda c: {'action': jnp.zeros((B, 3), jnp.float32)}
      return r.imagine(r.initial(B), pol, 3, training=True)
    _, feat, action = run(fn)
    assert feat['deter'].shape == (B, 3, DETER)
    assert action['action'].shape == (B, 3, 3)

  def test_the_policy_sees_a_stop_gradiented_carry(self):
    """`policy(sg(carry))` -- the actor input must not backprop into the state.

    This is what makes the actor REINFORCE-only; a gradient here would be the
    Director `actor_grad_cont: backprop` pathway instead.
    """
    import inspect
    src = inspect.getsource(RSSM.imagine)
    assert 'policy(sg(carry))' in src.replace(' ', '').replace(
        'policy(sg(carry))', 'policy(sg(carry))') or 'sg(carry)' in src


class TestKLRouting:
  """`dyn` trains the prior only; `rep` trains the posterior only."""

  @staticmethod
  def loss_fn(r, tok, act, res):
    carry = r.initial(B)
    _, _, losses, _, _ = r.loss(carry, tok, act, res, training=True)
    return losses

  def test_both_kl_terms_are_produced_with_the_right_shape(self):
    losses = run(self.loss_fn, *inputs())
    assert set(losses) == {'dyn', 'rep'}
    for k, v in losses.items():
      assert v.shape == (B, T), (k, v.shape)

  def test_kl_terms_are_non_negative(self):
    losses = run(self.loss_fn, *inputs())
    for k, v in losses.items():
      assert np.all(np.asarray(v, np.float32) >= -1e-3), k

  def test_dyn_does_not_train_the_observation_head_within_a_timestep(self):
    """`dyn` stop-gradients the posterior, so the DIRECT path is cut.

    Checked at T=1, where the direct KL term is the only route from `obslogit`
    to `dyn`. At T>1 a gradient does legitimately reach `obslogit`, but only
    through the recurrent path: `logit_t -> stoch_t` (the OneHot
    straight-through sample) `-> deter_{t+1} -> prior_{t+1}`. That is the
    dynamics gradient, not a leak, and it is asserted separately below.
    """
    tokens, action, reset = inputs(T_=1)
    g = grads_of(
        lambda r, *a: self.loss_fn(r, *a)['dyn'].mean(),
        tokens, action, reset)
    obs_keys = [k for k in g if 'obslogit' in k]
    assert obs_keys, 'expected an obslogit parameter'
    for k in obs_keys:
      assert np.all(np.abs(np.asarray(g[k], np.float32)) < 1e-8), (
          f'{k} received gradient from the dyn term at T=1; the posterior must '
          'be stop-gradiented in the direct KL path')

  def test_dyn_reaches_the_observation_head_only_through_the_recurrent_path(self):
    """Documents the T>1 behaviour so the T=1 test above is not read too broadly."""
    tokens, action, reset = inputs(T_=4)
    g = grads_of(
        lambda r, *a: self.loss_fn(r, *a)['dyn'].mean(),
        tokens, action, reset)
    obs = [np.abs(np.asarray(g[k], np.float32)).sum()
           for k in g if 'obslogit' in k]
    assert sum(obs) > 0, (
        'expected the straight-through stoch sample to carry dyn gradient '
        'into obslogit across timesteps')

  def test_rep_does_not_train_the_prior_head(self):
    tokens, action, reset = inputs()
    g = grads_of(
        lambda r, *a: self.loss_fn(r, *a)['rep'].mean(),
        tokens, action, reset)
    prior_keys = [k for k in g if 'imglogit' in k or 'prior' in k]
    for k in prior_keys:
      assert np.all(np.abs(np.asarray(g[k], np.float32)) < 1e-8), (
          f'{k} received gradient from the rep term')

  def test_dyn_does_reach_the_prior(self):
    tokens, action, reset = inputs()
    g = grads_of(
        lambda r, *a: self.loss_fn(r, *a)['dyn'].mean(),
        tokens, action, reset)
    total = sum(float(np.abs(np.asarray(v, np.float32)).sum()) for v in g.values())
    assert total > 0, 'the dyn term produced no gradient at all'


class TestFreeNats:

  def test_free_nats_clamps_both_terms_from_below(self):
    tokens, action, reset = inputs()
    rssm = RSSM(
        act_space(), deter=DETER, stoch=STOCH, classes=CLASSES, hidden=16,
        blocks=2, obslayers=1, imglayers=1, dynlayers=1, unroll=False,
        free_nats=100.0, name='rssm')
    losses = run(TestKLRouting.loss_fn, tokens, action, reset, module=rssm)
    for k, v in losses.items():
      np.testing.assert_allclose(np.asarray(v, np.float32), 100.0, rtol=1e-4)


class TestCarryHelpers:

  def test_truncate_takes_the_last_timestep(self):
    entries = {'deter': jnp.arange(B * T * DETER, dtype=jnp.float32).reshape(
        B, T, DETER)}
    out = run(lambda r, e: r.truncate(e), entries)
    np.testing.assert_allclose(
        np.asarray(out['deter']), np.asarray(entries['deter'][:, -1]))

  def test_truncate_rejects_a_non_sequence_input(self):
    with pytest.raises(AssertionError):
      run(lambda r, e: r.truncate(e), {'deter': jnp.zeros((B, DETER))})

  def test_starts_flattens_the_last_n_steps_into_the_batch(self):
    entries = {'deter': jnp.zeros((B, T, DETER), jnp.float32)}
    carry = {'deter': jnp.zeros((B, DETER), jnp.float32)}
    out = run(lambda r, e, c: r.starts(e, c, 3), entries, carry)
    assert out['deter'].shape == (B * 3, DETER)

  def test_entry_space_matches_the_carry_shapes(self):
    rssm = make()
    space = rssm.entry_space
    assert space['deter'].shape == (DETER,)
    assert space['stoch'].shape == (STOCH, CLASSES)
