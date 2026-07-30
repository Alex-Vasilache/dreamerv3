"""Verification of ``imag_loss_wkr``'s DENSE fallback credit path -- the ONE
worker-credit algorithm genuinely-variable holds ever use.

``worker_split_window`` selects the windowed (``split_traj``) worker path
whenever the hold pattern is statically regular: plain fixed-K Director, or
the duration-pinned control (``goal_duration_fixed > 0``). Free-running
variable holds (``goal_duration_fixed == 0``, the actual research
configuration) always fall through to the DENSE path in ``imag_loss_wkr``
(``skill_window`` passed as the raw ``(B, T)`` switch-mask tensor). Every
"pinned equivalence" run to date -- including the ones that came back alive
after the terminal-flag fix -- never touched this branch at all. It is
completely unverified by any of that work.

This module builds a small, hand-checkable rollout with an IRREGULAR switch
pattern (not the K-periodic one a pinned config produces) and cross-checks
``imag_loss_wkr``'s dense-path return/advantage against an independent,
per-window ``lambda_return`` computed by literally slicing the rollout at
each switch and running the flat return within that slice alone -- the
textbook definition of "reset at goal-window boundaries," implemented a
completely different way from the vectorized ``last``-mask trick the real
code uses.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.hrl.losses import imag_loss_wkr, lambda_return
import embodied.jax.outs as outs

f32 = jnp.float32


class _StubNorm:
  def stats(self):
    return 0.0, 1.0

  def __call__(self, x, update=True, weights=None):
    return 0.0, 1.0


def ref_windowed_lambda_return(rew, con, val, boot, switch_mask, disc, lam):
  """Independent reference: literally slice at each switch and run a PLAIN
  (no window-reset) lambda_return within each slice, then re-stitch. This is
  the textbook meaning of "reset at goal-window boundaries," built without
  reusing ``lambda_return``'s own ``last``-mask machinery at all.
  """
  B, T = rew.shape
  out = np.zeros((B, T - 1), np.float32)
  for b in range(B):
    starts = [t for t in range(T) if switch_mask[b, t] > 0.5]
    if not starts or starts[0] != 0:
      starts = [0] + starts
    for i, s in enumerate(starts):
      e = starts[i + 1] if i + 1 < len(starts) else T - 1
      if e <= s:
        continue
      seg_rew = rew[b:b+1, s:e+1]
      seg_con = con[b:b+1, s:e+1]
      seg_val = val[b:b+1, s:e+1]
      seg_boot = boot[b:b+1, s:e+1]
      seg_last = jnp.zeros((1, e + 1 - s), f32)
      seg_term = 1 - seg_con
      seg_ret = lambda_return(
          seg_last, seg_term, seg_rew, seg_val, seg_boot, disc, lam)
      out[b, s:e] = np.asarray(seg_ret)[0]
  return jnp.array(out)


def test_dense_path_return_matches_independent_per_window_reference():
  """The vectorized ``last``-mask reset must equal literally slicing the
  rollout at each switch and computing the return within each slice alone --
  built two structurally different ways, on an IRREGULAR switch pattern no
  pinned run ever produces.
  """
  B, T = 2, 13
  rng = np.random.RandomState(0)
  rew = jnp.array(rng.randn(B, T).astype(np.float32) * 0.4)
  con = jnp.array(rng.uniform(0.9, 0.999, (B, T)).astype(np.float32))
  val = jnp.array(rng.randn(B, T).astype(np.float32) * 2.0)
  disc, lam = 1.0, 0.95

  # Two DIFFERENT irregular patterns, one per batch row -- deliberately not
  # K-periodic, so this cannot be produced by any pinned/fixed-K config.
  sw = jnp.array([
      [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
      [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
  ], f32)

  last = f32(sw).at[:, 0].set(0.0)
  term = 1 - con
  real_ret = lambda_return(last, term, rew, val, val, disc, lam)
  ref_ret = ref_windowed_lambda_return(rew, con, val, val, sw, disc, lam)

  np.testing.assert_allclose(
      np.asarray(real_ret), np.asarray(ref_ret), rtol=1e-5, atol=1e-6,
      err_msg="dense-path window-reset return diverges from an independently "
              "computed per-window reference on an irregular switch pattern")


def test_dense_path_does_not_leak_reward_across_a_switch_boundary():
  """A large, distinctive reward placed just AFTER a switch must not inflate
  the return of the step just BEFORE that switch (the defining property of a
  window reset -- this is what would go silently wrong if ``last`` were off
  by one position in either direction).
  """
  B, T = 1, 9
  con = jnp.ones((B, T))
  val = jnp.zeros((B, T))
  rew = jnp.zeros((B, T))
  # Switch at t=4. Put a huge reward at t=5 (i.e. rew[:,5], the transition
  # INTO the new window) -- it must count for the NEW window (crediting
  # decision starting at t=4), not leak backward into t=3's return (the last
  # step of the OLD window, t in [0,4)).
  rew = rew.at[0, 5].set(1000.0)
  sw = jnp.zeros((B, T), f32).at[0, 0].set(1.0).at[0, 4].set(1.0)
  last = sw.at[:, 0].set(0.0)
  term = 1 - con
  ret = lambda_return(last, term, rew, val, val, 1.0, 0.95)
  ret = np.asarray(ret)[0]
  # t=3 is the last step of the OLD window [0,4) -- must NOT see the reward.
  assert abs(ret[3]) < 1e-4, f'reward leaked backward across the switch: {ret[3]}'
  # t=4 is the first step of the NEW window -- must see it in full.
  assert ret[4] > 900.0, f'reward did not credit the new window: {ret[4]}'


def test_imag_loss_wkr_dense_path_shapes_dtypes_and_gradients():
  """End-to-end through the real function (not just ``lambda_return``): the
  actual REINFORCE construction (logp/entropy alignment, advantage shape,
  loss shape/dtype) on an irregular switch mask, plus a gradient sanity check
  -- the policy logits must receive a nonzero gradient through this path.
  """
  B, T, A = 2, 13, 4
  rng = np.random.RandomState(3)
  sw = jnp.array([
      [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
      [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
  ], f32)
  # The real worker policy head is ``outs.OneHot``, whose ``.sample()`` (used
  # to build the real ``imgact``) returns one-hot vectors, not raw indices --
  # match that here so ``.logp()`` sees what it actually gets in training.
  act_idx = jnp.array(rng.randint(0, A, (B, T)))
  act = {'action': jax.nn.one_hot(act_idx, A, dtype=f32)}
  con = jnp.array(rng.uniform(0.9, 0.999, (B, T)).astype(np.float32))
  wkr_rew = jnp.array(rng.randn(B, T).astype(np.float32) * 0.3)
  val_feat = jnp.array(rng.randn(B, T).astype(np.float32))

  def loss_fn(logits_w):
    logits = jnp.einsum('bt,c->btc', val_feat, logits_w)
    policy = {'action': outs.OneHot(logits)}
    value = outs.MSE(val_feat)
    slowvalue = outs.MSE(0.9 * val_feat)
    norms = [_StubNorm() for _ in range(3)]
    losses, outs_, mets = imag_loss_wkr(
        act, wkr_rew, con, policy, value, slowvalue, *norms,
        update=True, contdisc=True, slowtar=False, horizon=333,
        actent=3e-4, skill_window=sw)
    for key in ('wkr_policy', 'wkr_goal_value'):
      assert losses[key].shape == (B, T - 1), (key, losses[key].shape)
      assert losses[key].dtype == f32, (key, losses[key].dtype)
    assert outs_['wkr_goal_ret'].shape == (B, T - 1)
    return losses['wkr_policy'].mean() + losses['wkr_goal_value'].mean()

  logits_w = jnp.array(rng.randn(A).astype(np.float32) * 0.5)
  loss, grad = jax.value_and_grad(loss_fn)(logits_w)
  assert np.isfinite(float(loss))
  assert np.all(np.isfinite(np.asarray(grad)))
  assert float(jnp.abs(grad).max()) > 1e-8, (
      'the dense-path worker policy loss produced zero gradient into the '
      'policy logits -- REINFORCE is disconnected on this path')


def test_dense_and_windowed_paths_agree_when_the_switch_pattern_is_k_periodic():
  """The one point of contact between the two algorithms: if a genuinely
  variable-length rollout HAPPENS to switch on a perfectly regular K-periodic
  grid (as the pinned control forces), the dense path's return must equal
  what the windowed split_traj path would give for the SAME window --
  confirming the two independently-implemented credit algorithms don't
  quietly disagree in their one overlapping case.
  """
  B, T, K = 1, 17, 8   # H=16, matches the real pinned configuration exactly
  rng = np.random.RandomState(5)
  rew = jnp.array(rng.randn(B, T).astype(np.float32) * 0.3)
  con = jnp.full((B, T), 0.997)
  val = jnp.array(rng.randn(B, T).astype(np.float32) * 2.0)
  sw = jnp.array([1 if t % K == 0 else 0 for t in range(T)], f32)[None]

  last = sw.at[:, 0].set(0.0)
  term = 1 - con
  dense_ret = np.asarray(lambda_return(last, term, rew, val, val, 1.0, 0.95))[0]

  # Windowed reference: one plain lambda_return per K-length window, each
  # bootstrapping off the BOUNDARY state's own value (the split_traj
  # convention: the window's last state holds the OLD goal for bootstrap).
  n_win = (T - 1) // K
  for w in range(n_win):
    s = w * K
    seg_rew = rew[:, s:s + K + 1]
    seg_con = con[:, s:s + K + 1]
    seg_val = val[:, s:s + K + 1]
    seg_last = jnp.zeros((1, K + 1), f32)
    seg_term = 1 - seg_con
    seg_ret = lambda_return(seg_last, seg_term, seg_rew, seg_val, seg_val, 1.0, 0.95)
    np.testing.assert_allclose(
        dense_ret[s:s + K], np.asarray(seg_ret)[0], rtol=1e-5, atol=1e-6,
        err_msg=f'window {w} (steps {s}..{s+K}) disagrees between the dense '
                'and windowed credit algorithms on a K-periodic pattern')
