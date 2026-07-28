"""Director-equivalence tests for the block-pooled variable-K manager path.

Pin ``agent.goal_duration_fixed = agent.manager_sample_freq`` (= K) with
the block-pooled variable-K path
becomes a pure *reparameterization* of fixed-K Director: every manager decision
is held for exactly K steps, so the two paths see the same decision states, the
same block-pooled rewards, the same block continuations and the same policy
distributions. Under that pinning the manager losses, the statistics fed to the
running normalizers, and the gradients must all match fixed-K Director
**exactly**. Anything that differs is a bug in the variable-K plumbing, not a
difference in algorithm -- which is precisely the configuration (e334/e335,
rerun as e342/e343) that trains far below true Director.

The existing ``test_variable_goals.py`` suite checks the pooling helpers in
isolation (segment ids, pooled rewards, pooled continuation, downsampled
states) and they agree. These tests instead compare what actually reaches the
optimizer: the reduced loss tensors that ``Agent.loss`` builds
(``{k: v.mean(1) ...}`` over ``imag_loss_mgr``'s output), the arguments handed
to ``retnorm``/``advnorm``, and the gradients. The gap between the two levels
is the padded decision axis: ``downsample_at_switch_mask`` packs decisions into
a static ``T``-wide buffer and forward-fills the tail, so at H=16/K=8 only 3 of
17 columns are real.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.agent import (
    aggregate_mgr_cont,
    aggregate_mgr_extr_rew,
    decision_mean_rescale,
    downsample_at_switch_mask,
    downsample_manager_states,
    imag_loss_mgr,
    imag_reward_pad,
    patch_trailing_replay_state,
    variable_block_director_tensors,
    worker_split_window,
)
import embodied.jax.outs as outs

f32 = jnp.float32

# Imagination geometry of the runs under test (BIG scale: imag_length=16,
# manager_sample_freq=8, goal_duration_fixed=8).
M, H, K, D, C = 4, 16, 8, 6, 8
T = H + 1


def _fixed_switch_mask(B, length, k):
  pos = jnp.arange(length)
  return jnp.broadcast_to(f32(jnp.equal(pos % k, 0)), (B, length))


def _rollout(seed=0, length=T, batch=M):
  """A synthetic imagined rollout shared by both manager paths."""
  k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 4)
  return dict(
      feat=jax.random.normal(k1, (batch, length, D)),
      rew=jax.random.normal(k2, (batch, length)) * 0.3,
      expl=jax.random.normal(k3, (batch, length)) * 0.1,
      con=jnp.full((batch, length), 0.997),
      logit_w=jax.random.normal(k4, (D, C)) * 0.5,
  )


class RecordingNorm:
  """Stand-in for ``embodied.jax.Normalize`` with ``impl='none'``.

  Returns the identity statistics (offset 0, scale 1) like the configured
  ``valnorm``/``advnorm``, and records every tensor it is asked to normalize so
  the tests can compare what the *real* running normalizers would accumulate.
  """

  def __init__(self):
    self.seen = []

  def stats(self):
    return 0.0, 1.0

  def __call__(self, x, update=True, weights=None):
    self.seen.append((x, weights))
    return 0.0, 1.0

  @staticmethod
  def moments(entry):
    """Mean/std the real ``Normalize`` would accumulate from a recorded call."""
    x, weights = entry
    if weights is None:
      return x.mean(), x.std()
    w = jnp.broadcast_to(f32(weights), x.shape)
    den = jnp.maximum(w.sum(), 1e-8)
    mean = (x * w).sum() / den
    var = (jnp.square(x) * w).sum() / den - jnp.square(mean)
    return mean, jnp.sqrt(jnp.maximum(var, 0.0))


def _manager_tensors(kind, roll):
  """Manager AC inputs, built exactly as ``Agent.loss`` builds them.

  ``kind='fixed'`` mirrors the ``else`` branch of the ``variable_goal_length``
  test in ``Agent.loss`` (fixed-K Director); ``kind='block'`` mirrors the
  block-pooled branch with a switch mask pinned to every K steps
  (what ``goal_duration_fixed=K`` produces -- see
  ``test_switch_mask_from_skills_ignores_duration_head_under_fixed_duration``).
  """
  feat, rew, expl, con = roll['feat'], roll['rew'], roll['expl'], roll['con']
  logits_full = jnp.einsum('mtd,dc->mtc', feat, roll['logit_w'])
  skills_full = {'skill': jax.nn.one_hot(jnp.argmax(logits_full, -1), C)}
  if kind == 'fixed':
    return dict(
        feat=feat[:, ::K],
        skills=jax.tree.map(lambda s: s[:, ::K], skills_full),
        rew=imag_reward_pad(
            aggregate_mgr_extr_rew(rew, con, K, without_zeros=True)),
        expl=imag_reward_pad(
            aggregate_mgr_extr_rew(expl, con, K, without_zeros=True)),
        con=aggregate_mgr_cont(con, K, without_zeros=True),
        switch=None)
  sw = _fixed_switch_mask(feat.shape[0], feat.shape[1], K)
  feat_eff = downsample_at_switch_mask({'x': feat}, sw)['x']
  n_mgr = feat_eff.shape[1]
  b_rew, b_expl, b_con, b_switch = variable_block_director_tensors(
      rew, con, expl, sw, n_mgr, agg_mode='mean')
  return dict(
      feat=feat_eff,
      skills=downsample_at_switch_mask(skills_full, sw),
      rew=b_rew, expl=b_expl, con=b_con, switch=b_switch)


def _manager_losses(tensors, logit_w, val_w):
  """Run the real ``imag_loss_mgr`` and reduce as ``Agent.loss`` reduces it."""
  feat = tensors['feat']
  policy = {'skill': outs.OneHot(jnp.einsum('mtd,dc->mtc', feat, logit_w))}
  val = outs.MSE(feat @ val_w)
  slowval = outs.MSE(0.9 * (feat @ val_w))
  norms = {k: RecordingNorm() for k in
           ('extr_ret', 'expl_ret', 'extr_val', 'expl_val', 'adv')}
  losses, out, metrics = imag_loss_mgr(
      tensors['skills'], tensors['rew'], tensors['expl'], tensors['con'],
      policy, val, slowval, val, slowval,
      norms['extr_ret'], norms['expl_ret'], norms['extr_val'],
      norms['expl_val'], norms['adv'],
      update=True, contdisc=True, slowtar=False, horizon=333,
      mgr_expl_weight=0.1, actent=3e-4, switch_mask=tensors['switch'])
  # ``Agent.loss``: losses.update({k: v.mean(1).reshape(...) for ...})
  reduced = {k: v.mean(1) for k, v in losses.items()}
  return reduced, norms, metrics


def _both_paths(seed=0):
  roll = _rollout(seed)
  val_w = jax.random.normal(jax.random.PRNGKey(seed + 100), (D,)) * 0.4
  out = {}
  for kind in ('fixed', 'block'):
    tensors = _manager_tensors(kind, roll)
    out[kind] = _manager_losses(tensors, roll['logit_w'], val_w) + (tensors,)
  return roll, val_w, out


def test_block_pooled_decision_states_and_rewards_match_fixed_k():
  """Sanity floor: the two paths agree on every REAL decision column.

  This is the part the existing helper-level tests already cover; asserting it
  here makes the failures below unambiguously about the padded tail rather than
  about the pooling itself.
  """
  roll = _rollout()
  fx = _manager_tensors('fixed', roll)
  bl = _manager_tensors('block', roll)
  n = fx['feat'].shape[1]
  np.testing.assert_allclose(fx['feat'], bl['feat'][:, :n], rtol=1e-6)
  np.testing.assert_allclose(fx['rew'], bl['rew'][:, :n], rtol=1e-5, atol=1e-7)
  np.testing.assert_allclose(fx['con'], bl['con'][:, :n], rtol=1e-6)
  np.testing.assert_allclose(
      fx['skills']['skill'], bl['skills']['skill'][:, :n])
  # ...and that the block path's tail really is padding, not new information.
  np.testing.assert_allclose(
      bl['feat'][:, n:],
      jnp.broadcast_to(bl['feat'][:, n - 1:n], bl['feat'][:, n:].shape))


def test_block_pooled_valid_decision_count_excludes_empty_trailing_segment():
  """``mgr_switch`` must count decisions that actually own realized steps.

  At H=16/K=8 the rollout holds switches at t=0, 8 and 16. The t=16 switch
  starts a segment with ZERO steps inside the rollout -- it is the bootstrap
  state, exactly the column fixed-K drops via ``[:, :-1]``. Fixed-K therefore
  trains the manager on 2 decisions; the block path must agree, or it credits a
  decision whose pooled reward is identically zero.
  """
  roll = _rollout()
  bl = _manager_tensors('block', roll)
  n_credited = int(bl['switch'][0].sum())
  n_fixed_decisions = _manager_tensors('fixed', roll)['rew'].shape[1] - 1
  assert n_fixed_decisions == 2
  assert n_credited == n_fixed_decisions, (
      f'block-pooled path marks {n_credited} decisions as trainable but '
      f'fixed-K Director trains {n_fixed_decisions}; the extra one is the '
      f'switch on the rollout\'s final step, whose segment holds no steps')


@pytest.mark.parametrize('loss_key', [
    'mgr_policy', 'mgr_extr_value', 'mgr_expl_value'])
def test_block_pooled_manager_losses_match_fixed_k(loss_key):
  """The loss that reaches the optimizer must be identical, not proportional.

  ``Agent.loss`` reduces every manager loss with ``v.mean(1)`` over the
  decision axis. Fixed-K's axis is exactly the decision count (2 here); the
  block-pooled path's axis is the padded buffer width (16 here), so the same
  per-decision losses come out ~K times smaller -- an unintended, K-dependent
  down-weighting of the manager against the world model and the worker, which
  under true variable-K also drifts with the realized hold length.
  """
  _, _, both = _both_paths()
  fx = both['fixed'][0][loss_key]
  bl = both['block'][0][loss_key]
  np.testing.assert_allclose(bl, fx, rtol=1e-5, atol=1e-7)


def test_block_pooled_manager_gradients_match_fixed_k():
  """Full gradient path: same rollout, same params -> same manager gradient."""
  roll = _rollout()
  val_w = jax.random.normal(jax.random.PRNGKey(100), (D,)) * 0.4
  scales = {'mgr_policy': 1.0, 'mgr_extr_value': 1.0, 'mgr_expl_value': 1.0}

  def total(kind, logit_w, val_w):
    tensors = _manager_tensors(kind, {**roll, 'logit_w': logit_w})
    reduced, _, _ = _manager_losses(tensors, logit_w, val_w)
    return sum(reduced[k].mean() * s for k, s in scales.items())

  g_fixed = jax.grad(total, argnums=(1, 2))('fixed', roll['logit_w'], val_w)
  g_block = jax.grad(total, argnums=(1, 2))('block', roll['logit_w'], val_w)
  np.testing.assert_allclose(g_block[0], g_fixed[0], rtol=1e-4, atol=1e-6)
  np.testing.assert_allclose(g_block[1], g_fixed[1], rtol=1e-4, atol=1e-6)


def test_block_pooled_return_normalizer_statistics_match_fixed_k():
  """``mgr_retnorm`` (impl ``meanstd``) must see the real decisions only.

  ``imag_loss_mgr`` hands the whole return tensor to the running normalizer
  with no mask, so in the block-pooled path 13 of 16 columns are the
  forward-filled padding tail (all equal to the bootstrap value). That shrinks
  the running std the advantage is divided by, inflating every manager
  advantage relative to Director's.
  """
  _, _, both = _both_paths()
  for key in ('extr_ret', 'expl_ret', 'adv'):
    fx_mean, fx_std = RecordingNorm.moments(both['fixed'][1][key].seen[0])
    bl_mean, bl_std = RecordingNorm.moments(both['block'][1][key].seen[0])
    np.testing.assert_allclose(
        bl_mean, fx_mean, rtol=1e-4, atol=1e-6,
        err_msg=f'{key}: normalizer accumulates a different mean')
    np.testing.assert_allclose(
        bl_std, fx_std, rtol=1e-4, atol=1e-6,
        err_msg=f'{key}: normalizer accumulates a different spread')


def test_block_pooled_entropy_statistics_match_fixed_k():
  """Manager entropy metrics/adapters must be valid-decision weighted.

  The manager's adaptive entropy regularizer (``manager_actent_impl: mult``)
  drives its Lagrange multiplier from the MEAN entropy over the decision axis.
  On the padded axis that mean is dominated by repeated copies of the last real
  decision, so the controller tracks the wrong quantity.
  """
  _, _, both = _both_paths()
  fx = both['fixed'][2]['mgr_ent/skill']
  bl = both['block'][2]['mgr_ent/skill']
  np.testing.assert_allclose(bl, fx, rtol=1e-4, atol=1e-6)


def test_block_pooled_replay_manager_value_loss_matches_fixed_k():
  """Replay-side manager critic: same batch window -> same loss magnitude.

  Mirrors ``Agent.loss``'s ``repval_loss`` manager branch for a
  ``batch_length=64`` window. Fixed-K downsamples to 10 boundary states (9
  training columns); the block path packs its 9 decisions into a 64-wide buffer
  and averages over all of it.
  """
  B, T_rep = 3, 64
  roll = _rollout(seed=7, length=T_rep, batch=B)
  feat, rew, con, expl = roll['feat'], roll['rew'], roll['con'], roll['expl']
  last = jnp.zeros((B, T_rep), bool)
  val_w = jax.random.normal(jax.random.PRNGKey(11), (D,)) * 0.4

  # --- fixed-K (Agent.loss lines under ``else: feat_down = downsample_...``)
  fx_feat = downsample_manager_states({'x': feat}, K)['x']
  Tm = T_rep - 1
  n = Tm // K
  rem = Tm - n * K
  idx = [0] + [1 + i * K - 1 for i in range(1, n + 1)]
  idx += [n * K, Tm] if rem > 0 else [Tm]
  idx = sorted(set(idx))
  fx_last = last[:, idx]
  fx_rew = imag_reward_pad(
      aggregate_mgr_extr_rew(rew, con, K, without_zeros=True))
  fx_weight = f32(~fx_last)

  # --- block-pooled variable-K with the hold pinned to K
  sw = _fixed_switch_mask(B, T_rep, K)
  bl_feat = patch_trailing_replay_state(
      downsample_at_switch_mask({'x': feat}, sw)['x'], feat, sw)
  n_mgr = bl_feat.shape[1]
  bl_rew, _, bl_con, valid = variable_block_director_tensors(
      rew, con, expl, sw, n_mgr, agg_mode='mean')
  bl_last = patch_trailing_replay_state(
      downsample_at_switch_mask({'l': f32(last)}, sw)['l'], f32(last), sw)
  # Exactly the weight ``Agent.loss`` builds for this branch.
  bl_weight = (
      f32(bl_last < 0.5) * valid * decision_mean_rescale(valid[:, :-1]))

  assert fx_feat.shape[1] == int(valid[0].sum()) + 1, (
      'the two replay paths disagree on how many manager decisions the window '
      f'contains: fixed-K {fx_feat.shape[1]}, block-pooled {int(valid[0].sum())}')

  def value_loss(feat_down, weight, rew_down, w_val):
    pred = feat_down @ w_val
    target = jnp.concatenate([rew_down[:, :-1], 0 * rew_down[:, -1:]], 1)
    return (weight[:, :-1] * jnp.square(pred - target)[:, :-1]).mean()

  fx_loss = value_loss(fx_feat, fx_weight, fx_rew, val_w)
  bl_loss = value_loss(bl_feat, bl_weight, bl_rew, val_w)
  np.testing.assert_allclose(bl_loss, fx_loss, rtol=1e-4, atol=1e-7)


def test_duration_pinned_rollout_switches_on_the_static_every_k_grid():
  """The premise behind reusing Director's ``split_traj`` under variable K.

  ``_imagine_with_manager``'s variable-K branch starts with ``remaining = 0``
  (switch at t=0), then reloads the countdown with ``_duration_steps``, which
  ``goal_duration_fixed`` pins to K. So the realized switch pattern is exactly
  the fixed-K grid -- the property the windowed worker credit needs.
  """
  remaining, switches = 0, []
  for _ in range(T):
    update = remaining <= 0
    switches.append(float(update))
    remaining = (K if update else remaining) - 1
  np.testing.assert_array_equal(
      jnp.array(switches), _fixed_switch_mask(1, T, K)[0])


def test_worker_split_traj_applies_to_the_duration_pinned_control():
  """Fixed-K Director and duration-pinned variable-K must pick the same worker
  credit algorithm.

  Before this fix the branch condition was ``not self.variable_goal_length``,
  so the duration-pinned control -- whose rollout switches on the identical
  every-K grid (test above) -- fell through to the dense fallback: a different
  worker objective (boundary state conditioned on the NEW goal, single
  lambda-return with resets, different per-window loss weighting) than the
  Director baseline it is meant to reproduce.
  """
  # Fixed-K Director: windowed, as always.
  assert worker_split_window(True, False, 0, K, H) == K
  # Duration-pinned variable K: same grid -> same windowing.
  assert worker_split_window(True, True, K, K, H) == K
  # Genuinely variable holds: no static grid, dense fallback.
  assert worker_split_window(True, True, 0, K, H) == 0
  # Feature switched off, or a horizon that is not a whole number of windows.
  assert worker_split_window(False, False, 0, K, H) == 0
  assert worker_split_window(True, False, 0, 5, H) == 0
  # A pinned hold that does divide the horizon is used even if it differs from
  # manager_sample_freq (the duration head, not the fixed-K knob, sets timing).
  assert worker_split_window(True, True, 4, 8, H) == 4


class _StubAdapter:
  """Minimal ``AutoAdapt`` stand-in: records calls, returns a linear penalty."""

  def __init__(self):
    self.calls = []

  def __call__(self, reg, update=True, target=None, weights=None):
    self.calls.append(reg)
    return 0.5 * reg, {}

  def scale(self):
    return jnp.float32(0.5)


def _mgr_loss_with_duration_head(
    duration_fixed, dur_actent, dur_reg, include_duration=True):
  """Manager loss for a two-head (skill + duration) manager under a pinned hold."""
  roll = _rollout(seed=3)
  tensors = _manager_tensors('block', roll)
  feat = tensors['feat']
  n_dur = 16
  w_dur = jax.random.normal(jax.random.PRNGKey(5), (D, n_dur)) * 0.3
  # The real duration head is a scalar Categorical over hold classes (indices),
  # not a factorized OneHot like the skill head.
  policy = {
      'skill': outs.OneHot(jnp.einsum('mtd,dc->mtc', feat, roll['logit_w'])),
  }
  dur_head = outs.Categorical(jnp.einsum('mtd,dc->mtc', feat, w_dur))
  if include_duration:
    policy['duration'] = dur_head
  # ``embodied.jax.heads`` attaches these entropy bounds to the head outputs;
  # ``imag_loss_mgr``'s adaptive entropy loop keys off them.
  policy['skill'].minent, policy['skill'].maxent = 0.0, float(np.log(C))
  dur_head.minent, dur_head.maxent = 0.0, float(np.log(n_dur))
  skills = dict(tensors['skills'])
  skills['duration'] = jnp.argmax(dur_head.logits, -1)
  val = outs.MSE(feat @ jnp.ones((D,), f32) * 0.3)
  norms = [RecordingNorm() for _ in range(5)]
  losses, _, _ = imag_loss_mgr(
      skills, tensors['rew'], tensors['expl'], tensors['con'], policy,
      val, val, val, val, *norms,
      update=True, contdisc=True, slowtar=False, horizon=333,
      mgr_expl_weight=0.1, actent=3e-4, switch_mask=tensors['switch'],
      mgr_actent_adapter=_StubAdapter(), mgr_actent_perdim=True,
      mgr_dur_actent_adapter=dur_actent, duration_fixed=duration_fixed,
      dur_reg_weight=dur_reg, dur_reg_target=8.0, dur_min=1)
  return losses


def test_pinned_duration_head_receives_no_entropy_or_prior_gradient():
  """With the hold pinned, the duration head must be fully inert.

  ``manager_reinforce_policy`` (2026-07-27) dropped 'duration' from the
  REINFORCE sum under ``goal_duration_fixed``, but two other channels kept
  training it against a shared trunk: the adaptive duration-entropy
  regularizer and the ``goal_duration_reg`` prior on E[dur]. Neither can affect
  the trajectory once ``_duration_steps`` overrides the sample, and fixed-K
  Director has no duration head at all -- so under the pinning the manager loss
  must equal the skill-only manager loss exactly.
  """
  adapter = _StubAdapter()
  pinned = _mgr_loss_with_duration_head(True, adapter, dur_reg=0.01)
  assert not adapter.calls, (
      'the duration-entropy adapter was still stepped under a pinned hold')
  assert 'goal_duration_prior' not in pinned

  # The free-duration configuration is unaffected (contrast, so the assertion
  # above cannot pass by the mechanism simply being dead everywhere).
  adapter_free = _StubAdapter()
  free = _mgr_loss_with_duration_head(False, adapter_free, dur_reg=0.01)
  assert adapter_free.calls, 'contrast failed: adapter never runs at all'
  assert float(jnp.abs(free['mgr_policy'] - pinned['mgr_policy']).max()) > 1e-8

  # ...and the pinned loss matches a manager with no duration head whatsoever,
  # which is what fixed-K Director actually is.
  skill_only = _mgr_loss_with_duration_head(
      True, _StubAdapter(), dur_reg=0.01, include_duration=False)
  np.testing.assert_allclose(
      pinned['mgr_policy'], skill_only['mgr_policy'], rtol=1e-5, atol=1e-7)


def test_block_pooled_lambda_returns_match_fixed_k_elementwise():
  """The manager's lambda-returns must agree decision-by-decision, not just in
  aggregate.

  The loss-level tests above would still pass if two compensating errors
  cancelled inside the return construction, and the lambda-return is the named
  remaining suspect if the padded-axis fixes do not close the training gap. So
  compare the returns themselves, elementwise, on the real decisions.

  This also pins down the one place the padded tail can still influence a real
  decision: the last credited decision bootstraps THROUGH the forward-filled
  tail (pooled reward 0, continuation identity 1.0), where fixed-K bootstraps
  directly off its final column. The two agree only because that tail is a
  reward-free, non-terminating repeat of the bootstrap state -- exactly the
  property ``aggregate_mgr_cont_variable``'s empty-segment identity provides.
  """
  _, _, both = _both_paths()
  # norms record (tensor, weights); the first extr_ret call is the retnorm one.
  fx_ret = both['fixed'][1]['extr_ret'].seen[0][0]
  bl_ret = both['block'][1]['extr_ret'].seen[0][0]
  n = fx_ret.shape[1]
  np.testing.assert_allclose(bl_ret[:, :n], fx_ret, rtol=1e-5, atol=1e-7)
  # The tail past the real decisions is the constant bootstrap value, i.e. it
  # adds no reward and never terminates -- if it did either, the last real
  # decision's return above would not have matched.
  tail = bl_ret[:, n:]
  np.testing.assert_allclose(
      tail, jnp.broadcast_to(tail[:, :1], tail.shape), rtol=1e-5, atol=1e-6)
