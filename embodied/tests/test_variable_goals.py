import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.agent import (
    aggregate_mgr_cont,
    aggregate_mgr_cont_variable,
    aggregate_mgr_extr_rew,
    aggregate_mgr_extr_rew_variable,
    align_skill_events,
    downsample_at_switch_mask,
    downsample_manager_states,
    head_logp_time,
    imag_reward_pad,
    manager_reinforce_policy,
    masked_cumprod,
    patch_trailing_replay_state,
    relabel_truncated_last_duration,
    switch_valid_mask,
    variable_block_director_tensors,
    variable_segment_ids,
)
import embodied.jax.outs as outs

f32 = jnp.float32


def _fixed_switch_mask(B, T, k):
  pos = jnp.arange(T)
  return jnp.broadcast_to(jnp.equal(pos % k, 0).astype(jnp.float32), (B, T))


def test_variable_segment_ids_fixed_k():
  sw = _fixed_switch_mask(2, 17, 8)
  seg = variable_segment_ids(sw)
  # A switch STARTS its segment (``cumsum(sw)-1``): switches at 0/8/16 -> the held
  # window [0,8) is segment 0, [8,16) segment 1, matching the fixed-K blocks.
  np.testing.assert_array_equal(seg[0, :8], 0)
  np.testing.assert_array_equal(seg[0, 8:16], 1)


def test_masked_cumprod_resets_at_switches():
  c = jnp.ones(8)
  sw = jnp.array([1, 0, 0, 0, 1, 0, 0, 0, 0], f32)
  cprod = masked_cumprod(c, sw)
  assert cprod.shape == (8,)
  np.testing.assert_allclose(cprod[:4], 1.0)
  np.testing.assert_allclose(cprod[4:8], 1.0)


def test_aggregate_variable_matches_fixed_k_blocks():
  # rew and con share the length-T convention (r=rew[1:], c=con[:-1]); rew[0] is the
  # throwaway pre-action slot, as in the real imagination/replay call sites.
  B, T = 2, 17
  k = 8
  rew = jnp.arange(T, dtype=jnp.float32)[None, :].repeat(B, 0)
  con = jnp.ones((B, T))
  sw = _fixed_switch_mask(B, T, k)
  var_agg = aggregate_mgr_extr_rew_variable(rew, con, sw, without_zeros=True)
  fix_agg = aggregate_mgr_extr_rew(rew, con, k, without_zeros=True)
  # On a fixed-K mask the variable per-segment means must equal the fixed-K block
  # means (the segmentation off-by-one bug made these differ).
  np.testing.assert_allclose(
      var_agg[:, :fix_agg.shape[1]], fix_agg, rtol=1e-5, atol=1e-5)


def test_downsample_at_switch_mask_fixed_k():
  B, T = 1, 17
  k = 8
  feat = {'x': jnp.arange(B * T, dtype=jnp.float32).reshape(B, T)}
  sw = _fixed_switch_mask(B, T, k)
  down = downsample_at_switch_mask(feat, sw)['x']
  expected_idx = [0, 8, 16]
  np.testing.assert_array_equal(down[0, :len(expected_idx)], jnp.asarray(expected_idx, f32))
  # Padded tail should hold the last real switch value, not zero.
  np.testing.assert_allclose(down[0, len(expected_idx):], down[0, len(expected_idx) - 1])


def test_variable_block_director_matches_fixed_k():
  # End-to-end: on a fixed-K mask the variable block-rew tensors must reproduce the
  # fixed-K Director credit (block-pooled rewards + abstract_traj continuation).
  B, T = 2, 17
  k = 8
  rew = jnp.arange(T, dtype=jnp.float32)[None, :].repeat(B, 0)
  con = jnp.full((B, T), 0.9, jnp.float32)  # <1 so the continuation product bites
  sw = _fixed_switch_mask(B, T, k)
  extr, _, cont, _ = variable_block_director_tensors(rew, con, rew, sw, T)
  fix_extr = imag_reward_pad(aggregate_mgr_extr_rew(rew, con, k, without_zeros=True))
  fix_cont = aggregate_mgr_cont(con, k, without_zeros=True)
  np.testing.assert_allclose(
      extr[:, :fix_extr.shape[1]], fix_extr, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(
      cont[:, :fix_cont.shape[1]], fix_cont, rtol=1e-5, atol=1e-5)


def test_aggregate_mgr_cont_variable_pools_segment_product():
  """Per-segment continuation must be the PRODUCT over the held-goal segment, not
  its first step. With cont<1 and length-8 segments the compounded product
  (~0.8**8 ≈ 0.17) is far below the first-step value (0.8); the previous
  ``segment_max`` returned the first step (≈ no discounting over the segment).
  Constant cont makes this independent of the segment-boundary convention: a
  product of >1 terms <1 is strictly below the per-step value a max-pool returns.
  """
  B, T, k = 1, 17, 8
  con = jnp.full((B, T), 0.8, jnp.float32)
  sw = _fixed_switch_mask(B, T, k)
  agg = aggregate_mgr_cont_variable(con, sw, without_zeros=True)
  # At least one full-length segment shows the compounded product, well below the
  # per-step 0.8 that a max-pool (the old bug) would have returned.
  assert float(jnp.min(agg[0, :2])) < 0.5
  # Pooled continuation never exceeds a single-step continuation.
  assert float(jnp.max(agg[0, :2])) <= 0.8 + 1e-5
  # Empty trailing segments are finite zeros (no +inf segment_min identity that
  # would become NaN under the downstream ``* block_mask``).
  assert bool(jnp.isfinite(agg).all())


def test_switch_valid_mask_pads_tail():
  sw = jnp.array([[1, 0, 0, 1, 0, 0, 0, 0, 0]], f32)
  valid = switch_valid_mask(sw, 9)
  np.testing.assert_array_equal(valid[0], [1, 1, 0, 0, 0, 0, 0, 0, 0])


def test_switch_rate_matches_mean_duration():
  """Held duration p implies switch_rate ≈ 1/p on long sequences."""
  from dreamerv3.agent import Agent  # noqa: circular import guard in test only

  class _Cfg:
    variable_goal_length = True
    goal_duration_min = 1

  p = 8
  T = 64
  durations = jnp.full((1, T), p - 1, dtype=jnp.int32)  # class index for p=8
  skills = {'skill': jnp.zeros((1, T, 8, 8)), 'duration': durations}
  agent = object.__new__(Agent)
  agent.variable_goal_length = True
  agent.goal_duration_min = 1
  agent.goal_duration_fixed = 0
  agent.manager_sample_freq = 8
  sw = agent._switch_mask_from_skills(skills)
  switch_rate = float(sw.mean())
  assert abs(switch_rate - 1.0 / p) < 0.02


if __name__ == '__main__':
  pytest.main([__file__, '-xvs'])


def test_last_decision_bootstrap_when_switch_lands_on_final_step():
  """Regression probe: does a switch landing exactly on the rollout's last
  timestep (avail=0 real transitions for the segment it would open) zero out
  the RETURN for that (real, valid) decision instead of letting it bootstrap
  purely off its own value? This is the deterministic case
  ``goal_duration_fixed=8`` hits every single rollout at ``imag_length=16``
  (switches land exactly at 0, 8, 16 -- t=16 is the last valid index)."""
  from dreamerv3.agent import lambda_return
  B, T, k = 1, 17, 8
  rew = jnp.zeros((B, T), jnp.float32)
  con = jnp.full((B, T), 0.997, jnp.float32)  # mimic contdisc-baked-in discount
  sw = _fixed_switch_mask(B, T, k)  # switches at 0, 8, 16
  n_sw = int(sw.sum())
  print('n_sw =', n_sw)

  extr, _, cont, switch = variable_block_director_tensors(rew, con, rew, sw, T)
  print('mgr_extr_rew[0] =', np.asarray(extr[0]))
  print('mgr_cont[0]     =', np.asarray(cont[0]))
  print('mgr_switch[0]   =', np.asarray(switch[0]))

  # Fake per-node values: distinct so we can see which one leaks into the return.
  val = jnp.array([[10.0, 20.0, 30.0] + [30.0] * (T - 3)], jnp.float32)
  last = jnp.zeros_like(con)
  term = 1 - cont
  ret = lambda_return(last, term, extr, val, val, 1.0, 0.95)
  print('mgr_extr_ret[0] =', np.asarray(ret[0]))

  # Node 2 (state index 2 = t=16, the LAST real decision) has no real trailing
  # steps within this rollout. Its return should equal its own bootstrap value
  # (30.0, i.e. a neutral 0-advantage terminal node) -- NOT collapse to 0.
  # With disc=1.0 (matching the real contdisc=True config, where all decay
  # lives in ``con`` itself) and the empty-segment identity fix (1.0, not 0.0),
  # the padding tail beyond node 2 is a lossless V=V=V... chain: this must now
  # be EXACT, not approximate.
  r2 = float(ret[0, 2])
  print('R at last real decision (expected exactly 30.0, i.e. == its own value):', r2)
  assert abs(r2 - 30.0) < 1e-4, (
      f'last real decision return={r2}, expected 30.0 (its own bootstrap value); '
      'a switch landing exactly on the final timestep is zeroing the return '
      'instead of bootstrapping.')


def test_final_bootstrap_bug_is_specific_to_block_pooled_varK():
  """Does the same 'switch lands on the final timestep' scenario also corrupt
  (a) the fixed-K Director path and (b) the full-resolution (non-block-pooled)
  var-K path, or is it specific to variable_goal_block_rew's padded-to-static-
  horizon construction? Same T=17, k=8 fixture as the previous test."""
  from dreamerv3.agent import lambda_return
  B, T, k = 1, 17, 8
  rew = jnp.zeros((B, T), jnp.float32)
  con = jnp.full((B, T), 0.997, jnp.float32)
  sw = _fixed_switch_mask(B, T, k)

  # --- (a) Fixed-K Director path: exactly mirrors the real call site
  # (agent.py ~2769-2773): _mgr_cont/_mgr_extr_rew, without_zeros=True, no padding.
  fixedk_cont = aggregate_mgr_cont(con, k, without_zeros=True)          # (B, 3): no padding
  fixedk_rew = imag_reward_pad(aggregate_mgr_extr_rew(rew, con, k, without_zeros=True))
  print('fixedk_cont  =', np.asarray(fixedk_cont[0]))
  print('fixedk_rew   =', np.asarray(fixedk_rew[0]))
  val_fixedk = jnp.array([[10.0, 20.0, 30.0]], jnp.float32)
  last_fixedk = jnp.zeros_like(fixedk_cont)
  term_fixedk = 1 - fixedk_cont
  ret_fixedk = lambda_return(
      last_fixedk, term_fixedk, fixedk_rew, val_fixedk, val_fixedk, 1.0, 0.95)
  print('fixedk_ret   =', np.asarray(ret_fixedk[0]))
  r2_fixedk = float(ret_fixedk[0, -1])
  # Fixed-K's array is EXACTLY sized to its 3 real states (no padding), so its
  # own last exposed slot (index 1, "decision 1") gets rets[-1]=boot[-1]=30.0
  # as a direct, undiminished base case -- but that base case is then combined
  # through ONE real (non-identity) transition of its own (con~0.976, a real
  # 8-step hold), so the expected value is con*30 (a real decay), not exactly
  # 30 -- 30*0.9762503 = 29.2875.
  print('fixed-K last-decision return (expect ~29.29, real decay over its own hold):',
        r2_fixedk)

  # --- (b) Full-resolution var-K path (variable_goal_block_rew=False): mirrors
  # the real call site (agent.py ~2704-2711) -- mgr_cont = con directly, no
  # pooling, no padding at all; every one of the T real positions is genuine.
  val_full = jnp.full((B, T), 30.0, jnp.float32)
  val_full = val_full.at[0, 0].set(10.0).at[0, 8].set(20.0)
  last_full = jnp.zeros_like(con)
  term_full = 1 - con
  ret_full = lambda_return(
      last_full, term_full, rew, val_full, val_full, 1.0, 0.95)
  r_last_full = float(ret_full[0, -1])
  # Same reasoning: one real (con=0.997) transition into the terminal bootstrap.
  print('full-resolution var-K final-step return (expect ~29.91 = 0.997*30):',
        r_last_full)

  # --- (c) Block-pooled var-K path, post both fixes (n_blocks=n_sw AND the
  # empty-segment continuation identity). The real equivalence claim: decisions
  # 0 and 1 (the two FULLY-REAL, non-degenerate decisions -- both fixed-K and
  # block-pooled model these identically) must match fixed-K's own R_0/R_1
  # exactly, since this T=17/k=8 fixture is exactly what fixed-K computes too.
  extr, _, cont, _ = variable_block_director_tensors(rew, con, rew, sw, T)
  val_blockpooled = jnp.array([[10.0, 20.0, 30.0] + [30.0] * (T - 3)], jnp.float32)
  last_bp = jnp.zeros_like(cont)
  term_bp = 1 - cont
  ret_bp = lambda_return(last_bp, term_bp, extr, val_blockpooled, val_blockpooled, 1.0, 0.95)
  print('block-pooled var-K ret[0:3] =', np.asarray(ret_bp[0, :3]))
  r0_bp, r1_bp = float(ret_bp[0, 0]), float(ret_bp[0, 1])
  r2_bp = float(ret_bp[0, 2])

  assert abs(r2_fixedk - 30.0 * 0.9762503) < 1e-3, (
      f'fixed-K last-decision return={r2_fixedk}, expected ~29.29 (a real, '
      'expected decay over its own 8-step hold) -- sanity check on the test '
      'fixture itself, not on any code under test.')
  assert abs(r_last_full - 30.0 * 0.997) < 1e-3, (
      f'full-resolution var-K final-step return={r_last_full}, expected ~29.91 '
      '(a real, expected decay over one step) -- sanity check on the fixture.')
  assert abs(r0_bp - float(ret_fixedk[0, 0])) < 1e-3, (
      f'block-pooled R_0={r0_bp} != fixed-K R_0={float(ret_fixedk[0, 0])}: '
      'the two credit-assignment paths disagree on a fully-real decision.')
  assert abs(r1_bp - r2_fixedk) < 1e-3, (
      f'block-pooled R_1={r1_bp} != fixed-K R_1={r2_fixedk}: '
      'the two credit-assignment paths disagree on a fully-real decision.')
  assert abs(r2_bp - 30.0) < 1e-4, (
      f'block-pooled R_2 (the degenerate, zero-trailing-steps 3rd decision '
      f'fixed-K never has to model) = {r2_bp}, expected exactly 30.0.')
  print('CONFIRMED: block-pooled var-K (post-fix) matches fixed-K exactly on '
        'every fully-real decision, and correctly bootstraps the one decision '
        '(the degenerate trailing one) that fixed-K structurally never creates.')


def test_switch_mask_from_skills_ignores_duration_head_under_fixed_duration():
  """e334/e335 (BIG-scale 'equivalence check', EXPERIMENTS.md §2 2026-07-27) pin
  ``goal_duration_fixed=8`` so block-pooled var-K should switch on exactly the same
  schedule as fixed-K Director (``manager_sample_freq=8``). ``_duration_steps``
  (agent.py) is supposed to make this true by construction -- when
  ``goal_duration_fixed > 0`` it ignores the manager's own duration-head output
  entirely and returns the constant. This test checks that claim directly at the
  ``_switch_mask_from_skills`` level (the function the REPLAY-side block-rew path
  calls to reconstruct decision boundaries, agent.py ~3228): feed it two
  DIFFERENT, random duration-class arrays (standing in for whatever a real,
  still-training duration head might emit) and confirm the resulting switch mask
  is identical to the static period-8 mask in both cases -- i.e. the duration
  head's content has provably zero effect on switch timing under a fixed
  duration, so this specific mechanism cannot be the source of e334/e335's gap
  against the true Director baselines (e124 hopper, e191 cheetah)."""
  import jax
  from dreamerv3.agent import Agent

  B, T, K = 3, 40, 8
  expected = jnp.broadcast_to(
      jnp.equal(jnp.arange(T) % K, 0).astype(f32), (B, T))

  agent = object.__new__(Agent)
  agent.variable_goal_length = True
  agent.goal_duration_min = 1
  agent.goal_duration_fixed = K
  agent.manager_sample_freq = K

  key1, key2 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
  dur1 = jax.random.randint(key1, (B, T), 0, 16)
  dur2 = jax.random.randint(key2, (B, T), 0, 16)  # genuinely different content

  sw1 = agent._switch_mask_from_skills(
      {'skill': jnp.zeros((B, T, 8, 8)), 'duration': dur1})
  sw2 = agent._switch_mask_from_skills(
      {'skill': jnp.zeros((B, T, 8, 8)), 'duration': dur2})

  np.testing.assert_array_equal(sw1, expected)
  np.testing.assert_array_equal(sw2, expected)
  np.testing.assert_array_equal(sw1, sw2)


def test_variable_block_director_expl_and_switch_outputs_match_fixed_k():
  """The existing equivalence test (``test_variable_block_director_matches_
  fixed_k``) only asserts on the ``extr``/``cont`` outputs of
  ``variable_block_director_tensors``, and it passes the SAME array as both
  ``rew`` and ``expl`` -- a bug isolated to the exploration-reward branch (a
  real, separately-computed quantity feeding ``mgr_expl_weight`` mixing,
  agent.py:3951/3960) would be invisible to it. It also discards the returned
  ``mgr_switch`` with ``_``. Close both gaps."""
  B, T = 2, 17
  k = 8
  rew = jnp.arange(T, dtype=f32)[None, :].repeat(B, 0)
  expl = ((T - 1) - jnp.arange(T, dtype=f32))[None, :].repeat(B, 0)  # different pattern
  con = jnp.full((B, T), 0.9, f32)
  sw = _fixed_switch_mask(B, T, k)

  extr, expl_out, cont, switch = variable_block_director_tensors(
      rew, con, expl, sw, T, agg_mode='mean')

  fix_extr = imag_reward_pad(aggregate_mgr_extr_rew(rew, con, k, without_zeros=True))
  fix_expl = imag_reward_pad(aggregate_mgr_extr_rew(expl, con, k, without_zeros=True))
  fix_cont = aggregate_mgr_cont(con, k, without_zeros=True)

  np.testing.assert_allclose(
      extr[:, :fix_extr.shape[1]], fix_extr, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(
      expl_out[:, :fix_expl.shape[1]], fix_expl, rtol=1e-5, atol=1e-5,
      err_msg='expl output diverges from the fixed-K reference when expl != rew')
  np.testing.assert_allclose(
      cont[:, :fix_cont.shape[1]], fix_cont, rtol=1e-5, atol=1e-5)

  # ``mgr_switch`` marks the decisions that own realized transitions, which is
  # ``switch_valid_mask`` MINUS any switch landing on the sequence's final
  # timestep: rewards are indexed ``rew[:, 1:]``, so such a switch pools an
  # empty segment and is only a bootstrap anchor -- the column fixed-K drops
  # with ``[:, :-1]``. Here switches sit at t=0/8/16 of a length-17 rollout, so
  # 2 of the 3 are trainable decisions (fixed 2026-07-27; see
  # ``test_block_pooled_valid_decision_count_excludes_empty_trailing_segment``
  # and the "trains the manager 8x too weakly" entry in EXPERIMENTS.md §7).
  expected_switch = switch_valid_mask(sw[:, :-1], T)
  np.testing.assert_array_equal(switch, expected_switch)
  assert float(switch[0].sum()) == 2.0


def test_variable_block_director_batched_heterogeneous_phase_matches_per_row():
  """Every existing test drives the whole batch through one shared, lockstep
  switch pattern (``_fixed_switch_mask`` broadcasts the SAME mask to every row).
  Real training batches don't look like this: different rows are sampled from
  different points in different episodes, so they sit at different phases of
  their hold when a window starts (see EXPERIMENTS.md §2 2026-07-27 discussion
  of ``_manager_skill_step``'s per-row countdown carry). Check that batching two
  rows with genuinely DIFFERENT switch patterns gives exactly the same per-row
  result as running each row through the function alone -- i.e. no row's
  pooling leaks into another's via a batch-shared quantity (``n_blocks``,
  ``max_seg``, etc. are all meant to be per-row already, but nothing currently
  exercises that with a non-uniform batch)."""
  T = 17
  rew_a = jnp.arange(T, dtype=f32)
  rew_b = 2.0 * jnp.arange(T, dtype=f32) + 1.0
  con_a = jnp.full((T,), 0.9, f32)
  con_b = jnp.full((T,), 0.8, f32)
  sw_a = jnp.equal(jnp.arange(T) % 8, 0).astype(f32)  # switches at 0, 8, 16
  sw_b = jnp.array(  # a deliberately different phase: switches at 0, 5, 13
      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], f32)

  rew = jnp.stack([rew_a, rew_b])
  con = jnp.stack([con_a, con_b])
  sw = jnp.stack([sw_a, sw_b])

  batched = variable_block_director_tensors(rew, con, rew, sw, T, agg_mode='mean')
  alone_a = variable_block_director_tensors(
      rew_a[None], con_a[None], rew_a[None], sw_a[None], T, agg_mode='mean')
  alone_b = variable_block_director_tensors(
      rew_b[None], con_b[None], rew_b[None], sw_b[None], T, agg_mode='mean')

  names = ('extr', 'expl', 'cont', 'switch')
  for name, b, a in zip(names, batched, alone_a):
    np.testing.assert_allclose(
        b[0], a[0], rtol=1e-5, atol=1e-5,
        err_msg=f'row 0 ({name}) differs between batched and solo call')
  for name, b, a in zip(names, batched, alone_b):
    np.testing.assert_allclose(
        b[1], a[0], rtol=1e-5, atol=1e-5,
        err_msg=f'row 1 ({name}) differs between batched and solo call')


def test_duration_head_reinforce_is_live_under_fixed_duration():
  """Stage 4 (EXPERIMENTS.md §2 2026-07-27 / paper 'block-pooled var-K credit
  path' discussion of e334/e335). Test A
  (``test_switch_mask_from_skills_ignores_duration_head_under_fixed_duration``)
  proved the duration head's SAMPLE cannot move switch TIMING once
  ``goal_duration_fixed`` overrides it. That is not the same claim as "the
  duration head receives no gradient" -- ``imag_loss_mgr``'s REINFORCE term
  (agent.py ~3980-3981) builds ``mgr_logpi`` as an UNCONDITIONAL sum over every
  key in ``manager_policy`` (``sum(head_logp_time(v, skill_events[k]) for k, v
  in manager_policy.items())``), then multiplies the WHOLE sum by one shared
  per-decision advantage (agent.py ~4046, ``mgr_reinforce = mgr_logpi *
  sg(mgr_adv_normed)``) -- there is no branch anywhere in that function that
  drops 'duration' from the sum when ``goal_duration_fixed>0``. So the duration
  head is still being pushed by REINFORCE, using an advantage that in the real
  system is caused entirely by the skill/code choice (duration has zero causal
  effect on the trajectory once fixed) -- a nuisance/nonstationary-target
  training signal true Director never has to carry at all (it has no duration
  head).

  This reproduces that exact wiring -- real ``outs.OneHot`` heads,
  ``align_skill_events``/``head_logp_time`` from agent.py itself (not
  reimplemented), only the "shared trunk feeding two heads" part is a minimal
  stand-in for the real network -- and confirms the duration head's own
  parameters, and the shared trunk, actually receive nonzero gradient from this
  channel. This is a measurement, not a pass/fail correctness check: the
  log-prob-sum-times-shared-advantage pattern is standard/correct for factorized
  action heads in general (DreamerV3 uses it elsewhere too); what's specific to
  this path is that one of the factors is provably causally inert, yet still
  shares the advantage and the trunk with the one that isn't."""
  B, Tm, n_skill, n_dur = 4, 6, 8, 8
  key = jax.random.PRNGKey(0)
  k_trunk, k_wskill, k_wdur, k_skill_ev, k_dur_ev = jax.random.split(key, 5)

  trunk = jax.random.normal(k_trunk, (5,))
  w_skill = jax.random.normal(k_wskill, (5, n_skill)) * 0.1
  w_dur = jax.random.normal(k_wdur, (5, n_dur)) * 0.1

  # A fixed, nonzero per-decision advantage -- stands in for `mgr_adv_normed`,
  # which in the real system is driven by the extrinsic/exploration reward the
  # CHOSEN GOAL earned, never by which duration class happened to be sampled.
  adv = jnp.linspace(-1.0, 1.0, B * Tm).reshape(B, Tm)

  skill_idx = jax.random.randint(k_skill_ev, (B, Tm), 0, n_skill)
  dur_idx = jax.random.randint(k_dur_ev, (B, Tm), 0, n_dur)
  skill_event = jax.nn.one_hot(skill_idx, n_skill)
  dur_event = jax.nn.one_hot(dur_idx, n_dur)

  def loss(trunk, w_skill, w_dur, include_duration):
    logits_skill = jnp.broadcast_to(trunk @ w_skill, (B, Tm, n_skill))
    logits_dur = jnp.broadcast_to(trunk @ w_dur, (B, Tm, n_dur))
    manager_policy = {'skill': outs.OneHot(logits_skill)}
    skills = {'skill': skill_event}
    if include_duration:
      manager_policy['duration'] = outs.OneHot(logits_dur)
      skills['duration'] = dur_event
    # Real agent.py helpers, not reimplemented (agent.py:3964, 3980-3981).
    skill_events = align_skill_events(skills, manager_policy)
    mgr_logpi = sum(
        head_logp_time(v, skill_events[k]) for k, v in manager_policy.items())
    # Real construction (agent.py:4046): reinforce = logpi * sg(advantage).
    mgr_reinforce = mgr_logpi * jax.lax.stop_gradient(adv[:, :-1])
    return -mgr_reinforce.mean()

  grad_fn = jax.grad(loss, argnums=(0, 1, 2))
  g_trunk_with, _, g_wdur_with = grad_fn(trunk, w_skill, w_dur, True)
  g_trunk_without, _, g_wdur_without = grad_fn(trunk, w_skill, w_dur, False)

  # 1. The duration head's own params get real, nonzero REINFORCE gradient from
  # an advantage that has nothing causally to do with duration once
  # goal_duration_fixed is set.
  assert float(jnp.abs(g_wdur_with).max()) > 1e-6, (
      'expected the duration head to receive nonzero REINFORCE gradient from '
      'the shared advantage in this reproduction of agent.py:3980-4046')
  # Confirms the "without" branch is a valid contrast (no duration term -> no
  # duration-head gradient at all).
  np.testing.assert_allclose(g_wdur_without, jnp.zeros_like(g_wdur_without))

  # 2. That gradient reaches the SHARED TRUNK too (not quarantined to the
  # duration head's own private output layer) -- i.e. it can push the same
  # representations the skill/code head reads from.
  trunk_diff = float(jnp.abs(g_trunk_with - g_trunk_without).max())
  assert trunk_diff > 1e-6, (
      'the duration branch contributes no gradient to the shared trunk in '
      'this reproduction -- if the real network fully separates per-head '
      'trunks (unlike this minimal stand-in), Stage 4 would not apply there')
  print(f'duration-head own-param grad magnitude: '
        f'{float(jnp.abs(g_wdur_with).max()):.4f}; '
        f'shared-trunk grad shift from including duration: {trunk_diff:.4f}')


def test_downsample_at_switch_mask_matches_downsample_manager_states_exact_multiple():
  """Stage 5 (replay-side reconstruction). The block-pooled var-K replay path
  selects manager decision states via ``downsample_at_switch_mask(feat,
  repl_switch)`` (agent.py:3229); fixed-K's OWN replay path (agent.py:3247-3261)
  computes an inline static index list with the SAME formula as the
  module-level ``downsample_manager_states`` helper, rather than calling it --
  itself a code-duplication drift risk, not exercised by any existing test.
  Confirm the two independently-written index-selection paths agree on the
  case that matters for e334/e335's own IMAGINATION rollout (H=16, k=8: an
  exact multiple, no trailing remainder)."""
  for T, k in [(17, 8), (9, 8)]:  # H+1 for imag_length in {16, 8}, both exact
    B = 2
    feat = {'deter': jnp.arange(B * T, dtype=f32).reshape(B, T)}
    sw = _fixed_switch_mask(B, T, k)
    via_mask = downsample_at_switch_mask(feat, sw)['deter']
    via_static = downsample_manager_states(feat, k)['deter']
    n = via_static.shape[1]
    np.testing.assert_allclose(
        via_mask[:, :n], via_static, rtol=1e-6, atol=1e-6,
        err_msg=f'T={T}, k={k}: replay-side downsample disagrees with the '
        'fixed-K static index list on the real decision states')


def test_downsample_at_switch_mask_omits_fixed_k_trailing_state_on_remainder():
  """Companion to the test above, but for the REPLAY window's actual shape:
  ``batch_length=64`` almost never divides evenly by an 8-step hold, so this
  (not the clean H=16 imagination case) is what real training exercises every
  step. This is a DOCUMENTING test for a genuine, newly-found asymmetry, not a
  pass/fail correctness check -- there is no existing test (before this
  session) that exercises the remainder branch of either function at all.

  ``downsample_manager_states``'s docstring says it selects "index 0,
  boundaries of blocks, AND THE FINAL STATE" -- it unconditionally appends the
  window's true last index even when that index isn't a real switch boundary
  (agent.py:416-432, the ``idx.append(int(Tm))`` in both the ``rem>0`` and
  ``rem==0`` branches). ``downsample_at_switch_mask`` has no such provision: it
  only ever places values at real switch positions and forward-fills the rest
  (agent.py:359-383) -- so on a remainder window it returns ONE FEWER distinct
  state than fixed-K's replay path does, missing a value prediction at the
  window's true end. Fixed-K's replay-side critic target gets an extra,
  non-switch anchor state for free; block-pooled var-K's does not, and instead
  falls back on a broadcast bootstrap value from the imagination side
  (agent.py:3299-3300, 3307-3310: ``boot_extr_down = broadcast(boot_extr[:,
  -1:], ...)``) for its last decision -- a coarser signal than a real value
  prediction at the actual final replayed state. This is a live, not yet
  resolved, candidate contributor to e334/e335's gap specific to the REPLAY
  value loss (which runs every training step, unlike the clean imagination
  rollout) -- flagged here for the next round of investigation, not fixed."""
  B, k = 2, 8
  T = 20  # switches at 0, 8, 16 (n_sw=3); Tm=19, rem=3 -> a genuine remainder
  feat = {'deter': jnp.arange(B * T, dtype=f32).reshape(B, T)}
  sw = _fixed_switch_mask(B, T, k)  # switches at 0, 8, 16 (not at 19, the true end)
  via_mask = downsample_at_switch_mask(feat, sw)['deter']
  via_static = downsample_manager_states(feat, k)['deter']

  n_sw = int(sw[0].sum())
  n_static = via_static.shape[1]
  # The asymmetry itself: fixed-K's static list has one MORE distinct state
  # than there are real switches (the appended non-switch final index).
  assert n_static == n_sw + 1, (
      f'expected downsample_manager_states to append exactly one extra '
      f'(non-switch) trailing state on a remainder window: n_switches={n_sw}, '
      f'n_static={n_static}')
  # That extra state is genuinely the window's TRUE final value, not a repeat
  # of the last real switch's value (which is what downsample_at_switch_mask's
  # forward-fill would give instead).
  last_switch_value = via_mask[0, n_sw - 1]
  true_final_value = feat['deter'][0, -1]
  static_final_value = via_static[0, -1]
  np.testing.assert_allclose(static_final_value, true_final_value)
  assert abs(float(last_switch_value - true_final_value)) > 1e-6, (
      'fixture is degenerate (last switch value happens to equal the true '
      'final value) -- pick different T/k so the asymmetry is visible')
  print(f'CONFIRMED asymmetry: fixed-K replay downsample includes the true '
        f'final state ({float(true_final_value)}) as an extra anchor; '
        f'block-pooled var-K\'s downsample_at_switch_mask has no equivalent '
        f'and would forward-fill the last switch\'s value '
        f'({float(last_switch_value)}) into that slot instead.')


# ---------------------------------------------------------------------------
# Fix verification (2026-07-27): both issues found above are now fixed in
# agent.py. These tests confirm the fixes using the REAL fix functions
# (``manager_reinforce_policy``, ``patch_trailing_replay_state``), not a
# reproduction, so they'd fail if the fix were reverted or broken later.
# ---------------------------------------------------------------------------


def test_fix_manager_reinforce_policy_drops_duration_only_when_fixed():
  """Fix for Issue 1 (duration-head REINFORCE noise). Before the fix,
  ``imag_loss_mgr`` summed EVERY manager head's log-prob into the REINFORCE
  term regardless of ``goal_duration_fixed``
  (``test_duration_head_reinforce_is_live_under_fixed_duration`` demonstrated
  this gives the duration head a nonzero, causally-meaningless gradient).
  ``manager_reinforce_policy`` is the real function ``imag_loss_mgr`` now
  calls to build the REINFORCE head set. Confirm: (a) with
  ``duration_fixed=False`` it's a no-op (unchanged behavior for every
  existing non-fixed-duration run); (b) with ``duration_fixed=True`` it drops
  exactly 'duration' and nothing else."""
  policy = {'skill': object(), 'duration': object(), 'mask': object()}

  unfixed = manager_reinforce_policy(policy, duration_fixed=False)
  assert unfixed is policy, (
      'duration_fixed=False must be a true no-op (same object), matching '
      'every pre-fix call site exactly -- not just an equal-content copy')

  fixed = manager_reinforce_policy(policy, duration_fixed=True)
  assert set(fixed.keys()) == {'skill', 'mask'}, (
      f'expected duration_fixed=True to drop only "duration", got keys '
      f'{set(fixed.keys())}')
  assert fixed['skill'] is policy['skill'] and fixed['mask'] is policy['mask']


def test_fix_duration_head_reinforce_gradient_is_exactly_zero_when_fixed():
  """End-to-end confirmation of the Issue 1 fix at the gradient level: rerun
  the exact reproduction from
  ``test_duration_head_reinforce_is_live_under_fixed_duration`` (same shared
  trunk, same two OneHot heads, same shared advantage), but build
  ``manager_policy``/``skill_events`` through the REAL
  ``manager_reinforce_policy`` fix function instead of a hand-toggled
  ``include_duration`` flag. With the fix applied, the duration head's own
  parameters must receive EXACTLY zero gradient (not just "smaller") from the
  manager's REINFORCE loss, while the skill head's gradient (and the shared
  trunk's gradient restricted to what the skill branch alone would produce)
  is completely unaffected."""
  B, Tm, n_skill, n_dur = 4, 6, 8, 8
  key = jax.random.PRNGKey(0)
  k_trunk, k_wskill, k_wdur, k_skill_ev, k_dur_ev = jax.random.split(key, 5)

  trunk = jax.random.normal(k_trunk, (5,))
  w_skill = jax.random.normal(k_wskill, (5, n_skill)) * 0.1
  w_dur = jax.random.normal(k_wdur, (5, n_dur)) * 0.1
  adv = jnp.linspace(-1.0, 1.0, B * Tm).reshape(B, Tm)
  skill_idx = jax.random.randint(k_skill_ev, (B, Tm), 0, n_skill)
  dur_idx = jax.random.randint(k_dur_ev, (B, Tm), 0, n_dur)
  skill_event = jax.nn.one_hot(skill_idx, n_skill)
  dur_event = jax.nn.one_hot(dur_idx, n_dur)

  def loss(trunk, w_skill, w_dur, duration_fixed):
    logits_skill = jnp.broadcast_to(trunk @ w_skill, (B, Tm, n_skill))
    logits_dur = jnp.broadcast_to(trunk @ w_dur, (B, Tm, n_dur))
    manager_policy = {
        'skill': outs.OneHot(logits_skill),
        'duration': outs.OneHot(logits_dur),
    }
    skills = {'skill': skill_event, 'duration': dur_event}
    # The REAL fix function, not a reimplementation.
    reinforce_policy = manager_reinforce_policy(manager_policy, duration_fixed)
    skill_events = align_skill_events(skills, reinforce_policy)
    mgr_logpi = sum(
        head_logp_time(v, skill_events[k]) for k, v in reinforce_policy.items())
    mgr_reinforce = mgr_logpi * jax.lax.stop_gradient(adv[:, :-1])
    return -mgr_reinforce.mean()

  grad_fn = jax.grad(loss, argnums=(0, 1, 2))
  g_trunk_fixed, g_wskill_fixed, g_wdur_fixed = grad_fn(
      trunk, w_skill, w_dur, True)
  g_trunk_unfixed, g_wskill_unfixed, g_wdur_unfixed = grad_fn(
      trunk, w_skill, w_dur, False)

  # The fix: duration's own gradient is now EXACTLY zero (not just smaller).
  np.testing.assert_array_equal(g_wdur_fixed, jnp.zeros_like(g_wdur_fixed))
  # Sanity: without the fix, it's the same nonzero signal already measured in
  # test_duration_head_reinforce_is_live_under_fixed_duration.
  assert float(jnp.abs(g_wdur_unfixed).max()) > 1e-6

  # The skill head's own gradient, and the shared trunk's gradient, are
  # UNCHANGED by the fix relative to what the skill-only computation gives
  # (i.e. dropping 'duration' doesn't perturb the surviving head's own math).
  def skill_only_loss(trunk, w_skill):
    logits_skill = jnp.broadcast_to(trunk @ w_skill, (B, Tm, n_skill))
    manager_policy = {'skill': outs.OneHot(logits_skill)}
    skill_events = align_skill_events({'skill': skill_event}, manager_policy)
    mgr_logpi = head_logp_time(manager_policy['skill'], skill_events['skill'])
    mgr_reinforce = mgr_logpi * jax.lax.stop_gradient(adv[:, :-1])
    return -mgr_reinforce.mean()

  g_trunk_ref, g_wskill_ref = jax.grad(skill_only_loss, argnums=(0, 1))(
      trunk, w_skill)
  np.testing.assert_allclose(g_wskill_fixed, g_wskill_ref, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(g_trunk_fixed, g_trunk_ref, rtol=1e-6, atol=1e-6)
  print(f'CONFIRMED fix: duration-head gradient exactly {float(jnp.abs(g_wdur_fixed).max())} '
        f'when duration_fixed=True (was {float(jnp.abs(g_wdur_unfixed).max()):.4f} before); '
        f'skill/trunk gradients unaffected by the fix.')


def test_fix_patch_trailing_replay_state_uses_true_final_value():
  """Fix for Issue 2 (replay downsample trailing-state asymmetry). Same
  fixture as ``test_downsample_at_switch_mask_omits_fixed_k_trailing_state_on_
  remainder`` (T=20, k=8, switches at 0/8/16, true end at 19) but now applies
  the REAL fix function, ``patch_trailing_replay_state``, and confirms the
  padding slot now holds the window's true final value (19.0) instead of the
  stale forward-filled repeat (16.0) -- i.e. it now matches what fixed-K's
  ``downsample_manager_states`` would have given for the same window."""
  B, k = 2, 8
  T = 20  # switches at 0, 8, 16 (n_sw=3); Tm=19, rem=3 -> a genuine remainder
  feat = {'deter': jnp.arange(B * T, dtype=f32).reshape(B, T)}
  sw = _fixed_switch_mask(B, T, k)

  before = downsample_at_switch_mask(feat, sw)['deter']
  after = patch_trailing_replay_state(
      downsample_at_switch_mask(feat, sw), feat, sw)['deter']

  n_sw = int(sw[0].sum())
  true_final_value = feat['deter'][0, -1]

  # Before the fix: stale repeat of the last real switch's value.
  np.testing.assert_allclose(before[0, n_sw], before[0, n_sw - 1])
  # After the fix: the window's true final value.
  np.testing.assert_allclose(after[0, n_sw], true_final_value)
  # Everything BEFORE the patched slot (the real switch states themselves) is
  # untouched by the fix.
  np.testing.assert_allclose(after[0, :n_sw], before[0, :n_sw])
  print(f'CONFIRMED fix: padding slot now holds the true final value '
        f'({float(true_final_value)}) instead of the stale repeat '
        f'({float(before[0, n_sw])}).')


def test_fix_patch_trailing_replay_state_is_noop_on_exact_multiple():
  """The fix must not change anything for e334/e335's OWN actual imagination-
  rollout shape (H=16, k=8: an exact multiple, no trailing remainder) -- the
  patch should be a true no-op whenever the last switch already lands on the
  window's final timestep, matching fixed-K's own convention there too
  (already confirmed clean by
  ``test_downsample_at_switch_mask_matches_downsample_manager_states_exact_
  multiple``, before any fix existed)."""
  for T, k in [(17, 8), (9, 8)]:
    B = 2
    feat = {'deter': jnp.arange(B * T, dtype=f32).reshape(B, T)}
    sw = _fixed_switch_mask(B, T, k)
    before = downsample_at_switch_mask(feat, sw)['deter']
    after = patch_trailing_replay_state(
        downsample_at_switch_mask(feat, sw), feat, sw)['deter']
    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Further equivalence checks (2026-07-27, requested follow-up): additional
# pipeline surface not yet exercised by the tests above -- per-row-independent
# behaviour of the new fix under a genuinely heterogeneous batch, correct
# handling of a real episode boundary landing in the patched slot, and
# composition with ``relabel_truncated_last_duration`` (both operate on data
# derived from the same ``switch_mask`` at the real replay call site,
# agent.py ~3225-3242).
# ---------------------------------------------------------------------------


def test_patch_trailing_replay_state_heterogeneous_batch_per_row_correctness():
  """Every ``patch_trailing_replay_state`` test so far uses a batch where both
  rows share the SAME switch pattern (``_fixed_switch_mask`` broadcasts one
  mask to every row), so a bug that mixed up rows (e.g. patching row 1 with
  row 0's true-final value or vice versa) would be invisible. Real replay
  batches are exactly this heterogeneous: different rows are sampled from
  different episodes/offsets, so they have different ``n_sw`` and different
  true final states. Construct two rows with genuinely different remainder
  amounts and distinct data, and confirm each row is patched with ITS OWN
  true final value, matching what patching each row alone would give (same
  cross-check style as
  ``test_variable_block_director_batched_heterogeneous_phase_matches_per_row``)."""
  T, k = 20, 8
  feat_a = {'deter': jnp.arange(T, dtype=f32)}                    # 0..19
  feat_b = {'deter': 100.0 + jnp.arange(T, dtype=f32)}             # 100..119
  sw_a = jnp.equal(jnp.arange(T) % k, 0).astype(f32)               # switches 0,8,16 (n_sw=3)
  sw_b = jnp.array(                                                # switches 0,5 only (n_sw=2)
      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], f32)

  feat = {'deter': jnp.stack([feat_a['deter'], feat_b['deter']])}
  sw = jnp.stack([sw_a, sw_b])

  batched = patch_trailing_replay_state(
      downsample_at_switch_mask(feat, sw), feat, sw)['deter']
  alone_a = patch_trailing_replay_state(
      downsample_at_switch_mask({'deter': feat_a['deter'][None]}, sw_a[None]),
      {'deter': feat_a['deter'][None]}, sw_a[None])['deter']
  alone_b = patch_trailing_replay_state(
      downsample_at_switch_mask({'deter': feat_b['deter'][None]}, sw_b[None]),
      {'deter': feat_b['deter'][None]}, sw_b[None])['deter']

  np.testing.assert_allclose(batched[0], alone_a[0], rtol=1e-6, atol=1e-6,
                              err_msg='row 0 patched differently in a batch than alone')
  np.testing.assert_allclose(batched[1], alone_b[0], rtol=1e-6, atol=1e-6,
                              err_msg='row 1 patched differently in a batch than alone')
  # And each row's patched slot really is ITS OWN true final value (19 vs 119),
  # not the other row's -- the failure mode this test targets specifically.
  np.testing.assert_allclose(batched[0, 3], 19.0)   # row A: n_sw=3, true final=19
  np.testing.assert_allclose(batched[1, 2], 119.0)  # row B: n_sw=2, true final=119


def test_patch_trailing_replay_state_preserves_true_terminal_flag():
  """The fix is applied to BOTH ``feat_down`` (state/value target) and
  ``last_down`` (episode-boundary flag) at the real replay call site
  (agent.py ~3237-3239). This checks the ``last_down`` side specifically: if
  the window's TRUE final step is a genuine episode end (``is_last=True``)
  but the last real switch itself was not, the pre-fix forward-fill would
  have silently carried the switch's OWN (False) flag into the padding slot,
  hiding a real termination from ``weight_down = ~last_down`` downstream.
  Confirm the patch correctly propagates the true terminal flag instead."""
  T, k = 20, 8
  last = jnp.zeros((1, T), f32).at[0, -1].set(1.0)  # only the TRUE final step ends the episode
  sw = _fixed_switch_mask(1, T, k)                  # switches at 0, 8, 16 (none of which end it)

  before = downsample_at_switch_mask({'last': last}, sw)['last']
  after = patch_trailing_replay_state(
      {'last': downsample_at_switch_mask({'last': last}, sw)['last']},
      {'last': last}, sw)['last']

  n_sw = int(sw[0].sum())
  # Pre-fix: the padding slot forward-fills the last SWITCH's flag (False,
  # since none of 0/8/16 end the episode) -- the true termination is lost.
  np.testing.assert_allclose(before[0, n_sw], 0.0)
  # Post-fix: the padding slot now correctly shows the episode really ended.
  np.testing.assert_allclose(after[0, n_sw], 1.0)


def test_patch_trailing_replay_state_composes_with_relabel_truncated_duration():
  """Both ``patch_trailing_replay_state`` (new fix) and
  ``relabel_truncated_last_duration`` (existing, 2026-07-24) read from the
  SAME ``switch_mask`` at the real replay call site, but operate on entirely
  different fields (state/terminal-flag vs. the manager's sampled duration
  class) -- confirm they don't interfere when used together on the same
  fixture, and that relabeling still correctly identifies the truncated
  amount independent of the state patch."""
  T, k, dur_min, dur_max = 20, 8, 1, 16
  feat = {'deter': jnp.arange(T, dtype=f32)[None]}
  sw = _fixed_switch_mask(1, T, k)  # switches at 0, 8, 16; n_sw=3, last hold runs 16..19 (4 steps)

  # Sanity: patch_trailing_replay_state still behaves normally alongside relabel.
  patched = patch_trailing_replay_state(
      downsample_at_switch_mask(feat, sw), feat, sw)
  np.testing.assert_allclose(patched['deter'][0, 3], 19.0)

  # duration class sampled 8 (index 7 given dur_min=1) for all 3 decisions --
  # the last one only ran 4 real steps (16..19) before the window ended.
  mgr_skills_eff = {'duration': jnp.full((1, 3), 7, jnp.int32)}
  relabeled = relabel_truncated_last_duration(mgr_skills_eff, sw, dur_min, dur_max)
  # First two decisions ran their full sampled 8-step hold -- untouched.
  np.testing.assert_array_equal(relabeled['duration'][0, :2], [7, 7])
  # Last decision is relabeled to its REALIZED length (4 steps -> class 3).
  np.testing.assert_array_equal(relabeled['duration'][0, 2], 3)
  print('CONFIRMED: state patch and duration relabeling compose cleanly on '
        'the same fixture -- relabeled last-decision duration = 4 steps '
        '(class 3), true final state = 19.0, neither affects the other.')
