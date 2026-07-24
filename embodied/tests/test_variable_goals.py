import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.agent import (
    aggregate_mgr_cont,
    aggregate_mgr_cont_variable,
    aggregate_mgr_extr_rew,
    aggregate_mgr_extr_rew_variable,
    downsample_at_switch_mask,
    imag_reward_pad,
    masked_cumprod,
    switch_valid_mask,
    variable_block_director_tensors,
    variable_segment_ids,
)

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
