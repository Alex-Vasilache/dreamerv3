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
