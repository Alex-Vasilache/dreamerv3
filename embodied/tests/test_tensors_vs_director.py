"""Fixed-K credit-assignment tensors against TF Director's `abstract_traj`.

`test_block_pooled_equivalence.py` and `test_block_pooled_stepwise.py` already
check that the variable-K path reproduces the fixed-K path. Nothing checked that
the fixed-K path reproduces *Director*. If fixed-K were wrong, both would be
wrong together and every one of those tests would still pass. This file closes
that gap by transcribing `hierarchy.Hierarchy.abstract_traj` into numpy and
testing against it.

Reference: `code/director/embodied/agents/director/hierarchy.py:431`.

    k = self.config.train_skill_duration
    reshape = lambda x: x.reshape([x.shape[0] // k, k] + x.shape[1:])
    weights = tf.math.cumprod(reshape(traj['cont'][:-1]), 1)
    for key, value in list(traj.items()):
      if 'reward' in key:
        traj[key] = (reshape(value) * weights).mean(1)
      elif key == 'cont':
        traj[key] = tf.concat([value[:1], reshape(value[1:]).prod(1)], 0)
      else:
        traj[key] = tf.concat([reshape(value[:-1])[:, 0], value[-1:]], 0)

Director asserts `len(action) % train_skill_duration == 1`, so `T - 1` is always
an exact multiple of `k` and there is never a remainder. Our implementation adds
a remainder branch; it is tested separately for self-consistency, not against
Director, because Director has no such case.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.hrl.tensors import (
    aggregate_mgr_cont,
    aggregate_mgr_extr_rew,
    downsample_manager_states,
    goal_reward_cosine_max,
    imag_reward_pad,
    masked_cumprod,
    skill_switch,
)

# T - 1 must be divisible by k for the Director reference to apply.
EXACT = [(9, 4), (17, 8), (17, 4), (13, 2), (25, 8)]


def director_abstract_reward(rew, cont, k):
  """`(reshape(value) * weights).mean(1)` for one batch row.

  Director's reward array has length T-1 and its cont array length T; `weights`
  comes from `cont[:-1]`.
  """
  T = len(cont)
  n = (T - 1) // k
  w = np.cumprod(cont[:-1].reshape(n, k), axis=1)
  return (rew.reshape(n, k) * w).mean(1)


def director_abstract_cont(cont, k):
  """`concat([value[:1], reshape(value[1:]).prod(1)], 0)` for one batch row."""
  T = len(cont)
  n = (T - 1) // k
  return np.concatenate([cont[:1], cont[1:].reshape(n, k).prod(1)])


def director_abstract_state(value, k):
  """`concat([reshape(value[:-1])[:, 0], value[-1:]], 0)` for one batch row."""
  T = len(value)
  n = (T - 1) // k
  return np.concatenate([value[:-1].reshape(n, k, *value.shape[1:])[:, 0],
                         value[-1:]])


def sample(T, seed, cont_lo=0.9):
  rng = np.random.default_rng(seed)
  rew = rng.normal(size=(1, T)).astype(np.float32)
  cont = rng.uniform(cont_lo, 1.0, size=(1, T)).astype(np.float32)
  return rew, cont


class TestBlockRewardMatchesDirector:

  @pytest.mark.parametrize('T,k', EXACT)
  def test_pooled_reward_equals_abstract_traj(self, T, k):
    rew, cont = sample(T, seed=T * 100 + k)
    got = np.asarray(aggregate_mgr_extr_rew(
        jnp.asarray(rew), jnp.asarray(cont), k, without_zeros=True))
    # Ours pairs rew[:, 1:] with cont[:, :-1]; Director's `reward` array is
    # already the length-(T-1) transition rewards, i.e. our rew[1:].
    want = director_abstract_reward(rew[0, 1:], cont[0], k)
    np.testing.assert_allclose(got[0], want, rtol=1e-5, atol=1e-6)

  @pytest.mark.parametrize('T,k', EXACT)
  def test_cumprod_restarts_inside_each_block(self, T, k):
    """Director's `cumprod(reshape(...), 1)` restarts per block, not globally."""
    rew = np.ones((1, T), np.float32)
    cont = np.full((1, T), 0.5, np.float32)
    got = np.asarray(aggregate_mgr_extr_rew(
        jnp.asarray(rew), jnp.asarray(cont), k, without_zeros=True))
    # Every block sees the same weights 0.5, 0.25, ... so every block is equal.
    assert np.allclose(got[0], got[0][0]), got[0]
    want = np.mean([0.5 ** (j + 1) for j in range(k)])
    assert got[0][0] == pytest.approx(want, rel=1e-5)

  def test_sum_mode_is_the_block_sum_not_the_mean(self):
    T, k = 17, 8
    rew, cont = sample(T, seed=1)
    mean = np.asarray(aggregate_mgr_extr_rew(
        jnp.asarray(rew), jnp.asarray(cont), k, True, agg_mode='mean'))
    total = np.asarray(aggregate_mgr_extr_rew(
        jnp.asarray(rew), jnp.asarray(cont), k, True, agg_mode='sum'))
    np.testing.assert_allclose(total, mean * k, rtol=1e-5)


class TestBlockContMatchesDirector:

  @pytest.mark.parametrize('T,k', EXACT)
  def test_pooled_cont_equals_abstract_traj(self, T, k):
    _, cont = sample(T, seed=T + k)
    got = np.asarray(aggregate_mgr_cont(jnp.asarray(cont), k, without_zeros=True))
    want = director_abstract_cont(cont[0], k)
    np.testing.assert_allclose(got[0], want, rtol=1e-5, atol=1e-6)

  @pytest.mark.parametrize('T,k', EXACT)
  def test_first_column_is_the_raw_first_continuation(self, T, k):
    _, cont = sample(T, seed=7)
    got = np.asarray(aggregate_mgr_cont(jnp.asarray(cont), k, without_zeros=True))
    assert got[0, 0] == pytest.approx(cont[0, 0], rel=1e-6)

  def test_a_terminal_inside_a_block_zeroes_that_block(self):
    T, k = 17, 8
    cont = np.ones((1, T), np.float32)
    cont[0, 5] = 0.0
    got = np.asarray(aggregate_mgr_cont(jnp.asarray(cont), k, without_zeros=True))
    assert got[0, 1] == pytest.approx(0.0)   # block containing index 5
    assert got[0, 2] == pytest.approx(1.0)   # later block unaffected


class TestDecisionStatesMatchDirector:

  @pytest.mark.parametrize('T,k', EXACT)
  def test_downsampled_indices_equal_abstract_traj(self, T, k):
    rng = np.random.default_rng(T * k)
    feat = {'deter': rng.normal(size=(1, T, 3)).astype(np.float32)}
    got = np.asarray(downsample_manager_states(
        {k2: jnp.asarray(v) for k2, v in feat.items()}, k)['deter'])
    want = director_abstract_state(feat['deter'][0], k)
    np.testing.assert_allclose(got[0], want, rtol=1e-6)

  @pytest.mark.parametrize('T,k', EXACT)
  def test_decision_count_is_one_more_than_the_block_count(self, T, k):
    feat = {'deter': jnp.zeros((2, T, 3), jnp.float32)}
    got = downsample_manager_states(feat, k)['deter']
    assert got.shape == (2, (T - 1) // k + 1, 3)

  def test_pooled_reward_and_decision_states_line_up(self):
    """One pooled reward per decision, after `imag_reward_pad`."""
    T, k = 17, 8
    rew, cont = sample(T, seed=3)
    pooled = aggregate_mgr_extr_rew(
        jnp.asarray(rew), jnp.asarray(cont), k, without_zeros=True)
    padded = imag_reward_pad(pooled)
    states = downsample_manager_states({'d': jnp.zeros((1, T, 2))}, k)['d']
    conts = aggregate_mgr_cont(jnp.asarray(cont), k, without_zeros=True)
    assert padded.shape[1] == states.shape[1] == conts.shape[1]


class TestSparseScatterForm:
  """`without_zeros=False` places each block value at its block-start index."""

  @pytest.mark.parametrize('T,k', EXACT)
  def test_sparse_and_compact_forms_agree_on_the_nonzero_entries(self, T, k):
    rew, cont = sample(T, seed=11)
    compact = np.asarray(aggregate_mgr_extr_rew(
        jnp.asarray(rew), jnp.asarray(cont), k, without_zeros=True))
    sparse = np.asarray(aggregate_mgr_extr_rew(
        jnp.asarray(rew), jnp.asarray(cont), k, without_zeros=False))
    n = (T - 1) // k
    idx = 1 + np.arange(n) * k
    np.testing.assert_allclose(sparse[0, idx], compact[0][:n], rtol=1e-5)

  @pytest.mark.parametrize('T,k', EXACT)
  def test_every_non_block_start_column_is_exactly_zero(self, T, k):
    rew, cont = sample(T, seed=12)
    sparse = np.asarray(aggregate_mgr_extr_rew(
        jnp.asarray(rew), jnp.asarray(cont), k, without_zeros=False))
    n = (T - 1) // k
    idx = set((1 + np.arange(n) * k).tolist())
    zeros = [j for j in range(T) if j not in idx]
    assert np.all(sparse[0, zeros] == 0.0), sparse[0]


class TestRemainderBranch:
  """Ours handles `T-1 % k != 0`; Director never does. Self-consistency only."""

  @pytest.mark.parametrize('T,k', [(10, 4), (15, 8), (7, 4)])
  def test_remainder_produces_one_extra_block(self, T, k):
    rew, cont = sample(T, seed=T)
    got = aggregate_mgr_extr_rew(
        jnp.asarray(rew), jnp.asarray(cont), k, without_zeros=True)
    n, rem = (T - 1) // k, (T - 1) % k
    assert rem > 0
    assert got.shape == (1, n + 1)

  @pytest.mark.parametrize('T,k', [(10, 4), (15, 8), (7, 4)])
  def test_remainder_block_is_the_mean_over_its_actual_length(self, T, k):
    T_ = T
    rew = np.ones((1, T_), np.float32)
    cont = np.ones((1, T_), np.float32)
    got = np.asarray(aggregate_mgr_extr_rew(
        jnp.asarray(rew), jnp.asarray(cont), k, without_zeros=True))
    # With cont == 1 and rew == 1, every block mean is 1 regardless of length.
    np.testing.assert_allclose(got[0], 1.0, rtol=1e-6)

  @pytest.mark.parametrize('T,k', [(10, 4), (15, 8)])
  def test_remainder_reward_and_cont_lengths_still_line_up(self, T, k):
    rew, cont = sample(T, seed=T + 1)
    padded = imag_reward_pad(aggregate_mgr_extr_rew(
        jnp.asarray(rew), jnp.asarray(cont), k, without_zeros=True))
    conts = aggregate_mgr_cont(jnp.asarray(cont), k, without_zeros=True)
    states = downsample_manager_states({'d': jnp.zeros((1, T, 2))}, k)['d']
    assert padded.shape[1] == conts.shape[1] == states.shape[1]


class TestShapesAndDtypes:

  @pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
  def test_dtype_is_preserved_through_pooling(self, dtype):
    rew = jnp.ones((2, 17), dtype)
    cont = jnp.ones((2, 17), dtype)
    assert aggregate_mgr_extr_rew(rew, cont, 8, True).dtype == dtype
    assert aggregate_mgr_cont(cont, 8, True).dtype == dtype

  def test_sparse_form_keeps_the_input_shape(self):
    rew = jnp.ones((3, 17), jnp.float32)
    cont = jnp.ones((3, 17), jnp.float32)
    assert aggregate_mgr_extr_rew(rew, cont, 8).shape == (3, 17)
    assert aggregate_mgr_cont(cont, 8).shape == (3, 17)

  def test_batch_rows_are_independent(self):
    rew = np.zeros((2, 17), np.float32)
    rew[0] = 1.0
    cont = np.ones((2, 17), np.float32)
    got = np.asarray(aggregate_mgr_extr_rew(
        jnp.asarray(rew), jnp.asarray(cont), 8, True))
    assert np.allclose(got[0], 1.0) and np.allclose(got[1], 0.0)

  def test_k_below_one_is_clamped_rather_than_dividing_by_zero(self):
    rew = jnp.ones((1, 9), jnp.float32)
    cont = jnp.ones((1, 9), jnp.float32)
    assert aggregate_mgr_extr_rew(rew, cont, 0, True).shape == (1, 8)

  def test_block_longer_than_the_sequence_returns_the_input_unchanged(self):
    rew = jnp.arange(5, dtype=jnp.float32)[None]
    cont = jnp.ones((1, 5), jnp.float32)
    # n == 0 branch: the sparse form falls back to the raw reward.
    np.testing.assert_allclose(
        np.asarray(aggregate_mgr_extr_rew(rew, cont, 16)), np.asarray(rew))


class TestGradients:
  """Pooling is linear in the reward; the Jacobian is the block weight / k."""

  def test_reward_gradient_is_the_block_weight_over_k(self):
    T, k = 17, 8
    cont = jnp.ones((1, T), jnp.float32)
    def total(r):
      return aggregate_mgr_extr_rew(r, cont, k, without_zeros=True).sum()
    g = np.asarray(jax.grad(total)(jnp.zeros((1, T), jnp.float32)))
    # rew[0] is dropped (`rew[:, 1:]`); every other entry contributes 1/k.
    assert g[0, 0] == pytest.approx(0.0)
    np.testing.assert_allclose(g[0, 1:], 1.0 / k, rtol=1e-5)

  def test_first_reward_column_never_receives_gradient(self):
    T, k = 17, 4
    cont = jnp.ones((1, T), jnp.float32)
    fn = lambda r: aggregate_mgr_extr_rew(r, cont, k, True).sum()
    g = np.asarray(jax.grad(fn)(jnp.ones((1, T), jnp.float32)))
    assert g[0, 0] == 0.0

  def test_last_continuation_column_never_receives_gradient(self):
    """Ours uses `con[:, :-1]`, so the final continuation is unused."""
    T, k = 17, 4
    rew = jnp.ones((1, T), jnp.float32)
    fn = lambda c: aggregate_mgr_extr_rew(rew, c, k, True).sum()
    g = np.asarray(jax.grad(fn)(jnp.full((1, T), 0.9, jnp.float32)))
    assert g[0, -1] == 0.0

  def test_pooled_cont_gradient_reaches_every_step_in_its_block(self):
    T, k = 9, 4
    fn = lambda c: aggregate_mgr_cont(c, k, True)[0, 1]
    g = np.asarray(jax.grad(fn)(jnp.full((1, T), 0.9, jnp.float32)))
    assert np.all(g[0, 1:1 + k] > 0), g
    assert g[0, 0] == 0.0
    assert np.all(g[0, 1 + k:] == 0.0)


class TestMaskedCumprod:

  def test_resets_at_every_switch(self):
    c = jnp.full((1, 8), 0.5, jnp.float32)
    sw = jnp.asarray([[1, 0, 0, 0, 1, 0, 0, 0]], jnp.float32)
    got = np.asarray(masked_cumprod(c, sw))[0]
    want = [0.5, 0.25, 0.125, 0.0625] * 2
    np.testing.assert_allclose(got, want, rtol=1e-6)

  def test_without_switches_it_is_a_plain_cumprod(self):
    c = jnp.full((1, 6), 0.9, jnp.float32)
    sw = jnp.zeros((1, 6), jnp.float32)
    np.testing.assert_allclose(
        np.asarray(masked_cumprod(c, sw))[0],
        np.cumprod(np.full(6, 0.9, np.float32)), rtol=1e-6)

  def test_accepts_a_switch_mask_one_longer_than_the_continuation(self):
    c = jnp.full((1, 6), 0.9, jnp.float32)
    sw = jnp.zeros((1, 7), jnp.float32)
    assert masked_cumprod(c, sw).shape == (1, 6)


class TestGoalRewardMatchesDirector:

  def test_cosine_max_equals_the_director_formula(self):
    rng = np.random.default_rng(0)
    goal = rng.normal(size=(2, 5, 8)).astype(np.float32)
    feat = rng.normal(size=(2, 5, 8)).astype(np.float32)
    gnorm = np.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
    fnorm = np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
    norm = np.maximum(gnorm, fnorm)
    want = np.einsum('...i,...i->...', goal / norm, feat / norm)
    got = np.asarray(goal_reward_cosine_max(jnp.asarray(goal), jnp.asarray(feat)))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

  def test_identical_vectors_give_reward_one(self):
    x = jnp.asarray(np.random.default_rng(1).normal(size=(1, 3, 6)), jnp.float32)
    got = np.asarray(goal_reward_cosine_max(x, x))
    np.testing.assert_allclose(got, 1.0, rtol=1e-5)

  def test_a_shorter_feat_is_penalised_by_the_max_norm(self):
    """`max(||g||, ||f||)` means matching direction is not enough."""
    goal = jnp.asarray([[[3.0, 0.0]]], jnp.float32)
    feat = jnp.asarray([[[1.0, 0.0]]], jnp.float32)
    got = float(np.asarray(goal_reward_cosine_max(goal, feat))[0, 0])
    assert got == pytest.approx(3.0 / 9.0, rel=1e-5)

  def test_zero_vectors_do_not_produce_nan(self):
    z = jnp.zeros((1, 2, 4), jnp.float32)
    assert np.all(np.isfinite(np.asarray(goal_reward_cosine_max(z, z))))


class TestSmallHelpers:

  def test_imag_reward_pad_prepends_one_zero_column(self):
    x = jnp.ones((2, 5), jnp.float32)
    out = imag_reward_pad(x)
    assert out.shape == (2, 6)
    assert np.all(np.asarray(out)[:, 0] == 0.0)
    np.testing.assert_allclose(np.asarray(out)[:, 1:], 1.0)

  def test_skill_switch_keeps_the_old_skill_where_update_is_false(self):
    new = jnp.ones((3, 4), jnp.float32)
    old = jnp.zeros((3, 4), jnp.float32)
    upd = jnp.asarray([True, False, True])
    got = np.asarray(skill_switch(upd, new, old))
    np.testing.assert_allclose(got[0], 1.0)
    np.testing.assert_allclose(got[1], 0.0)
    np.testing.assert_allclose(got[2], 1.0)

  def test_skill_switch_broadcasts_over_trailing_dims(self):
    new = jnp.ones((2, 3, 4), jnp.float32)
    old = jnp.zeros((2, 3, 4), jnp.float32)
    got = np.asarray(skill_switch(jnp.asarray([True, False]), new, old))
    assert got.shape == (2, 3, 4)
    np.testing.assert_allclose(got[1], 0.0)

  def test_skill_switch_maps_over_a_dict_of_skills(self):
    new = {'a': jnp.ones((2, 3)), 'b': jnp.ones((2, 5))}
    old = {'a': jnp.zeros((2, 3)), 'b': jnp.zeros((2, 5))}
    got = skill_switch(jnp.asarray([True, False]), new, old)
    assert set(got) == {'a', 'b'}
    assert np.all(np.asarray(got['b'])[1] == 0.0)
