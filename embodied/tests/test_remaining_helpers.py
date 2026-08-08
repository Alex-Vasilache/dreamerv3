"""The last helpers with no coverage anywhere in the suite.

Collected by cross-referencing every top-level `def` in `dreamerv3/hrl/` against
every test file. These were the remainder: `mgr_as_dict`, `pairwise_cosmax`,
`vq._agg`, and the `outs.Agg` wrappers in `goal_ae.py`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import embodied.jax.outs as outs
from dreamerv3.hrl.goal_ae import _wrap_code, _wrap_code_onehot, _wrap_recon
from dreamerv3.hrl.tensors import mgr_as_dict, pairwise_cosmax
from dreamerv3.hrl.vq import _agg

f32 = jnp.float32


class TestMgrAsDict:

  def test_a_bare_tensor_is_wrapped_under_skill(self):
    x = jnp.ones((2, 3))
    assert mgr_as_dict(x)['skill'] is x

  def test_an_existing_dict_passes_through_untouched(self):
    d = {'skill': jnp.ones((2, 3)), 'duration': jnp.zeros((2,))}
    assert mgr_as_dict(d) is d

  def test_an_empty_dict_is_still_a_dict(self):
    assert mgr_as_dict({}) == {}


class TestVqAgg:

  def test_sum_reduces_the_trailing_axes(self):
    x = jnp.ones((2, 3, 4, 5))
    assert _agg(x, 2, 'sum').shape == (2, 3)
    np.testing.assert_allclose(np.asarray(_agg(x, 2, 'sum')), 20.0)

  def test_mean_reduces_the_trailing_axes(self):
    x = jnp.ones((2, 3, 4, 5))
    np.testing.assert_allclose(np.asarray(_agg(x, 2, 'mean')), 1.0)

  def test_sum_and_mean_differ_by_the_reduced_element_count(self):
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(2, 3, 4)), f32)
    np.testing.assert_allclose(
        np.asarray(_agg(x, 1, 'sum')), np.asarray(_agg(x, 1, 'mean')) * 4,
        rtol=1e-5)

  def test_reducing_three_axes(self):
    x = jnp.ones((2, 3, 4, 5))
    assert _agg(x, 3, 'sum').shape == (2,)

  def test_an_unknown_impl_is_rejected(self):
    with pytest.raises(NotImplementedError):
      _agg(jnp.ones((2, 3)), 1, 'median')

  def test_dtype_is_preserved(self):
    assert _agg(jnp.ones((2, 3), jnp.bfloat16), 1, 'sum').dtype == jnp.bfloat16


class TestPairwiseCosmax:
  """All-pairs magnitude-aware similarity used by the goal-geometry diagnostic."""

  def test_matches_the_pairwise_definition(self):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 7)).astype(np.float32)
    got = np.asarray(pairwise_cosmax(jnp.asarray(x)))
    n = np.linalg.norm(x, axis=-1) + 1e-12
    want = np.empty((5, 5), np.float32)
    for i in range(5):
      for j in range(5):
        m = max(n[i], n[j])
        want[i, j] = x[i] @ x[j] / (m * m)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)

  def test_the_matrix_is_symmetric(self):
    rng = np.random.default_rng(1)
    got = np.asarray(pairwise_cosmax(
        jnp.asarray(rng.normal(size=(6, 4)), f32)))
    np.testing.assert_allclose(got, got.T, rtol=1e-5)

  def test_the_diagonal_is_one(self):
    rng = np.random.default_rng(2)
    got = np.asarray(pairwise_cosmax(
        jnp.asarray(rng.normal(size=(6, 4)), f32)))
    np.testing.assert_allclose(np.diag(got), 1.0, rtol=1e-4)

  def test_it_agrees_with_the_worker_goal_reward_on_a_pair(self):
    from dreamerv3.hrl.tensors import goal_reward_cosine_max
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(2, 4)), f32)
    pair = float(np.asarray(pairwise_cosmax(x))[0, 1])
    direct = float(np.asarray(
        goal_reward_cosine_max(x[0][None], x[1][None]))[0])
    assert pair == pytest.approx(direct, rel=1e-4)

  def test_shorter_vectors_score_below_one_against_longer_ones(self):
    x = jnp.asarray([[1.0, 0.0], [3.0, 0.0]], f32)
    got = np.asarray(pairwise_cosmax(x))
    assert got[0, 1] == pytest.approx(3.0 / 9.0, rel=1e-4)

  def test_zero_rows_do_not_produce_nan(self):
    x = jnp.zeros((3, 4), f32)
    assert np.all(np.isfinite(np.asarray(pairwise_cosmax(x))))

  def test_output_shape_is_square_in_the_pool_size(self):
    assert pairwise_cosmax(jnp.ones((9, 3), f32)).shape == (9, 9)


class TestAggWrappers:
  """`outs.Agg` wrapping keeps `_head_inner` able to find the real head."""

  def test_wrap_code_is_unwrapped_back_to_the_original(self):
    from dreamerv3.hrl.heads import _head_inner
    inner = outs.OneHot(jnp.zeros((2, 3, 4), f32))
    assert _head_inner(_wrap_code(inner)) is inner

  def test_wrap_code_onehot_is_unwrapped_back_to_the_original(self):
    from dreamerv3.hrl.heads import _head_inner
    inner = outs.OneHot(jnp.zeros((2, 3, 4), f32))
    assert _head_inner(_wrap_code_onehot(inner)) is inner

  def test_wrap_recon_sums_squared_error_over_the_feature_axis(self):
    pred = jnp.zeros((2, 5), f32)
    wrapped = _wrap_recon(pred)
    target = jnp.ones((2, 5), f32)
    np.testing.assert_allclose(np.asarray(wrapped.loss(target)), 5.0, rtol=1e-5)

  def test_wrap_recon_pred_round_trips_the_input(self):
    pred = jnp.asarray([[1.0, 2.0, 3.0]], f32)
    np.testing.assert_allclose(
        np.asarray(_wrap_recon(pred).pred()), np.asarray(pred), rtol=1e-6)

  def test_wrap_recon_gradient_reaches_the_prediction(self):
    target = jnp.ones((2, 5), f32)
    fn = lambda p: _wrap_recon(p).loss(target).sum()
    g = np.asarray(jax.grad(fn)(jnp.zeros((2, 5), f32)))
    np.testing.assert_allclose(g, -2.0, rtol=1e-5)

  def test_wrap_recon_target_is_not_differentiated(self):
    """`MSE.loss` stop-gradients its own target, so callers cannot leak.

    This matters for the goal autoencoder: the reconstruction target is the
    world model's `deter`, and a gradient path from the AE loss back into the
    world model would let the goal AE reshape the representation it is
    supposed to be summarising.
    """
    pred = jnp.zeros((2, 5), f32)
    fn = lambda t: _wrap_recon(pred).loss(t).sum()
    g = np.asarray(jax.grad(fn)(jnp.ones((2, 5), f32)))
    assert np.all(g == 0.0)

  def test_every_regression_head_stop_gradients_its_target(self):
    for make in (lambda p: outs.MSE(p), lambda p: outs.Huber(p)):
      pred = jnp.zeros((4,), f32)
      fn = lambda t: make(pred).loss(t).sum()
      g = np.asarray(jax.grad(fn)(jnp.ones((4,), f32)))
      assert np.all(g == 0.0), make

  def test_agg_preserves_the_leading_batch_dims(self):
    wrapped = _wrap_recon(jnp.zeros((3, 7, 5), f32))
    assert wrapped.loss(jnp.ones((3, 7, 5), f32)).shape == (3, 7)


class TestCodeGeometryEquivalence:
  """`agent.py`'s `goal/struct_corr_code` rests on a documented identity.

  For the Director head the decoder input is a flattened one-hot whose norm is
  sqrt(L) for every code, so `pairwise_cosmax` of those inputs equals the
  fraction of matching blocks (1 - Hamming). The comment in `agent.py` claims
  this makes `struct_corr_code` comparable to the Hamming-based baselines; the
  claim is asserted here rather than trusted.
  """

  @staticmethod
  def _codes(n, L, C, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, C, size=(n, L))

  @pytest.mark.parametrize('L,C', [(8, 8), (4, 16), (8, 4)])
  def test_cosmax_of_flat_onehots_equals_hamming_similarity(self, L, C):
    ids = self._codes(12, L, C, seed=L * C)
    onehot = np.eye(C, dtype=np.float32)[ids]              # (N, L, C)
    got = np.asarray(pairwise_cosmax(
        jnp.asarray(onehot.reshape(len(ids), -1))))
    want = 1.0 - (ids[:, None, :] != ids[None, :, :]).mean(-1)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

  def test_identical_codes_score_one_and_fully_disjoint_codes_score_zero(self):
    L, C = 8, 8
    a = np.zeros((1, L), int)
    b = np.ones((1, L), int)
    ids = np.concatenate([a, a, b], 0)
    onehot = np.eye(C, dtype=np.float32)[ids]
    got = np.asarray(pairwise_cosmax(jnp.asarray(onehot.reshape(3, -1))))
    assert got[0, 1] == pytest.approx(1.0, rel=1e-5)
    assert got[0, 2] == pytest.approx(0.0, abs=1e-6)

  def test_a_flat_onehot_has_norm_sqrt_L(self):
    L, C = 8, 8
    ids = self._codes(5, L, C, seed=1)
    onehot = np.eye(C, dtype=np.float32)[ids].reshape(5, -1)
    np.testing.assert_allclose(
        np.linalg.norm(onehot, axis=-1), np.sqrt(L), rtol=1e-6)

  def test_quantized_codes_do_not_reduce_to_hamming(self):
    """The reason the metric was changed: embeddings carry codebook geometry."""
    L, C, D = 4, 8, 3
    rng = np.random.default_rng(0)
    table = rng.normal(size=(L, C, D)).astype(np.float32)
    ids = self._codes(10, L, C, seed=2)
    onehot = np.eye(C, dtype=np.float32)[ids]
    emb = np.einsum('nlc,lcd->nld', onehot, table).reshape(10, -1)
    cos = np.asarray(pairwise_cosmax(jnp.asarray(emb)))
    ham = 1.0 - (ids[:, None, :] != ids[None, :, :]).mean(-1)
    assert not np.allclose(cos, ham, atol=1e-3)
