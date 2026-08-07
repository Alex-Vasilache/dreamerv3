"""Unit tests for the blockwise VQ / circular-SOM bottleneck.

Three things are checked exhaustively because they are the ones that silently
produce a plausible-but-wrong model in JAX:

  * indexing -- every gather is compared against an explicit Python loop over
    blocks, so a transposed axis cannot pass;
  * gradient routing -- for each loss term, which parameters receive gradient
    is asserted entry by entry, including which entries must receive EXACTLY
    zero (stop-gradients and the non-differentiable argmin);
  * numerics -- distances against the algebraic expansion, tie-breaking,
    large-magnitude inputs, and bfloat16 input handling.
"""
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from dreamerv3.hrl import vq

f32 = jnp.float32

L, C, D = 4, 5, 3


def make(blocks=L, classes=C, dim=D, seed=0, **kw):
  cb = vq.BlockCodebook(blocks, classes, dim, name='cb', **kw)
  z = jnp.asarray(
      np.random.default_rng(seed).normal(0, 0.1, (2, blocks, dim)), f32)
  params = nj.init(cb.distances)({}, z, seed=0)
  return cb, params, z


def run(cb, fn, params, *args):
  return nj.pure(fn)(params, *args)[1]


class TestTable:

  def test_shape_and_scale(self):
    cb, params, z = make()
    table = params['cb/table']
    assert table.shape == (L, C, D)
    assert table.dtype == f32
    # truncated_normal(-2, 2) * 0.05 -> hard support [-0.1, 0.1].
    assert np.abs(np.asarray(table)).max() <= 0.1 + 1e-6
    assert np.abs(np.asarray(table)).max() > 1e-3

  def test_custom_stddev(self):
    cb, params, z = make(stddev=1.0)
    assert np.abs(np.asarray(params['cb/table'])).max() > 0.5

  def test_blocks_and_entries_are_distinct(self):
    cb, params, z = make(stddev=1.0)
    table = np.asarray(params['cb/table'])
    for l in range(L):
      for c in range(C - 1):
        assert not np.allclose(table[l, c], table[l, c + 1])
    for l in range(L - 1):
      assert not np.allclose(table[l], table[l + 1])

  def test_rejects_degenerate_sizes(self):
    with pytest.raises(AssertionError):
      vq.BlockCodebook(0, C, D, name='bad')
    with pytest.raises(AssertionError):
      vq.BlockCodebook(L, C, 0, name='bad2')


class TestDistances:

  def test_matches_explicit_loop(self):
    cb, params, z = make()
    got = np.asarray(run(cb, cb.distances, params, z))
    table = np.asarray(params['cb/table'])
    zz = np.asarray(z)
    assert got.shape == (2, L, C)
    for b in range(2):
      for l in range(L):
        for c in range(C):
          expect = ((zz[b, l] - table[l, c]) ** 2).sum()
          np.testing.assert_allclose(got[b, l, c], expect, rtol=1e-5, atol=1e-7)

  def test_matches_algebraic_expansion(self):
    # |z|^2 - 2 z.e + |e|^2, a fully independent formula.
    cb, params, z = make(stddev=1.0)
    got = run(cb, cb.distances, params, z)
    table = params['cb/table']
    expect = (
        jnp.square(z).sum(-1)[..., None]
        - 2 * jnp.einsum('...ld,lcd->...lc', z, table)
        + jnp.square(table).sum(-1))
    np.testing.assert_allclose(got, expect, rtol=1e-4, atol=1e-6)

  def test_nonnegative_and_zero_on_exact_entry(self):
    cb, params, z = make()
    table = params['cb/table']
    # Feed entry 2 of every block: that distance must be exactly 0.
    z_exact = jnp.broadcast_to(table[:, 2, :], (2, L, D))
    got = run(cb, cb.distances, params, z_exact)
    assert (np.asarray(got) >= 0).all()
    np.testing.assert_array_equal(np.asarray(got)[:, :, 2], 0.0)

  def test_per_block_independence(self):
    # Perturbing block 0's input must not change block 1's distances.
    cb, params, z = make()
    z2 = z.at[:, 0, :].add(10.0)
    a = np.asarray(run(cb, cb.distances, params, z))
    b = np.asarray(run(cb, cb.distances, params, z2))
    assert not np.allclose(a[:, 0], b[:, 0])
    np.testing.assert_array_equal(a[:, 1:], b[:, 1:])

  def test_large_magnitude_inputs_stay_finite(self):
    cb, params, z = make()
    big = jnp.full((2, L, D), 1e12, f32)
    got = run(cb, cb.distances, params, big)
    assert jnp.isfinite(got).all()

  def test_accepts_bfloat16_input(self):
    cb, params, z = make()
    got = run(cb, cb.distances, params, z.astype(jnp.bfloat16))
    ref = run(cb, cb.distances, params, z)
    assert got.dtype == f32
    np.testing.assert_allclose(got, ref, rtol=2e-2, atol=1e-5)

  def test_rejects_wrong_shape(self):
    cb, params, z = make()
    with pytest.raises(AssertionError):
      run(cb, cb.distances, params, jnp.zeros((2, L, D + 1), f32))
    with pytest.raises(AssertionError):
      run(cb, cb.distances, params, jnp.zeros((2, L + 1, D), f32))


class TestQuantize:

  def test_ids_match_numpy_argmin(self):
    cb, params, z = make(stddev=1.0)
    out = run(cb, cb.quantize, params, z)
    dist = np.asarray(run(cb, cb.distances, params, z))
    np.testing.assert_array_equal(np.asarray(out['ids']), dist.argmin(-1))
    assert out['ids'].dtype == jnp.int32

  def test_onehot_is_a_valid_code(self):
    cb, params, z = make(stddev=1.0)
    out = run(cb, cb.quantize, params, z)
    oh = np.asarray(out['onehot'])
    assert oh.shape == (2, L, C)
    np.testing.assert_array_equal(oh.sum(-1), np.ones((2, L)))
    np.testing.assert_array_equal(oh.argmax(-1), np.asarray(out['ids']))
    assert set(np.unique(oh)) <= {0.0, 1.0}

  def test_zq_equals_explicit_gather(self):
    cb, params, z = make(stddev=1.0)
    out = run(cb, cb.quantize, params, z)
    table = np.asarray(params['cb/table'])
    ids = np.asarray(out['ids'])
    for b in range(2):
      for l in range(L):
        np.testing.assert_array_equal(
            np.asarray(out['z_q'])[b, l], table[l, ids[b, l]])

  def test_selected_distance_is_the_minimum(self):
    cb, params, z = make(stddev=1.0)
    out = run(cb, cb.quantize, params, z)
    sel = jnp.take_along_axis(out['dist'], out['ids'][..., None], -1)[..., 0]
    np.testing.assert_allclose(sel, out['dist'].min(-1), rtol=1e-6)
    np.testing.assert_allclose(
        sel, jnp.square(z - out['z_q']).sum(-1), rtol=1e-4, atol=1e-7)

  def test_tie_breaks_to_lowest_index(self):
    # Documented argmin semantics; a change here would silently reshuffle codes.
    cb = vq.BlockCodebook(1, 3, 2, name='cb')
    params = {'cb/table': jnp.asarray([[[1.0, 0.0], [1.0, 0.0], [9.0, 9.0]]], f32)}
    z = jnp.asarray([[[1.0, 0.0]]], f32)
    out = run(cb, cb.quantize, params, z)
    assert int(out['ids'][0, 0]) == 0

  def test_jit_and_vmap(self):
    cb, params, z = make(stddev=1.0)
    ref = run(cb, cb.quantize, params, z)
    fn = jax.jit(lambda p, x: nj.pure(cb.quantize)(p, x)[1])
    np.testing.assert_array_equal(fn(params, z)['ids'], ref['ids'])
    vfn = jax.vmap(lambda x: nj.pure(cb.quantize)(params, x)[1]['ids'])
    np.testing.assert_array_equal(vfn(z[:, None])[:, 0], ref['ids'])

  def test_extra_batch_dims(self):
    cb, params, z = make(stddev=1.0)
    z4 = jnp.broadcast_to(z[None, None], (3, 2, 2, L, D))
    out = run(cb, cb.quantize, params, z4)
    assert out['ids'].shape == (3, 2, 2, L)
    assert out['z_q'].shape == (3, 2, 2, L, D)


class TestLookup:

  def test_lookup_matches_gather(self):
    cb, params, z = make(stddev=1.0)
    table = np.asarray(params['cb/table'])
    ids = np.array([[0, 1, 2, 3], [4, 0, 1, 2]])
    oh = jnp.asarray(np.eye(C)[ids], f32)
    got = np.asarray(run(cb, cb.lookup, params, oh))
    for b in range(2):
      for l in range(L):
        np.testing.assert_array_equal(got[b, l], table[l, ids[b, l]])

  def test_soft_code_gives_convex_combination(self):
    cb, params, z = make(stddev=1.0)
    table = params['cb/table']
    soft = jnp.full((1, L, C), 1.0 / C, f32)
    got = run(cb, cb.lookup, params, soft)
    np.testing.assert_allclose(got[0], table.mean(1), rtol=1e-5, atol=1e-7)

  def test_lookup_rejects_wrong_code_shape(self):
    cb, params, z = make()
    with pytest.raises(AssertionError):
      run(cb, cb.lookup, params, jnp.zeros((2, L, C + 1), f32))


class TestNeighbors:

  def _table_params(self):
    # Distinct, identifiable entries: table[l, c] = [l*100 + c].
    table = np.arange(L * C, dtype=np.float32).reshape(L, C, 1)
    table = table + np.arange(L)[:, None, None] * 100.0
    return {'cb/table': jnp.asarray(table, f32)}

  def test_circular_wraparound(self):
    cb = vq.BlockCodebook(L, C, 1, name='cb')
    params = self._table_params()
    table = np.asarray(params['cb/table'])
    ids = jnp.asarray([[0, 1, C - 2, C - 1]], jnp.int32)
    got = np.asarray(run(cb, cb.neighbors, params, ids))
    assert got.shape == (1, L, 2, 1)
    expect_idx = [[C - 1, 1], [0, 2], [C - 3, C - 1], [C - 2, 0]]
    for l in range(L):
      for n in range(2):
        np.testing.assert_array_equal(
            got[0, l, n], table[l, expect_idx[l][n]])

  def test_winner_excluded_by_default(self):
    cb = vq.BlockCodebook(L, C, 1, name='cb')
    params = self._table_params()
    table = np.asarray(params['cb/table'])
    ids = jnp.asarray([[2, 2, 2, 2]], jnp.int32)
    got = np.asarray(run(cb, cb.neighbors, params, ids))
    for l in range(L):
      assert not (got[0, l] == table[l, 2]).any()

  def test_include_self_puts_winner_first(self):
    cb = vq.BlockCodebook(L, C, 1, name='cb', include_self=True)
    params = self._table_params()
    table = np.asarray(params['cb/table'])
    ids = jnp.asarray([[2, 2, 2, 2]], jnp.int32)
    got = np.asarray(run(cb, cb.neighbors, params, ids))
    assert got.shape == (1, L, 3, 1)
    for l in range(L):
      np.testing.assert_array_equal(got[0, l, 0], table[l, 2])
      np.testing.assert_array_equal(got[0, l, 1], table[l, 1])
      np.testing.assert_array_equal(got[0, l, 2], table[l, 3])

  def test_neighbor_blocks_do_not_leak(self):
    # Block l's neighbors must come from codebook l, never a different block.
    cb = vq.BlockCodebook(L, C, 1, name='cb')
    params = self._table_params()
    ids = jnp.asarray([[0, 0, 0, 0]], jnp.int32)
    got = np.asarray(run(cb, cb.neighbors, params, ids))
    for l in range(L):
      assert (got[0, l] // 100 == l).all(), got[0, l]


class TestSoftProbsAndMetrics:

  def test_soft_probs_normalized_and_peaked_at_the_winner(self):
    cb, params, z = make(stddev=1.0)
    probs = run(cb, lambda x: cb.soft_probs(x, 1.0), params, z)
    ids = run(cb, cb.encode, params, z)
    np.testing.assert_allclose(probs.sum(-1), np.ones((2, L)), rtol=1e-5)
    assert (np.asarray(probs) >= 0).all()
    np.testing.assert_array_equal(probs.argmax(-1), ids)

  def test_temperature_sharpens(self):
    cb, params, z = make(stddev=1.0)
    hot = run(cb, lambda x: cb.soft_probs(x, 10.0), params, z)
    cold = run(cb, lambda x: cb.soft_probs(x, 0.01), params, z)
    assert cold.max(-1).mean() > hot.max(-1).mean()
    np.testing.assert_allclose(cold.max(-1), np.ones((2, L)), rtol=1e-3)

  def test_soft_probs_rejects_nonpositive_temp(self):
    cb, params, z = make()
    with pytest.raises(AssertionError):
      run(cb, lambda x: cb.soft_probs(x, 0.0), params, z)

  def test_metrics_on_uniform_and_collapsed_usage(self):
    cb = vq.BlockCodebook(1, C, 1, name='cb')
    params = {'cb/table': jnp.zeros((1, C, 1), f32)}
    uniform = jnp.asarray(np.arange(C).reshape(C, 1), jnp.int32)
    mets = run(cb, cb.metrics, params, uniform)
    np.testing.assert_allclose(mets['used_frac'], 1.0)
    np.testing.assert_allclose(mets['perplexity'], C, rtol=1e-4)
    np.testing.assert_allclose(mets['max_prob'], 1.0 / C, rtol=1e-5)
    collapsed = jnp.zeros((C, 1), jnp.int32)
    mets = run(cb, cb.metrics, params, collapsed)
    np.testing.assert_allclose(mets['used_frac'], 1.0 / C, rtol=1e-5)
    np.testing.assert_allclose(mets['perplexity'], 1.0, rtol=1e-4)
    np.testing.assert_allclose(mets['max_prob'], 1.0, rtol=1e-5)


class TestGradientRouting:
  """Which parameter each term is allowed to move -- entry by entry."""

  def _setup(self, **kw):
    cb = vq.BlockCodebook(L, C, D, name='cb', stddev=1.0, **kw)
    z = jnp.asarray(
        np.random.default_rng(1).normal(0, 1, (2, L, D)), f32)
    params = nj.init(cb.distances)({}, z, seed=0)
    ids = np.asarray(run(cb, cb.encode, params, z))
    return cb, params, z, ids

  def test_argmin_blocks_gradient_to_the_encoder(self):
    # z_q carries NO gradient to z_e: there is no straight-through estimator,
    # which is exactly why the SOM arm needs a second reconstruction from z_e.
    cb, params, z, ids = self._setup()
    def fn(z):
      return nj.pure(cb.quantize)(params, z)[1]['z_q'].sum()
    np.testing.assert_array_equal(np.asarray(jax.grad(fn)(z)), 0.0)

  def test_lookup_gradient_reaches_only_selected_entries(self):
    cb, params, z, ids = self._setup()
    def fn(params):
      out = nj.pure(cb.quantize)(params, z)[1]
      return out['z_q'].sum()
    g = np.asarray(jax.grad(fn)(params)['cb/table'])
    for l in range(L):
      for c in range(C):
        hit = (ids[:, l] == c).sum()
        if hit:
          np.testing.assert_allclose(g[l, c], float(hit), rtol=1e-5)
        else:
          np.testing.assert_array_equal(g[l, c], 0.0)

  def test_codebook_term_moves_only_the_codebook(self):
    cb, params, z, ids = self._setup()
    def fn(params, z):
      out = nj.pure(cb.quantize)(params, z)[1]
      return vq.vq_losses(z, out['z_q'])['codebook'].sum()
    gp, gz = jax.grad(fn, (0, 1))(params, z)
    assert np.abs(np.asarray(gp['cb/table'])).max() > 1e-6
    np.testing.assert_array_equal(np.asarray(gz), 0.0)

  def test_commit_term_moves_only_the_encoder(self):
    cb, params, z, ids = self._setup()
    def fn(params, z):
      out = nj.pure(cb.quantize)(params, z)[1]
      return vq.vq_losses(z, out['z_q'])['commit'].sum()
    gp, gz = jax.grad(fn, (0, 1))(params, z)
    np.testing.assert_array_equal(np.asarray(gp['cb/table']), 0.0)
    assert np.abs(np.asarray(gz)).max() > 1e-6

  def test_commit_gradient_value_is_exact(self):
    cb, params, z, ids = self._setup()
    def fn(z):
      out = nj.pure(cb.quantize)(params, z)[1]
      return vq.vq_losses(z, out['z_q'])['commit'].sum()
    got = np.asarray(jax.grad(fn)(z))
    z_q = np.asarray(run(cb, cb.quantize, params, z)['z_q'])
    np.testing.assert_allclose(got, 2 * (np.asarray(z) - z_q), rtol=1e-5)

  def test_som_term_moves_only_the_two_neighbors(self):
    # The defining property of the SOM regularizer: it pulls the WINNER'S
    # NEIGHBORS toward the encoder output, and touches nothing else.
    cb, params, z, ids = self._setup()
    def fn(params, z):
      out = nj.pure(cb.quantize)(params, z)[1]
      nb = nj.pure(cb.neighbors)(params, out['ids'])[1]
      return vq.vq_losses(z, out['z_q'], nb)['som'].sum()
    gp, gz = jax.grad(fn, (0, 1))(params, z)
    np.testing.assert_array_equal(np.asarray(gz), 0.0)  # sg[z_e]
    g = np.asarray(gp['cb/table'])
    touched = {(l, c) for l in range(L) for c in range(C)
               if np.abs(g[l, c]).max() > 1e-7}
    expect = set()
    for b in range(2):
      for l in range(L):
        expect.add((l, int((ids[b, l] - 1) % C)))
        expect.add((l, int((ids[b, l] + 1) % C)))
    assert touched == expect
    # And in particular the winner itself is untouched by this term.
    for b in range(2):
      for l in range(L):
        k = int(ids[b, l])
        if (l, k) not in expect:
          np.testing.assert_array_equal(g[l, k], 0.0)

  def test_som_term_with_include_self_also_moves_the_winner(self):
    cb, params, z, ids = self._setup(include_self=True)
    def fn(params):
      out = nj.pure(cb.quantize)(params, z)[1]
      nb = nj.pure(cb.neighbors)(params, out['ids'])[1]
      return vq.vq_losses(z, out['z_q'], nb)['som'].sum()
    g = np.asarray(jax.grad(fn)(params)['cb/table'])
    for b in range(2):
      for l in range(L):
        assert np.abs(g[l, int(ids[b, l])]).max() > 1e-7

  def test_split_equals_unsplit_gradients(self):
    # motivation.tex writes a single alpha * ||z_e - z_q||^2 (and the official
    # SOM-VAE's loss_commit has no stop-gradient at all). With equal weights
    # the split codebook+commit form has identical gradients everywhere.
    cb, params, z, ids = self._setup()
    def split(params, z):
      out = nj.pure(cb.quantize)(params, z)[1]
      los = vq.vq_losses(z, out['z_q'])
      return (los['codebook'] + los['commit']).sum()
    def unsplit(params, z):
      out = nj.pure(cb.quantize)(params, z)[1]
      return jnp.square(z - out['z_q']).sum()
    a = jax.grad(split, (0, 1))(params, z)
    b = jax.grad(unsplit, (0, 1))(params, z)
    np.testing.assert_allclose(a[0]['cb/table'], b[0]['cb/table'], rtol=1e-5)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-5)

  def test_soft_probs_are_differentiable_wrt_the_encoder(self):
    cb, params, z, ids = self._setup()
    def fn(z):
      return nj.pure(lambda x: cb.soft_probs(x, 1.0))(params, z)[1].max(-1).sum()
    assert np.abs(np.asarray(jax.grad(fn)(z))).max() > 1e-6


class TestVQLosses:

  def test_shapes_reduce_to_batch_dims(self):
    z_e = jnp.zeros((3, 7, L, D), f32)
    z_q = jnp.zeros((3, 7, L, D), f32)
    nb = jnp.zeros((3, 7, L, 2, D), f32)
    los = vq.vq_losses(z_e, z_q, nb)
    for k, v in los.items():
      assert v.shape == (3, 7), (k, v.shape)

  def test_values_against_hand_computation(self):
    z_e = jnp.asarray([[[1.0, 0.0], [0.0, 0.0]]], f32)   # (1, L=2, D=2)
    z_q = jnp.asarray([[[0.0, 0.0], [0.0, 2.0]]], f32)
    los = vq.vq_losses(z_e, z_q, agg='sum')
    np.testing.assert_allclose(los['codebook'], [1.0 + 4.0], rtol=1e-6)
    np.testing.assert_allclose(los['commit'], [1.0 + 4.0], rtol=1e-6)
    np.testing.assert_allclose(los['som'], [0.0])

  def test_sum_vs_mean_scale_factor(self):
    rng = np.random.default_rng(2)
    z_e = jnp.asarray(rng.normal(0, 1, (5, L, D)), f32)
    z_q = jnp.asarray(rng.normal(0, 1, (5, L, D)), f32)
    nb = jnp.asarray(rng.normal(0, 1, (5, L, 2, D)), f32)
    s = vq.vq_losses(z_e, z_q, nb, agg='sum')
    m = vq.vq_losses(z_e, z_q, nb, agg='mean')
    np.testing.assert_allclose(s['commit'], m['commit'] * (L * D), rtol=1e-4)
    np.testing.assert_allclose(s['som'], m['som'] * (L * 2 * D), rtol=1e-4)

  def test_som_value_against_hand_computation(self):
    z_e = jnp.asarray([[[0.0]]], f32)                    # (1, L=1, D=1)
    z_q = jnp.asarray([[[0.0]]], f32)
    nb = jnp.asarray([[[[3.0], [4.0]]]], f32)            # (1, 1, N=2, 1)
    los = vq.vq_losses(z_e, z_q, nb, agg='sum')
    np.testing.assert_allclose(los['som'], [9.0 + 16.0], rtol=1e-6)

  def test_none_neighbors_gives_zero_som(self):
    z_e = jnp.ones((2, L, D), f32)
    los = vq.vq_losses(z_e, jnp.zeros((2, L, D), f32), None)
    np.testing.assert_array_equal(los['som'], np.zeros(2))

  def test_all_terms_vanish_at_a_perfect_fit(self):
    z = jnp.asarray(np.random.default_rng(3).normal(0, 1, (2, L, D)), f32)
    los = vq.vq_losses(z, z, jnp.broadcast_to(z[..., None, :], (2, L, 2, D)))
    for k, v in los.items():
      np.testing.assert_allclose(v, np.zeros(2), atol=1e-6, err_msg=k)

  def test_terms_are_nonnegative(self):
    rng = np.random.default_rng(4)
    z_e = jnp.asarray(rng.normal(0, 5, (8, L, D)), f32)
    z_q = jnp.asarray(rng.normal(0, 5, (8, L, D)), f32)
    nb = jnp.asarray(rng.normal(0, 5, (8, L, 2, D)), f32)
    for k, v in vq.vq_losses(z_e, z_q, nb).items():
      assert (np.asarray(v) >= 0).all(), k

  def test_rejects_mismatched_shapes(self):
    z_e = jnp.zeros((2, L, D), f32)
    with pytest.raises(AssertionError):
      vq.vq_losses(z_e, jnp.zeros((2, L, D + 1), f32))
    with pytest.raises(AssertionError):
      vq.vq_losses(z_e, z_e, jnp.zeros((2, L, 2, D + 1), f32))
    with pytest.raises(AssertionError):
      vq.vq_losses(z_e, z_e, jnp.zeros((2, L + 1, 2, D), f32))

  def test_rejects_unknown_agg(self):
    z = jnp.zeros((2, L, D), f32)
    with pytest.raises(NotImplementedError):
      vq.vq_losses(z, z, agg='median')


class TestLineTopology:
  """Open path instead of a ring: end entries have one neighbor, not two."""

  def _cb(self, **kw):
    cb = vq.BlockCodebook(L, C, 1, name='cb', topology='line', **kw)
    table = np.arange(L * C, dtype=np.float32).reshape(L, C, 1)
    table = table + np.arange(L)[:, None, None] * 100.0
    return cb, {'cb/table': jnp.asarray(table, f32)}

  def test_no_wraparound(self):
    # The defining difference from a ring: entry 0's "left" neighbor does not
    # exist, so it must not be entry C-1.
    cb, params = self._cb()
    table = np.asarray(params['cb/table'])
    ids = jnp.asarray([[0, 1, C - 2, C - 1]], jnp.int32)
    got = np.asarray(run(cb, cb.neighbors, params, ids))
    # Index is clamped, so the out-of-range side duplicates the entry itself;
    # the mask is what removes it from the loss.
    np.testing.assert_array_equal(got[0, 0, 0], table[0, 0])      # clamped
    np.testing.assert_array_equal(got[0, 0, 1], table[0, 1])      # real
    np.testing.assert_array_equal(got[0, 3, 1], table[3, C - 1])  # clamped

  def test_mask_marks_only_the_missing_ends(self):
    cb, params = self._cb()
    ids = jnp.asarray([[0, 1, C - 2, C - 1]], jnp.int32)
    m = np.asarray(run(cb, cb.neighbor_mask, params, ids))
    assert m.shape == (1, L, 2)
    np.testing.assert_array_equal(m[0, 0], [0.0, 1.0])   # no left of entry 0
    np.testing.assert_array_equal(m[0, 1], [1.0, 1.0])   # interior
    np.testing.assert_array_equal(m[0, 2], [1.0, 1.0])   # interior
    np.testing.assert_array_equal(m[0, 3], [1.0, 0.0])   # no right of C-1

  def test_ring_mask_is_all_ones(self):
    cb = vq.BlockCodebook(L, C, 1, name='cb')      # default topology='ring'
    params = {'cb/table': jnp.zeros((L, C, 1), f32)}
    ids = jnp.asarray([[0, 1, C - 2, C - 1]], jnp.int32)
    m = np.asarray(run(cb, cb.neighbor_mask, params, ids))
    np.testing.assert_array_equal(m, np.ones((1, L, 2)))

  def test_masked_neighbor_contributes_nothing_to_the_loss(self):
    z_e = jnp.ones((1, L, 1), f32)
    nb = jnp.stack([jnp.full((1, L, 1), 5.0, f32),
                    jnp.full((1, L, 1), 2.0, f32)], -2)     # (1, L, 2, 1)
    mask = jnp.zeros((1, L, 2), f32).at[..., 1].set(1.0)     # keep only the 2nd
    los = vq.vq_losses(z_e, z_e, nb, 'sum', nb_mask=mask)
    np.testing.assert_allclose(los['som'], [L * 1.0], rtol=1e-6)  # (2-1)^2 each
    full = vq.vq_losses(z_e, z_e, nb, 'sum')
    np.testing.assert_allclose(full['som'], [L * (16.0 + 1.0)], rtol=1e-6)

  def test_masked_neighbor_receives_no_gradient(self):
    # An entry past the end of the line is a clamped duplicate; if it were not
    # masked it would be pulled twice as hard as any interior entry.
    cb, params = self._cb()
    z_e = jnp.zeros((1, L, 1), f32)
    ids = jnp.zeros((1, L), jnp.int32)          # every block picks entry 0
    def loss(params):
      nb = nj.pure(cb.neighbors)(params, ids)[1]
      m = nj.pure(cb.neighbor_mask)(params, ids)[1]
      return vq.vq_losses(z_e, z_e, nb, 'sum', nb_mask=m)['som'].sum()
    g = np.asarray(jax.grad(loss)(params)['cb/table'])
    for l in range(L):
      np.testing.assert_array_equal(g[l, 0], 0.0)   # the clamped duplicate
      assert np.abs(g[l, 1]).max() > 0.0            # the real neighbor

  def test_line_spans_a_wider_distance_range_than_a_ring(self):
    # Evenly spaced points: a path's adjacent/arbitrary ratio is lower than a
    # ring's, because a ring's wrap keeps its farthest pairs closer together.
    def ratio(points):
      nb = np.linalg.norm(np.diff(points, axis=0), axis=-1).mean()
      d = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
      return nb / d[~np.eye(len(points), dtype=bool)].mean()
    RC = 8                      # the project's C; the module constant here is 5
    line = np.arange(RC, dtype=np.float64)[:, None]
    ang = 2 * np.pi * np.arange(RC) / RC
    ring = np.stack([np.cos(ang), np.sin(ang)], 1)
    nb_r = np.linalg.norm(ring - np.roll(ring, 1, 0), axis=-1).mean()
    d_r = np.linalg.norm(ring[:, None] - ring[None, :], axis=-1)
    ring_ratio = nb_r / d_r[~np.eye(RC, dtype=bool)].mean()
    assert ratio(line) < ring_ratio
    np.testing.assert_allclose(ratio(line), 1 / 3, rtol=1e-6)
    np.testing.assert_allclose(ring_ratio, 0.533, atol=5e-3)


class TestJointCommitment:
  """The original SOM-VAE's single ``alpha ||z_e - e_k||^2`` term."""

  def _pair(self, seed=0):
    rng = np.random.default_rng(seed)
    z_e = jnp.asarray(rng.normal(0, 1, (2, 3, L, D)), f32)
    z_q = jnp.asarray(rng.normal(0, 1, (2, 3, L, D)), f32)
    return z_e, z_q

  def test_value_is_the_undivided_squared_distance(self):
    z_e, z_q = self._pair()
    out = vq.vq_losses(z_e, z_q, commit_joint=True)
    np.testing.assert_allclose(
        np.asarray(out['commit']),
        np.asarray(jnp.square(z_e - z_q).sum((-2, -1))), rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(out['codebook']), 0.0)

  def test_gradients_are_identical_to_the_equal_weighted_split(self):
    # The claim that lets us call the split faithful to the source paper. The
    # split sends 2(z_e - z_q) to z_e and 2(z_q - z_e) to the codebook, which
    # is exactly the joint term's two partial derivatives.
    z_e, z_q = self._pair(1)
    joint = lambda a, b: vq.vq_losses(a, b, commit_joint=True)['commit'].sum()
    split = lambda a, b: (lambda o: (o['codebook'] + o['commit']).sum())(
        vq.vq_losses(a, b, commit_joint=False))
    for argnum in (0, 1):
      gj = np.asarray(jax.grad(joint, argnum)(z_e, z_q))
      gs = np.asarray(jax.grad(split, argnum)(z_e, z_q))
      np.testing.assert_allclose(gj, gs, rtol=1e-5, atol=1e-6)

  def test_the_split_double_counts_the_logged_value(self):
    # Same gradients, but the split's reported number is twice the joint's.
    z_e, z_q = self._pair(2)
    j = vq.vq_losses(z_e, z_q, commit_joint=True)
    s = vq.vq_losses(z_e, z_q, commit_joint=False)
    np.testing.assert_allclose(
        np.asarray(s['codebook'] + s['commit']),
        2 * np.asarray(j['commit']), rtol=1e-5)

  def test_the_joint_term_moves_both_sides(self):
    # No stop-gradient anywhere: unlike either half alone, it is nonzero in
    # both arguments.
    z_e, z_q = self._pair(3)
    joint = lambda a, b: vq.vq_losses(a, b, commit_joint=True)['commit'].sum()
    for argnum in (0, 1):
      g = np.asarray(jax.grad(joint, argnum)(z_e, z_q))
      assert np.abs(g).max() > 1e-6, argnum

  def test_the_som_term_is_untouched_by_the_flag(self):
    z_e, z_q = self._pair(4)
    nb = jnp.asarray(np.random.default_rng(5).normal(0, 1, (2, 3, L, 2, D)), f32)
    a = vq.vq_losses(z_e, z_q, nb, commit_joint=True)['som']
    b = vq.vq_losses(z_e, z_q, nb, commit_joint=False)['som']
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)
