"""Unit tests for the LipVQ-VAE Lipschitz weight normalization.

The properties checked here are the ones the architecture actually relies on:
the normalization bounds the layer's inf-norm operator norm, the bound
composes multiplicatively through 1-Lipschitz activations, and the trainable
bound only receives forward-pass gradient while the constraint is active.
"""
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

import embodied.jax.nets as nets
from dreamerv3.hrl import lipschitz as lip

f32 = jnp.float32


@pytest.fixture(autouse=True)
def f32_compute():
  """Run the MLP trunk in f32 so the tests measure math, not bfloat16 noise."""
  old = nets.COMPUTE_DTYPE
  nets.COMPUTE_DTYPE = jnp.float32
  yield
  nets.COMPUTE_DTYPE = old


def inf_norm(x):
  return np.abs(np.asarray(x)).max(-1)


class TestSoftplusInv:

  @pytest.mark.parametrize('y', [1e-6, 1e-3, 0.1, 1.0, 2.4, 19.0, 21.0, 100.0])
  def test_roundtrip(self, y):
    c = lip.softplus_inv(jnp.asarray(y, f32))
    back = jax.nn.softplus(c)
    np.testing.assert_allclose(back, y, rtol=1e-5, atol=1e-6)

  def test_large_branch_is_identity(self):
    # softplus(y) == y to f32 precision above ~20, so the inverse must be too;
    # log(expm1(y)) would overflow to inf there.
    y = jnp.asarray([25.0, 60.0, 400.0], f32)
    assert jnp.isfinite(lip.softplus_inv(y)).all()
    np.testing.assert_allclose(lip.softplus_inv(y), y, rtol=1e-6)

  def test_vector_shape_preserved(self):
    y = jnp.asarray([[0.5, 30.0], [2.0, 7.0]], f32)
    assert lip.softplus_inv(y).shape == (2, 2)


class TestRowSumAndScale:

  def test_abs_row_sum_reduces_input_axis(self):
    # Kernel is (in, out): column j is the LipVQ "row" W_j. Distinct column
    # sums catch an axis swap, which a square kernel would hide.
    kernel = jnp.asarray([[1.0, -2.0, 0.0],
                          [-3.0, 0.5, 4.0]], f32)     # (in=2, out=3)
    got = lip.abs_row_sum(kernel)
    assert got.shape == (3,)
    np.testing.assert_allclose(got, [4.0, 2.5, 4.0], rtol=1e-6)

  def test_scale_clamps_only_violating_rows(self):
    kernel = jnp.asarray([[1.0, 0.1],
                          [1.0, 0.1]], f32)           # row sums: [2.0, 0.2]
    scale = lip.lip_scale(kernel, jnp.asarray(1.0, f32), clamp=True)
    # First output unit is over budget -> scaled to exactly 1.0/2.0.
    # Second is under budget -> untouched (exactly 1.0, not 1.0/0.2 = 5.0).
    np.testing.assert_allclose(scale, [0.5, 1.0], rtol=1e-6)

  def test_scale_without_clamp_rescales_both_ways(self):
    kernel = jnp.asarray([[1.0, 0.1],
                          [1.0, 0.1]], f32)
    scale = lip.lip_scale(kernel, jnp.asarray(1.0, f32), clamp=False)
    np.testing.assert_allclose(scale, [0.5, 5.0], rtol=1e-5)

  def test_per_row_bound_vector(self):
    kernel = jnp.asarray([[1.0, 1.0],
                          [1.0, 1.0]], f32)           # row sums: [2.0, 2.0]
    scale = lip.lip_scale(kernel, jnp.asarray([1.0, 4.0], f32), clamp=True)
    np.testing.assert_allclose(scale, [0.5, 1.0], rtol=1e-6)

  def test_normalize_hits_the_bound_exactly(self):
    rng = np.random.default_rng(0)
    kernel = jnp.asarray(rng.normal(0, 1, (16, 5)), f32)
    c = lip.softplus_inv(jnp.asarray(0.25, f32))      # bound well below row sums
    normed = lip.lip_normalize(kernel, c, clamp=True)
    np.testing.assert_allclose(lip.abs_row_sum(normed), 0.25, rtol=1e-5)

  def test_normalize_is_inert_when_bound_is_slack(self):
    kernel = jnp.asarray([[0.1, 0.1], [0.1, 0.1]], f32)
    c = lip.softplus_inv(jnp.asarray(100.0, f32))
    normed = lip.lip_normalize(kernel, c, clamp=True)
    np.testing.assert_array_equal(normed, kernel)

  def test_normalize_preserves_sign_and_direction(self):
    kernel = jnp.asarray([[1.0, -2.0], [-3.0, 4.0]], f32)
    normed = lip.lip_normalize(kernel, lip.softplus_inv(jnp.asarray(1.0, f32)))
    # Column-wise positive rescale: sign pattern and within-column ratios kept.
    assert (jnp.sign(normed) == jnp.sign(kernel)).all()
    np.testing.assert_allclose(
        normed[0, 0] / normed[1, 0], kernel[0, 0] / kernel[1, 0], rtol=1e-5)


class TestPenalty:

  def test_prod_and_sum(self):
    bounds = [jnp.asarray(2.0), jnp.asarray(3.0), jnp.asarray(4.0)]
    np.testing.assert_allclose(lip.lip_penalty(bounds, 'prod'), 24.0)
    np.testing.assert_allclose(lip.lip_penalty(bounds, 'sum'), 9.0)

  def test_empty_is_zero_for_both(self):
    # Off switch: gamma * penalty must vanish, so an empty product is 0, not 1.
    np.testing.assert_allclose(lip.lip_penalty([], 'prod'), 0.0)
    np.testing.assert_allclose(lip.lip_penalty([], 'sum'), 0.0)

  def test_unknown_impl_raises(self):
    with pytest.raises(NotImplementedError):
      lip.lip_penalty([jnp.asarray(1.0)], 'mean')


class TestLipLinear:

  def _make(self, D=6, U=4, **kw):
    net = lip.LipLinear(U, name='lin', **kw)
    x = jnp.asarray(np.random.default_rng(0).normal(0, 1, (3, D)), f32)
    params = nj.init(net)({}, x, seed=0)
    return net, params, x

  def test_param_shapes_per_row(self):
    net, params, x = self._make(D=6, U=4)
    assert set(params) == {'lin/kernel', 'lin/bias', 'lin/c'}
    assert params['lin/kernel'].shape == (6, 4)
    assert params['lin/c'].shape == (4,)

  def test_param_shape_scalar_bound(self):
    net, params, x = self._make(D=6, U=4, per_row=False)
    assert params['lin/c'].shape == ()

  def test_default_init_is_inert(self):
    # cinit < 0 initializes softplus(c) to the layer's own row sums, so the
    # constraint must not touch the forward pass at step 0.
    net, params, x = self._make(D=6, U=4)
    _, out = nj.pure(net)(params, x)
    ref = x @ params['lin/kernel'] + params['lin/bias']
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
    _, scale = nj.pure(net.scale)(params)
    np.testing.assert_allclose(scale, np.ones(4), rtol=1e-5)

  def test_scalar_bound_init_is_inert_too(self):
    net, params, x = self._make(D=6, U=4, per_row=False)
    _, scale = nj.pure(net.scale)(params)
    # A single scalar bound is initialized from the LARGEST row sum, so no row
    # is rescaled, though the slack rows sit strictly below the bound.
    np.testing.assert_allclose(scale, np.ones(4), rtol=1e-5)

  def test_bound_is_max_over_rows(self):
    net, params, x = self._make(D=6, U=4)
    _, bound = nj.pure(net.bound)(params)
    expect = jax.nn.softplus(params['lin/c']).max()
    np.testing.assert_allclose(bound, expect, rtol=1e-6)

  def test_enforces_inf_norm_lipschitz(self):
    net, params, x = self._make(D=6, U=4, cinit=0.3)
    _, bound = nj.pure(net.bound)(params)
    rng = np.random.default_rng(1)
    x1 = jnp.asarray(rng.normal(0, 3, (64, 6)), f32)
    x2 = jnp.asarray(rng.normal(0, 3, (64, 6)), f32)
    _, y1 = nj.pure(net)(params, x1)
    _, y2 = nj.pure(net)(params, x2)
    lhs = inf_norm(y1 - y2)
    rhs = float(bound) * inf_norm(x1 - x2)
    assert (lhs <= rhs * (1 + 1e-5)).all(), (lhs.max(), rhs.min())
    # The bound must be tight enough to be meaningful, not vacuous.
    assert (lhs.max() / rhs.max()) > 0.05

  def test_bias_does_not_affect_the_bound(self):
    # Lipschitz continuity is a property of differences; a bias shifts both
    # outputs equally and must cancel.
    net, params, x = self._make(D=6, U=4, cinit=0.3)
    shifted = dict(params)
    shifted['lin/bias'] = params['lin/bias'] + 7.0
    rng = np.random.default_rng(2)
    x1 = jnp.asarray(rng.normal(0, 1, (8, 6)), f32)
    x2 = jnp.asarray(rng.normal(0, 1, (8, 6)), f32)
    d0 = nj.pure(net)(params, x1)[1] - nj.pure(net)(params, x2)[1]
    d1 = nj.pure(net)(shifted, x1)[1] - nj.pure(net)(shifted, x2)[1]
    np.testing.assert_allclose(d0, d1, rtol=1e-5, atol=1e-6)

  def test_no_bias_variant(self):
    net, params, x = self._make(D=6, U=4, bias=False)
    assert 'lin/bias' not in params

  def test_forward_gradient_reaches_c_only_when_active(self):
    D, U = 6, 4
    x = jnp.asarray(np.random.default_rng(3).normal(0, 1, (3, D)), f32)

    def grad_c(cinit):
      net = lip.LipLinear(U, name='lin', cinit=cinit)
      params = nj.init(net)({}, x, seed=0)
      def loss(params):
        _, out = nj.pure(net)(params, x)
        return (out ** 2).sum()
      return np.asarray(jax.grad(loss)(params)['lin/c'])

    # Active constraint (tight bound): c changes the effective weights.
    assert np.abs(grad_c(0.3)).max() > 1e-6
    # Slack constraint: min() saturates at 1.0 and c is disconnected from the
    # forward pass -- only the L_Lipschitz penalty can move it. This is the
    # behavior that makes the penalty necessary rather than merely optional.
    assert np.abs(grad_c(1e4)).max() == 0.0

  def test_kernel_gradient_flows_through_normalization(self):
    D, U = 6, 4
    x = jnp.asarray(np.random.default_rng(4).normal(0, 1, (3, D)), f32)
    net = lip.LipLinear(U, name='lin', cinit=0.3)
    params = nj.init(net)({}, x, seed=0)
    def loss(params):
      _, out = nj.pure(net)(params, x)
      return (out ** 2).sum()
    g = jax.grad(loss)(params)
    assert np.abs(np.asarray(g['lin/kernel'])).max() > 1e-6

  def test_unclamped_option_scales_up(self):
    # motivation.tex's unconditional form: a slack layer is scaled UP to the
    # bound instead of left alone.
    D, U = 6, 4
    x = jnp.asarray(np.random.default_rng(5).normal(0, 1, (3, D)), f32)
    net = lip.LipLinear(U, name='lin', cinit=50.0, clamp=False)
    params = nj.init(net)({}, x, seed=0)
    _, scale = nj.pure(net.scale)(params)
    assert (np.asarray(scale) > 5.0).all()


class TestLipMLP:

  def _mlp(self, layers=3, units=8, D=6, **kw):
    net = lip.LipMLP(layers, units, name='mlp', **kw)
    x = jnp.asarray(np.random.default_rng(0).normal(0, 1, (5, D)), f32)
    params = nj.init(net)({}, x, seed=0)
    return net, params, x

  def test_plain_mode_matches_nets_mlp_exactly(self):
    # lip=False must be a bit-for-bit drop-in for the existing trunk, so the
    # non-Lipschitz ablation arms are unaffected by this module.
    D = 6
    x = jnp.asarray(np.random.default_rng(0).normal(0, 1, (5, D)), f32)
    a = lip.LipMLP(3, 8, name='mlp', act='silu', norm='rms', lip=False)
    b = nets.MLP(3, 8, name='mlp', act='silu', norm='rms')
    pa = nj.init(a)({}, x, seed=0)
    pb = nj.init(b)({}, x, seed=0)
    assert set(pa) == set(pb)
    for k in pa:
      np.testing.assert_array_equal(pa[k], pb[k])
    np.testing.assert_array_equal(nj.pure(a)(pa, x)[1], nj.pure(b)(pb, x)[1])

  def test_plain_mode_has_no_bound_params(self):
    net, params, x = self._mlp(lip=False, norm='rms')
    assert not [k for k in params if k.endswith('/c')]
    _, _ = nj.pure(net)(params, x)
    assert net.bounds() == []

  def test_lip_mode_creates_one_bound_per_layer(self):
    net, params, x = self._mlp(layers=3, lip=True, act='relu')
    cs = sorted(k for k in params if k.endswith('/c'))
    assert cs == ['mlp/linear0/c', 'mlp/linear1/c', 'mlp/linear2/c']

  def test_strict_bound_requires_no_norm(self):
    # Without strict_bound a rescaling norm is allowed (and is the shipped
    # default: an unnormalized trunk is not trainable at this project's scale),
    # but then prod(bounds) bounds only the linear layers, not the trunk.
    lip.LipMLP(2, 8, name='permissive', lip=True, norm='rms')
    with pytest.raises(AssertionError, match='norm=none'):
      lip.LipMLP(2, 8, name='bad', lip=True, norm='rms', strict_bound=True)

  def test_strict_bound_accepts_no_norm(self):
    net = lip.LipMLP(2, 8, name='strict', lip=True, norm='none',
                     strict_bound=True)
    x = jnp.asarray(np.random.default_rng(0).normal(0, 1, (5, 6)), f32)
    params = nj.init(net)({}, x, seed=0)
    assert [k for k in params if k.endswith('/c')]

  def test_composed_bound_holds_end_to_end(self):
    # The whole point: with 1-Lipschitz activations the network's inf-norm
    # Lipschitz constant is bounded by the product of the per-layer bounds.
    L = 3
    net = lip.LipMLP(L, 8, name='mlp', lip=True, act='relu', norm='none',
                     cinit=0.5)
    x = jnp.asarray(np.random.default_rng(0).normal(0, 1, (5, 6)), f32)
    params = nj.init(net)({}, x, seed=0)

    def bounds_fn(x):
      net(x)
      return lip.lip_penalty(net.bounds(), 'prod')
    _, total = nj.pure(bounds_fn)(params, x)
    np.testing.assert_allclose(total, 0.5 ** L, rtol=1e-4)

    rng = np.random.default_rng(6)
    x1 = jnp.asarray(rng.normal(0, 5, (128, 6)), f32)
    x2 = jnp.asarray(rng.normal(0, 5, (128, 6)), f32)
    y1 = nj.pure(net)(params, x1)[1]
    y2 = nj.pure(net)(params, x2)[1]
    lhs = inf_norm(y1 - y2)
    rhs = float(total) * inf_norm(x1 - x2)
    assert (lhs <= rhs * (1 + 1e-4)).all(), (lhs.max(), rhs.min())

  def test_bound_shrinks_output_variation(self):
    # A tighter bound must produce a strictly smoother map, which is the
    # mechanism the ablation is testing.
    x = jnp.asarray(np.random.default_rng(0).normal(0, 1, (5, 6)), f32)
    rng = np.random.default_rng(7)
    x1 = jnp.asarray(rng.normal(0, 5, (64, 6)), f32)
    x2 = jnp.asarray(rng.normal(0, 5, (64, 6)), f32)
    spreads = []
    for cinit in (0.2, 2.0):
      net = lip.LipMLP(3, 8, name='mlp', lip=True, act='relu', norm='none',
                       cinit=cinit)
      params = nj.init(net)({}, x, seed=0)
      y1 = nj.pure(net)(params, x1)[1]
      y2 = nj.pure(net)(params, x2)[1]
      spreads.append(inf_norm(y1 - y2).mean())
    assert spreads[0] < spreads[1]

  def test_bounds_are_differentiable_wrt_c(self):
    net = lip.LipMLP(2, 8, name='mlp', lip=True, act='relu', norm='none',
                     cinit=0.5)
    x = jnp.asarray(np.random.default_rng(0).normal(0, 1, (5, 6)), f32)
    params = nj.init(net)({}, x, seed=0)
    def penalty(params):
      def fn(x):
        net(x)
        return lip.lip_penalty(net.bounds(), 'prod')
      return nj.pure(fn)(params, x)[1]
    g = jax.grad(penalty)(params)
    for k in ('mlp/linear0/c', 'mlp/linear1/c'):
      assert np.abs(np.asarray(g[k])).max() > 1e-6

  def test_bound_holds_under_bfloat16_compute(self):
    # Training runs the trunk in bfloat16; the guarantee should survive it up
    # to bfloat16 rounding (~1e-2 relative), not break outright.
    old = nets.COMPUTE_DTYPE
    nets.COMPUTE_DTYPE = jnp.bfloat16
    try:
      net = lip.LipMLP(3, 8, name='mlp', lip=True, act='relu', norm='none',
                       cinit=0.5)
      x = jnp.asarray(np.random.default_rng(0).normal(0, 1, (5, 6)), f32)
      params = nj.init(net)({}, x, seed=0)
      rng = np.random.default_rng(8)
      x1 = jnp.asarray(rng.normal(0, 5, (128, 6)), f32)
      x2 = jnp.asarray(rng.normal(0, 5, (128, 6)), f32)
      y1 = nj.pure(net)(params, x1)[1].astype(f32)
      y2 = nj.pure(net)(params, x2)[1].astype(f32)
      lhs = inf_norm(y1 - y2)
      rhs = (0.5 ** 3) * inf_norm(x1 - x2)
      assert (lhs <= rhs * 1.05).all(), (lhs.max(), rhs.min())
    finally:
      nets.COMPUTE_DTYPE = old
