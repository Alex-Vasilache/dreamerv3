"""Optimizer transforms and the running normalizers.

`clip_by_agc` is what replaced Director's global-norm-100 clip, and
`Normalize(impl='perc')` is what replaced Director's `std` return normalizer.
Both were listed as untested differences in `docs/VERIFICATION.md`. The
normalizer in particular decides the scale of every advantage, so its behaviour
when returns are near-degenerate is worth pinning.
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from embodied.jax.opt import clip_by_agc, scale_by_momentum, scale_by_rms
from embodied.jax.utils import Normalize

f32 = jnp.float32


class TestAdaptiveGradientClipping:

  @staticmethod
  def apply(params, grads, clip=0.3, pmin=1e-3):
    tx = clip_by_agc(clip, pmin)
    out, _ = tx.update(grads, tx.init(params), params)
    return out

  def test_a_small_update_passes_through_unchanged(self):
    p = {'w': jnp.ones((4,), f32)}          # ||p|| = 2, upper = 0.6
    g = {'w': jnp.full((4,), 0.01, f32)}    # ||g|| = 0.02 < 0.6
    np.testing.assert_allclose(
        np.asarray(self.apply(p, g)['w']), np.asarray(g['w']), rtol=1e-6)

  def test_a_large_update_is_scaled_to_the_threshold(self):
    p = {'w': jnp.ones((4,), f32)}          # ||p|| = 2
    g = {'w': jnp.ones((4,), f32) * 10.0}   # ||g|| = 20
    out = np.asarray(self.apply(p, g)['w'])
    assert np.linalg.norm(out) == pytest.approx(0.3 * 2.0, rel=1e-5)

  def test_the_clipped_update_keeps_its_direction(self):
    rng = np.random.default_rng(0)
    p = {'w': jnp.asarray(rng.normal(size=(6,)), f32)}
    g = {'w': jnp.asarray(rng.normal(size=(6,)) * 100, f32)}
    out = np.asarray(self.apply(p, g)['w'])
    raw = np.asarray(g['w'])
    cos = out @ raw / (np.linalg.norm(out) * np.linalg.norm(raw))
    assert cos == pytest.approx(1.0, rel=1e-5)

  def test_pmin_floors_the_threshold_for_near_zero_parameters(self):
    p = {'w': jnp.zeros((4,), f32)}
    g = {'w': jnp.ones((4,), f32)}
    out = np.asarray(self.apply(p, g, clip=0.3, pmin=1e-3)['w'])
    # upper = 0.3 * max(1e-3, 0) = 3e-4
    assert np.linalg.norm(out) == pytest.approx(3e-4, rel=1e-4)

  def test_clip_zero_disables_the_transform(self):
    p = {'w': jnp.ones((4,), f32)}
    g = {'w': jnp.full((4,), 1e6, f32)}
    np.testing.assert_allclose(
        np.asarray(self.apply(p, g, clip=0)['w']), np.asarray(g['w']))

  def test_clipping_is_per_parameter_not_global(self):
    """Unlike Director's global-norm clip, each tensor is scaled on its own."""
    p = {'a': jnp.ones((4,), f32), 'b': jnp.ones((4,), f32)}
    g = {'a': jnp.full((4,), 1e3, f32), 'b': jnp.full((4,), 1e-6, f32)}
    out = self.apply(p, g)
    assert np.linalg.norm(np.asarray(out['a'])) == pytest.approx(0.6, rel=1e-4)
    np.testing.assert_allclose(
        np.asarray(out['b']), np.asarray(g['b']), rtol=1e-6)

  def test_shapes_and_dtypes_are_preserved(self):
    p = {'w': jnp.ones((3, 5), f32)}
    g = {'w': jnp.ones((3, 5), f32) * 50}
    out = self.apply(p, g)['w']
    assert out.shape == (3, 5) and out.dtype == f32

  def test_a_zero_gradient_stays_zero_and_finite(self):
    p = {'w': jnp.ones((4,), f32)}
    g = {'w': jnp.zeros((4,), f32)}
    out = np.asarray(self.apply(p, g)['w'])
    assert np.all(out == 0.0) and np.all(np.isfinite(out))

  def test_the_threshold_scales_with_the_parameter_norm(self):
    """Big weights tolerate proportionally bigger updates."""
    small = self.apply({'w': jnp.ones((4,), f32)},
                       {'w': jnp.full((4,), 1e3, f32)})['w']
    large = self.apply({'w': jnp.full((4,), 10.0, f32)},
                       {'w': jnp.full((4,), 1e3, f32)})['w']
    assert np.linalg.norm(np.asarray(large)) == pytest.approx(
        10 * np.linalg.norm(np.asarray(small)), rel=1e-4)


class TestScaleByRms:

  def test_a_constant_gradient_converges_to_unit_scale(self):
    tx = scale_by_rms(beta=0.9, eps=1e-8)
    p = {'w': jnp.ones((3,), f32)}
    g = {'w': jnp.full((3,), 2.0, f32)}
    state = tx.init(p)
    for _ in range(200):
      out, state = tx.update(g, state, p)
    np.testing.assert_allclose(np.asarray(out['w']), 1.0, rtol=1e-3)

  def test_bias_correction_makes_the_first_step_unit_scale(self):
    tx = scale_by_rms(beta=0.999, eps=1e-12)
    p = {'w': jnp.ones((3,), f32)}
    g = {'w': jnp.full((3,), 7.0, f32)}
    out, _ = tx.update(g, tx.init(p), p)
    np.testing.assert_allclose(np.asarray(out['w']), 1.0, rtol=1e-4)

  def test_state_dtype_is_float32(self):
    tx = scale_by_rms()
    _, nu = tx.init({'w': jnp.ones((3,), jnp.bfloat16)})
    assert jax.tree.leaves(nu)[0].dtype == f32


class TestScaleByMomentum:

  def test_a_constant_gradient_converges_to_that_gradient(self):
    tx = scale_by_momentum(beta=0.9)
    p = {'w': jnp.ones((3,), f32)}
    g = {'w': jnp.full((3,), 3.0, f32)}
    state = tx.init(p)
    for _ in range(200):
      out, state = tx.update(g, state, p)
    np.testing.assert_allclose(np.asarray(out['w']), 3.0, rtol=1e-3)

  def test_bias_correction_makes_the_first_step_the_raw_gradient(self):
    tx = scale_by_momentum(beta=0.9)
    p = {'w': jnp.ones((3,), f32)}
    g = {'w': jnp.full((3,), 5.0, f32)}
    out, _ = tx.update(g, tx.init(p), p)
    np.testing.assert_allclose(np.asarray(out['w']), 5.0, rtol=1e-5)


def run_norm(impl, xs, weights=None, **kw):
  """Feed a sequence of batches through a Normalize and return final stats."""
  norm = Normalize(impl, name='norm', **kw)

  def fn(seq, w):
    for i in range(len(seq)):
      norm(seq[i], update=True, weights=None if w is None else w[i])
    return norm.stats()

  params = nj.init(fn)({}, xs, weights, seed=0)
  _, out = nj.pure(fn)(params, xs, weights, seed=0)
  return [float(np.asarray(v)) for v in out]


class TestNormalizeNone:

  def test_identity_offset_and_scale(self):
    xs = jnp.ones((3, 4, 5), f32)
    off, scale = run_norm('none', xs)
    assert (off, scale) == (0.0, 1.0)


class TestNormalizeMeanStd:

  def test_recovers_the_mean_and_std_of_a_stationary_stream(self):
    rng = np.random.default_rng(0)
    xs = jnp.asarray(rng.normal(5.0, 2.0, size=(400, 32)), f32)
    off, scale = run_norm('meanstd', xs, rate=0.05)
    assert off == pytest.approx(5.0, abs=0.3)
    assert scale == pytest.approx(2.0, abs=0.3)

  def test_the_scale_is_floored_by_limit(self):
    xs = jnp.full((50, 8), 3.0, f32)
    _, scale = run_norm('meanstd', xs, rate=0.1, limit=0.5)
    assert scale == pytest.approx(0.5)

  def test_weights_restrict_the_statistics_to_real_entries(self):
    """The padded-axis fix: forward-filled tails must not drag the mean."""
    xs = np.zeros((60, 10), np.float32)
    xs[:, :5] = 1.0     # real data
    xs[:, 5:] = 100.0   # padding
    w = np.zeros((60, 10), np.float32)
    w[:, :5] = 1.0
    off, _ = run_norm('meanstd', jnp.asarray(xs), jnp.asarray(w), rate=0.2)
    assert off == pytest.approx(1.0, abs=0.05)

  def test_without_weights_the_padding_does_drag_the_mean(self):
    xs = np.zeros((60, 10), np.float32)
    xs[:, :5] = 1.0
    xs[:, 5:] = 100.0
    off, _ = run_norm('meanstd', jnp.asarray(xs), None, rate=0.2)
    assert off > 10.0


class TestNormalizePercentile:
  """`retnorm: perc` -- the 5/95 percentile range used for every advantage."""

  def test_recovers_the_percentile_range_of_a_uniform_stream(self):
    rng = np.random.default_rng(1)
    xs = jnp.asarray(rng.uniform(0.0, 10.0, size=(400, 64)), f32)
    off, scale = run_norm('perc', xs, rate=0.05)
    assert off == pytest.approx(0.5, abs=0.6)     # 5th percentile
    assert scale == pytest.approx(9.0, abs=1.0)   # 95th - 5th

  def test_a_degenerate_all_equal_stream_is_floored_by_limit(self):
    """Early training: every return is ~0, so hi - lo collapses."""
    xs = jnp.zeros((80, 32), f32)
    off, scale = run_norm('perc', xs, rate=0.2, limit=1.0)
    assert scale == pytest.approx(1.0), (
        'the advantage scale must not fall below `limit`, or advantages '
        'explode when returns are degenerate')

  def test_the_scale_never_goes_below_limit_for_a_narrow_stream(self):
    rng = np.random.default_rng(2)
    xs = jnp.asarray(rng.normal(0.0, 1e-6, size=(80, 32)), f32)
    _, scale = run_norm('perc', xs, rate=0.2, limit=1.0)
    assert scale >= 1.0

  def test_the_scale_does_exceed_limit_once_returns_spread_out(self):
    rng = np.random.default_rng(3)
    xs = jnp.asarray(rng.normal(0.0, 50.0, size=(300, 64)), f32)
    _, scale = run_norm('perc', xs, rate=0.05, limit=1.0)
    assert scale > 10.0

  def test_offset_and_scale_are_stop_gradiented(self):
    """`perc` returns `sg(lo)` and `sg(hi - lo)`; no gradient into the stats."""
    import inspect
    src = inspect.getsource(Normalize.stats)
    perc = src.split("elif self.impl == 'perc':")[1]
    assert 'sg(lo)' in perc and 'sg(' in perc

  def test_weights_restrict_the_percentiles(self):
    xs = np.zeros((60, 20), np.float32)
    xs[:, :10] = np.linspace(0, 1, 10)
    xs[:, 10:] = 1000.0
    w = np.zeros((60, 20), np.float32)
    w[:, :10] = 1.0
    _, scale = run_norm('perc', jnp.asarray(xs), jnp.asarray(w), rate=0.2)
    assert scale < 10.0, 'padding leaked into the percentile range'


class TestNormalizeDebias:

  def test_debias_makes_the_first_update_unbiased(self):
    xs = jnp.full((1, 16), 4.0, f32)
    off, _ = run_norm('meanstd', xs, rate=0.01, debias=True)
    assert off == pytest.approx(4.0, rel=1e-3)

  def test_without_debias_the_first_update_is_shrunk_by_the_rate(self):
    xs = jnp.full((1, 16), 4.0, f32)
    off, _ = run_norm('meanstd', xs, rate=0.01, debias=False)
    assert off == pytest.approx(0.04, rel=1e-3)

  def test_an_unknown_impl_is_rejected(self):
    with pytest.raises(NotImplementedError):
      run_norm('bogus', jnp.zeros((2, 3), f32))
