"""Parameter initialization scale (`embodied/jax/nets.py::Initializer`).

Every weight in the world model, the actor, both critics and the goal
autoencoder comes from here (`winit: trunc_normal_in` throughout). A
mis-scaled init is one of the few things that could plausibly make outcomes
seed-dependent -- some seeds would start in a much worse region than others --
so the variance investigation should either confirm the scaling or rule it out.
This file confirms it: the realised standard deviation is `1/sqrt(fan_in)` to
within a few percent at every layer width used in production.

`1.1368` is the correction factor for truncating a standard normal at +-2
sigma, whose remaining mass has std ~0.8796; 1/0.8796 = 1.1368.
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

import embodied.jax.nets as nn

f32 = jnp.float32


def sample(dist='trunc_normal', fan='in', scale=1.0, shape=(1024, 1024), seed=0):
  initr = nn.Initializer(dist, fan, scale)
  fn = lambda: initr(shape, f32)
  params = nj.init(fn)({}, seed=seed)
  return np.asarray(nj.pure(fn)(params, seed=seed)[1], np.float64)


class TestTruncNormalIn:
  """The production initializer."""

  @pytest.mark.parametrize('fan_in', [64, 512, 1024, 8192])
  def test_std_is_one_over_sqrt_fan_in(self, fan_in):
    x = sample(shape=(fan_in, 512))
    assert x.std() == pytest.approx(1.0 / np.sqrt(fan_in), rel=0.05)

  def test_the_truncation_correction_is_what_makes_that_true(self):
    """Without the 1.1368 factor the realised std would be ~12% low."""
    x = sample(shape=(1024, 1024))
    raw_std = x.std() / 1.1368 * np.sqrt(1024)
    assert raw_std == pytest.approx(0.8796, rel=0.02)

  def test_values_are_truncated_at_two_sigma(self):
    x = sample(shape=(1024, 1024))
    limit = 2 * 1.1368 * np.sqrt(1 / 1024)
    assert np.abs(x).max() <= limit * 1.0001

  def test_the_mean_is_zero(self):
    x = sample(shape=(1024, 1024))
    assert abs(x.mean()) < 5e-4

  def test_fan_in_is_the_leading_axis_for_a_matrix(self):
    """Kernels are (in, out); scaling must follow the input axis."""
    a = sample(shape=(4096, 64))
    b = sample(shape=(64, 4096))
    assert a.std() < b.std()
    assert a.std() == pytest.approx(1 / np.sqrt(4096), rel=0.05)
    assert b.std() == pytest.approx(1 / np.sqrt(64), rel=0.05)

  def test_different_seeds_give_different_draws(self):
    a = sample(seed=0)
    b = sample(seed=1)
    assert not np.allclose(a, b)

  def test_but_the_scale_is_seed_independent(self):
    """Seeds must differ in draw, not in scale -- otherwise init is a variance source."""
    stds = [sample(shape=(1024, 1024), seed=s).std() for s in range(5)]
    assert max(stds) / min(stds) < 1.02, stds


class TestOtherDistributions:

  def test_zeros(self):
    assert np.all(sample('zeros', shape=(16, 16)) == 0.0)

  def test_uniform_respects_its_limit(self):
    x = sample('uniform', shape=(512, 512))
    limit = np.sqrt(1 / 512)
    assert np.abs(x).max() <= limit * 1.0001
    assert x.std() == pytest.approx(limit / np.sqrt(3), rel=0.05)

  def test_normal_has_the_unscaled_std(self):
    x = sample('normal', shape=(512, 512))
    assert x.std() == pytest.approx(1 / np.sqrt(512), rel=0.05)

  def test_normed_gives_unit_column_norms(self):
    x = sample('normed', shape=(64, 32))
    norms = np.linalg.norm(x.reshape((-1, 32)), 2, 0)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-4)

  def test_an_unknown_distribution_is_rejected(self):
    with pytest.raises(NotImplementedError):
      sample('cauchy', shape=(8, 8))


class TestFanModes:

  def test_fan_out_uses_the_trailing_axis(self):
    x = sample(fan='out', shape=(64, 4096))
    assert x.std() == pytest.approx(1 / np.sqrt(4096), rel=0.05)

  def test_fan_avg_is_between_in_and_out(self):
    a = sample(fan='in', shape=(64, 4096)).std()
    b = sample(fan='out', shape=(64, 4096)).std()
    m = sample(fan='avg', shape=(64, 4096)).std()
    assert min(a, b) < m < max(a, b)

  def test_fan_none_applies_no_scaling(self):
    x = sample(fan='none', shape=(1024, 1024))
    assert x.std() == pytest.approx(0.8796 * 1.1368, rel=0.05)


class TestScale:

  def test_scale_multiplies_the_result(self):
    a = sample(scale=1.0, shape=(512, 512))
    b = sample(scale=0.1, shape=(512, 512))
    assert b.std() == pytest.approx(a.std() * 0.1, rel=0.05)

  def test_outscale_zero_produces_an_exactly_constant_head(self):
    """`outscale: 0.0` on the reward head is what makes pred() exactly zero."""
    x = sample(scale=0.0, shape=(256, 256))
    assert np.all(x == 0.0)


class TestShapeValidation:

  def test_a_zero_sized_axis_is_rejected(self):
    with pytest.raises(AssertionError):
      sample(shape=(0, 8))

  def test_a_non_integer_shape_is_rejected(self):
    with pytest.raises(AssertionError):
      sample(shape=(8.0, 8))

  def test_an_int_shape_is_promoted_to_a_tuple(self):
    assert sample(shape=64).shape == (64,)

  def test_dtype_is_honoured(self):
    initr = nn.Initializer('trunc_normal', 'in', 1.0)
    fn = lambda: initr((8, 8), jnp.bfloat16)
    params = nj.init(fn)({}, seed=0)
    assert nj.pure(fn)(params, seed=0)[1].dtype == jnp.bfloat16


class TestSignalPropagation:
  """The point of fan-in scaling: activations keep their scale through depth."""

  def test_a_deep_stack_neither_explodes_nor_vanishes(self):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 1024))
    for _ in range(8):
      w = sample(shape=(x.shape[-1], 1024), seed=int(rng.integers(1 << 20)))
      x = np.maximum(x @ w, 0.0) * np.sqrt(2.0)  # relu + gain
    assert 0.2 < x.std() < 5.0, x.std()

  def test_without_fan_scaling_the_same_stack_explodes(self):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 1024))
    for _ in range(8):
      w = sample(fan='none', shape=(x.shape[-1], 1024),
                 seed=int(rng.integers(1 << 20)))
      x = np.maximum(x @ w, 0.0) * np.sqrt(2.0)
    assert x.std() > 100.0, x.std()
