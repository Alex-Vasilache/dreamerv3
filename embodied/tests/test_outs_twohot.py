"""Numerics of the output distributions in `embodied/jax/outs.py`.

`TwoHot` had no test anywhere, despite being the reward head *and* both critic
heads (`symexp_twohot`, 255 bins). Everything the agent believes about reward
and value passes through `pred()` and `loss()` here. `Binary`, `Normal`,
`Huber`, `Frozen` and `Concat` were also uncovered.

Checked: the symmetric-sum trick that makes `pred()` exactly zero at init, the
two-hot interpolation weights, behaviour outside the bin range, gradient
routing, and dtype/shape handling.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import embodied.jax.outs as outs

f32 = jnp.float32


def bins(n, lim=1.0):
  """Bins built the way `heads.py::symexp_twohot` builds them.

  It mirrors a half-range rather than calling `linspace(-lim, lim, n)`, which
  makes every positive bin the exact floating-point negation of its negative
  counterpart. `TwoHot.pred()`'s symmetric-sum trick depends on that: with a
  naive single linspace the cancellation is only approximate and the head
  predicts ~1e-8 instead of 0 at initialisation.
  """
  if n % 2 == 1:
    half = jnp.linspace(-lim, 0, (n - 1) // 2 + 1, dtype=f32)
    return jnp.concatenate([half, -half[:-1][::-1]], 0)
  half = jnp.linspace(-lim, 0, n // 2, dtype=f32)
  return jnp.concatenate([half, -half[::-1]], 0)


def naive_bins(n, lim=1.0):
  return jnp.linspace(-lim, lim, n, dtype=f32)


class TestTwoHotPred:

  @pytest.mark.parametrize('n', [3, 5, 8, 255])
  def test_uniform_logits_over_symmetric_bins_predict_exactly_zero(self, n):
    """The reason `pred()` uses a symmetric sum instead of a plain dot product."""
    d = outs.TwoHot(jnp.zeros((4, n)), bins(n))
    got = np.asarray(d.pred())
    assert np.all(got == 0.0), got

  @pytest.mark.parametrize('n', [8, 255])
  def test_the_exact_zero_depends_on_mirrored_bins(self, n):
    """`linspace(-lim, lim, n)` is not exactly symmetric, and breaks the trick.

    This is why `heads.py::symexp_twohot` mirrors a half-range. If that is ever
    simplified to a single linspace, the reward head and both critics start
    predicting a small nonzero value at initialisation.
    """
    exact = np.asarray(outs.TwoHot(jnp.zeros((1, n)), bins(n)).pred())
    naive = np.asarray(outs.TwoHot(jnp.zeros((1, n)), naive_bins(n)).pred())
    assert np.all(exact == 0.0)
    assert np.any(naive != 0.0), (
        'naive linspace happened to cancel exactly here; the mirrored '
        'construction is still the one to rely on')

  @pytest.mark.parametrize('n', [3, 5, 8, 255])
  def test_mirrored_bins_are_exactly_antisymmetric(self, n):
    b = np.asarray(bins(n))
    np.testing.assert_array_equal(b, -b[::-1])

  def test_all_mass_on_one_bin_predicts_that_bin(self):
    n = 7
    for i in range(n):
      logits = jnp.full((n,), -1e9).at[i].set(1e9)
      got = float(np.asarray(outs.TwoHot(logits, bins(n)).pred()))
      assert got == pytest.approx(float(bins(n)[i]), abs=1e-5)

  def test_prediction_is_the_probability_weighted_bin_mean(self):
    rng = np.random.default_rng(0)
    n = 9
    logits = jnp.asarray(rng.normal(size=(3, n)), f32)
    d = outs.TwoHot(logits, bins(n))
    want = np.asarray(jax.nn.softmax(logits) * bins(n)).sum(-1)
    np.testing.assert_allclose(np.asarray(d.pred()), want, atol=1e-6)

  def test_unsquash_is_applied_to_the_prediction(self):
    n = 5
    d = outs.TwoHot(jnp.zeros((n,)), bins(n),
                    squash=lambda x: x, unsquash=lambda x: x + 10.0)
    assert float(np.asarray(d.pred())) == pytest.approx(10.0)

  def test_shapes_are_preserved_minus_the_bin_axis(self):
    d = outs.TwoHot(jnp.zeros((2, 3, 5)), bins(5))
    assert d.pred().shape == (2, 3)

  def test_probs_sum_to_one(self):
    rng = np.random.default_rng(1)
    d = outs.TwoHot(jnp.asarray(rng.normal(size=(4, 11)), f32), bins(11))
    np.testing.assert_allclose(np.asarray(d.probs).sum(-1), 1.0, rtol=1e-6)


class TestTwoHotLoss:

  def test_a_target_exactly_on_a_bin_is_a_one_hot_target(self):
    n = 5
    b = bins(n)
    logits = jnp.zeros((n,))
    d = outs.TwoHot(logits, b)
    loss = float(np.asarray(d.loss(jnp.asarray(b[2], f32))))
    # Uniform logits -> -log(1/n) for a pure one-hot target.
    assert loss == pytest.approx(float(np.log(n)), rel=1e-5)

  def test_a_target_midway_between_bins_splits_mass_evenly(self):
    n = 5
    b = bins(n)
    mid = jnp.asarray((b[1] + b[2]) / 2, f32)
    # Put all mass on bin 1: the loss should equal that of all mass on bin 2.
    l1 = outs.TwoHot(jnp.zeros(n).at[1].set(5.0), b).loss(mid)
    l2 = outs.TwoHot(jnp.zeros(n).at[2].set(5.0), b).loss(mid)
    assert float(np.asarray(l1)) == pytest.approx(float(np.asarray(l2)), rel=1e-5)

  def test_interpolation_weights_are_linear_in_the_target(self):
    """A target 1/4 of the way from bin i to bin i+1 gets weights (3/4, 1/4)."""
    n = 5
    b = bins(n)
    t = jnp.asarray(b[1] + 0.25 * (b[2] - b[1]), f32)
    # Compare against a hand-built two-hot target through the same NLL.
    logits = jnp.asarray(np.random.default_rng(2).normal(size=n), f32)
    d = outs.TwoHot(logits, b)
    logp = np.asarray(jax.nn.log_softmax(logits))
    want = -(0.75 * logp[1] + 0.25 * logp[2])
    assert float(np.asarray(d.loss(t))) == pytest.approx(want, rel=1e-5)

  def test_loss_is_minimised_when_the_prediction_matches_the_target(self):
    n = 9
    b = bins(n)
    t = jnp.asarray(b[4], f32)
    good = outs.TwoHot(jnp.zeros(n).at[4].set(10.0), b).loss(t)
    bad = outs.TwoHot(jnp.zeros(n).at[0].set(10.0), b).loss(t)
    assert float(np.asarray(good)) < float(np.asarray(bad))

  def test_loss_is_non_negative(self):
    rng = np.random.default_rng(3)
    n = 11
    d = outs.TwoHot(jnp.asarray(rng.normal(size=(5, n)), f32), bins(n))
    t = jnp.asarray(rng.uniform(-1, 1, size=5), f32)
    assert np.all(np.asarray(d.loss(t)) >= 0.0)

  def test_a_target_below_the_lowest_bin_is_clipped_not_nan(self):
    n = 7
    d = outs.TwoHot(jnp.zeros((n,)), bins(n))
    got = float(np.asarray(d.loss(jnp.asarray(-5.0, f32))))
    assert np.isfinite(got)

  def test_a_target_above_the_highest_bin_is_clipped_not_nan(self):
    n = 7
    d = outs.TwoHot(jnp.zeros((n,)), bins(n))
    got = float(np.asarray(d.loss(jnp.asarray(5.0, f32))))
    assert np.isfinite(got)

  def test_out_of_range_targets_collapse_onto_the_end_bin(self):
    n = 7
    b = bins(n)
    lo = outs.TwoHot(jnp.zeros(n).at[0].set(10.0), b)
    assert float(np.asarray(lo.loss(jnp.asarray(-5.0, f32)))) == pytest.approx(
        float(np.asarray(lo.loss(jnp.asarray(float(b[0]), f32)))), rel=1e-4)

  def test_the_target_is_stop_gradiented(self):
    n = 7
    d = outs.TwoHot(jnp.zeros((n,)), bins(n))
    g = np.asarray(jax.grad(lambda t: d.loss(t).sum())(jnp.asarray(0.3, f32)))
    assert g == 0.0, 'the regression target must not receive gradient'

  def test_gradient_reaches_the_logits(self):
    n = 7
    b = bins(n)
    fn = lambda lg: outs.TwoHot(lg, b).loss(jnp.asarray(0.3, f32)).sum()
    g = np.asarray(jax.grad(fn)(jnp.zeros((n,), f32)))
    assert np.any(g != 0.0)

  def test_loss_shape_matches_the_target_shape(self):
    n = 7
    d = outs.TwoHot(jnp.zeros((2, 3, n)), bins(n))
    assert d.loss(jnp.zeros((2, 3), f32)).shape == (2, 3)

  def test_a_non_float32_target_is_rejected(self):
    n = 5
    d = outs.TwoHot(jnp.zeros((n,)), bins(n))
    with pytest.raises(AssertionError):
      d.loss(jnp.zeros((), jnp.bfloat16))

  def test_under_uniform_logits_the_loss_is_log_n_for_any_target(self):
    """The two-hot weights sum to 1, so a uniform predictor pays log(n) flat.

    Worth pinning because it makes uniform logits useless for any test that
    tries to tell two in-range targets apart.
    """
    n = 5
    b = bins(n)
    d = outs.TwoHot(jnp.zeros((n,)), b)
    for t in (0.0, 0.3, -0.75, 1.0):
      got = float(np.asarray(d.loss(jnp.asarray(t, f32))))
      assert got == pytest.approx(float(np.log(n)), rel=1e-5), t

  def test_squash_is_applied_to_the_target(self):
    n = 5
    b = bins(n)
    # Non-uniform logits: see the test above for why uniform ones cannot
    # distinguish two in-range targets.
    logits = jnp.asarray(np.random.default_rng(4).normal(size=n), f32)
    target = jnp.asarray(0.3, f32)
    plain = outs.TwoHot(logits, b).loss(target)
    squashed = outs.TwoHot(
        logits, b, squash=lambda x: x * 0.0).loss(target)
    at_zero = outs.TwoHot(logits, b).loss(jnp.asarray(0.0, f32))
    assert float(np.asarray(squashed)) == pytest.approx(
        float(np.asarray(at_zero)), rel=1e-6)
    assert abs(float(np.asarray(plain)) - float(np.asarray(at_zero))) > 1e-4


class TestSymexpTwohotRoundTrip:
  """The configured reward/critic head: symlog squash with 255 bins."""

  @staticmethod
  def head(logits):
    n = logits.shape[-1]
    b = jnp.linspace(-20.0, 20.0, n, dtype=f32)
    symlog = lambda x: jnp.sign(x) * jnp.log1p(jnp.abs(x))
    symexp = lambda x: jnp.sign(x) * jnp.expm1(jnp.abs(x))
    return outs.TwoHot(logits, b, squash=symlog, unsquash=symexp)

  def test_prediction_is_zero_at_initialisation(self):
    d = self.head(jnp.zeros((4, 255)))
    np.testing.assert_allclose(np.asarray(d.pred()), 0.0, atol=1e-6)

  @pytest.mark.parametrize('target', [0.0, 1.0, -1.0, 37.5, -250.0])
  def test_fitting_a_target_recovers_it_through_the_squash(self, target):
    """Optimise the logits directly and check `pred()` converges to the target."""
    b = jnp.linspace(-20.0, 20.0, 255, dtype=f32)
    symlog = lambda x: jnp.sign(x) * jnp.log1p(jnp.abs(x))
    symexp = lambda x: jnp.sign(x) * jnp.expm1(jnp.abs(x))
    t = jnp.asarray(target, f32)

    def loss(lg):
      return outs.TwoHot(lg, b, squash=symlog, unsquash=symexp).loss(t).sum()

    lg = jnp.zeros((255,), f32)
    for _ in range(400):
      lg = lg - 2.0 * jax.grad(loss)(lg)
    got = float(np.asarray(
        outs.TwoHot(lg, b, squash=symlog, unsquash=symexp).pred()))
    assert got == pytest.approx(target, rel=0.02, abs=1e-3)

  def test_very_large_targets_are_recovered_only_approximately(self):
    """`pred()` symexps an average taken in symlog space, which is biased.

    At |target| ~ 1e4 the 255 bins over [-20, 20] are ~0.157 apart in symlog
    space, and symexp of the interpolated value is not the interpolation of
    the symexps. Measured error is ~2.3% -- fine for a reward/value head, but
    it means `pred()` is not an unbiased estimator at the extremes.
    """
    b = jnp.linspace(-20.0, 20.0, 255, dtype=f32)
    symlog = lambda x: jnp.sign(x) * jnp.log1p(jnp.abs(x))
    symexp = lambda x: jnp.sign(x) * jnp.expm1(jnp.abs(x))
    t = jnp.asarray(1e4, f32)
    loss = lambda lg: outs.TwoHot(lg, b, squash=symlog, unsquash=symexp).loss(t).sum()
    lg = jnp.zeros((255,), f32)
    for _ in range(400):
      lg = lg - 2.0 * jax.grad(loss)(lg)
    got = float(np.asarray(
        outs.TwoHot(lg, b, squash=symlog, unsquash=symexp).pred()))
    assert got == pytest.approx(1e4, rel=0.05)

  def test_large_magnitudes_stay_finite(self):
    d = self.head(jnp.zeros((255,)))
    assert np.isfinite(float(np.asarray(d.loss(jnp.asarray(1e8, f32)))))


class TestOtherOutputs:

  def test_mse_pred_and_loss(self):
    d = outs.MSE(jnp.asarray([1.0, 2.0], f32))
    np.testing.assert_allclose(np.asarray(d.pred()), [1.0, 2.0])
    np.testing.assert_allclose(
        np.asarray(d.loss(jnp.asarray([0.0, 0.0], f32))), [1.0, 4.0])

  def test_huber_is_linear_far_from_the_target(self):
    d = outs.Huber(jnp.asarray([100.0], f32))
    a = float(np.asarray(d.loss(jnp.asarray([0.0], f32)))[0])
    d2 = outs.Huber(jnp.asarray([200.0], f32))
    b = float(np.asarray(d2.loss(jnp.asarray([0.0], f32)))[0])
    assert b < 2.5 * a, 'Huber must grow sub-quadratically'

  def test_binary_loss_is_minimised_at_the_right_label(self):
    d = outs.Binary(jnp.asarray([5.0], f32))
    pos = float(np.asarray(d.loss(jnp.asarray([1.0], f32)))[0])
    neg = float(np.asarray(d.loss(jnp.asarray([0.0], f32)))[0])
    assert pos < neg

  def test_binary_pred_is_a_thresholded_probability(self):
    assert bool(np.asarray(outs.Binary(jnp.asarray([5.0], f32)).pred())[0])
    assert not bool(np.asarray(outs.Binary(jnp.asarray([-5.0], f32)).pred())[0])

  def test_normal_loss_is_minimised_at_the_mean(self):
    d = outs.Normal(jnp.zeros((1,), f32))
    at = float(np.asarray(d.loss(jnp.zeros((1,), f32)))[0])
    off = float(np.asarray(d.loss(jnp.ones((1,), f32)))[0])
    assert at < off

  def test_onehot_sample_is_a_valid_onehot(self):
    d = outs.OneHot(jnp.zeros((4, 6), f32))
    s = np.asarray(d.sample(jax.random.PRNGKey(0)), np.float32)
    np.testing.assert_allclose(s.sum(-1), 1.0, rtol=1e-6)
    assert set(np.unique(s)).issubset({0.0, 1.0})

  def test_onehot_sample_has_the_same_shape_as_pred(self):
    d = outs.OneHot(jnp.zeros((4, 6), f32))
    assert d.sample(jax.random.PRNGKey(0)).shape == d.pred().shape

  def test_onehot_entropy_is_maximal_for_uniform_logits(self):
    uni = float(np.asarray(outs.OneHot(jnp.zeros((6,), f32)).entropy()))
    peaked = float(np.asarray(
        outs.OneHot(jnp.zeros(6).at[0].set(10.0).astype(f32)).entropy()))
    assert uni == pytest.approx(float(np.log(6)), rel=1e-4)
    assert peaked < uni

  def test_categorical_logp_matches_log_softmax(self):
    logits = jnp.asarray([1.0, 2.0, 3.0], f32)
    d = outs.Categorical(logits)
    want = float(np.asarray(jax.nn.log_softmax(logits))[2])
    got = float(np.asarray(d.logp(jnp.asarray(2, jnp.int32))))
    assert got == pytest.approx(want, rel=1e-5)
