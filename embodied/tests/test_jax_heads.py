"""`embodied/jax/heads.py` -- the heads that produce every prediction.

`Head` builds the reward head, both critics, the actor and the goal
encoder/decoder. `test_outs_twohot.py` covers the `TwoHot` distribution itself;
this file covers the head that constructs it -- in particular the mirrored bin
range that `TwoHot.pred()`'s exact-zero-at-init property depends on, and the
`minent`/`maxent` constants the entropy controller normalizes by.
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

import elements
import embodied.jax.heads as jheads
import embodied.jax.nets as nn

f32 = jnp.float32
CD = nn.COMPUTE_DTYPE


def run(space, output, x_shape=(2, 16), **kw):
  head = jheads.Head(space, output, name='head', **kw)
  fn = lambda x: head(x)
  x = jnp.asarray(np.random.default_rng(0).normal(size=x_shape), CD)
  params = nj.init(fn)({}, x, seed=0)
  return nj.pure(fn)(params, x, seed=0)[1]


class TestSymexpTwohotBins:
  """The reward head and both critics. `output: symexp_twohot`, `bins: 255`."""

  def test_bins_are_exactly_antisymmetric(self):
    out = run(elements.Space(np.float32, ()), 'symexp_twohot')
    b = np.asarray(out.bins)
    np.testing.assert_array_equal(b, -b[::-1])

  def test_prediction_is_exactly_zero_at_initialisation(self):
    """`outscale` defaults to 0.0 for the reward head, so logits start uniform."""
    out = run(elements.Space(np.float32, ()), 'symexp_twohot', outscale=0.0)
    got = np.asarray(out.pred(), np.float32)
    assert np.all(got == 0.0), got

  def test_the_bin_count_matches_the_config(self):
    out = run(elements.Space(np.float32, ()), 'symexp_twohot', bins=255)
    assert out.bins.shape == (255,)

  def test_an_even_bin_count_is_also_antisymmetric(self):
    out = run(elements.Space(np.float32, ()), 'symexp_twohot', bins=64)
    b = np.asarray(out.bins)
    assert b.shape == (64,)
    np.testing.assert_array_equal(b, -b[::-1])

  def test_bins_live_in_raw_reward_space_not_symlog_space(self):
    """`symexp_twohot` symexps the bins; it does NOT squash the target.

    `heads.py` builds `linspace(-20, 0, ...)`, applies `symexp`, mirrors it, and
    passes it to `TwoHot(logits, bins)` with no `squash`/`unsquash`. So the
    two-hot interpolation happens in raw reward units over geometrically spaced
    bins -- fine resolution near zero, coarse far away -- rather than in symlog
    units over uniform bins. Easy to assume the other way round.
    """
    out = run(elements.Space(np.float32, ()), 'symexp_twohot')
    b = np.asarray(out.bins)
    assert b[-1] == pytest.approx(float(np.expm1(20.0)), rel=1e-4)
    assert b[0] == pytest.approx(-float(np.expm1(20.0)), rel=1e-4)
    assert float(b[-1]) > 1e8

  def test_bin_spacing_is_fine_near_zero_and_coarse_far_away(self):
    out = run(elements.Space(np.float32, ()), 'symexp_twohot')
    b = np.asarray(out.bins, np.float64)
    mid = len(b) // 2
    near = b[mid + 1] - b[mid]
    far = b[-1] - b[-2]
    assert near < 0.5, near
    assert far > 1e6, far
    assert far / near > 1e6

  def test_the_head_applies_no_squash(self):
    out = run(elements.Space(np.float32, ()), 'symexp_twohot')
    assert out.squash(jnp.asarray(3.7, f32)) == pytest.approx(3.7)
    assert out.unsquash(jnp.asarray(3.7, f32)) == pytest.approx(3.7)

  def test_bin_spacing_at_a_dmc_scale_return_is_tens_of_units(self):
    """Records the actual resolution, which is coarse at hopper's scale.

    At a return of 300 the neighbouring bins are ~49 apart. That does NOT make
    the critic's prediction quantized: the two-hot target splits mass between
    the two adjacent bins in proportion to the distance, so its expectation --
    and therefore the optimum of the cross-entropy -- is the exact target. The
    spacing bounds resolution of the *distribution*, not of `pred()`.
    """
    out = run(elements.Space(np.float32, ()), 'symexp_twohot')
    b = np.asarray(out.bins, np.float64)
    i = int(np.searchsorted(b, 300.0))
    gap = b[i] - b[i - 1]
    assert 20.0 < gap < 100.0, gap

  def test_the_two_hot_mean_is_exact_between_widely_spaced_bins(self):
    """The property that makes the coarse spacing above harmless."""
    import embodied.jax.outs as outs
    bins = jnp.asarray([0.0, 100.0], f32)
    for target in (0.0, 25.0, 50.0, 99.0, 100.0):
      t = jnp.asarray(target, f32)
      # Fit two logits to the two-hot target and check pred() recovers it.
      lg = jnp.zeros((2,), f32)
      loss = lambda l: outs.TwoHot(l, bins).loss(t).sum()
      for _ in range(500):
        lg = lg - 0.5 * jax.grad(loss)(lg)
      got = float(np.asarray(outs.TwoHot(lg, bins).pred()))
      assert got == pytest.approx(target, abs=0.5), (target, got)

  def test_loss_against_a_realistic_return_is_finite(self):
    out = run(elements.Space(np.float32, ()), 'symexp_twohot')
    loss = out.loss(jnp.full((2,), 350.0, f32))
    assert np.all(np.isfinite(np.asarray(loss)))


class TestEntropyBounds:
  """`minent`/`maxent` are the denominators the entropy controller uses."""

  def test_onehot_maxent_is_blocks_times_log_classes(self):
    out = run(elements.Space(np.float32, (8, 8), 0.0, 1.0), 'onehot')
    inner = out.output if hasattr(out, 'output') else out
    assert inner.maxent == pytest.approx(float(8 * np.log(8)))
    assert inner.minent == 0.0

  @pytest.mark.parametrize('blocks,classes', [(8, 8), (4, 16), (1, 6), (8, 4)])
  def test_maxent_scales_with_both_blocks_and_classes(self, blocks, classes):
    out = run(elements.Space(np.float32, (blocks, classes), 0.0, 1.0), 'onehot')
    inner = out.output if hasattr(out, 'output') else out
    assert inner.maxent == pytest.approx(float(blocks * np.log(classes)))

  def test_a_single_block_has_no_outer_factor(self):
    out = run(elements.Space(np.float32, (6,), 0.0, 1.0), 'onehot')
    inner = out.output if hasattr(out, 'output') else out
    assert inner.maxent == pytest.approx(float(np.log(6)))

  def test_a_uniform_onehot_head_sits_at_maxent(self):
    """Fresh heads start near uniform, so normalized entropy starts near 1."""
    out = run(elements.Space(np.float32, (8, 8), 0.0, 1.0), 'onehot', outscale=0.0)
    inner = out.output if hasattr(out, 'output') else out
    ent = float(np.asarray(inner.entropy()).sum(-1).mean())
    assert ent == pytest.approx(inner.maxent, rel=1e-4)

  def test_categorical_maxent_is_log_classes(self):
    out = run(elements.Space(np.int32, (), 0, 6), 'categorical')
    inner = out.output if hasattr(out, 'output') else out
    assert inner.maxent == pytest.approx(float(np.log(6)))


class TestShapesAndAggregation:

  def test_a_scalar_space_gives_a_scalar_prediction(self):
    out = run(elements.Space(np.float32, ()), 'symexp_twohot')
    assert out.pred().shape == (2,)

  def test_a_vector_space_is_aggregated_over_its_own_dims(self):
    out = run(elements.Space(np.float32, (5,)), 'mse')
    assert out.pred().shape == (2, 5)
    assert out.loss(jnp.zeros((2, 5), f32)).shape == (2,)

  def test_an_onehot_space_keeps_its_class_axis_in_pred(self):
    out = run(elements.Space(np.float32, (8, 8), 0.0, 1.0), 'onehot')
    assert out.pred().shape == (2, 8, 8)

  def test_a_time_axis_passes_through(self):
    out = run(elements.Space(np.float32, ()), 'symexp_twohot', x_shape=(2, 7, 16))
    assert out.pred().shape == (2, 7)

  def test_binary_head_shape(self):
    out = run(elements.Space(bool, ()), 'binary')
    assert out.pred().shape == (2,)

  def test_an_unknown_output_is_rejected(self):
    with pytest.raises(NotImplementedError):
      run(elements.Space(np.float32, ()), 'quantile')


class TestOutscale:
  """`outscale: 0.0` is what makes a fresh head predict exactly its prior."""

  def test_zero_outscale_gives_identical_logits_everywhere(self):
    out = run(elements.Space(np.float32, (4, 6), 0.0, 1.0), 'onehot',
              outscale=0.0, x_shape=(3, 16))
    inner = out.output if hasattr(out, 'output') else out
    lg = np.asarray(inner.logits if hasattr(inner, 'logits') else inner.dist.logits,
                    np.float32)
    # Every batch element sees the same (uniform) distribution.
    np.testing.assert_allclose(lg[0], lg[1], atol=1e-6)

  def test_nonzero_outscale_makes_the_head_input_dependent(self):
    out = run(elements.Space(np.float32, (4, 6), 0.0, 1.0), 'onehot',
              outscale=1.0, x_shape=(3, 16))
    inner = out.output if hasattr(out, 'output') else out
    lg = np.asarray(inner.logits if hasattr(inner, 'logits') else inner.dist.logits,
                    np.float32)
    assert not np.allclose(lg[0], lg[1], atol=1e-6)


class TestUnimix:
  """`unimix` floors the actor's probabilities so no action becomes impossible."""

  def test_unimix_zero_leaves_a_peaked_head_peaked(self):
    out = run(elements.Space(np.float32, (1, 4), 0.0, 1.0), 'onehot',
              unimix=0.0, outscale=1.0)
    inner = out.output if hasattr(out, 'output') else out
    p = np.asarray(jax.nn.softmax(
        inner.logits if hasattr(inner, 'logits') else inner.dist.logits), np.float32)
    np.testing.assert_allclose(p.sum(-1), 1.0, rtol=1e-5)

  def test_unimix_keeps_probabilities_normalised(self):
    out = run(elements.Space(np.float32, (1, 4), 0.0, 1.0), 'onehot',
              unimix=0.01, outscale=1.0)
    inner = out.output if hasattr(out, 'output') else out
    p = np.asarray(jax.nn.softmax(
        inner.logits if hasattr(inner, 'logits') else inner.dist.logits), np.float32)
    np.testing.assert_allclose(p.sum(-1), 1.0, rtol=1e-5)


class TestGradients:

  def test_gradient_reaches_the_head_parameters(self):
    head = jheads.Head(elements.Space(np.float32, ()), 'symexp_twohot',
                       name='head')
    def fn(x):
      return head(x).loss(jnp.ones((2,), f32)).sum()
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 16)), CD)
    params = nj.init(fn)({}, x, seed=0)
    g = nj.pure(jax.grad(fn))(params, x, seed=0)[1]
    assert np.any(np.abs(np.asarray(g, np.float32)) > 0)

  def test_the_regression_target_receives_no_gradient(self):
    head = jheads.Head(elements.Space(np.float32, (5,)), 'mse', name='head')
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 16)), CD)
    def fn(x, t):
      return head(x).loss(t).sum()
    params = nj.init(fn)({}, x, jnp.zeros((2, 5), f32), seed=0)
    gfn = lambda t: nj.pure(fn)(params, x, t, seed=0)[1]
    g = np.asarray(jax.grad(gfn)(jnp.ones((2, 5), f32)))
    assert np.all(g == 0.0)
