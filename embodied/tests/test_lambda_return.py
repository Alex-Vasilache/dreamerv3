"""`lambda_return` against a transcription of TF Director's `gve`.

Returns are where an off-by-one in reward/value alignment hides best: the shapes
stay valid, training still runs, and the only symptom is a biased target. This
file pins our implementation to the original by differential test rather than by
reading the two side by side.

Reference: `code/director/embodied/agents/director/agent.py::VFunction.target`,
`impl == 'gve'`, and `tfutils.lambda_return`.
"""

import numpy as np
import pytest

from dreamerv3.hrl.losses import lambda_return


def director_gve(reward, value, disc, lam):
  """Verbatim transcription of Director's `gve` branch, in numpy.

  `reward` and `disc` have length T-1; `value` has length T. Returns length T-1.
  """
  vals = [value[-1]]
  interm = reward + disc * value[1:] * (1 - lam)
  for t in reversed(range(len(disc))):
    vals.append(interm[t] + disc[t] * lam * vals[-1])
  return np.stack(list(reversed(vals))[:-1])


def ours(rew, val, cont, lam, disc, last=None, term=None):
  """Call our implementation with Director's convention laid out on axis 1.

  Ours indexes rewards from 1 (reward aligned with the state it leads to), so
  Director's `reward[t]` is our `rew[t + 1]`.
  """
  T = len(val)
  rew_a = np.zeros((1, T), np.float32)
  rew_a[0, 1:] = rew
  term_a = np.zeros((1, T), np.float32)
  if term is not None:
    term_a[0, 1:] = term
  last_a = np.zeros((1, T), np.float32)
  if last is not None:
    last_a[0, 1:] = last
  val_a = val[None].astype(np.float32)
  out = lambda_return(
      last_a, term_a, rew_a, val_a, val_a, np.float32(disc), np.float32(lam))
  return np.asarray(out)[0]


class TestMatchesDirector:

  @pytest.mark.parametrize('lam', [0.0, 0.5, 0.95, 1.0])
  @pytest.mark.parametrize('disc', [0.99, 0.997])
  def test_equals_gve_with_constant_discount(self, lam, disc):
    rng = np.random.default_rng(0)
    T = 17
    val = rng.normal(size=T).astype(np.float32)
    rew = rng.normal(size=T - 1).astype(np.float32)
    want = director_gve(rew, val, np.full(T - 1, disc, np.float32), lam)
    got = ours(rew, val, None, lam, disc)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

  def test_equals_gve_with_a_terminal_partway_through(self):
    rng = np.random.default_rng(1)
    T, lam, disc = 12, 0.95, 0.99
    val = rng.normal(size=T).astype(np.float32)
    rew = rng.normal(size=T - 1).astype(np.float32)
    term = np.zeros(T - 1, np.float32)
    term[6] = 1.0
    # Director folds termination into `disc` via traj['cont'].
    want = director_gve(rew, val, (1 - term) * disc, lam)
    got = ours(rew, val, None, lam, disc, term=term)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


class TestLimitCases:
  """The two limits named in Director's own docstring."""

  def test_lambda_one_is_the_discounted_monte_carlo_return(self):
    T, disc = 9, 0.9
    val = np.zeros(T, np.float32)
    rew = np.ones(T - 1, np.float32)
    got = ours(rew, val, None, 1.0, disc)
    # With a zero bootstrap, R_t = sum_{k>=0} disc^k for the remaining steps.
    want = np.array(
        [sum(disc ** k for k in range(T - 1 - t)) for t in range(T - 1)],
        np.float32)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

  def test_lambda_zero_is_the_one_step_td_target(self):
    rng = np.random.default_rng(2)
    T, disc = 9, 0.9
    val = rng.normal(size=T).astype(np.float32)
    rew = rng.normal(size=T - 1).astype(np.float32)
    got = ours(rew, val, None, 0.0, disc)
    want = rew + disc * val[1:]
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


class TestHorizonSensitivity:
  """Why `horizon: 333` and Director's `discount: 0.99` are not interchangeable."""

  def test_lambda_truncates_the_horizon_long_before_the_discount_does(self):
    """The discount difference is mostly masked by `return_lambda`.

    gamma 0.99 vs 0.997 is a 100- vs 333-step horizon, but lambda 0.95 already
    caps the effective horizon at ~1/(1-lambda) = 20 steps. Over a 64-step
    window the two discounts differ by ~11%, not by the 3.3x the horizons
    suggest -- so `horizon: 333` is a weak explanation for return variance.
    """
    T = 65
    val = np.zeros(T, np.float32)
    rew = np.ones(T - 1, np.float32)
    short = ours(rew, val, None, 0.95, 0.99)         # Director
    long_ = ours(rew, val, None, 0.95, 1 - 1 / 333)  # ours
    ratio = long_[0] / short[0]
    assert 1.05 < ratio < 1.20, ratio

  def test_even_at_lambda_one_a_64_step_window_cannot_express_the_horizon(self):
    """At lambda=1 only gamma sets the horizon, but the window truncates both.

    Over 64 steps, gamma 0.99 accumulates 47.4 and gamma 0.997 accumulates
    58.3 -- a 1.23x gap, far short of the 3.3x the nominal horizons imply,
    because neither 100 nor 333 steps fit in the window.
    """
    T = 65
    val = np.zeros(T, np.float32)
    rew = np.ones(T - 1, np.float32)
    short = ours(rew, val, None, 1.0, 0.99)
    long_ = ours(rew, val, None, 1.0, 1 - 1 / 333)
    assert long_[0] / short[0] == pytest.approx(1.229, abs=0.01)

  def test_at_the_real_imagination_length_the_discount_barely_matters(self):
    """`imag_length: 16` is the window the manager actually sees."""
    T = 17
    val = np.zeros(T, np.float32)
    rew = np.ones(T - 1, np.float32)
    short = ours(rew, val, None, 0.95, 0.99)
    long_ = ours(rew, val, None, 0.95, 1 - 1 / 333)
    # 4.5% apart, against a 3.3x difference in nominal horizon: the discount is
    # not a plausible variance source at this imagination length.
    assert long_[0] / short[0] == pytest.approx(1.045, abs=0.005)

  def test_longer_horizon_amplifies_value_error_into_the_return(self):
    """A fixed per-step value error propagates further at gamma ~0.997."""
    rng = np.random.default_rng(3)
    T = 65
    rew = np.zeros(T - 1, np.float32)
    noise = rng.normal(size=T).astype(np.float32)
    short = ours(rew, np.zeros(T, np.float32) + noise, None, 0.95, 0.99)
    long_ = ours(rew, np.zeros(T, np.float32) + noise, None, 0.95, 1 - 1 / 333)
    assert long_.std() > short.std()


class TestTrajectoryBoundaries:
  """`last` is ours alone -- Director has no equivalent, so pin its behaviour."""

  def test_last_flag_stops_bootstrapping_across_the_boundary(self):
    T, lam, disc = 12, 0.95, 0.99
    val = np.ones(T, np.float32)
    rew = np.zeros(T - 1, np.float32)
    last = np.zeros(T - 1, np.float32)
    last[5] = 1.0
    got = ours(rew, val, None, lam, disc, last=last)
    # At a `last` step lambda is zeroed, so the return is the plain TD target
    # rew + disc * val, with no contribution from beyond the boundary.
    assert got[5] == pytest.approx(disc * 1.0, rel=1e-5)

  def test_without_last_flags_we_reduce_to_director_exactly(self):
    rng = np.random.default_rng(4)
    T, lam, disc = 20, 0.95, 0.99
    val = rng.normal(size=T).astype(np.float32)
    rew = rng.normal(size=T - 1).astype(np.float32)
    want = director_gve(rew, val, np.full(T - 1, disc, np.float32), lam)
    got = ours(rew, val, None, lam, disc, last=np.zeros(T - 1, np.float32))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


class TestShapesAndDtypes:

  def test_output_is_one_shorter_than_the_input_time_axis(self):
    T = 10
    val = np.zeros(T, np.float32)
    rew = np.zeros(T - 1, np.float32)
    assert ours(rew, val, None, 0.95, 0.99).shape == (T - 1,)

  def test_float32_is_preserved(self):
    T = 10
    val = np.zeros(T, np.float32)
    rew = np.zeros(T - 1, np.float32)
    assert ours(rew, val, None, 0.95, 0.99).dtype == np.float32
