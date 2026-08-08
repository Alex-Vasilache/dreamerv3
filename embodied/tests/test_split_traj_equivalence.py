"""Worker credit: our boundary-masked return vs Director's `split_traj`.

Director cuts the imagined trajectory into overlapping windows of `K+1` states
and computes a separate return inside each one:

    (1 2 3 4) (4 5 6 7) (7 8 9 10)

so a window's return bootstraps off its own last value and never reaches across
a goal switch. We never physically split: `imag_loss_wkr` keeps one long
trajectory and zeroes lambda at each boundary step (`skill_window`), which is
supposed to have the same effect.

"Supposed to" was the entire basis for it -- nothing compared the two. This file
transcribes `hierarchy.Hierarchy.split_traj` into numpy, computes Director's
per-window returns with the same `gve` recursion, and checks our masked return
equals it elementwise.

Reference: `code/director/embodied/agents/director/hierarchy.py:410`.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.hrl.losses import lambda_return
from dreamerv3.hrl.tensors import worker_split_window

f32 = jnp.float32


def director_split_returns(rew, val, k, disc, lam):
  """Director's per-window returns, for one batch row.

  `rew[t]` is the reward for the transition out of state `t`; `val` has one
  entry per state. With `T = n*k + 1` states there are `n` windows of `k+1`
  states, each yielding `k` returns, so `n*k == T-1` returns overall -- the same
  count our single masked call produces.
  """
  T = len(val)
  assert (T - 1) % k == 0, (T, k)
  n = (T - 1) // k
  out = []
  for w in range(n):
    lo = w * k
    v = val[lo:lo + k + 1]        # k+1 states
    r = rew[lo:lo + k]            # k transitions
    # Director's `gve`, restricted to this window.
    vals = [v[-1]]
    interm = r + disc * v[1:] * (1 - lam)
    for t in reversed(range(k)):
      vals.append(interm[t] + disc * lam * vals[-1])
    out.extend(list(reversed(vals))[:-1])
  return np.asarray(out, np.float32)


def ours_masked_returns(rew, val, k, disc, lam):
  """`imag_loss_wkr`'s return: one long trajectory, lambda zeroed at boundaries."""
  T = len(val)
  # Our reward convention indexes from 1: rew[t+1] is the reward for the
  # transition out of state t.
  rew_a = np.zeros((1, T), np.float32)
  rew_a[0, 1:] = rew
  val_a = val[None].astype(np.float32)
  # Exactly the mask `imag_loss_wkr` builds for an integer `skill_window`.
  pos = np.arange(T)
  boundary = ((pos % k == 0) & (pos > 0)).astype(np.float32)
  last = np.broadcast_to(boundary, (1, T)).copy()
  term = np.zeros((1, T), np.float32)
  out = lambda_return(
      jnp.asarray(last), jnp.asarray(term), jnp.asarray(rew_a),
      jnp.asarray(val_a), jnp.asarray(val_a), np.float32(disc), np.float32(lam))
  return np.asarray(out)[0]


CASES = [(9, 4), (17, 8), (13, 4), (17, 4), (21, 5), (25, 8)]


class TestMatchesDirectorSplitTraj:

  @pytest.mark.parametrize('T,k', CASES)
  @pytest.mark.parametrize('lam', [0.0, 0.5, 0.95, 1.0])
  def test_masked_return_equals_per_window_return(self, T, k, lam):
    rng = np.random.default_rng(T * 100 + k)
    val = rng.normal(size=T).astype(np.float32)
    rew = rng.normal(size=T - 1).astype(np.float32)
    want = director_split_returns(rew, val, k, 0.99, lam)
    got = ours_masked_returns(rew, val, k, 0.99, lam)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

  @pytest.mark.parametrize('T,k', CASES)
  def test_the_two_produce_the_same_number_of_returns(self, T, k):
    rng = np.random.default_rng(1)
    val = rng.normal(size=T).astype(np.float32)
    rew = rng.normal(size=T - 1).astype(np.float32)
    assert len(director_split_returns(rew, val, k, 0.99, 0.95)) == T - 1
    assert len(ours_masked_returns(rew, val, k, 0.99, 0.95)) == T - 1

  def test_a_boundary_step_gets_the_one_step_td_target(self):
    """The last state of a window bootstraps off the next value, nothing more."""
    T, k, disc, lam = 9, 4, 0.99, 0.95
    val = np.arange(T, dtype=np.float32)
    rew = np.ones(T - 1, np.float32)
    got = ours_masked_returns(rew, val, k, disc, lam)
    # State 3 is the last state of window 0, so its return is rew + disc*V(4).
    assert got[3] == pytest.approx(1.0 + disc * val[4], rel=1e-5)

  def test_credit_does_not_leak_across_a_goal_boundary(self):
    """A reward after the boundary must not change a return before it."""
    T, k = 9, 4
    val = np.zeros(T, np.float32)
    a = np.zeros(T - 1, np.float32)
    b = a.copy()
    b[6] = 100.0  # well past the boundary at state 4
    ra = ours_masked_returns(a, val, k, 0.99, 0.95)
    rb = ours_masked_returns(b, val, k, 0.99, 0.95)
    np.testing.assert_allclose(ra[:4], rb[:4], atol=1e-6)
    assert not np.allclose(ra[4:], rb[4:])

  def test_without_the_mask_credit_does_leak(self):
    """Control: the isolation comes from the mask, not from the horizon."""
    T = 9
    val = np.zeros(T, np.float32)
    a = np.zeros(T - 1, np.float32)
    b = a.copy()
    b[6] = 100.0
    # skill_window larger than T means no interior boundary at all.
    ra = ours_masked_returns(a, val, 100, 0.99, 0.95)
    rb = ours_masked_returns(b, val, 100, 0.99, 0.95)
    assert not np.allclose(ra[:4], rb[:4])


class TestWorkerSplitWindow:
  """`worker_split_window` decides whether the windowed path applies at all."""

  def test_disabled_when_split_traj_is_off(self):
    assert worker_split_window(False, False, 0, 8, 16) == 0

  def test_fixed_k_returns_the_manager_period(self):
    assert worker_split_window(True, False, 0, 8, 16) == 8

  def test_a_pinned_duration_uses_that_duration(self):
    assert worker_split_window(True, True, 4, 8, 16) == 4

  def test_genuinely_variable_holds_fall_back_to_the_dense_path(self):
    assert worker_split_window(True, True, 0, 8, 16) == 0

  def test_a_horizon_not_divisible_by_k_disables_the_window(self):
    """Director asserts len(action) % k == 1; a ragged horizon cannot split."""
    assert worker_split_window(True, False, 0, 5, 16) == 0

  def test_a_horizon_shorter_than_k_disables_the_window(self):
    assert worker_split_window(True, False, 0, 32, 16) == 0

  @pytest.mark.parametrize('H,k', [(16, 8), (16, 4), (24, 8), (32, 8)])
  def test_the_production_combinations_are_enabled(self, H, k):
    assert worker_split_window(True, False, 0, k, H) == k


class TestDirectorWindowShape:
  """Pin the window structure itself, independent of the return values."""

  def test_windows_overlap_by_exactly_one_state(self):
    k, T = 4, 9
    n = (T - 1) // k
    windows = [list(range(w * k, w * k + k + 1)) for w in range(n)]
    assert windows == [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]]
    for a, b in zip(windows, windows[1:]):
      assert a[-1] == b[0]

  def test_every_state_except_the_boundaries_appears_once(self):
    k, T = 8, 17
    n = (T - 1) // k
    seen = [i for w in range(n) for i in range(w * k, w * k + k + 1)]
    from collections import Counter
    counts = Counter(seen)
    assert counts[0] == 1 and counts[T - 1] == 1
    assert counts[k] == 2, 'the shared boundary state appears in both windows'
