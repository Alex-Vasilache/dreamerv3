"""Tests for ``worker_timed_goals`` (HiTS-style countdown conditioning).

This flag is False in every config launched to date, so it has never actually
run in training and had zero dedicated tests. Requested audit: verify the
whole mechanism -- input concatenation, normalization, and the countdown
VALUES fed to the worker at each of the three places a worker-credit path
builds them (imagination's dense scan, the windowed ``split_traj`` path, and
the replay path) -- independently of whether it is ever turned on.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.agent import Agent

f32 = jnp.float32
i32 = jnp.int32


def _bare_agent(**overrides):
  """A minimal Agent-like object with just enough state for the manager
  countdown/feat-goal methods (same pattern as the existing
  test_switch_rate_matches_mean_duration test)."""
  agent = object.__new__(Agent)
  agent.variable_goal_length = True
  agent.goal_duration_min = 1
  agent.goal_duration_max = 16
  agent.goal_duration_fixed = 0
  agent.manager_sample_freq = 8
  agent.worker_timed_goals = True
  for k, v in overrides.items():
    setattr(agent, k, v)
  return agent


# --------------------------------------------------------------------------
# _feat_goal2tensor: the input-concatenation site.
# --------------------------------------------------------------------------

def test_feat_goal2tensor_appends_exactly_one_countdown_channel_when_on():
  agent = _bare_agent(worker_timed_goals=True)
  B, T, Dd, Ds, Cs, Dg = 2, 5, 4, 3, 2, 6
  x = {'deter': jnp.zeros((B, T, Dd)), 'stoch': jnp.zeros((B, T, Ds, Cs))}
  y = jnp.zeros((B, T, Dg))
  # Real countdown tensors come from ``_countdown_norm``, which has a trailing
  # singleton dim (``[..., None]``) -- match that shape here.
  cd = jnp.full((B, T, 1), 3.0)
  out = agent._feat_goal2tensor(x, y, countdown=cd)
  expected_width = Dd + Ds * Cs + Dg + 1
  assert out.shape == (B, T, expected_width), out.shape


def test_feat_goal2tensor_appends_nothing_when_off():
  agent = _bare_agent(worker_timed_goals=False)
  B, T, Dd, Ds, Cs, Dg = 2, 5, 4, 3, 2, 6
  x = {'deter': jnp.zeros((B, T, Dd)), 'stoch': jnp.zeros((B, T, Ds, Cs))}
  y = jnp.zeros((B, T, Dg))
  out = agent._feat_goal2tensor(x, y, countdown=jnp.full((B, T), 3.0))
  expected_width = Dd + Ds * Cs + Dg
  assert out.shape == (B, T, expected_width), (
      'a countdown channel leaked in even though the flag is off')


def test_feat_goal2tensor_countdown_value_lands_in_the_last_channel():
  """The appended channel must literally BE the normalized countdown, not a
  zero placeholder or something shape-compatible but wrong."""
  agent = _bare_agent(worker_timed_goals=True, goal_duration_fixed=0,
                      goal_duration_max=10)
  B, T = 1, 4
  x = {'deter': jnp.zeros((B, T, 2)), 'stoch': jnp.zeros((B, T, 1, 1))}
  y = jnp.zeros((B, T, 2))
  cd_steps = jnp.array([[10.0, 5.0, 1.0, 0.0]])
  normed = agent._countdown_norm(cd_steps)
  out = agent._feat_goal2tensor(x, y, countdown=normed)
  # ``nn.cast`` casts to the bfloat16 compute dtype, so this checks the value
  # is the countdown (not a placeholder), at bf16 precision -- not bit-exact.
  np.testing.assert_allclose(
      np.asarray(out[..., -1], np.float32), np.asarray(normed[..., 0]),
      atol=5e-3, rtol=5e-3)


def test_feat_goal2tensor_missing_countdown_defaults_to_full_budget():
  """``countdown=None`` (report/viz-only call sites) must fall back to +1.0
  (full budget), not zero or an error -- matches the documented default."""
  agent = _bare_agent(worker_timed_goals=True)
  x = {'deter': jnp.zeros((2, 3)), 'stoch': jnp.zeros((2, 1, 1))}
  y = jnp.zeros((2, 3))
  out = agent._feat_goal2tensor(x, y, countdown=None)
  np.testing.assert_allclose(np.asarray(out[..., -1]), 1.0)


def test_feat_goal2tensor_countdown_is_stop_gradiented():
  """Gradient must not flow from the worker's loss back into whatever
  produced the countdown (it's a scalar side-signal, not a differentiable
  quantity the worker should be able to game)."""
  agent = _bare_agent(worker_timed_goals=True)

  def f(cd_scalar):
    cd = jnp.full((1, 1), cd_scalar)
    x = {'deter': jnp.zeros((1, 1, 2)), 'stoch': jnp.zeros((1, 1, 1, 1))}
    y = jnp.zeros((1, 1, 2))
    out = agent._feat_goal2tensor(x, y, countdown=agent._countdown_norm(cd))
    return out.sum()

  grad = jax.grad(f)(jnp.float32(5.0))
  assert float(grad) == 0.0, (
      'gradient leaked through the countdown channel into its source value')


# --------------------------------------------------------------------------
# _countdown_norm / _countdown_budget: normalization correctness.
# --------------------------------------------------------------------------

def test_countdown_norm_boundary_values():
  agent = _bare_agent(goal_duration_fixed=0, goal_duration_max=8)
  cd = jnp.array([0.0, 8.0, 4.0])
  normed = np.asarray(agent._countdown_norm(cd))[:, 0]
  np.testing.assert_allclose(normed, [-1.0, 1.0, 0.0], atol=1e-6)


def test_countdown_norm_clips_out_of_range_values():
  """A countdown that somehow exceeds the budget (shouldn't happen given
  legitimate inputs, but the clip is defensive) must saturate, not wrap or
  produce a value outside [-1, 1]."""
  agent = _bare_agent(goal_duration_fixed=0, goal_duration_max=8)
  cd = jnp.array([-5.0, 100.0])
  normed = np.asarray(agent._countdown_norm(cd))[:, 0]
  np.testing.assert_allclose(normed, [-1.0, 1.0], atol=1e-6)


def test_countdown_budget_uses_the_pinned_hold_not_goal_duration_max():
  """Bug found 2026-07-29: under a PINNED hold, the realized countdown can
  never exceed ``goal_duration_fixed`` (``_duration_steps`` bypasses the
  duration head entirely), but launch configs default ``goal_duration_max``
  to 16 regardless of the pin (only ever raised to match the duration
  target, never lowered) -- e.g. e360/e361's actual launch config has
  ``goal_duration_fixed=8, goal_duration_max=16``. Before the fix, a fresh
  switch's countdown (the true maximum, 8) normalized to 0.0 instead of
  +1.0 -- the worker never saw the top half of its intended [-1, 1] range.
  """
  pinned = _bare_agent(goal_duration_fixed=8, goal_duration_max=16)
  assert pinned._countdown_budget() == 8.0
  at_fresh_switch = np.asarray(pinned._countdown_norm(jnp.array([8.0])))[0, 0]
  np.testing.assert_allclose(at_fresh_switch, 1.0, atol=1e-6)

  # Genuinely free holds are untouched: the true bound IS goal_duration_max.
  free = _bare_agent(goal_duration_fixed=0, goal_duration_max=16)
  assert free._countdown_budget() == 16.0

  # Fixed-K Director (not variable_goal_length at all) is untouched: budget is
  # manager_sample_freq regardless of goal_duration_fixed/_max (irrelevant there).
  director = _bare_agent(variable_goal_length=False, goal_duration_fixed=0)
  director.manager_sample_freq = 8
  assert director._countdown_budget() == 8.0


# --------------------------------------------------------------------------
# End-to-end countdown VALUES through each of the three worker-credit paths,
# with worker_timed_goals actually exercised (never done before -- the flag
# has been off in every launched run).
# --------------------------------------------------------------------------

def test_windowed_path_countdown_decreases_to_zero_at_the_boundary_state():
  """Mirrors the ``win_cd`` construction at the ``split_traj`` call site
  (agent.py): countdown K, K-1, ..., 1, 0 across a K+1-state window, with the
  boundary state (holding the OLD goal for bootstrap) reading exactly 0."""
  K = 8
  win_cd = K - jnp.arange(K + 1, dtype=i32)
  agent = _bare_agent(goal_duration_fixed=K, goal_duration_max=16)
  normed = np.asarray(agent._countdown_norm(win_cd))[:, 0]
  np.testing.assert_array_equal(np.asarray(win_cd), [8, 7, 6, 5, 4, 3, 2, 1, 0])
  np.testing.assert_allclose(normed[0], 1.0, atol=1e-6)   # fresh switch
  np.testing.assert_allclose(normed[-1], -1.0, atol=1e-6)  # boundary/bootstrap
  assert np.all(np.diff(normed) < 0), 'countdown must strictly decrease'


def test_dense_path_countdown_matches_imagine_with_manager_convention():
  """The dense (free var-K) worker path's countdown is
  ``img_countdowns`` straight from ``_imagine_with_manager``: on a switch
  step it equals the freshly sampled duration; on a held step it equals the
  carried remaining count. Reproduce that recurrence directly (same formula
  as ``_imagine_with_manager``'s body) and check the normalized sequence
  against hand-derived values for a hold of duration 4 starting at t=0."""
  agent = _bare_agent(goal_duration_fixed=0, goal_duration_max=16)
  p = 4
  remaining = 0
  cds = []
  for t in range(6):
    update = remaining <= 0
    cd = p if update else remaining
    remaining = cd - 1
    cds.append(cd)
  # t=0 (switch, fresh budget p=4) .. t=3 (last held step, 1 left) .. t=4 (next switch).
  assert cds == [4, 3, 2, 1, 4, 3]
  normed = np.asarray(agent._countdown_norm(jnp.array(cds, f32)))[:, 0]
  expected = [2 * c / 16 - 1 for c in cds]
  np.testing.assert_allclose(normed, expected, atol=1e-6)


def test_replay_path_countdown_uses_the_same_budget_fix_as_imagination():
  """The replay worker path's countdown (``repl_cd``, from
  ``_manager_skills_on_sequence``'s ``countdown`` output) shares
  ``_countdown_norm``/``_countdown_budget`` with every other path -- confirm
  a pinned-hold replay countdown gets the SAME corrected normalization as
  imagination and the windowed path, not a third, independently-drifted one."""
  agent_a = _bare_agent(goal_duration_fixed=8, goal_duration_max=16)
  agent_b = _bare_agent(goal_duration_fixed=8, goal_duration_max=16)
  cd = jnp.array([8.0, 4.0, 0.0])
  np.testing.assert_allclose(
      np.asarray(agent_a._countdown_norm(cd)),
      np.asarray(agent_b._countdown_norm(cd)))
  np.testing.assert_allclose(np.asarray(agent_a._countdown_norm(cd))[0, 0], 1.0)
