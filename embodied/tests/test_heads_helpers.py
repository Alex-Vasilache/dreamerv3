"""Pure helpers in `hrl/heads.py`.

`_shift_up`, `_shift_down`, `head_entropy_time`, `head_entropy_perdim_time` and
`policy_time_slice` had no test anywhere in the suite. They sit directly on the
manager's REINFORCE path, so a wrong axis or a wrong slice would silently
mis-credit every manager update.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.hrl.heads import (
    _head_inner,
    _shift_down,
    _shift_up,
    align_skill_events,
    head_entropy_perdim_time,
    head_entropy_time,
    head_logp_time,
    manager_reinforce_policy,
    policy_time_slice,
    smooth_skill_event,
)


class TestShifts:

  def test_shift_up_moves_mass_to_the_next_class(self):
    e = jnp.asarray([[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(np.asarray(_shift_up(e))[0], [0, 1, 0, 0])

  def test_shift_down_moves_mass_to_the_previous_class(self):
    e = jnp.asarray([[0.0, 0.0, 1.0, 0.0]])
    np.testing.assert_allclose(np.asarray(_shift_down(e))[0], [0, 1, 0, 0])

  def test_mass_at_the_top_end_falls_off_under_shift_up(self):
    e = jnp.asarray([[0.0, 0.0, 0.0, 1.0]])
    assert np.asarray(_shift_up(e)).sum() == pytest.approx(0.0)

  def test_mass_at_the_bottom_end_falls_off_under_shift_down(self):
    e = jnp.asarray([[1.0, 0.0, 0.0, 0.0]])
    assert np.asarray(_shift_down(e)).sum() == pytest.approx(0.0)

  def test_shifts_act_on_the_last_axis_only(self):
    e = jnp.zeros((2, 3, 5)).at[:, :, 0].set(1.0)
    assert _shift_up(e).shape == (2, 3, 5)
    np.testing.assert_allclose(np.asarray(_shift_up(e))[:, :, 1], 1.0)

  def test_shifts_preserve_dtype(self):
    e = jnp.zeros((1, 4), jnp.bfloat16).at[0, 0].set(1)
    assert _shift_up(e).dtype == jnp.bfloat16


class TestSmoothSkillEvent:

  def test_alpha_zero_is_exactly_the_identity(self):
    e = jnp.asarray([[0.0, 1.0, 0.0, 0.0]])
    assert smooth_skill_event(e, 0.0, 'ring') is e
    assert smooth_skill_event(e, 0.0, 'line') is e

  def test_ring_kernel_has_the_documented_weights(self):
    e = jnp.zeros((1, 5)).at[0, 2].set(1.0)
    got = np.asarray(smooth_skill_event(e, 0.1, 'ring'))[0]
    np.testing.assert_allclose(got, [0, 0.1, 0.8, 0.1, 0], atol=1e-6)

  def test_ring_wraps_around_at_the_endpoints(self):
    e = jnp.zeros((1, 5)).at[0, 0].set(1.0)
    got = np.asarray(smooth_skill_event(e, 0.1, 'ring'))[0]
    assert got[-1] == pytest.approx(0.1), 'entry 0 must share with entry C-1'

  def test_line_does_not_wrap_around(self):
    e = jnp.zeros((1, 5)).at[0, 0].set(1.0)
    got = np.asarray(smooth_skill_event(e, 0.1, 'line'))[0]
    assert got[-1] == pytest.approx(0.0), (
        'a line topology must not share credit between the two most distant '
        'codes')

  def test_line_endpoint_row_is_renormalised_to_one(self):
    e = jnp.zeros((1, 5)).at[0, 0].set(1.0)
    got = np.asarray(smooth_skill_event(e, 0.1, 'line'))[0]
    assert got.sum() == pytest.approx(1.0, rel=1e-6)
    # Documented endpoint kernel: ((1-2a)/(1-a), a/(1-a)).
    assert got[0] == pytest.approx(0.8 / 0.9, rel=1e-5)
    assert got[1] == pytest.approx(0.1 / 0.9, rel=1e-5)

  def test_every_row_remains_a_distribution(self):
    for topo in ('ring', 'line'):
      e = jnp.eye(6)
      got = np.asarray(smooth_skill_event(e, 0.15, topo))
      np.testing.assert_allclose(got.sum(-1), 1.0, rtol=1e-6)

  def test_blocks_are_smoothed_independently(self):
    e = jnp.zeros((1, 3, 5)).at[0, :, 2].set(1.0)
    got = np.asarray(smooth_skill_event(e, 0.1, 'ring'))
    assert got.shape == (1, 3, 5)
    for b in range(3):
      np.testing.assert_allclose(got[0, b], [0, 0.1, 0.8, 0.1, 0], atol=1e-6)

  def test_an_unknown_topology_is_rejected(self):
    with pytest.raises(AssertionError):
      smooth_skill_event(jnp.eye(3), 0.1, 'grid')


class TestPolicyTimeSlice:

  def test_trailing_axes_are_summed_then_the_last_step_dropped(self):
    x = jnp.ones((2, 5, 3, 4))
    got = policy_time_slice(x)
    assert got.shape == (2, 4)
    np.testing.assert_allclose(np.asarray(got), 12.0)

  def test_a_two_dimensional_input_is_only_sliced(self):
    x = jnp.arange(10, dtype=jnp.float32).reshape(2, 5)
    np.testing.assert_allclose(
        np.asarray(policy_time_slice(x)), np.asarray(x)[:, :-1])

  def test_a_one_dimensional_input_is_promoted_then_sliced_to_empty(self):
    got = policy_time_slice(jnp.arange(4, dtype=jnp.float32))
    assert got.shape == (4, 0)

  def test_dtype_is_preserved(self):
    assert policy_time_slice(jnp.ones((2, 3), jnp.bfloat16)).dtype == jnp.bfloat16


class FakeHead:
  def __init__(self, ent, logits=None):
    self._ent = ent
    self._logits = logits

  def entropy(self):
    return self._ent

  def logp(self, event):
    return (self._logits * event).sum(-1)

  def pred(self):
    return jnp.zeros_like(self._logits)


class TestEntropyHelpers:

  def test_head_entropy_time_sums_block_axes_and_drops_the_last_step(self):
    head = FakeHead(jnp.ones((2, 6, 8)))
    got = head_entropy_time(head)
    assert got.shape == (2, 5)
    np.testing.assert_allclose(np.asarray(got), 8.0)

  def test_head_entropy_perdim_time_keeps_the_block_axis(self):
    head = FakeHead(jnp.ones((2, 6, 8)))
    got = head_entropy_perdim_time(head)
    assert got.shape == (2, 5, 8), 'the per-block axis must survive'

  def test_perdim_and_summed_entropy_agree_after_summing_blocks(self):
    rng = np.random.default_rng(0)
    ent = jnp.asarray(rng.uniform(size=(2, 6, 8)), jnp.float32)
    head = FakeHead(ent)
    np.testing.assert_allclose(
        np.asarray(head_entropy_perdim_time(head)).sum(-1),
        np.asarray(head_entropy_time(head)), rtol=1e-5)

  def test_a_scalar_entropy_head_is_promoted_rather_than_crashing(self):
    head = FakeHead(jnp.ones(6))
    assert head_entropy_perdim_time(head).shape == (1, 5)


class TestManagerReinforcePolicy:

  def test_all_heads_participate_when_duration_is_free(self):
    pol = {'skill': 1, 'duration': 2}
    assert manager_reinforce_policy(pol, duration_fixed=False) == pol

  def test_the_duration_head_is_dropped_when_the_hold_is_pinned(self):
    pol = {'skill': 1, 'duration': 2}
    assert manager_reinforce_policy(pol, duration_fixed=True) == {'skill': 1}

  def test_dropping_duration_does_not_mutate_the_caller_dict(self):
    pol = {'skill': 1, 'duration': 2}
    manager_reinforce_policy(pol, duration_fixed=True)
    assert 'duration' in pol

  def test_a_policy_without_a_duration_head_is_unchanged(self):
    pol = {'skill': 1}
    assert manager_reinforce_policy(pol, duration_fixed=True) == pol


class TestAlignSkillEvents:

  def test_matching_shapes_pass_through_unchanged(self):
    pol = {'skill': FakeHead(None, jnp.zeros((2, 6, 8)))}
    ev = align_skill_events({'skill': jnp.ones((2, 6, 8))}, pol)
    assert ev['skill'].shape == (2, 6, 8)

  def test_a_missing_trailing_axis_is_broadcast(self):
    """Regression: the new axis must go last, not at position 1."""
    pol = {'skill': FakeHead(None, jnp.zeros((2, 6, 8)))}
    ev = align_skill_events({'skill': jnp.ones((2, 6))}, pol)
    assert ev['skill'].shape == (2, 6, 8)

  def test_broadcasting_copies_along_the_class_axis_not_the_time_axis(self):
    """With T == C the old `e[:, None]` broadcast silently, and wrongly."""
    pol = {'skill': FakeHead(None, jnp.zeros((2, 5, 5)))}
    e = jnp.asarray(np.arange(10, dtype=np.float32).reshape(2, 5))
    ev = np.asarray(align_skill_events({'skill': e}, pol)['skill'])
    # Every class entry of a given timestep holds that timestep's value.
    for t in range(5):
      np.testing.assert_allclose(ev[0, t, :], np.float32(t))

  def test_a_bare_tensor_is_accepted_instead_of_a_dict(self):
    pol = {'skill': FakeHead(None, jnp.zeros((2, 6, 8)))}
    ev = align_skill_events(jnp.ones((2, 6, 8)), pol)
    assert ev['skill'].shape == (2, 6, 8)

  def test_events_are_stop_gradiented(self):
    import jax
    pol = {'skill': FakeHead(None, jnp.zeros((2, 6, 8)))}
    def fn(x):
      return align_skill_events({'skill': x}, pol)['skill'].sum()
    g = np.asarray(jax.grad(fn)(jnp.ones((2, 6, 8), jnp.float32)))
    assert np.all(g == 0.0), 'the sampled skill must not receive gradient'


class TestHeadLogpTime:

  def test_logp_is_sliced_to_one_shorter_than_time(self):
    head = FakeHead(None, jnp.ones((2, 6, 8)))
    got = head_logp_time(head, jnp.ones((2, 6, 8)))
    assert got.shape == (2, 5)

  def test_the_event_is_stop_gradiented_inside_logp(self):
    import jax
    head = FakeHead(None, jnp.ones((2, 6, 8)))
    fn = lambda e: head_logp_time(head, e).sum()
    g = np.asarray(jax.grad(fn)(jnp.ones((2, 6, 8), jnp.float32)))
    assert np.all(g == 0.0)


class TestHeadInner:

  def test_a_plain_head_is_returned_unchanged(self):
    head = FakeHead(jnp.ones((2, 3)))
    assert _head_inner(head) is head

  def test_an_agg_wrapper_is_unwrapped(self):
    import embodied.jax.outs as outs
    inner = FakeHead(jnp.ones((2, 3)))
    agg = outs.Agg(inner, 1, jnp.float32) if hasattr(outs, 'Agg') else None
    if agg is None:
      pytest.skip('outs.Agg not available')
    assert _head_inner(agg) is inner
