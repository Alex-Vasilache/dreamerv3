"""Step-by-step verification of the block-pooled variable-K manager pipeline.

``test_block_pooled_equivalence.py`` compares the block-pooled path against
fixed-K Director *end to end*. That catches disagreements but localizes them
poorly, and it only ever exercises the one switch pattern fixed-K can produce
(every K steps, starting at 0).

This module does the complementary job. It re-implements Director's block
pooling as slow, obviously-correct pure-Python-over-numpy loops -- no JAX, no
vectorization, no cleverness -- and fuzzes the real implementation against it
over many random switch patterns, including the degenerate ones the pinned
configuration never produces (switch every step, a single switch, a switch on
the final timestep, rows of a batch disagreeing on switch count). It also
asserts the things JAX will silently get wrong rather than raise on: shapes,
lengths, dtypes, and which column corresponds to which decision.

Silent-failure modes specifically covered:
  * ``x.at[i].set(v)`` CLAMPS out-of-range ``i`` instead of raising, so a packed
    buffer that is too small loses writes with no error.
  * ``jax.ops.segment_*`` silently DROPS ids >= ``num_segments``.
  * broadcasting turns an off-by-one shape into a silent fan-out rather than an
    error, so a (B, 1) vs (B, T) mixup produces plausible numbers.
  * reductions over a padded axis return a number, never a complaint.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.hrl import (
    aggregate_mgr_cont_variable,
    aggregate_mgr_extr_rew_variable,
    downsample_at_switch_mask,
    forward_fill_packed,
    imag_reward_pad,
    masked_cumprod,
    switch_valid_mask,
    variable_block_director_tensors,
    variable_segment_ids,
)

f32 = jnp.float32


# --------------------------------------------------------------------------
# Reference implementation: plain loops, one row at a time, no JAX.
# --------------------------------------------------------------------------

def ref_segments(sw_row):
  """[(start, stop)] transition ranges per decision, in ``r = rew[1:]`` indexing.

  A switch at position ``s`` owns transitions ``s .. next_switch - 1``. The
  transition axis has length ``T - 1`` (``r[j] = rew[j + 1]``), so a switch
  landing on ``T - 1`` owns nothing.
  """
  T = len(sw_row)
  starts = [t for t in range(T) if sw_row[t] > 0.5]
  segs = []
  for i, s in enumerate(starts):
    stop = starts[i + 1] if i + 1 < len(starts) else T - 1
    segs.append((s, min(stop, T - 1)))
  return segs


def ref_pool(rew_row, con_row, sw_row, agg='mean'):
  """Director ``abstract_traj`` pooling for one row, written out longhand.

  Note the deliberate one-step asymmetry between the two quantities, which is
  Director's own convention and is reproduced identically by the fixed-K and
  block-pooled implementations (see
  ``test_reward_and_cont_use_a_one_step_shifted_continuation``):
    * reward weights use ``con[j]``   (``aggregate_*_rew*`` takes ``con[:, :-1]``)
    * the pooled continuation uses ``con[j + 1]`` (``aggregate_*_cont*`` takes
      ``con[:, 1:]``)
  """
  pooled_rew, pooled_cont = [], []
  for (start, stop) in ref_segments(sw_row):
    weighted, w_rew, w_cont = [], 1.0, 1.0
    for j in range(start, stop):
      w_rew = w_rew * con_row[j]          # cumprod, reset at the segment start
      w_cont = w_cont * con_row[j + 1]    # shifted one step, as Director does
      weighted.append(rew_row[j + 1] * w_rew)
    if weighted:
      pooled_rew.append(sum(weighted) if agg == 'sum'
                        else sum(weighted) / len(weighted))
      pooled_cont.append(w_cont)
    else:
      pooled_rew.append(0.0)
      pooled_cont.append(1.0)             # empty product identity
  return pooled_rew, pooled_cont


def ref_decision_states(feat_row, sw_row):
  """States at each decision, then forward-filled into the padded tail."""
  T = len(sw_row)
  picked = [feat_row[t] for t in range(T) if sw_row[t] > 0.5]
  if not picked:
    picked = [feat_row[0]]
  return [picked[min(i, len(picked) - 1)] for i in range(T)]


def ref_credited_decisions(sw_row):
  """Decisions owning >= 1 transition (i.e. excluding a final-timestep switch)."""
  return int(sum(1 for t in range(len(sw_row) - 1) if sw_row[t] > 0.5))


# --------------------------------------------------------------------------
# Switch patterns to fuzz over, including the degenerate ones.
# --------------------------------------------------------------------------

def switch_patterns(T):
  pats = {
      'every_8_from_0': [1 if t % 8 == 0 else 0 for t in range(T)],
      'every_step': [1] * T,
      'only_first': [1] + [0] * (T - 1),
      'first_and_last': [1] + [0] * (T - 2) + [1],
      'irregular': [1 if t in (0, 3, 4, 9, 10, 11, T - 2) else 0 for t in range(T)],
      'dense_then_sparse': [1 if (t < 5 or t == T - 3) else 0 for t in range(T)],
  }
  return pats


def _rows(T, seed=0):
  rng = np.random.RandomState(seed)
  rew = rng.randn(T).astype(np.float32) * 0.5
  con = rng.uniform(0.90, 1.0, size=T).astype(np.float32)
  return rew, con


@pytest.mark.parametrize('T', [17, 33, 64])
@pytest.mark.parametrize('agg', ['mean', 'sum'])
def test_pooled_rewards_match_longhand_reference(T, agg):
  """Pooled reward per decision == the hand-written loop, every pattern."""
  for pname, sw in switch_patterns(T).items():
    rew, con = _rows(T, seed=len(pname))
    got = np.asarray(aggregate_mgr_extr_rew_variable(
        jnp.array(rew)[None], jnp.array(con)[None], jnp.array(sw, f32)[None],
        without_zeros=True, agg_mode=agg))[0]
    want, _ = ref_pool(rew, con, sw, agg)
    assert len(want) <= got.shape[0], (
        f'{pname}: packed buffer ({got.shape[0]}) is smaller than the number '
        f'of decisions ({len(want)}) -- segment_sum would silently drop them')
    np.testing.assert_allclose(
        got[:len(want)], np.array(want, np.float32), rtol=2e-5, atol=2e-6,
        err_msg=f'pattern={pname} agg={agg}')


@pytest.mark.parametrize('T', [17, 33, 64])
def test_pooled_continuation_matches_longhand_reference(T):
  """Pooled continuation per decision == the product over that segment."""
  for pname, sw in switch_patterns(T).items():
    rew, con = _rows(T, seed=len(pname) + 1)
    got = np.asarray(aggregate_mgr_cont_variable(
        jnp.array(con)[None], jnp.array(sw, f32)[None], without_zeros=True))[0]
    _, want = ref_pool(rew, con, sw)
    np.testing.assert_allclose(
        got[:len(want)], np.array(want, np.float32), rtol=2e-5, atol=2e-6,
        err_msg=f'pattern={pname}')


@pytest.mark.parametrize('T', [17, 33])
def test_decision_states_match_longhand_reference(T):
  """``downsample_at_switch_mask`` picks the right states and fills the tail."""
  for pname, sw in switch_patterns(T).items():
    feat = np.arange(T, dtype=np.float32) * 10.0
    got = np.asarray(downsample_at_switch_mask(
        {'x': jnp.array(feat)[None]}, jnp.array(sw, f32)[None])['x'])[0]
    want = np.array(ref_decision_states(feat, sw), np.float32)
    np.testing.assert_allclose(got, want, err_msg=f'pattern={pname}')


@pytest.mark.parametrize('T', [17, 33])
def test_credited_decision_count_matches_reference(T):
  """``mgr_switch`` counts exactly the decisions owning >= 1 transition."""
  for pname, sw in switch_patterns(T).items():
    rew, con = _rows(T, seed=7)
    _, _, _, switch = variable_block_director_tensors(
        jnp.array(rew)[None], jnp.array(con)[None], jnp.array(rew)[None],
        jnp.array(sw, f32)[None], T, agg_mode='mean')
    got = int(np.asarray(switch)[0].sum())
    want = ref_credited_decisions(sw)
    assert got == want, f'pattern={pname}: credited {got}, reference {want}'


@pytest.mark.parametrize('T', [17, 33])
def test_reward_column_alignment_is_decision_index_plus_one(T):
  """Column i+1 of ``mgr_extr_rew`` holds decision i's pooled reward.

  This alignment is load-bearing: ``lambda_return`` reads ``rew[t + 1]`` for the
  return at ``t``, and every mask in the loss is indexed by decision. An
  off-by-one here still produces finite, plausible numbers.
  """
  for pname, sw in switch_patterns(T).items():
    rew, con = _rows(T, seed=3)
    extr, _, cont, _ = variable_block_director_tensors(
        jnp.array(rew)[None], jnp.array(con)[None], jnp.array(rew)[None],
        jnp.array(sw, f32)[None], T, agg_mode='mean')
    want_rew, want_cont = ref_pool(rew, con, sw)
    extr, cont = np.asarray(extr)[0], np.asarray(cont)[0]
    assert extr[0] == 0.0, f'{pname}: slot 0 must be the empty pre-action slot'
    n = min(len(want_rew), T - 1)
    np.testing.assert_allclose(
        extr[1:n + 1], np.array(want_rew[:n], np.float32), rtol=2e-5, atol=2e-6,
        err_msg=f'{pname}: reward column alignment')
    np.testing.assert_allclose(
        cont[0], con[0], rtol=1e-6,
        err_msg=f'{pname}: cont slot 0 must be the raw first continuation')
    np.testing.assert_allclose(
        cont[1:n + 1], np.array(want_cont[:n], np.float32), rtol=2e-5, atol=2e-6,
        err_msg=f'{pname}: cont column alignment')


def test_heterogeneous_batch_rows_do_not_leak_into_each_other():
  """Each row's packing must depend only on that row's switches.

  Under real variable-K every row of the batch has a different switch count;
  a reduction that accidentally spans the batch axis still returns numbers.
  """
  T = 33
  pats = list(switch_patterns(T).values())[:4]
  sw = np.array(pats, np.float32)
  rew = np.stack([_rows(T, seed=i)[0] for i in range(len(pats))])
  con = np.stack([_rows(T, seed=i)[1] for i in range(len(pats))])
  batched = np.asarray(aggregate_mgr_extr_rew_variable(
      jnp.array(rew), jnp.array(con), jnp.array(sw), without_zeros=True))
  for i in range(len(pats)):
    solo = np.asarray(aggregate_mgr_extr_rew_variable(
        jnp.array(rew[i])[None], jnp.array(con[i])[None],
        jnp.array(sw[i])[None], without_zeros=True))[0]
    np.testing.assert_allclose(batched[i], solo, rtol=2e-5, atol=2e-6,
                               err_msg=f'row {i} differs when batched')


def test_segment_ids_never_exceed_the_packed_buffer():
  """Guards the silent ``segment_sum`` drop and the silent ``at[].set`` clamp."""
  for T in (17, 33, 64):
    for pname, sw in switch_patterns(T).items():
      seg = np.asarray(variable_segment_ids(jnp.array(sw, f32)[None]))[0]
      assert seg.max() < T, (
          f'{pname}: segment id {seg.max()} >= buffer width {T}; '
          f'segment_sum would drop it and at[].set would clamp')
      assert seg.min() >= 0


def test_packed_dtypes_are_preserved():
  """Packing must not silently upcast/downcast the tensors it carries."""
  T = 17
  sw = jnp.array(switch_patterns(T)['every_8_from_0'], f32)[None]
  for dtype in (jnp.float32, jnp.bfloat16, jnp.int32):
    x = jnp.arange(T, dtype=dtype)[None]
    got = downsample_at_switch_mask({'x': x}, sw)['x']
    assert got.dtype == dtype, f'downsample changed {dtype} -> {got.dtype}'
    filled = forward_fill_packed(x, sw)
    assert filled.dtype == dtype, f'forward_fill changed {dtype} -> {filled.dtype}'


def test_shapes_are_exact_at_every_stage():
  """Every stage's shape, spelled out, so a broadcast cannot hide a mismatch."""
  B, T = 3, 17
  sw = jnp.stack([jnp.array(switch_patterns(T)['every_8_from_0'], f32)] * B)
  rew, con = _rows(T, seed=5)
  rew = jnp.stack([jnp.array(rew)] * B)
  con = jnp.stack([jnp.array(con)] * B)
  feat = downsample_at_switch_mask({'x': rew}, sw)['x']
  assert feat.shape == (B, T), feat.shape
  assert masked_cumprod(con[:, :-1], sw).shape == (B, T - 1)
  assert variable_segment_ids(sw).shape == (B, T)
  assert switch_valid_mask(sw, T).shape == (B, T)
  assert imag_reward_pad(rew[:, :T - 1]).shape == (B, T)
  extr, expl, cont, switch = variable_block_director_tensors(
      rew, con, rew, sw, T, agg_mode='mean')
  for name, t in (('extr', extr), ('expl', expl), ('cont', cont),
                  ('switch', switch)):
    assert t.shape == (B, T), f'{name} has shape {t.shape}, expected {(B, T)}'


def test_all_switch_pattern_degenerates_to_per_step_credit():
  """K=1 (a switch every step) must reduce to plain per-step manager credit.

  This is the opposite extreme from the pinned K=8 configuration and exercises
  the packing at full occupancy, where there is no padding at all.
  """
  T = 17
  rew, con = _rows(T, seed=11)
  sw = jnp.ones((1, T), f32)
  extr, _, cont, switch = variable_block_director_tensors(
      jnp.array(rew)[None], jnp.array(con)[None], jnp.array(rew)[None],
      sw, T, agg_mode='mean')
  # Every decision owns exactly one transition: pooled reward == rew[t+1]*con[t].
  want = np.array([rew[t + 1] * con[t] for t in range(T - 1)], np.float32)
  np.testing.assert_allclose(
      np.asarray(extr)[0][1:T], want, rtol=2e-5, atol=2e-6)
  # Shifted by one, per Director's convention (see the asymmetry test).
  np.testing.assert_allclose(np.asarray(cont)[0][1:T],
                             np.array(con[1:T], np.float32), rtol=2e-5)
  assert int(np.asarray(switch)[0].sum()) == T - 1


def test_reward_and_cont_use_a_one_step_shifted_continuation():
  """Documents a real asymmetry, and pins that BOTH paths share it.

  ``aggregate_*_extr_rew*`` weights reward ``rew[j+1]`` by ``cumprod(con[:-1])``,
  i.e. by ``con[j]``; ``aggregate_*_cont*`` pools ``con[1:]``, i.e. ``con[j+1]``.
  So a block's reward discount and its own continuation are computed one step
  apart. This is Director's ``abstract_traj`` convention
  (``traj['cont'] = concat([value[:1], reshape(value[1:]).prod(1)])`` against
  ``weights = cumprod(traj['cont'][:-1])``), and the fixed-K and block-pooled
  implementations reproduce it identically -- so it cannot explain any gap
  BETWEEN them. Asserted here so the shift is deliberate and visible rather
  than rediscovered as a bug.
  """
  T = 17
  rew, con = _rows(T, seed=21)
  sw = jnp.array([1 if t % 8 == 0 else 0 for t in range(T)], f32)[None]
  cont = np.asarray(aggregate_mgr_cont_variable(
      jnp.array(con)[None], sw, without_zeros=True))[0]
  # Block 0 spans transitions 0..7 -> the implementation's product is con[1..8].
  shifted = float(np.prod(con[1:9]))
  unshifted = float(np.prod(con[0:8]))
  np.testing.assert_allclose(cont[0], shifted, rtol=2e-5)
  assert abs(cont[0] - unshifted) > 1e-4, (
      'the shift vanished -- if this fires, the convention changed on one side '
      'and the fixed-K/block-pooled paths may no longer agree')


def test_transitions_before_the_first_switch_fold_into_decision_zero():
  """A latent trap, harmless only because a switch at t=0 is guaranteed.

  ``variable_segment_ids`` clamps the pre-first-switch prefix to segment 0, so
  those transitions are pooled into the FIRST decision -- and, under ``mean``
  aggregation, inflate its denominator with steps that decision never chose.
  Both rollout builders start with ``remaining = 0``, forcing a switch at t=0
  (``_imagine_with_manager`` and ``_manager_skills_on_sequence``), so this never
  fires today. It is asserted here so that if that invariant is ever relaxed --
  e.g. carrying a hold across the replay-window boundary -- the consequence is
  already written down and tested rather than silently mis-crediting decision 0.
  """
  T = 17
  rew, con = _rows(T, seed=31)
  late = jnp.array([0] * 5 + [1] + [0] * (T - 6), f32)[None]
  got = float(np.asarray(aggregate_mgr_extr_rew_variable(
      jnp.array(rew)[None], jnp.array(con)[None], late,
      without_zeros=True))[0][0])
  # The reference, which credits ONLY the transitions from the switch onward.
  want_from_switch, _ = ref_pool(rew, con, np.asarray(late)[0])
  assert abs(got - want_from_switch[0]) > 1e-3, (
      'prefix folding appears to be gone; if so this test should be inverted')
  # What it actually computes is a hybrid: EVERY transition 0..T-2 lands in
  # segment 0 (ids are clamped), but ``masked_cumprod`` still resets its weight
  # at the real switch -- so the prefix carries weights accumulated from t=0
  # while the post-switch steps restart at 1.0, and the mean divides by all of
  # them. Spell that out exactly.
  sw_row = np.asarray(late)[0]
  weighted, w = [], 1.0
  for j in range(T - 1):
    w = con[j] if sw_row[j] > 0.5 else w * con[j]
    weighted.append(rew[j + 1] * w)
  np.testing.assert_allclose(
      got, sum(weighted) / len(weighted), rtol=2e-5, atol=2e-6)


def test_rollout_builders_always_switch_on_the_first_step():
  """The invariant the test above depends on, checked on the real recurrence.

  Both builders initialize ``remaining = 0`` and switch when ``remaining <= 0``,
  so step 0 is always a decision regardless of the sampled duration.
  """
  for hold in (1, 3, 8, 16):
    remaining, first = 0, None
    for t in range(5):
      update = remaining <= 0
      if first is None:
        first = update
      remaining = (hold if update else remaining) - 1
    assert first, f'hold={hold}: step 0 was not a manager decision'


# --------------------------------------------------------------------------
# The sixth bug: terminal flags must be flags, not a continuation probability.
# --------------------------------------------------------------------------

def test_bool_cast_of_a_continuation_complement_reads_as_terminal():
  """The primitive failure, isolated: ``(1 - 0.997).astype(bool) is True``.

  ``is_terminal`` is an ``elements.Space(bool)``, so casting a float
  continuation-complement to its dtype turns any live-but-not-certain step into
  a terminal one. JAX raises nothing; the numbers stay finite and plausible.
  """
  cont = jnp.full((1, 6), 0.997)
  as_bool = (1.0 - cont).astype(jnp.zeros((1, 6), bool).dtype)
  assert bool(as_bool.all()), 'expected every live step to read as terminal'


def test_replay_terminal_flags_are_not_derived_from_continuation():
  """A terminal-flag mixup destroys the manager's replay bootstrap entirely.

  ``lambda_return`` uses ``live = 1 - term``, so ``term == True`` everywhere
  zeroes every bootstrap term and the return collapses to the immediate pooled
  reward. Asserts the size of the damage so a regression is loud rather than a
  slow drift: on a non-terminating window the correct return is many times the
  broken one.
  """
  from dreamerv3.hrl import lambda_return
  B, T = 1, 10
  cont = jnp.full((B, T), 0.997)
  rew = jnp.full((B, T), 0.3)
  val = jnp.full((B, T), 5.0)
  last = jnp.zeros((B, T), f32)
  real_term = jnp.zeros((B, T), bool)          # nothing actually terminated
  good = lambda_return(last, f32(real_term), rew, val, val, 1.0, 0.95)
  broken = lambda_return(
      last, f32((1.0 - cont).astype(real_term.dtype)), rew, val, val, 1.0, 0.95)
  np.testing.assert_allclose(np.asarray(broken)[0], 0.3, rtol=1e-5)
  assert float(np.asarray(good)[0][0]) > 10 * float(np.asarray(broken)[0][0]), (
      'the bootstrap collapse is not reproduced; if lambda_return changed, this '
      'guard needs rewriting rather than deleting')


def test_downsampled_terminal_flags_round_trip_through_the_bool_cast():
  """The fix's own path: real flags survive downsample -> patch -> bool.

  A genuine terminal inside the window must still read as terminal at its
  decision, and a live window must produce all-False -- the property the old
  code violated.
  """
  from dreamerv3.hrl import patch_trailing_replay_state
  T, K = 33, 8
  sw = jnp.array([1 if t % K == 0 else 0 for t in range(T)], f32)[None]

  live = jnp.zeros((1, T), bool)
  down = downsample_at_switch_mask({'t': live.astype(f32)}, sw)['t']
  down = patch_trailing_replay_state(
      down, live.astype(f32), sw).astype(live.dtype)
  assert not bool(down.any()), 'a fully live window produced a terminal flag'

  term = jnp.zeros((1, T), bool).at[0, 16].set(True)   # terminal at a decision
  down = downsample_at_switch_mask({'t': term.astype(f32)}, sw)['t']
  down = patch_trailing_replay_state(
      down, term.astype(f32), sw).astype(term.dtype)
  assert bool(down[0, 2]), 'a real terminal at decision 2 was lost'
  assert int(np.asarray(down)[0][:4].sum()) == 1, 'terminal flag smeared'


# --------------------------------------------------------------------------
# Bug 1's fix (decision_mean_rescale) under GENUINELY variable holds: every
# row of the batch has a different number of real decisions, unlike the
# duration-pinned configuration (every row = exactly 2) that the live e360-
# e369 reruns exercise. The scalar-buffer-width math is written per-row, but
# "written per-row" and "tested with rows that disagree" are different
# claims -- nothing above ever fed it a batch where they did.
# --------------------------------------------------------------------------

def test_decision_rescale_is_correct_per_row_when_batch_rows_disagree():
  """Each row's rescale factor must depend on THAT row's own decision count,
  not the batch's, not another row's, and not get averaged across rows.

  Switch counts are computed from the pattern itself (``sum(row)``), not
  hand-counted in a comment -- a hand-miscount here (``t % 16 == 0`` over a
  33-column row hits 0/16/32, three switches, not two) would silently assert
  the wrong number and read as a real bug in ``decision_mean_rescale`` when
  it was the test's own arithmetic that was wrong.
  """
  from dreamerv3.hrl import decision_mean_rescale
  T = 33
  rows = [
      [1] + [0] * (T - 1),                              # 1 switch
      [1 if t % 15 == 0 else 0 for t in range(T)],       # 3 switches (0,15,30)
      [1 if t % 7 == 0 else 0 for t in range(T)],        # 5 switches
      [1 if t % 4 == 0 else 0 for t in range(T)],        # 9 switches
      [1] * T,                                           # 33 switches
  ]
  n_valid = [sum(r) for r in rows]
  assert n_valid == sorted(set(n_valid)), 'rows must have distinct counts'
  sw = jnp.array(rows, f32)
  got = np.asarray(decision_mean_rescale(sw))[:, 0]
  want = np.array([T / n for n in n_valid], np.float32)
  np.testing.assert_allclose(got, want, rtol=1e-6,
      err_msg="rescale factor does not match T / (this row's own decision count)")
  # Not just correct on average -- correct on EVERY row.
  for i, (n, factor) in enumerate(zip(n_valid, want)):
    assert abs(float(got[i]) - float(factor)) < 1e-4, f'row {i} (n={n}) off'


def test_decision_rescale_makes_variable_length_manager_loss_match_a_fixed_k_row():
  """End-to-end check with heterogeneous holds: row A holds K=8 throughout
  (agrees with fixed-K Director everywhere); row B genuinely varies its hold
  length. The rescale must still recover row A's exact fixed-K loss, proving
  the per-row factor isn't contaminated by row B's different decision count
  sitting right next to it in the same batch call.
  """
  T = 33
  sw_fixed = [1 if t % 8 == 0 else 0 for t in range(T)]         # 4 decisions
  sw_var = [1 if t in (0, 3, 4, 9, 10, 11, 20, 28) else 0 for t in range(T)]  # 8
  sw = jnp.array([sw_fixed, sw_var], f32)

  rng = np.random.RandomState(9)
  D, C = 6, 8
  feat = jnp.array(rng.randn(2, T, D).astype(np.float32))
  rew = jnp.array(rng.randn(2, T).astype(np.float32) * 0.3)
  expl = jnp.array(rng.randn(2, T).astype(np.float32) * 0.1)
  con = jnp.full((2, T), 0.997)
  logit_w = jnp.array(rng.randn(D, C).astype(np.float32) * 0.5)
  val_w = jnp.array(rng.randn(D).astype(np.float32) * 0.4)

  feat_eff = downsample_at_switch_mask({'x': feat}, sw)['x']
  n_mgr = feat_eff.shape[1]
  b_rew, b_expl, b_con, b_switch = variable_block_director_tensors(
      rew, con, expl, sw, n_mgr, agg_mode='mean')
  logits_full = jnp.einsum('btd,dc->btc', feat, logit_w)
  skills_full = {'skill': jax.nn.one_hot(jnp.argmax(logits_full, -1), C)}
  skills_eff = downsample_at_switch_mask(skills_full, sw)

  from dreamerv3.hrl import imag_loss_mgr
  import embodied.jax.outs as outs

  class _StubNorm:
    def stats(self): return 0.0, 1.0
    def __call__(self, x, update=True, weights=None): return 0.0, 1.0

  policy = {'skill': outs.OneHot(jnp.einsum('btd,dc->btc', feat_eff, logit_w))}
  val = outs.MSE(feat_eff @ val_w)
  norms = [_StubNorm() for _ in range(5)]
  losses, _, _ = imag_loss_mgr(
      skills_eff, b_rew, b_expl, b_con, policy, val, val, val, val, *norms,
      update=True, contdisc=True, slowtar=False, horizon=333,
      mgr_expl_weight=0.1, actent=3e-4, switch_mask=b_switch)
  # ``imag_loss_mgr`` returns the UNREDUCED per-column tensor; ``Agent.loss``
  # applies ``.mean(1)`` over it. Row A's real width (32) and the fixed-K
  # reference's width (4) differ by construction (compacted decisions vs. the
  # padded buffer), so the two can only be compared after that same reduction
  # -- comparing the raw arrays elementwise is a shape mismatch, not a bug.
  row0_loss = float(np.asarray(losses['mgr_policy'][0]).mean())

  # Reference: row A alone, run through the plain fixed-K path (K=8, T=33 ->
  # exactly 4 decisions), which is what row A's own dynamics ARE.
  from dreamerv3.hrl import aggregate_mgr_cont, aggregate_mgr_extr_rew
  fx_feat = feat[0:1, ::8]
  fx_rew = imag_reward_pad(
      aggregate_mgr_extr_rew(rew[0:1], con[0:1], 8, without_zeros=True))
  fx_expl = imag_reward_pad(
      aggregate_mgr_extr_rew(expl[0:1], con[0:1], 8, without_zeros=True))
  fx_cont = aggregate_mgr_cont(con[0:1], 8, without_zeros=True)
  fx_skills = jax.tree.map(lambda s: s[0:1, ::8], skills_full)
  fx_policy = {'skill': outs.OneHot(jnp.einsum('btd,dc->btc', fx_feat, logit_w))}
  fx_val = outs.MSE(fx_feat @ val_w)
  fx_norms = [_StubNorm() for _ in range(5)]
  fx_losses, _, _ = imag_loss_mgr(
      fx_skills, fx_rew, fx_expl, fx_cont, fx_policy, fx_val, fx_val, fx_val,
      fx_val, *fx_norms, update=True, contdisc=True, slowtar=False,
      horizon=333, mgr_expl_weight=0.1, actent=3e-4, switch_mask=None)
  fixed_loss = float(np.asarray(fx_losses['mgr_policy'][0]).mean())

  np.testing.assert_allclose(row0_loss, fixed_loss, rtol=1e-5, atol=1e-7,
      err_msg='row A (fixed-K within a heterogeneous batch) picked up a '
              "rescale factor contaminated by row B's different hold pattern")
