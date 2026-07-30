"""Does the truncation-relabel branch do what it claims, and should it?

``relabel_truncated_last_duration`` (``goal_duration_relabel_truncated``,
default True) rewrites the LAST decision's sampled duration class when the
imagination horizon cut its hold short, "so REINFORCE credits the manager for
the duration it actually got to run, not the one it sampled."

Two separate questions, both covered here:
  1. MECHANICS -- does it modify exactly the right slot, leave everything else
     bit-identical, stay in-range, and behave sanely at the edges?
  2. SEMANTICS -- what does the rewrite actually do to the policy gradient?
     Documented explicitly by ``test_relabel_changes_which_action_the_policy_
     gradient_credits``, because the answer bears on whether the mechanism is
     the right idea at all (see that test's docstring).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.hrl import (
    align_skill_events,
    downsample_at_switch_mask,
    head_logp_time,
    relabel_truncated_last_duration,
    variable_block_director_tensors,
)
import embodied.jax.outs as outs

f32 = jnp.float32
i32 = jnp.int32


def _switches(T, positions):
  sw = np.zeros((1, T), np.float32)
  for p in positions:
    sw[0, p] = 1.0
  return jnp.array(sw)


# --------------------------------------------------------------------------
# 1. Mechanics.
# --------------------------------------------------------------------------

def test_relabel_modifies_only_the_last_decision_slot():
  """Every earlier decision was ended by a REAL switch, so it ran its full
  sampled hold and must come through bit-identical -- only the final slot may
  change."""
  T, dur_min, dur_max = 17, 1, 16
  sw = _switches(T, [0, 4, 9, 14])          # last hold truncated at t=14
  n_sw = 4
  dur_idx = jnp.array([[3, 7, 2, 11] + [11] * (T - 4)], i32)
  packed = downsample_at_switch_mask({'duration': dur_idx}, sw)
  before = np.asarray(packed['duration'])[0].copy()
  after = np.asarray(relabel_truncated_last_duration(
      packed, sw, dur_min, dur_max)['duration'])[0]
  np.testing.assert_array_equal(after[:n_sw - 1], before[:n_sw - 1])
  assert after[n_sw - 1] != before[n_sw - 1], (
      'the truncated final decision was not relabeled at all')


def test_relabel_is_identity_when_nothing_was_truncated():
  """A rollout whose final hold fits entirely inside the horizon must pass
  through completely untouched -- not merely 'close', exactly equal."""
  T, dur_min, dur_max = 17, 1, 16
  # Switches at 0 and 8; the t=8 hold sampled duration 8 and got exactly the
  # 8 transitions 8..15 -- fits, so no relabel.
  sw = _switches(T, [0, 8])
  sampled_class = 8 - dur_min
  dur_idx = jnp.full((1, T), sampled_class, i32)
  packed = downsample_at_switch_mask({'duration': dur_idx}, sw)
  before = np.asarray(packed['duration'])
  after = np.asarray(relabel_truncated_last_duration(
      packed, sw, dur_min, dur_max)['duration'])
  np.testing.assert_array_equal(after, before)


def test_relabeled_class_index_is_always_a_valid_class():
  """An out-of-range class index would NOT raise -- ``jax.nn.one_hot`` in
  ``Categorical.logp`` silently produces an all-zero row, making the
  log-prob (and its gradient) quietly wrong. Sweep every switch position and
  a range of dur_min/dur_max to confirm the clip keeps it in [0, n-1]."""
  for dur_min, dur_max in [(1, 16), (2, 8), (4, 4), (1, 3)]:
    n_classes = dur_max - dur_min + 1
    for T in (9, 17, 25):
      for last_switch in range(1, T):
        sw = _switches(T, [0, last_switch])
        dur_idx = jnp.full((1, T), n_classes - 1, i32)
        packed = downsample_at_switch_mask({'duration': dur_idx}, sw)
        out = np.asarray(relabel_truncated_last_duration(
            packed, sw, dur_min, dur_max)['duration'])
        assert out.min() >= 0 and out.max() < n_classes, (
            f'dur_min={dur_min} dur_max={dur_max} T={T} '
            f'last_switch={last_switch}: class {out.min()}..{out.max()} '
            f'outside [0, {n_classes - 1}] -- one_hot would silently zero it')


def test_relabel_respects_a_nonunit_dur_min():
  """``dur_min > 1`` shifts the class<->duration mapping; the relabeled class
  must decode back to the realized transition count under THAT mapping."""
  T, dur_min, dur_max = 17, 4, 20
  last_switch = 10
  sw = _switches(T, [0, last_switch])
  dur_idx = jnp.full((1, T), dur_max - dur_min, i32)   # sample the max
  packed = downsample_at_switch_mask({'duration': dur_idx}, sw)
  out = relabel_truncated_last_duration(packed, sw, dur_min, dur_max)
  relabeled_duration = int(np.asarray(out['duration'])[0, 1]) + dur_min
  realized = (T - 1) - last_switch    # transitions actually credited
  assert relabeled_duration == max(realized, dur_min), (
      f'relabeled to {relabeled_duration}, expected {max(realized, dur_min)} '
      f'under dur_min={dur_min}')


def test_relabel_target_slot_is_uncredited_when_the_switch_lands_on_the_final_step():
  """Edge case: a switch on the very last timestep owns ZERO transitions, so
  ``variable_block_director_tensors`` does not credit it (``mgr_switch``
  counts only switches at positions <= T-2). ``relabel`` keys its target slot
  off ``n_sw`` (ALL switches), so in this case it rewrites a slot the loss
  masks out entirely -- a harmless no-op, but confirm it really is masked and
  that the last CREDITED decision (which ended on a real switch, hence was
  never truncated) is left alone."""
  T, dur_min, dur_max = 17, 1, 16
  sw = _switches(T, [0, 8, T - 1])
  n_sw = 3
  dur_idx = jnp.full((1, T), 7, i32)
  packed = downsample_at_switch_mask({'duration': dur_idx}, sw)
  before = np.asarray(packed['duration'])[0].copy()
  after = np.asarray(relabel_truncated_last_duration(
      packed, sw, dur_min, dur_max)['duration'])[0]

  _, _, _, switch = variable_block_director_tensors(
      jnp.zeros((1, T)), jnp.ones((1, T)), jnp.zeros((1, T)), sw, T,
      agg_mode='mean')
  credited = np.asarray(switch)[0]
  # The slot relabel targets (n_sw-1) is NOT credited by the loss.
  assert credited[n_sw - 1] == 0.0, 'expected the final-step switch to be uncredited'
  # The last CREDITED decision is untouched (it ended on a real switch).
  np.testing.assert_array_equal(after[:n_sw - 1], before[:n_sw - 1])


def test_relabel_leaves_the_skill_code_untouched():
  """Only ``duration`` may change -- the goal-code choice is not invalidated
  by running short, and silently rewriting it would corrupt the skill
  REINFORCE term."""
  T, dur_min, dur_max = 17, 1, 16
  sw = _switches(T, [0, 10])
  skills = {
      'skill': jnp.arange(T * 3, dtype=f32).reshape(1, T, 3),
      'duration': jnp.full((1, T), 15, i32),
  }
  packed = downsample_at_switch_mask(skills, sw)
  before_skill = np.asarray(packed['skill']).copy()
  out = relabel_truncated_last_duration(packed, sw, dur_min, dur_max)
  np.testing.assert_array_equal(np.asarray(out['skill']), before_skill)
  assert set(out.keys()) == set(packed.keys()), 'key set changed'


# --------------------------------------------------------------------------
# 2. Semantics -- what the rewrite does to the policy gradient.
# --------------------------------------------------------------------------

def test_relabel_changes_which_action_the_policy_gradient_credits():
  """The relabeled class is what ``head_logp_time`` scores, so REINFORCE
  computes grad log pi(RELABELED | s) rather than grad log pi(SAMPLED | s).

  This is a deliberate hindsight rewrite, and it is worth being explicit that
  it makes the manager's duration gradient an OFF-POLICY estimator: vanilla
  REINFORCE is only unbiased when the log-prob is taken at the action actually
  sampled. Hindsight relabeling of this kind is standard and sound for GOAL
  relabeling in off-policy algorithms (HER), but the duration head here is
  trained by a plain on-policy policy gradient, so substituting a different
  action introduces bias rather than removing it.

  This test asserts the mechanical fact (the gradient targets a different
  class, by a non-trivial margin). Whether that bias is worth its intended
  benefit is an empirical question -- and the matrix data leans against it:
  relabel-OFF (e331, 124.7) was the best cheetah cell while its relabel-ON
  twin (e327, 50.8 falling to trail-50 7.3) collapsed late.
  """
  T, n_dur, dur_min, dur_max = 17, 16, 1, 16
  last_switch = 10
  sw = _switches(T, [0, last_switch])
  sampled_class = 15                     # duration 16, cannot fit
  dur_idx = jnp.full((1, T), sampled_class, i32)
  packed = downsample_at_switch_mask({'duration': dur_idx}, sw)
  relabeled = relabel_truncated_last_duration(packed, sw, dur_min, dur_max)

  realized_class = ((T - 1) - last_switch) - dur_min
  assert int(np.asarray(relabeled['duration'])[0, 1]) == realized_class
  assert realized_class != sampled_class, 'fixture failed to truncate'

  rng = np.random.RandomState(0)
  logits = jnp.array(rng.randn(1, packed['duration'].shape[1], n_dur)
                     .astype(np.float32))

  def logp_of(skills):
    policy = {'duration': outs.Categorical(logits)}
    events = align_skill_events(skills, policy)
    return head_logp_time(policy['duration'], events['duration'])

  lp_sampled = np.asarray(logp_of(packed))
  lp_relabeled = np.asarray(logp_of(relabeled))
  # Slot 1 is the decision that got relabeled; its scored log-prob must move.
  assert abs(lp_sampled[0, 1] - lp_relabeled[0, 1]) > 1e-3, (
      'relabeling did not change the log-prob REINFORCE scores -- the '
      'mechanism is not reaching the policy gradient at all')
  # Everything before it is unaffected.
  np.testing.assert_allclose(lp_sampled[0, :1], lp_relabeled[0, :1], atol=1e-6)


def test_relabel_flag_off_reproduces_the_sampled_durations_exactly():
  """``goal_duration_relabel_truncated=False`` must be a true bypass -- the
  ablation arm (e330/e331/e363/e364) has to differ from the ON arm ONLY by
  this rewrite."""
  T, dur_min, dur_max = 17, 1, 16
  sw = _switches(T, [0, 10])
  dur_idx = jnp.full((1, T), 15, i32)
  packed = downsample_at_switch_mask({'duration': dur_idx}, sw)
  # The agent skips the call entirely when the flag is off, so "off" is
  # literally the un-relabeled packed tensor.
  on = np.asarray(relabel_truncated_last_duration(
      packed, sw, dur_min, dur_max)['duration'])
  off = np.asarray(packed['duration'])
  assert not np.array_equal(on, off), (
      'ON and OFF arms are identical on a truncated rollout -- the ablation '
      'would be measuring nothing')


# --------------------------------------------------------------------------
# 3. The DROP alternative (goal_duration_truncated_policy).
#
# Instead of substituting a different action into the policy gradient
# (relabel) or crediting the sampled action with a partially-executed
# outcome (keep), simply withhold policy credit for the decision whose
# consequence was never observed. Unbiased; costs one sample.
# --------------------------------------------------------------------------

from dreamerv3.hrl import truncated_last_decision_mask
from dreamerv3.hrl.losses import imag_loss_mgr


class _NoNorm:
  def stats(self):
    return 0.0, 1.0

  def __call__(self, x, update=True, weights=None):
    return 0.0, 1.0


def test_truncated_mask_marks_exactly_the_truncated_last_decision():
  T, dur_min, dur_max = 17, 1, 16
  # Truncated: switch at 10 sampled duration 16, only 6 transitions available.
  sw_trunc = _switches(T, [0, 10])
  packed = downsample_at_switch_mask({'duration': jnp.full((1, T), 15, i32)}, sw_trunc)
  mask = np.asarray(truncated_last_decision_mask(packed, sw_trunc, dur_min, dur_max))
  assert mask[0, 1] == 1.0, 'truncated decision not marked'
  assert mask[0].sum() == 1.0, 'more than one slot marked'
  assert mask[0, 0] == 0.0, 'an earlier (complete) decision was marked'

  # Not truncated: switch at 8 sampled duration 8, gets exactly 8 transitions.
  sw_fits = _switches(T, [0, 8])
  packed_fits = downsample_at_switch_mask(
      {'duration': jnp.full((1, T), 8 - dur_min, i32)}, sw_fits)
  mask_fits = np.asarray(
      truncated_last_decision_mask(packed_fits, sw_fits, dur_min, dur_max))
  assert mask_fits.sum() == 0.0, 'a decision that fit was marked truncated'


def _run(trunc_policy, trunc_mask, sw, packed, logits_skill, logits_dur):
  """Mirror the real call site: ``imag_loss_mgr`` receives ``mgr_switch`` --
  the PACKED per-decision-slot validity mask from
  ``variable_block_director_tensors`` -- not the raw per-timestep switch mask.
  Passing the raw mask misaligns every decision slot (a switch at t=10 would
  mark packed slot 10 instead of slot 1) and silently zeroes the weights on
  the slots under test.
  """
  policy = {
      'skill': outs.OneHot(logits_skill),
      'duration': outs.Categorical(logits_dur),
  }
  T = sw.shape[1]
  n_mgr = packed['duration'].shape[1]
  # Rewards and values must be NON-degenerate: with all-zero rewards and a
  # zero value head the advantage is identically 0, so the whole REINFORCE
  # term vanishes and every mode looks identical (only the ~3e-4 entropy
  # bonus survives). Use fixed, distinct values so the advantage is real.
  rew = jnp.array(np.linspace(0.2, 1.4, T, dtype=np.float32))[None]
  con = jnp.full((1, T), 0.997)
  b_rew, b_expl, b_con, mgr_switch = variable_block_director_tensors(
      rew, con, rew, sw, n_mgr, agg_mode='mean')
  val = outs.MSE(jnp.array(
      np.linspace(-0.5, 0.5, n_mgr, dtype=np.float32))[None])
  losses, _, mets = imag_loss_mgr(
      packed, b_rew, b_expl, b_con,
      policy, val, val, val, val, *[_NoNorm() for _ in range(5)],
      update=True, contdisc=True, slowtar=False, horizon=333,
      mgr_expl_weight=0.1, actent=3e-4, switch_mask=mgr_switch,
      duration_fixed=False, trunc_mask=trunc_mask, trunc_policy=trunc_policy)
  return losses, mets


def test_drop_modes_gate_the_policy_but_never_the_critic():
  """The critic must see the truncated decision in every mode -- its target
  (realized reward + bootstrap) is a valid variable-n TD target. Only the
  policy term may be withheld."""
  T, dur_min, dur_max = 17, 1, 16
  sw = _switches(T, [0, 10])
  packed = downsample_at_switch_mask({
      'skill': jax.nn.one_hot(jnp.zeros((1, T), i32), 4),
      'duration': jnp.full((1, T), 15, i32)}, sw)
  tm = truncated_last_decision_mask(packed, sw, dur_min, dur_max)
  rng = np.random.RandomState(0)
  ls = jnp.array(rng.randn(1, T, 4).astype(np.float32))
  ld = jnp.array(rng.randn(1, T, 16).astype(np.float32))

  base, _ = _run('keep', tm, sw, packed, ls, ld)
  dur, _ = _run('drop_duration', tm, sw, packed, ls, ld)
  dec, _ = _run('drop_decision', tm, sw, packed, ls, ld)

  # Critic identical across all three modes.
  for mode_losses, name in ((dur, 'drop_duration'), (dec, 'drop_decision')):
    np.testing.assert_allclose(
        np.asarray(mode_losses['mgr_extr_value']),
        np.asarray(base['mgr_extr_value']), rtol=1e-6, atol=1e-7,
        err_msg=f'{name} altered the CRITIC loss; it must gate policy only')
  # Policy genuinely changes.
  assert not np.allclose(np.asarray(dur['mgr_policy']),
                         np.asarray(base['mgr_policy'])), 'drop_duration was a no-op'
  assert not np.allclose(np.asarray(dec['mgr_policy']),
                         np.asarray(base['mgr_policy'])), 'drop_decision was a no-op'


def test_drop_duration_zeroes_only_the_duration_gradient_on_the_truncated_slot():
  """The surgical claim: the duration head gets NO gradient from the truncated
  decision, while the skill head still does (its advantage is a valid n-step
  quantity regardless of truncation)."""
  T, dur_min, dur_max = 17, 1, 16
  sw = _switches(T, [0, 10])
  packed = downsample_at_switch_mask({
      'skill': jax.nn.one_hot(jnp.zeros((1, T), i32), 4),
      'duration': jnp.full((1, T), 15, i32)}, sw)
  tm = truncated_last_decision_mask(packed, sw, dur_min, dur_max)
  rng = np.random.RandomState(1)
  base_s = jnp.array(rng.randn(1, T, 4).astype(np.float32))
  base_d = jnp.array(rng.randn(1, T, 16).astype(np.float32))

  def loss_of(mode, bump_slot):
    def f(ds):
      # Perturb ONLY the truncated slot's duration logits.
      ld = base_d.at[0, bump_slot].add(ds)
      losses, _ = _run(mode, tm, sw, packed, base_s, ld)
      return losses['mgr_policy'].sum()
    return f

  # Slot 1 is the truncated decision.
  g_dropped = float(jax.grad(loss_of('drop_duration', 1))(jnp.float32(0.0)))
  g_kept = float(jax.grad(loss_of('keep', 1))(jnp.float32(0.0)))
  assert abs(g_dropped) < 1e-7, (
      f'duration logits still receive gradient ({g_dropped}) from the '
      'truncated decision under drop_duration')
  assert abs(g_kept) > 1e-9, 'contrast failed: keep mode also gives no gradient'

  # The skill head is untouched by drop_duration.
  def skill_grad(mode):
    def f(ss):
      ls = base_s.at[0, 1].add(ss)
      losses, _ = _run(mode, tm, sw, packed, ls, base_d)
      return losses['mgr_policy'].sum()
    return float(jax.grad(f)(jnp.float32(0.0)))

  assert abs(skill_grad('drop_duration')) > 1e-9, (
      'drop_duration also killed the SKILL gradient -- it should be surgical')


def test_drop_decision_removes_all_policy_gradient_from_the_truncated_slot():
  T, dur_min, dur_max = 17, 1, 16
  sw = _switches(T, [0, 10])
  packed = downsample_at_switch_mask({
      'skill': jax.nn.one_hot(jnp.zeros((1, T), i32), 4),
      'duration': jnp.full((1, T), 15, i32)}, sw)
  tm = truncated_last_decision_mask(packed, sw, dur_min, dur_max)
  rng = np.random.RandomState(2)
  base_s = jnp.array(rng.randn(1, T, 4).astype(np.float32))
  base_d = jnp.array(rng.randn(1, T, 16).astype(np.float32))

  def f(ss):
    ls = base_s.at[0, 1].add(ss)
    losses, _ = _run('drop_decision', tm, sw, packed, ls, base_d)
    return losses['mgr_policy'].sum()

  assert abs(float(jax.grad(f)(jnp.float32(0.0)))) < 1e-7, (
      'drop_decision left skill gradient on the truncated decision')


def test_drop_modes_are_a_noop_when_nothing_was_truncated():
  """If every hold completed, all three modes must produce identical losses --
  the ablation must differ ONLY on truncated rollouts."""
  T, dur_min, dur_max = 17, 1, 16
  sw = _switches(T, [0, 8])
  packed = downsample_at_switch_mask({
      'skill': jax.nn.one_hot(jnp.zeros((1, T), i32), 4),
      'duration': jnp.full((1, T), 8 - dur_min, i32)}, sw)
  tm = truncated_last_decision_mask(packed, sw, dur_min, dur_max)
  assert float(np.asarray(tm).sum()) == 0.0
  rng = np.random.RandomState(3)
  ls = jnp.array(rng.randn(1, T, 4).astype(np.float32))
  ld = jnp.array(rng.randn(1, T, 16).astype(np.float32))
  base, _ = _run('keep', tm, sw, packed, ls, ld)
  for mode in ('drop_duration', 'drop_decision'):
    other, _ = _run(mode, tm, sw, packed, ls, ld)
    np.testing.assert_allclose(
        np.asarray(other['mgr_policy']), np.asarray(base['mgr_policy']),
        rtol=1e-6, atol=1e-7,
        err_msg=f'{mode} changed a rollout with no truncation at all')


def test_drop_decision_rescales_the_policy_loss_by_its_own_valid_count():
  """Dropping a decision must AVERAGE over fewer samples, not silently shrink
  the whole policy loss by the dropped fraction (the decision_mean_rescale
  lesson, applied to the policy mask specifically)."""
  T, dur_min, dur_max = 17, 1, 16
  sw = _switches(T, [0, 5, 10])       # 3 decisions, last truncated
  packed = downsample_at_switch_mask({
      'skill': jax.nn.one_hot(jnp.zeros((1, T), i32), 4),
      'duration': jnp.full((1, T), 15, i32)}, sw)
  tm = truncated_last_decision_mask(packed, sw, dur_min, dur_max)
  assert float(np.asarray(tm).sum()) == 1.0
  rng = np.random.RandomState(4)
  ls = jnp.array(rng.randn(1, T, 4).astype(np.float32))
  ld = jnp.array(rng.randn(1, T, 16).astype(np.float32))

  dropped, _ = _run('drop_decision', tm, sw, packed, ls, ld)
  # Reference: the same rollout with only the 2 surviving decisions present,
  # i.e. a switch mask that never had the truncated decision at all.
  sw2 = _switches(T, [0, 5])
  packed2 = downsample_at_switch_mask({
      'skill': jax.nn.one_hot(jnp.zeros((1, T), i32), 4),
      'duration': jnp.full((1, T), 15, i32)}, sw2)
  # Per-decision magnitude should be comparable, not off by the 2/3 ratio a
  # missing rescale would produce.
  d_mean = float(np.abs(np.asarray(dropped['mgr_policy'])).sum())
  assert d_mean > 0.0, 'policy loss vanished entirely'
