"""Helpers for reading policy-head distributions in the actor-critic losses.

The manager emits several heads at once (``skill``, and under variable goal
length ``duration``), each possibly wrapped in ``outs.Agg``. These functions
unwrap that, align sampled events to head shapes, and slice log-probs/entropies
to the ``(batch, time - 1)`` convention the AC losses use.
"""
import jax
import jax.numpy as jnp

import embodied.jax.outs as outs

f32 = jnp.float32
i32 = jnp.int32
# Stop-gradient helper: optionally pass gradients through (e.g. reward head).
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)


def manager_reinforce_policy(manager_policy, duration_fixed):
  """Heads that participate in the manager's REINFORCE log-prob sum.

  Excludes 'duration' when ``goal_duration_fixed`` makes its sample causally
  inert (it cannot move switch timing, see ``_duration_steps``) -- otherwise
  the duration head is pushed every decision by an advantage caused entirely
  by the skill/code choice, never by duration itself: a nuisance training
  signal true fixed-K Director never carries (it has no duration head at
  all). Confirmed live via a standalone gradient test before this fix
  (``test_duration_head_reinforce_is_live_under_fixed_duration``, 2026-07-27).
  Does NOT affect ``mgr_ents`` (still computed over every head by the caller,
  used only for logging/entropy-regularizer bookkeeping, never REINFORCE'd
  directly since the adaptive-entropy branch is always taken in practice) or
  the soft duration-prior branches (already separately gated on their own
  weight/adapter args, which callers leave at their off defaults whenever
  ``goal_duration_fixed`` is set).
  """
  if not duration_fixed:
    return manager_policy
  return {k: v for k, v in manager_policy.items() if k != 'duration'}


def align_skill_events(skills, policy):
  """Match skill samples to policy head ``pred()`` shape for ``logp`` / ``entropy``."""
  events = {}
  for k, v in policy.items():
    ref = v.pred()
    e = skills[k] if isinstance(skills, dict) else skills
    e = sg(e)
    if e.shape == ref.shape:
      events[k] = e
    elif e.ndim + 1 == ref.ndim and e.shape == ref.shape[:e.ndim]:
      events[k] = jnp.broadcast_to(e[:, None], ref.shape)
    elif e.ndim == ref.ndim and e.shape[0] == ref.shape[1] and e.shape[1] == ref.shape[0]:
      events[k] = jnp.swapaxes(e, 0, 1)
    else:
      events[k] = jnp.broadcast_to(e, ref.shape)
  return events


def _shift_up(event):
  """Move mass from class ``c`` to ``c+1``; mass at ``C-1`` falls off the end."""
  return jnp.concatenate(
      [jnp.zeros_like(event[..., :1]), event[..., :-1]], -1)


def _shift_down(event):
  """Move mass from class ``c`` to ``c-1``; mass at ``0`` falls off the end."""
  return jnp.concatenate(
      [event[..., 1:], jnp.zeros_like(event[..., :1])], -1)


def smooth_skill_event(event, alpha, topology='ring'):
  """Spread a one-hot skill event over its codebook neighbours.

  ``OneHot.logp`` is ``sum_c log pi_c * event_c``, so replacing the sampled
  one-hot with a kernel over the codebook turns the REINFORCE target into
  ``sum_c w(c, a) log pi_c``: an advantage credited to entry ``a`` is also
  credited, at weight ``alpha``, to ``a-1`` and ``a+1``. With a SOM-ordered
  codebook those entries decode to nearby goals with likely similar returns,
  which a policy over unordered labels cannot exploit -- raising ``a`` tells it
  nothing about ``a+1``.

  ``alpha = 0`` returns the event unchanged and reproduces the standard
  estimator exactly. Otherwise the target stays a distribution over the block's
  classes, so the update is biased in the manner of label smoothing.

  The kernel must match the codebook's ``topology``, because it encodes which
  entries the SOM term actually made adjacent:

  ``'ring'``
    Weights ``(1 - 2*alpha, alpha, alpha)`` on ``(a, a-1, a+1)`` taken mod
    ``C``. Every class has two neighbours, so no normalization is needed.

  ``'line'``
    Same weights, but the path has no wraparound: entry ``0`` and entry ``C-1``
    are the two most distant codes, and ``jnp.roll`` would share credit between
    exactly them. Shifting with zero fill drops the off-the-end weight instead,
    leaving the endpoints' rows summing to ``1 - alpha``; dividing by the row
    sum restores a distribution. The endpoint kernel is therefore
    ``((1 - 2*alpha) / (1 - alpha), alpha / (1 - alpha))``, which raises the
    surviving neighbour's share slightly above ``alpha`` -- an endpoint pick
    splits its credit two ways rather than three.

    Note this differs from the treatment of the same missing neighbour in the
    SOM loss (``vq.BlockCodebook.neighbor_mask``), which masks the term without
    renormalizing, so endpoint entries simply receive less total pull. A
    log-prob target has no such freedom: it must sum to one.

  Acts on the last axis, which is the class axis of an ``(..., L, C)`` code, so
  the ``L`` blocks are smoothed independently.
  """
  if not alpha:
    return event
  assert topology in ('ring', 'line'), topology
  if topology == 'ring':
    return ((1.0 - 2.0 * alpha) * event
            + alpha * jnp.roll(event, 1, -1)
            + alpha * jnp.roll(event, -1, -1))
  raw = ((1.0 - 2.0 * alpha) * event
         + alpha * _shift_up(event)
         + alpha * _shift_down(event))
  return raw / jnp.maximum(raw.sum(-1, keepdims=True), 1e-8)


def _head_inner(head):
  """Unwrap ``outs.Agg`` so skill axes are not confused with time."""
  return head.output if isinstance(head, outs.Agg) else head


def head_logp_time(head, event):
  """Policy log-prob with leading axes ``(batch, time)``."""
  lp = _head_inner(head).logp(sg(event))
  return policy_time_slice(lp)


def head_entropy_time(head):
  """Policy entropy with leading axes ``(batch, time)``."""
  return policy_time_slice(_head_inner(head).entropy())


def head_entropy_perdim_time(head):
  """Per-categorical entropy sliced to ``(batch, time-1, ...)`` (no outer-dim sum).

  For a OneHot head with logits ``(B, T, L, C)``, ``entropy()`` returns
  ``(B, T, L)`` (Categorical entropy already summed over the class axis).
  We slice the trailing time step so it aligns with the AC-style ``[:-1]``
  convention but otherwise leave the outer dims intact so AutoAdapt can adapt
  one Lagrange multiplier per categorical (Director ``actent_perdim=True``).
  """
  ent = _head_inner(head).entropy()
  if ent.ndim < 2:
    # Defensive fallback; shouldn't trigger for standard manager heads.
    ent = ent[None, :]
  return ent[:, :-1]


def policy_time_slice(x):
  """Reduce head outputs to ``(batch, time - 1)`` for AC losses."""
  while x.ndim > 2:
    x = x.sum(-1)
  if x.ndim == 1:
    x = x[:, None]
  return x[:, :-1]
