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
      # The event is missing the head's TRAILING (class) axis, so the new axis
      # belongs at the end. ``e[:, None]`` inserted it at position 1 instead,
      # which makes the broadcast fail outright for the usual (B, T) -> (B, T, C)
      # case and, when T happens to equal C, silently broadcasts the time axis
      # into the class axis. Dead in production only because every OneHot head's
      # ``sample()`` already returns a one-hot of ``pred()``'s shape, so the
      # first branch always wins. See test_heads_helpers.py.
      events[k] = jnp.broadcast_to(e[..., None], ref.shape)
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


def class_distance_matrix(classes):
  """``(C, C)`` squared distance between class indices along the codebook line.

  The manager's per-block categorical has ORDERED classes once the goal
  autoencoder is a SOM line: class ``a`` and class ``a+1`` decode to nearby
  goals. Entropy is blind to that order -- permuting the classes leaves it
  unchanged -- so it cannot tell "spread over three adjacent codes" from "split
  between the two ends". This matrix is what supplies the order to
  ``head_rao_perdim_time``.

  Class ``a`` sits at ``x_a = a / (C - 1)`` on the unit interval and
  ``d(a, b) = (x_a - x_b)^2``, so the endpoints are the two most distant codes.

  Line only, deliberately. A ring has no endpoints, and there the normalized
  Rao reduces to ``1 - R^2`` in the resultant length: it is maximized by ANY
  distribution with zero resultant, so a uniform and a two-atom antipodal split
  score identically and the measure cannot rank them. Ring topology is not used
  in this project.
  """
  x = jnp.arange(classes, dtype=f32) / jnp.maximum(f32(classes - 1), 1.0)
  return jnp.square(x[:, None] - x[None, :])


def rao_quadratic_entropy(probs, dist, other=None):
  """``sum_{a,b} p_a q_b d(a,b)`` over the last axis, normalized to ``[0, 1]``.

  Rao's quadratic entropy is the expected distance between two classes drawn
  independently from the distribution. Unlike entropy, which only counts how
  many classes carry mass, this is large only when that mass sits on classes
  that are FAR APART. ``other`` defaults to ``probs`` (the spread within one
  decision); pass a second distribution for the cross form (how far apart two
  decisions are).

  Normalizer: a self-Rao is maximized by half the mass on each of the two most
  distant classes, giving ``max(d) / 2``, so dividing by that puts the self
  form on ``[0, 1]``. The cross form is divided by the SAME constant so the two
  are directly comparable; it can legitimately reach 2 when two distributions
  sit at opposite ends.

  Whole-code vs per-block. The manager emits ``L`` independent categoricals, so
  ``p(z) = prod_l p_l(z_l)``, and for a distance that is additive over blocks
  ``Q(p) = sum_l Q(p_l)`` EXACTLY. Whole-code Rao is the sum of the per-block
  Raos, so the ``C^L`` (8^8 = 16.7M) squared pair sum is never formed; this
  works per block, at ``C x C`` = 64 pairs.
  """
  other = probs if other is None else other
  q = jnp.einsum('...a,ab,...b->...', probs, dist, other)
  return q / jnp.maximum(dist.max() / 2.0, 1e-8)


def head_probs(head):
  """Class probabilities of a categorical policy head, or None if it has none."""
  dist = getattr(_head_inner(head), 'dist', None)
  logits = getattr(dist, 'logits', None)
  return None if logits is None else jax.nn.softmax(f32(logits), -1)


def head_rao_perdim_time(head):
  """Normalized per-categorical Rao sliced to ``(batch, time - 1, ...)``.

  Same convention as ``head_entropy_perdim_time`` -- one value per block, the
  trailing time step dropped -- so one AutoAdapt can regulate it per block.
  The class count comes from the head's own logits, so callers never have to
  dig it out of a head wrapper. Returns None for heads with no class logits
  (nothing to regularize).
  """
  probs = head_probs(head)
  if probs is None:
    return None
  rao = rao_quadratic_entropy(probs, class_distance_matrix(probs.shape[-1]))
  if rao.ndim < 2:
    rao = rao[None, :]
  return rao[:, :-1]


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
