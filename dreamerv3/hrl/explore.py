"""Epsilon-greedy index jumps for the manager, on an ordered codebook.

At the entropy the regularizer holds, the manager's Gaussian samples land close
to its mode: two draws in the same state differ by ~5.7 of the 56 index steps
the code admits, and decode to goals ~0.84 similar. That concentration is what
makes credit transfer to neighbouring classes, so it cannot simply be widened
-- raising the entropy target buys reach only by flattening the distribution
until the ordering stops being used (at target 0.8 the mode/neighbour ratio
falls from 2.9 to 1.27), and a uniform mixture is absorbed by the entropy
controller shrinking sigma to compensate.

This buys reach on a separate axis instead. With probability ``eps`` the
sampled code is replaced by a deliberately distant one:

  1. take the mode ``c`` of the manager's current distribution;
  2. a block at class ``c_l`` can shift by at most ``max(c_l, C-1-c_l)``, so the
     largest reachable index distance is ``dmax(c) = sum_l max(c_l, C-1-c_l)``.
     Note this is a property of the CODE: over uniform codes it averages 44,
     not 56, and only a code with every block at an extreme reaches 56;
  3. draw a target distance with ``p(d) = 2d / [dmax (dmax + 1)]`` for
     ``d = 1..dmax``, which normalizes exactly and favours large jumps
     (``E[d] = (2/3)(dmax + 1/2)``, about 30 in practice);
  4. realize exactly that distance by spreading ``d`` unit steps over the
     blocks within each block's own capacity, then choosing a direction per
     block among those the bounds allow.

Only worth doing because the index is ordered. On Director's codebook a large
index gap does not imply a distant goal -- its measured curve is flat and even
rises at the far end -- so jumping along that axis would buy nothing.

APPLIED IN THE ENVIRONMENT ROLLOUT ONLY. The manager is trained by REINFORCE on
the log-probability of its own samples, so drawing from a different
distribution during training biases the gradient. Importance weighting is the
textbook fix but is empty here: the policy assigns a jumped code a probability
of order 1e-61, which underflows float32, so such a sample would contribute
nothing anyway. Restricting the jump to the actor changes which states reach
the replay buffer -- the coverage we want -- and leaves imagination on-policy.
"""
import jax
import jax.numpy as jnp

i32 = jnp.int32
f32 = jnp.float32


def capacities(ids, classes):
  """Per-block room to move, and the total reachable index distance.

  Returns ``(down, up, cap, dmax)`` where ``down``/``up`` are how far the block
  can move toward class 0 / class C-1, ``cap = max(down, up)`` is the largest
  magnitude it admits, and ``dmax`` sums that over the blocks.
  """
  ids = i32(ids)
  down = ids
  up = (classes - 1) - ids
  cap = jnp.maximum(down, up)
  return down, up, cap, cap.sum(-1)


def sample_distance(key, dmax):
  """Draw ``d`` with ``p(d) = 2d / [dmax (dmax + 1)]``, ``d = 1..dmax``.

  Inverse-CDF: ``P(D <= d) = d(d+1) / [dmax(dmax+1)]``, so inverting
  ``d(d+1) = u * dmax(dmax+1)`` gives ``d = ceil((-1 + sqrt(1 + 4uM)) / 2)``.
  """
  m = f32(dmax) * (f32(dmax) + 1.0)
  u = jax.random.uniform(key, jnp.shape(dmax), f32)
  d = jnp.ceil((-1.0 + jnp.sqrt(1.0 + 4.0 * u * m)) / 2.0)
  return jnp.clip(i32(d), 1, jnp.maximum(i32(dmax), 1))


def spread(key, cap, d, classes):
  """Split ``d`` into per-block magnitudes with ``|delta_l| <= cap_l``.

  Each block contributes ``cap_l`` unit slots; choosing ``d`` of the
  ``sum_l cap_l`` slots without replacement enforces both ``sum |delta_l| = d``
  and the per-block bound at once, and every valid assignment is reachable.
  Implemented by ranking uniform keys, so it stays a fixed-shape operation.
  """
  steps = jnp.arange(classes - 1)
  valid = steps < cap[..., None]                      # (..., L, C-1)
  noise = jax.random.uniform(key, valid.shape, f32)
  noise = jnp.where(valid, noise, 2.0)                # invalid slots sort last
  flat = noise.reshape(*noise.shape[:-2], -1)
  rank = jnp.argsort(jnp.argsort(flat, -1), -1).reshape(valid.shape)
  return (rank < d[..., None, None]).astype(i32).sum(-1) * valid.any(-1)


def jump(key, ids, classes):
  """A code at a random large index distance from ``ids``."""
  down, up, cap, dmax = capacities(ids, classes)
  k_d, k_s, k_dir = jax.random.split(key, 3)
  d = sample_distance(k_d, dmax)
  mag = spread(k_s, cap, d, classes)
  # Direction: take whichever side has room. cap is the max of the two, so at
  # least one always does; when both do, pick at random.
  can_up = mag <= up
  can_down = mag <= down
  coin = jax.random.uniform(k_dir, mag.shape, f32) < 0.5
  go_up = jnp.where(can_up & can_down, coin, can_up)
  delta = jnp.where(go_up, mag, -mag)
  return jnp.clip(ids + delta, 0, classes - 1), d


def apply_jump(key, onehot, eps, classes):
  """Replace the sampled code by a jump from its mode, with probability ``eps``.

  ``onehot`` is ``(..., L, C)``. The mode is taken per block; the decision to
  jump is made once per code, not once per block, so a jump moves the whole
  code coherently -- which is the point, and is what a per-class mixture like
  ``unimix`` cannot do.
  """
  if not eps:
    return onehot
  k_take, k_jump = jax.random.split(key, 2)
  ids = jnp.argmax(onehot, -1)
  jumped, _ = jump(k_jump, ids, classes)
  hot = jax.nn.one_hot(jumped, classes, dtype=onehot.dtype)
  take = jax.random.uniform(k_take, ids.shape[:-1], f32) < eps
  return jnp.where(take[..., None, None], hot, onehot)
