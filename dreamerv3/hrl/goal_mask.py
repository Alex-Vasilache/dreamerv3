"""Goal-space masking: a goal may constrain only some groups of ``deter``.

The goal ``deter`` vector is split into ``G`` contiguous groups. With ``G`` equal
to ``rssm.blocks`` these are exactly the RSSM's block-diagonal recurrent blocks
(block ``b`` owns dims ``b*D/G .. (b+1)*D/G - 1``). The mask lives inside the
goal code: the goal autoencoder is trained on randomly masked targets, and a
small head decodes the keep/ignore pattern from the code. The manager's action
stays a single code.

A decoded goal carries its mask as EXACT zeros in the ignored groups (the hard
mask multiplies the decoded values). A kept group of a decoded goal is never
exactly zero across all its dims, so the mask is recovered losslessly from the
goal itself, and every consumer (worker input, worker reward, held-goal carry)
keeps the plain ``(..., D)`` goal tensor.
"""
import jax
import jax.numpy as jnp

f32 = jnp.float32


def expand_groups(keep, dim):
  """``(..., G)`` per-group mask -> ``(..., D)`` per-dim mask."""
  groups = keep.shape[-1]
  assert dim % groups == 0, (dim, groups)
  return jnp.repeat(keep, dim // groups, axis=-1)


def group_keep(goal, groups):
  """Recover the ``(..., G)`` float keep mask of a decoded goal."""
  dim = goal.shape[-1]
  assert dim % groups == 0, (dim, groups)
  g = goal.reshape((*goal.shape[:-1], groups, dim // groups))
  return (jnp.abs(g).sum(-1) > 0).astype(f32)


def sample_train_mask(seed, shape, groups, full_prob):
  """Random keep masks for training the goal autoencoder, ``(*shape, G)``.

  With probability ``full_prob`` the whole goal is kept (the plain Director
  goal). Otherwise ``k`` is uniform on ``1 .. G-1`` and ``k`` groups chosen at
  random are kept. Never all-masked: an empty goal has nothing to achieve."""
  s1, s2, s3 = jax.random.split(seed, 3)
  k = jax.random.randint(s1, shape, 1, groups)               # 1 .. G-1
  rank = jnp.argsort(jnp.argsort(
      jax.random.uniform(s2, (*shape, groups)), -1), -1)     # random 0..G-1
  keep = rank < k[..., None]
  full = jax.random.uniform(s3, shape) < full_prob
  return jnp.where(full[..., None], True, keep).astype(f32)


def hard_keep(logit):
  """Keep groups with ``p > 0.5``; if none, keep the most likely one."""
  keep = logit > 0
  top = jax.nn.one_hot(jnp.argmax(logit, -1), logit.shape[-1], dtype=bool)
  none = ~keep.any(-1, keepdims=True)
  return jnp.where(none, top, keep).astype(f32)
