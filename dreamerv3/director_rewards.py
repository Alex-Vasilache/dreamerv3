"""Director reward definitions (`hierarchy.py` goal_reward, extr, expl), JAX."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

f32 = jnp.float32
EPS = 1e-12


def broadcast_context(feat_deter: jnp.ndarray) -> jnp.ndarray:
  """Director `tf.repeat(feat[0][None], 1 + imag_horizon, 0)` in (B, T, D) layout."""
  return jnp.broadcast_to(feat_deter[:, :1], feat_deter.shape)


def goal_reward(
    name: str,
    feat_deter: jnp.ndarray,
    goal: jnp.ndarray,
    skill: jnp.ndarray,
    context: jnp.ndarray,
    enc_fwd: Callable[..., Any] | None = None,
) -> jnp.ndarray:
  """Port of `Hierarchy.goal_reward` returning (B, T-1) for T timesteps of feat."""
  feat = feat_deter.astype(f32)
  goal = goal.astype(f32)
  skill = skill.astype(f32)
  context = context.astype(f32)
  g = jnp.swapaxes(goal, 0, 1)
  f = jnp.swapaxes(feat, 0, 1)
  s = jnp.swapaxes(skill, 0, 1)
  ctx = jnp.swapaxes(context, 0, 1)

  def out(x):
    return jnp.swapaxes(x, 0, 1)

  if name == 'dot':
    r = jnp.sum(g * f, axis=-1)[1:]
  elif name == 'dir':
    r = jnp.sum(jax.nn.l2_normalize(g, -1) * f, axis=-1)[1:]
  elif name == 'normed_inner':
    norm = jnp.linalg.norm(g, axis=-1, keepdims=True)
    r = jnp.sum((g / norm) * (f / norm), axis=-1)[1:]
  elif name == 'normed_squared':
    norm = jnp.linalg.norm(g, axis=-1, keepdims=True)
    r = -jnp.mean((g / norm - f / norm) ** 2, axis=-1)[1:]
  elif name == 'cosine_lower':
    gnorm = jnp.linalg.norm(g, axis=-1, keepdims=True) + EPS
    fnorm = jnp.linalg.norm(f, axis=-1, keepdims=True) + EPS
    fnorm = jnp.maximum(gnorm, fnorm)
    r = jnp.sum((g / gnorm) * (f / fnorm), axis=-1)[1:]
  elif name == 'cosine_lower_pos':
    gnorm = jnp.linalg.norm(g, axis=-1, keepdims=True) + EPS
    fnorm = jnp.linalg.norm(f, axis=-1, keepdims=True) + EPS
    fnorm = jnp.maximum(gnorm, fnorm)
    r = jnp.nn.relu(jnp.sum((g / gnorm) * (f / fnorm), axis=-1)[1:])
  elif name == 'cosine_frac':
    gnorm = jnp.linalg.norm(g, axis=-1) + EPS
    fnorm = jnp.linalg.norm(f, axis=-1) + EPS
    gn = g / gnorm[..., None]
    fn = f / fnorm[..., None]
    cos = jnp.sum(gn * fn, axis=-1)
    mag = jnp.minimum(gnorm, fnorm) / jnp.maximum(gnorm, fnorm)
    r = (cos * mag)[1:]
  elif name == 'cosine_frac_pos':
    gnorm = jnp.linalg.norm(g, axis=-1) + EPS
    fnorm = jnp.linalg.norm(f, axis=-1) + EPS
    gn = g / gnorm[..., None]
    fn = f / fnorm[..., None]
    cos = jnp.sum(gn * fn, axis=-1)
    mag = jnp.minimum(gnorm, fnorm) / jnp.maximum(gnorm, fnorm)
    r = jnp.nn.relu((cos * mag)[1:])
  elif name == 'cosine_max':
    gnorm = jnp.linalg.norm(g, axis=-1, keepdims=True) + EPS
    fnorm = jnp.linalg.norm(f, axis=-1, keepdims=True) + EPS
    norm = jnp.maximum(gnorm, fnorm)
    r = jnp.sum((g / norm) * (f / norm), axis=-1)[1:]
  elif name == 'cosine_max_pos':
    gnorm = jnp.linalg.norm(g, axis=-1, keepdims=True) + EPS
    fnorm = jnp.linalg.norm(f, axis=-1, keepdims=True) + EPS
    norm = jnp.maximum(gnorm, fnorm)
    cos = jnp.sum((g / norm) * (f / norm), axis=-1)[1:]
    r = jnp.nn.relu(cos)
  elif name == 'normed_inner_clip':
    norm = jnp.linalg.norm(g, axis=-1, keepdims=True)
    cosine = jnp.sum((g / norm) * (f / norm), axis=-1)[1:]
    r = jnp.clip(cosine, -1.0, 1.0)
  elif name == 'normed_inner_clip_pos':
    norm = jnp.linalg.norm(g, axis=-1, keepdims=True)
    cosine = jnp.sum((g / norm) * (f / norm), axis=-1)[1:]
    r = jnp.clip(cosine, 0.0, 1.0)
  elif name == 'diff':
    goal_n = jax.nn.l2_normalize(g[:-1], -1)
    diff = f[1:] - f[:-1]
    r = jnp.sum(goal_n * diff, axis=-1)
  elif name == 'norm':
    r = -jnp.linalg.norm(g - f, axis=-1)[1:]
  elif name == 'squared':
    r = -jnp.sum((g - f) ** 2, axis=-1)[1:]
  elif name == 'epsilon':
    r = ((g - f).mean(-1) < 1e-3).astype(f32)[1:]
  elif name in ('enclogprob', 'encprob', 'enc_normed_cos', 'enc_normed_squared'):
    if enc_fwd is None:
      raise ValueError(f'goal_reward {name} requires enc_fwd')
    raise NotImplementedError(
        f'goal_reward {name} needs encoder distribution port; use cosine_max')
  else:
    raise NotImplementedError(name)
  return out(r)
