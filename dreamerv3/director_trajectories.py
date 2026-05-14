"""Director `split_traj` / `abstract_traj` (ported from `hierarchy.py`, batch-major)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

f32 = jnp.float32


def _split_one_leaf(x, k: int, is_reward: bool):
  """Apply Director reshape to a single (B, T, ...) array."""
  B = x.shape[0]
  if is_reward:
    assert x.ndim >= 2
    t_m1 = x.shape[1]
    z = jnp.zeros_like(x[:, :1])
    val = jnp.concatenate([z, x], axis=1)
  else:
    val = x
  t = val.shape[1]
  head = val[:, :-1]
  new_shape = (B, (t - 1) // k, k) + head.shape[2:]
  resh = head.reshape(new_shape)
  tail = val[:, k::k]
  if tail.ndim < resh.ndim:
    tail = jnp.expand_dims(tail, 2)
  merged = jnp.concatenate([resh, tail], axis=2)
  perm = (0, 2, 1) + tuple(range(3, merged.ndim))
  tr = jnp.transpose(merged, perm)
  flat_t = tr.shape[1] * tr.shape[2]
  rest = tr.shape[3:]
  tr = jnp.reshape(tr, (B, flat_t) + rest)
  if is_reward:
    tr = tr[:, 1:]
  return tr


def split_traj(traj: dict, k: int, discount: float) -> dict:
  """Port of `Hierarchy.split_traj` with tensors (B, T, ...). `action` may be a pytree."""
  t_act = traj['deter'].shape[1]
  assert t_act % k == 1, (t_act, k, 'Director requires (H+1) % k == 1')
  out = {}
  for key, val in traj.items():
    is_reward = 'reward' in key
    if isinstance(val, dict):
      out[key] = jax.tree.map(
          lambda v: _split_one_leaf(v, k, is_reward), val)
    else:
      if is_reward:
        assert val.shape[1] == t_act - 1, (key, val.shape, t_act)
      else:
        assert val.shape[1] == t_act, (key, val.shape, t_act)
      out[key] = _split_one_leaf(val, k, is_reward)
  cont = out['cont']
  if isinstance(cont, dict):
    raise NotImplementedError('split_traj expects vector cont')
  disc = f32(discount)
  out['weight'] = jnp.cumprod(disc * cont, axis=1) / disc
  out['goal'] = jnp.concatenate([out['goal'][:, :-1], out['goal'][:, :1]], axis=1)
  return out


def _abstract_reward(value, k, t_act, B, weights):
  assert value.shape[1] == t_act - 1, (value.shape,)
  z = jnp.zeros_like(value[:, :1])
  rew_p = jnp.concatenate([z, value], axis=1)
  t = rew_p.shape[1]
  head = rew_p[:, :-1]
  sh = (B, (t - 1) // k, k) + head.shape[2:]
  resh = head.reshape(sh)
  return (resh * weights).mean(axis=2)


def abstract_traj(traj: dict, k: int, discount: float) -> dict:
  """Port of `Hierarchy.abstract_traj` (batch-major). `skill` becomes `action`."""
  if 'skill' not in traj:
    raise KeyError('abstract_traj expects traj["skill"]')
  out = {kk: vv for kk, vv in traj.items() if kk != 'skill'}
  out['action'] = traj['skill']
  B, t_act = out['action'].shape[:2]
  assert t_act % k == 1, (t_act, k)
  cont = out['cont']
  head_cont = cont[:, :-1]
  new_shape = (B, (t_act - 1) // k, k) + head_cont.shape[2:]
  resh_cont = head_cont.reshape(new_shape)
  weights = jnp.cumprod(resh_cont, axis=2)
  final = {}
  for key, value in out.items():
    if 'reward' in key:
      final[key] = _abstract_reward(value, k, t_act, B, weights)
    elif key == 'cont':
      first = value[:, :1]
      rest = value[:, 1:]
      sh = (B, (t_act - 1) // k, k) + rest.shape[2:]
      prod_chunk = jnp.prod(rest.reshape(sh), axis=2)
      final[key] = jnp.concatenate([first, prod_chunk], axis=1)
    else:
      assert value.shape[1] == t_act, (key, value.shape)
      head = value[:, :-1]
      sh = (B, (t_act - 1) // k, k) + head.shape[2:]
      resh = head.reshape(sh)[:, :, 0]
      last = value[:, -1:]
      final[key] = jnp.concatenate([resh, last], axis=1)
  cont_m = final['cont']
  disc = f32(discount)
  final['weight'] = jnp.cumprod(disc * cont_m, axis=1) / disc
  return final
