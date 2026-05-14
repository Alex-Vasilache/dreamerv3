"""Director-style HRL helpers (JAX). Used only when agent.use_director_hrl is True."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32
sg = jax.lax.stop_gradient


def deter_feat(repfeat_like):
  """Director-style feature: deterministic state only (B, T, D)."""
  return repfeat_like['deter']


def cosine_max_reward(goal, state_deter):
  """Max-cosine similarity (Director `cosine_max` on deter vectors)."""
  gnorm = jnp.linalg.norm(goal, axis=-1, keepdims=True) + 1e-8
  fnorm = jnp.linalg.norm(state_deter, axis=-1, keepdims=True) + 1e-8
  norm = jnp.maximum(gnorm, fnorm)
  return jnp.sum((goal / norm) * (state_deter / norm), axis=-1)


def categorical_kl_to_uniform(logits):
  """KL( Cat(logits) || Uniform ) along the class axis."""
  logp = jax.nn.log_softmax(logits, axis=-1)
  p = jnp.exp(logp)
  c = f32(logits.shape[-1])
  return jnp.sum(p * (logp + jnp.log(c)), axis=-1)


def initial_hrl_carry(batch: int, deter_dim: int, skill_syms: int):
  """Skill stored as int32 indices per symbol (B, skill_syms)."""
  return dict(
      step=jnp.zeros((batch,), jnp.int32),
      skill=jnp.zeros((batch, skill_syms), jnp.int32),
      goal=jnp.zeros((batch, deter_dim), f32),
  )


def reset_hrl_on_episode(hrl, is_first_b0, *, deter_dim: int, skill_syms: int):
  """Reset hierarchical carry rows where a new episode starts."""
  z = initial_hrl_carry(is_first_b0.shape[0], deter_dim, skill_syms)
  m = is_first_b0.astype(f32)
  return dict(
      step=jnp.where(is_first_b0, z['step'], hrl['step']),
      skill=jnp.where(m[:, None], z['skill'], hrl['skill']),
      goal=jnp.where(m[:, None], z['goal'], hrl['goal']),
  )


def goal_autoencoder_replay_loss(enc, dec, deter_bt, manager_step_k: int, skill_classes: int):
  """ELBO-style loss on replay: encode (ctx, goal) -> discrete skill, decode -> goal."""
  b, t, _ = deter_bt.shape
  k = int(manager_step_k)
  if t <= k:
    return jnp.zeros((b, t), f32), {}
  ctx = deter_bt[:, :-k]
  gol = deter_bt[:, k:]
  enc_in = jnp.concatenate([ctx, gol], axis=-1)
  enc_out = enc(enc_in, bdims=2)
  (key,) = enc_out.keys()
  dist = enc_out[key]
  skill_idx = dist.sample(nj.seed())
  oh = jax.nn.one_hot(skill_idx, skill_classes, dtype=f32)
  flat = oh.reshape(*oh.shape[:-1], -1)
  dec_in = jnp.concatenate([flat, ctx], axis=-1)
  rec = jnp.mean(dec(dec_in, bdims=2).loss(sg(gol)), axis=-1)
  kl_per_sym = categorical_kl_to_uniform(dist.logits)
  kl = jnp.sum(kl_per_sym, axis=-1)
  metrics = {f'director/ae_rec': rec.mean(), f'director/ae_kl': kl.mean()}
  pad = jnp.zeros((b, k), f32)
  full = jnp.concatenate([rec + kl, pad], axis=1)
  return full, metrics


def exploration_mse(dec, skill_idx, ctx_deter, next_deter, skill_classes: int):
  """Squared reconstruction error ||dec(onehot(skill), ctx) - s_{t+1}||^2 (mean over dims)."""
  oh = jax.nn.one_hot(skill_idx, skill_classes, dtype=f32)
  flat = oh.reshape(*oh.shape[:-1], -1)
  dec_in = jnp.concatenate([flat, ctx_deter], axis=-1)
  pred = dec(dec_in, bdims=2).pred()
  return jnp.mean((pred - sg(next_deter)) ** 2, axis=-1)


def reinforce_manager(man_out, skill_idx, advantage, key_skill: str):
  """REINFORCE loss on manager categorical (-log pi * A)."""
  dist = man_out[key_skill]
  logp = dist.logp(skill_idx)
  if logp.ndim > advantage.ndim:
    logp = jnp.sum(logp, axis=-1)
  return -sg(advantage) * logp


def elbo_adver_reward(enc, dec, deter_bt, skill_classes: int, adver_impl: str):
  """Director `Hierarchy.elbo_reward` / `expl_rew: adver` (B, T-1)."""
  from . import director_rewards as dr

  ctx = dr.broadcast_context(deter_bt)
  enc_in = jnp.concatenate([ctx, deter_bt], axis=-1)
  enc_out = enc(enc_in, bdims=2)
  (key,) = enc_out.keys()
  dist = enc_out[key]
  skill_idx = dist.sample(nj.seed())
  oh = jax.nn.one_hot(skill_idx, skill_classes, dtype=f32)
  flat = oh.reshape(*oh.shape[:-1], -1)
  dec_in = jnp.concatenate([flat, ctx], axis=-1)
  pred = dec(dec_in, bdims=2).pred()
  feat = deter_bt.astype(f32)
  if adver_impl == 'abs':
    r = jnp.mean(jnp.abs(pred - feat), axis=-1)
  elif adver_impl == 'squared':
    r = jnp.mean((pred - feat) ** 2, axis=-1)
  else:
    raise NotImplementedError(adver_impl)
  return r[:, 1:]


def disag_reward(deter_bt, heads):
  """Disagreement std across ensemble heads (B, T-1); `expl.py` without action in inputs."""
  preds = [h(deter_bt, bdims=2).pred() for h in heads]
  stack = jnp.stack(preds, axis=0)
  d = jnp.std(stack, axis=0).mean(axis=-1)
  return d[:, 1:]


def disag_replay_loss(heads, deter_bt, stoch_bt):
  """Port of `Disag.train` on replay (mean NLL of ensemble to next stoch)."""
  b, t, _ = deter_bt.shape
  st_flat = stoch_bt.reshape(b, t, -1)
  inp = deter_bt[:, :-1]
  tar = sg(st_flat[:, 1:])
  tot = jnp.zeros((b, t - 1), f32)
  for h in heads:
    tot = tot + jnp.mean(h(inp, 2).loss(tar), axis=-1)
  return jnp.concatenate([tot, jnp.zeros((b, 1), f32)], axis=1)
