"""DreamerV3-style agent: RSSM world model + policy/value on imagined rollouts.

Trains encoder, dynamics (RSSM), decoder, reward/continue heads, policy, and
value in one step from replay sequences. Actor-critic losses use imagined
trajectories from ``dyn.imagine``; optional ``repval_loss`` fits the value on
real replay tails with bootstrap from imagination.
"""
import math
import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import embodied.jax.outs as outs
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm

f32 = jnp.float32
i32 = jnp.int32
# Stop-gradient helper: optionally pass gradients through (e.g. reward head).
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
# Sample from a tree of distribution-like outputs (policy heads).
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
# Deterministic mode (argmax / mean) of a tree of distribution-like outputs.
mode = lambda xs: jax.tree.map(lambda x: x.pred(), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}


def mgr_as_dict(out):
  """Wrap a single manager head output as ``{'skill': ...}`` for shared code paths."""
  return out if isinstance(out, dict) else {'skill': out}


def skill_switch(update, new_skill, old_skill):
  """Director ``switch``: keep ``old_skill`` unless ``update`` (per batch element)."""
  u = jnp.asarray(update)
  expand = lambda n, o: jnp.where(
      u.reshape(u.shape + (1,) * (n.ndim - u.ndim)), n, o)
  return jax.tree.map(expand, new_skill, old_skill)


def imag_reward_pad(reward_step):
  """Pad Director ``[1:]`` rewards so ``lambda_return`` has shape ``(B, T+1)``."""
  return jnp.concatenate([jnp.zeros_like(reward_step[:, :1]), reward_step], axis=1)


def goal_reward_cosine_max(goal, feat):
  """Director ``goal_reward`` with ``cosine_max`` (``hierarchy.goal_reward``)."""
  gnorm = jnp.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
  fnorm = jnp.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
  norm = jnp.maximum(gnorm, fnorm)
  return jnp.sum((goal / norm) * (feat / norm), axis=-1)


def bernoulli_entropy(p, eps=1e-6):
  """Per-element Bernoulli entropy in nats (max ``ln 2`` at p=0.5)."""
  p = jnp.clip(p, eps, 1.0 - eps)
  return -(p * jnp.log(p) + (1.0 - p) * jnp.log(1.0 - p))


def bernoulli_kl(p, q, eps=1e-6):
  """KL(Bernoulli(p) || Bernoulli(q)) per element, in nats."""
  p = jnp.clip(p, eps, 1.0 - eps)
  q = float(min(max(q, eps), 1.0 - eps))
  return p * jnp.log(p / q) + (1.0 - p) * jnp.log((1.0 - p) / (1.0 - q))


def pairwise_cosmax(x):
  """All-pairs ``cosine_max`` over a pool of vectors ``x`` of shape ``(N, F)``.

  ``S[i, j] = (x_i . x_j) / max(||x_i||, ||x_j||)^2`` -- the same magnitude-aware
  similarity the worker reward uses (``goal_reward_cosine_max``), generalized to a
  full ``(N, N)`` Gram matrix. Used to match code-space geometry to deter geometry."""
  norm = jnp.linalg.norm(x, axis=-1) + 1e-12                  # (N,)
  dot = x @ x.T                                               # (N, N)
  nm = jnp.maximum(norm[:, None], norm[None, :])              # (N, N)
  return dot / (nm * nm)


def aggregate_mgr_extr_rew(rew, con, k, without_zeros=False, agg_mode='mean'):
  """Pool per-step extrinsic reward into skill windows (Director ``abstract_traj``).

  Each block of ``k`` transitions gets one reward:
  ``mean(rew_block * cumprod(con_block))`` placed at the block start index.
  Other timesteps are zero so ``lambda_return`` does not double-count.

  If without_zeros=True, returns only the per-window rewards in sequence,
  no zeros or padding. Shape: (B, n_blocks [+1 if remainder]).
  """
  k = max(1, int(k))
  B, T = rew.shape
  r = rew[:, 1:]
  c = con[:, :-1]
  Tm = r.shape[1]
  n = Tm // k
  result_blocks = []
  if n == 0:
    if without_zeros:
      return r.mean(axis=1, keepdims=True)  # Output is shape (B, 1)
    return rew
  r_blk = r[:, :n * k].reshape(B, n, k)
  c_blk = c[:, :n * k].reshape(B, n, k)
  weights = jnp.cumprod(c_blk, axis=-1)
  pooled = r_blk * weights
  agg = pooled.sum(axis=-1) if agg_mode == 'sum' else pooled.mean(axis=-1)  # (B, n)

  rem = Tm - n * k
  if rem > 0:
    r_rem = r[:, n * k:]
    c_rem = c[:, n * k:]
    w = jnp.cumprod(c_rem, axis=-1)
    pooled_rem = r_rem * w
    agg_rem = (pooled_rem.sum(-1, keepdims=True) if agg_mode == 'sum'
               else pooled_rem.mean(-1, keepdims=True))  # shape (B,1)
    agg_full = jnp.concatenate([agg, agg_rem], axis=1)
    idx = 1 + jnp.arange(n) * k
    out = jnp.zeros_like(rew)
    out = out.at[:, idx].set(agg)
    out = out.at[:, n * k + 1].set(agg_rem[:, 0])
  else:
    agg_full = agg
    idx = 1 + jnp.arange(n) * k
    out = jnp.zeros_like(rew)
    out = out.at[:, idx].set(agg)
  if without_zeros:
    return agg_full
  return out

def aggregate_mgr_cont(con, k, without_zeros=False):
  """
  Aggregate per-step continuation (Director `abstract_traj`, for skill window).

  Each block of `k` transitions is pooled via product, producing one continuation
  at the block start index. The other timesteps are filled appropriately to 
  avoid double counting by lambda_return.

  Mimics:
    traj['cont'] = tf.concat([value[:1], reshape(value[1:]).prod(1)], 0)

  If without_zeros=True, returns only the per-window continuation products in sequence,
  no zeros or padding. Output shape: (B, n_blocks [+1 if remainder]).
  Otherwise, output shape is identical to input con, with continuation block prods
  at each block start index and zeros elsewhere.
  """
  k = max(1, int(k))
  B, T = con.shape       # (B, T) input including first
  c = con[:, 1:]         # skip first (matches reward grouping as in abstract_traj)
  Tm = c.shape[1]
  n = Tm // k
  if n == 0:
    if without_zeros:
      return c.prod(axis=1, keepdims=True)
    return con
  # Full skill blocks, shape (B, n, k)
  c_blk = c[:, :n * k].reshape(B, n, k)
  block_prod = c_blk.prod(axis=-1)  # shape (B, n)
  # Handle any remaining elements (after last full block)
  rem = Tm - n * k
  if rem > 0:
    c_rem = c[:, n * k:]
    rem_prod = c_rem.prod(axis=-1, keepdims=True)  # shape (B, 1)
    block_prod_full = jnp.concatenate([block_prod, rem_prod], axis=1)  # (B, n+1)
    block_idx = 1 + jnp.arange(n) * k
    out = jnp.zeros_like(con)
    out = out.at[:, 0].set(con[:, 0])
    out = out.at[:, block_idx].set(block_prod)
    out = out.at[:, n * k + 1].set(rem_prod[:, 0])
  else:
    block_prod_full = block_prod
    block_idx = 1 + jnp.arange(n) * k
    out = jnp.zeros_like(con)
    out = out.at[:, 0].set(con[:, 0])
    out = out.at[:, block_idx].set(block_prod)
  if without_zeros:
    # Only return the concatenated block products (and remainder if any), not the sparse tensor
    return jnp.concatenate([con[:, :1], block_prod_full], axis=1)
  return out


def variable_segment_ids(switch_mask):
  """Per-step segment index from a manager switch mask ``(B, T)`` or ``(B, T+1)``.

  Segment ``i`` is the goal held from the i-th manager decision onward: a switch at
  position ``j`` STARTS segment ``cumsum(sw)[j]-1``, so each held window (switch +
  the steps until the next switch) shares one id. This is ``cumsum(sw)-1`` (clamped
  at 0 for any pre-first-switch prefix), matching the ``switch_idx = cumsum(sw)-1``
  used when scattering pooled values back. (The earlier ``cumsum(sw)-sw`` put each
  switch step in the *previous* segment -> a spurious singleton first segment and a
  one-step shift of every block, so block-pooled credit did not match fixed-K.)"""
  sw = f32(switch_mask)
  return jnp.maximum(jnp.cumsum(sw.astype(i32), axis=-1) - 1, 0)


def masked_cumprod(con, switch_mask):
  """Continuation product over transitions, resetting at each manager switch."""
  c = f32(con)
  sw = f32(switch_mask)
  if c.ndim == 1:
    sw = sw[:c.shape[0]] if sw.shape[0] == c.shape[0] + 1 else sw

    def body(carry, xt):
      ct, swt = xt
      cprod = jnp.where(swt > 0.5, 1.0, carry)
      cprod = cprod * ct
      return cprod, cprod

    _, cprod = jax.lax.scan(body, 1.0, (c, sw))
    return cprod

  def row(c_row, sw_row):
    sw_t = sw_row[:c_row.shape[0]] if sw_row.shape[0] == c_row.shape[0] + 1 else sw_row
    return masked_cumprod(c_row, sw_t)

  return jax.vmap(row)(c, sw)


def aggregate_mgr_extr_rew_variable(rew, con, switch_mask, without_zeros=False,
                                    agg_mode='mean'):
  """Pool per-step rewards between manager switches (variable segment lengths).

  Boundaries come from ``switch_mask`` (1 wherever a manager decision occurs) instead
  of a fixed ``k``. When ``without_zeros=True``, returns compact per-segment rewards
  with shape ``(B, n_seg)`` (padded to the max switch count in the batch).
  """
  sw = f32(switch_mask)
  r = f32(rew[:, 1:])
  c = f32(con[:, :-1])
  B, Tm = r.shape
  max_seg = sw.shape[-1]  # static upper bound (≤ one switch per timestep)
  seg = variable_segment_ids(sw)[:, :Tm]  # (B, Tm)
  cprod = masked_cumprod(c, sw)
  weighted = r * cprod

  def pool_row(w_row, seg_row):
    totals = jax.ops.segment_sum(w_row, seg_row, num_segments=max_seg)
    if agg_mode == 'sum':
      # SMDP option return: continuation-weighted SUM over the segment (no divide).
      return totals
    counts = jax.ops.segment_sum(jnp.ones_like(w_row), seg_row, num_segments=max_seg)
    return totals / jnp.maximum(counts, 1.0)

  agg = jax.vmap(pool_row)(weighted, seg)  # (B, max_seg)
  if without_zeros:
    return agg
  out = jnp.zeros_like(rew)
  switch_idx = jnp.clip(jnp.cumsum(sw.astype(i32), axis=-1) - 1, 0, max_seg - 1)
  placed = jnp.take_along_axis(agg, switch_idx[:, :Tm], axis=1)
  out = out.at[:, 1:Tm + 1].set(jnp.where(sw[:, :Tm] > 0.5, placed, 0.0))
  out = out.at[:, 0].set(rew[:, 0])
  return out


def aggregate_mgr_cont_variable(con, switch_mask, without_zeros=False):
  """Pool continuation between manager switches (variable segment lengths).

  The per-block continuation is the *product* of continuations over the segment
  (Director ``abstract_traj``'s ``c_blk.prod(-1)``). ``masked_cumprod`` resets at
  each switch, so its value at the segment's final step equals that product. Since
  continuations are in [0, 1] the cumulative product is non-increasing within a
  segment, so the product == ``segment_min``. (The earlier ``segment_max`` returned
  the segment's *first* step instead -> under-discounted held goals; benign only when
  cont≈1, e.g. non-terminating cartpole.) Empty segments (zero real steps -- either
  true padding far beyond the last real decision, or a trailing hold whose switch
  landed exactly on the rollout's final timestep) get the empty-product identity 1.0,
  not 0.0 (fixed 2026-07-24): with 0.0 the very first padding slot after the last
  real decision reads as ``live=0`` in ``lambda_return``, which caps that decision's
  own bootstrap contribution to a ``(1-lam)``-only fraction of its true value instead
  of the full lambda-weighted blend fixed-K's exactly-sized (unpadded) arrays get for
  free -- see EXPERIMENTS.md `2026-07-24` note and
  ``test_final_bootstrap_bug_is_specific_to_block_pooled_varK``. 1.0 is also finite,
  so the ``* block_mask`` downstream (which correctly zeroes genuinely-unused slots
  for the REWARD side) still stays finite.
  """
  sw = f32(switch_mask)
  c = f32(con[:, 1:])
  Tm = c.shape[1]
  max_seg = sw.shape[-1]
  seg = variable_segment_ids(sw)[:, :Tm]
  cprod = masked_cumprod(c, sw)

  def pool_row(cp_row, seg_row):
    seg_min = jax.ops.segment_min(cp_row, seg_row, num_segments=max_seg)
    counts = jax.ops.segment_sum(jnp.ones_like(cp_row), seg_row, num_segments=max_seg)
    return jnp.where(counts > 0, seg_min, 1.0)

  block_prod = jax.vmap(pool_row)(cprod, seg)
  if without_zeros:
    return block_prod
  out = jnp.zeros_like(con)
  switch_idx = jnp.clip(jnp.cumsum(sw.astype(i32), axis=-1) - 1, 0, max_seg - 1)
  placed = jnp.take_along_axis(block_prod, switch_idx[:, :Tm], axis=1)
  out = out.at[:, 1:Tm + 1].set(jnp.where(sw[:, :Tm] > 0.5, placed, 0.0))
  out = out.at[:, 0].set(con[:, 0])
  return out


def switch_valid_mask(switch_mask, n_cols):
  """Mask for packed switch timelines: 1 on real switch rows, 0 on padding."""
  counts = jnp.sum(f32(switch_mask), axis=-1, keepdims=True)
  return f32(jnp.arange(n_cols)[None, :] < counts)


def variable_block_director_tensors(rew, con, expl, switch_mask, horizon,
                                    agg_mode='mean'):
  """Director ``abstract_traj`` manager tensors for variable switch boundaries.

  Mirrors fixed-K ``_mgr_extr_rew`` / ``_mgr_cont`` + ``imag_reward_pad``: block-pooled
  rewards with shape ``(B, n_blocks)``, continuation ``concat([con[:, :1], blocks])``,
  and rewards padded to ``(B, n_blocks + 1)``. Only the first ``sum(switch_mask)``
  downsampled switch rows are real manager decisions; the static tail is masked out.
  """
  sw = f32(switch_mask)
  n_sw = jnp.sum(sw, axis=-1, keepdims=True)
  n_blocks = jnp.maximum(n_sw, 1)
  block_mask = f32(jnp.arange(horizon - 1)[None, :] < n_blocks)

  pooled_extr = aggregate_mgr_extr_rew_variable(
      rew, con, switch_mask, without_zeros=True,
      agg_mode=agg_mode)[:, :horizon - 1] * block_mask
  pooled_expl = aggregate_mgr_extr_rew_variable(
      expl, con, switch_mask, without_zeros=True,
      agg_mode=agg_mode)[:, :horizon - 1] * block_mask
  # NOT block_mask-multiplied (unlike the reward tensors above): a segment id
  # >= n_sw can never occur (variable_segment_ids is bounded by cumsum(switch_
  # mask)-1), so aggregate_mgr_cont_variable's own empty-segment identity (1.0,
  # fixed 2026-07-24) is already correct at every position past the real
  # content. Re-zeroing it here would starve the last real decision's
  # bootstrap back down to a (1-lam)-only fraction of its value -- the exact
  # bug the identity fix addresses; see that function's docstring.
  pooled_cont = aggregate_mgr_cont_variable(
      con, switch_mask, without_zeros=True)[:, :horizon - 1]

  mgr_cont = jnp.concatenate([con[:, :1], pooled_cont], axis=1)[:, :horizon]
  mgr_extr_rew = imag_reward_pad(pooled_extr)[:, :horizon]
  mgr_expl_rew = imag_reward_pad(pooled_expl)[:, :horizon]
  mgr_switch = f32(jnp.arange(horizon)[None, :] < n_sw)
  return mgr_extr_rew, mgr_expl_rew, mgr_cont, mgr_switch


def forward_fill_packed(buf, valid):
  """Hold the last valid switch row into trailing padding slots (not zeros)."""
  def row(buf_row, valid_row):
    def body(last, xt):
      bi, vi = xt
      last = jnp.where(vi > 0.5, bi, last)
      return last, last
    _, filled = jax.lax.scan(body, buf_row[0], (buf_row, valid_row))
    return filled
  valid = switch_valid_mask(valid, buf.shape[1])
  return jax.vmap(row)(buf, valid)


def downsample_at_switch_mask(feat, switch_mask):
  """Extract states at manager switch timesteps; pad to ``max switches`` per batch."""
  sw = f32(switch_mask)
  B, T = sw.shape
  max_sw = T

  def compact_row(x_row, sw_row):
    def body(carry, xt):
      buf, i = carry
      val, swt = xt
      buf = jax.lax.cond(
          swt > 0.5,
          lambda b: b.at[i].set(val),
          lambda b: b,
          buf)
      i = i + jnp.where(swt > 0.5, 1, 0)
      return (buf, i), None

    buf = jnp.zeros((max_sw,) + x_row.shape[1:], dtype=x_row.dtype)
    (buf, _), _ = jax.lax.scan(body, (buf, jnp.int32(0)), (x_row, sw_row))
    return forward_fill_packed(buf[None], sw_row)[0]

  return jax.tree.map(
      lambda x: jax.vmap(compact_row)(x, sw),
      feat)


def relabel_truncated_last_duration(mgr_skills_eff, switch_mask, dur_min, dur_max):
  """Hindsight-relabel the LAST decision's duration class when its hold was cut
  short by the imagination horizon rather than by a real switch, so REINFORCE
  credits the manager for the duration it actually got to run, not the one it
  sampled. Only the final real decision in a rollout can be truncated this way
  -- every earlier one is, by construction, followed by a genuine switch, so it
  always ran its full sampled duration and is left untouched. Only the
  ``duration`` field changes; the goal-code choice itself isn't invalidated by
  running short, only how long it got to hold."""
  if 'duration' not in mgr_skills_eff:
    return mgr_skills_eff
  sw = f32(switch_mask)                                    # (B, T)
  B, T = sw.shape
  n_sw = jnp.sum(sw, axis=-1)                               # (B,) real decisions
  idx = jnp.arange(T)[None, :]                              # (1, T)
  last_switch_pos = jnp.max(jnp.where(sw > 0.5, idx, -1), axis=-1)  # (B,)
  avail = f32(T) - f32(last_switch_pos)   # real steps from the last switch through rollout end, inclusive
  dur_idx = mgr_skills_eff['duration']                       # (B, n_mgr) int class index
  n_mgr = dur_idx.shape[1]
  is_last_slot = (jnp.arange(n_mgr)[None, :] == (n_sw - 1)[:, None])  # (B, n_mgr)
  orig_idx_at_last = jnp.sum(jnp.where(is_last_slot, dur_idx, 0), axis=-1)  # (B,)
  orig_dur = f32(dur_min) + f32(orig_idx_at_last)
  truncated = avail < orig_dur
  relabel_dur = jnp.clip(avail, dur_min, dur_max)
  relabel_idx = i32(relabel_dur - dur_min)
  new_last_idx = jnp.where(truncated, relabel_idx, orig_idx_at_last.astype(i32))
  new_dur = jnp.where(is_last_slot, new_last_idx[:, None], dur_idx)
  return {**mgr_skills_eff, 'duration': new_dur}


def downsample_manager_states(feat, k):
  """Extract the boundary states of the manager skill blocks.

  Selects index 0, boundaries of blocks, and the final state.
  """
  T = jax.tree.leaves(feat)[0].shape[1]
  Tm = T - 1
  n = Tm // k
  rem = Tm - n * k
  idx = [0] + [int(1 + i * k - 1) for i in range(1, n + 1)]
  if rem > 0:
    idx.append(int(n * k))
    idx.append(int(Tm))
  else:
    idx.append(int(Tm))
  idx = sorted(list(set(idx)))
  return jax.tree.map(lambda x: x[:, idx], feat)

  
# Concatenate pytrees along axis ``a`` (e.g. time) for feat/action sequences.
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


def _tb_video_grid(video_bthwc, time_stride=1, space_stride=1):
  """(batch, time, H, W, C) uint8 -> (time, H, batch*W, C) for TensorBoard video.

  ``time_stride``/``space_stride`` subsample frames and pixels before the grid
  reshape — both default to 1 (no-op). Encoding cost scales with output
  pixel count, so a 4× time stride drops report video work ~4×.
  """
  if time_stride > 1:
    video_bthwc = video_bthwc[:, ::time_stride]
  if space_stride > 1:
    video_bthwc = video_bthwc[:, :, ::space_stride, ::space_stride]
  rb, t, h, w, c = video_bthwc.shape
  return video_bthwc.transpose(1, 2, 0, 3, 4).reshape(t, h, rb * w, c)


def _vec_to_tb_rgb(vec_bt_d):
  """(B, T, D) float -> (B, T, H, W, 3) uint8 via min-max norm on flattened vector."""
  # Tolerate bool/int inputs (e.g. the discrete manager mask) — min-max needs float.
  vec_bt_d = jnp.asarray(vec_bt_d).astype(jnp.float32)
  d = vec_bt_d.shape[-1]
  h = max(1, int(math.floor(math.sqrt(float(d)))))
  cells = int(math.ceil(d / h) * h)
  w = cells // h
  flat = jnp.pad(vec_bt_d, [(0, 0)] * (vec_bt_d.ndim - 1) + [(0, cells - d)])
  lo = flat.min(axis=-1, keepdims=True)
  hi = flat.max(axis=-1, keepdims=True)
  g = (flat - lo) / (hi - lo + 1e-8)
  g = g.reshape(*vec_bt_d.shape[:-1], h, w, 1)
  u8 = (g * 255).astype(jnp.uint8)
  return jnp.repeat(u8, 3, axis=-1)


def _resize_frames(frames_bthwc, target_h, target_w):
  """Nearest-neighbor resize (B, T, H, W, C) uint8 to (B, T, target_h, target_w, C)."""
  B, T, H, W, C = frames_bthwc.shape
  if H == target_h and W == target_w:
    return frames_bthwc
  flat = frames_bthwc.reshape(B * T, H, W, C).astype(jnp.float32)
  resized = jax.image.resize(flat, (B * T, target_h, target_w, C), method='nearest')
  return jnp.clip(resized, 0, 255).reshape(B, T, target_h, target_w, C).astype(jnp.uint8)


class Agent(embodied.jax.Agent):
  """World model + policy/value; ``loss`` composes model ELBO and imag AC."""


  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    self.use_hrl = bool(getattr(config, 'use_hrl', True))
    # Masked goals (opt-in): manager additionally emits a discrete block mask and
    # the goal decoder produces per-block prototypes + continuous attention masks.
    # Only meaningful under HRL; ``False`` reproduces the plain-goal HRL path.
    self.use_masked_goals = self.use_hrl and bool(
        getattr(config, 'use_masked_goals', False))
    # Hard edit budget: >0 forces exactly this many edited blocks per manager
    # command (Gumbel-top-k over the mask head) and disables the soft sparsity
    # penalty. 0 keeps the adaptive sparsity behaviour.
    self.mask_topk = int(getattr(config, 'mask_topk', 0)) if self.use_masked_goals else 0
    # How the soft sparsity penalty trains the mask head (see configs.yaml):
    # 'sample' (legacy no-op), 'prob' (B2: penalize sigmoid probs, differentiable),
    # 'reinforce' (C: shape the manager reward by the realized edit fraction).
    self.mask_sparsity_mode = str(getattr(config, 'mask_sparsity_mode', 'prob'))
    # Implicit-sparsity controller (works without any edit/abstain mask, incl. plain
    # Director): 'reinforce' drives the measured implicit block sparsity toward
    # ``impl_sparsity_target`` via a per-decision REINFORCE change-cost. 'none' = off.
    self.impl_sparsity_mode = str(getattr(config, 'impl_sparsity_mode', 'none'))
    # Entropy mode only: whether the KL(mask || Bernoulli(sparse_prior)) "prefer not
    # to edit" term is added on top of the mask_actent anti-collapse entropy floor.
    self.mask_kl_enable = bool(getattr(config, 'mask_kl_enable', True))
    # B2 ablation: >0 replaces the adaptive sparsity target with a FIXED multiplier
    # at target 0 (constant pressure to edit no blocks). Only used in 'prob' mode.
    self.mask_sparsity_fixed = float(getattr(config, 'mask_sparsity_fixed_weight', 0.0))
    # Manager-policy input conditioning (masked goals only; all default off, so a run
    # with every flag False feeds the manager exactly ``feat2tensor(feat)`` as before).
    # Each appends a stop-gradient'd channel built from the PRE-edit running code Z, so
    # the manager can see what it is editing / whether the standing goal was reached.
    # ``mgr_cond_goalcode`` is also valid WITHOUT masked goals: a plain Director
    # manager that regenerates the whole code each decision needs to SEE the
    # previous goal to deliberately re-emit (reuse) blocks -- this is what makes
    # implicit sparsity learnable rather than incidental. The conditioning channel
    # is built generically from ``_running_goal_code`` (the emitted skill for plain
    # goals, the running code Z for masked), so no masked-only machinery is needed.
    self.mgr_cond_goalcode = bool(getattr(config, 'mgr_cond_goalcode', False))
    self.mgr_cond_achieve = self.use_masked_goals and bool(
        getattr(config, 'mgr_cond_achieve', False))
    # ``mgr_cond_decgoal`` is also valid WITHOUT masked goals, same reasoning as
    # ``mgr_cond_goalcode`` above: the previous DECODED goal is a generic
    # function of ``_running_goal_code`` regardless of recipe, and is required
    # input for ``goal_reuse_weight`` below (the manager can't reason about how
    # far it's moving the goal without seeing where the goal currently is).
    self.mgr_cond_decgoal = bool(getattr(config, 'mgr_cond_decgoal', False))
    # Delta goal generation (opt-in, F18 follow-up): instead of sampling the new
    # goal code from the manager's own categorical from scratch, its per-class
    # logits pass through an UNNORMALIZED sigmoid ("votes", independent per class,
    # can be driven to all-~0) that gets ADDED to the previous goal code's one-hot
    # before the categorical sample. All-~0 votes -> the sum is just the previous
    # one-hot -> the block is reproduced with ~1.0 probability: a clean, explicit,
    # DIFFERENTIABLE "reuse" action (unlike measuring reuse post hoc via a
    # stop-gradiented argmax==argmax comparison, F18's root cause). Actively
    # re-voting the same class still changes the winning margin and so still
    # carries gradient, so "reuse by default" and "actively re-predict the same
    # thing" remain distinguishable to any downstream loss. Only meaningful for
    # plain (non-masked, non-joint) goals -- it replaces the block-mask/abstain
    # mechanisms as the reuse channel, so it is mutually exclusive with both.
    self.goal_delta_mode = self.use_hrl and bool(
        getattr(config, 'goal_delta_mode', False))
    if self.goal_delta_mode and self.use_masked_goals:
      raise ValueError(
          'goal_delta_mode is mutually exclusive with use_masked_goals (which '
          'implies mask_joint_edit is unreachable too): it is itself the reuse '
          'mechanism, at the plain-goal skill head; combining it with an explicit '
          'edit/abstain mask would give two competing reuse channels.')
    # Delta mode needs to SEE the previous code (both as the additive target and,
    # via ``_mgr_input``, as the input the manager's votes are conditioned on) --
    # force the conditioning channel on rather than silently no-op'ing if the user
    # forgot to also set ``mgr_cond_goalcode``.
    self.mgr_cond_goalcode = self.mgr_cond_goalcode or self.goal_delta_mode
    # Cap on a single class's vote before it's added to the previous one-hot (which
    # always contributes exactly 1.0 to its class): <=1.0 means a single fresh vote
    # can at most tie the previous class's weight, not unilaterally overrule it
    # without the softmax-normalized result still giving the previous class real
    # mass. Purely a stability/tuning knob -- 1.0 (sigmoid's natural ceiling) is a
    # no-op clip.
    self.goal_delta_clip = float(getattr(config, 'goal_delta_clip', 1.0))
    # Direct (REINFORCE-free) implicit sparsity (2026-07-16, F18 follow-up #2):
    # instead of recombining the SAMPLE with the past (goal_delta_mode) or
    # measuring reuse post hoc with no gradient (impl_sparsity_mode), leave
    # plain (direct) sampling untouched and add an ordinary loss on the
    # OVERLAP between this decision's own softmax and the previous decision's
    # -- sum_c p_t[c]*p_{t-1}[c] per block, averaged over blocks -- plus feed
    # the manager the previous decision's soft distribution as an extra input.
    # No sampling, no decoder pass, no reward shaping: backprops straight into
    # both steps' own logits through their softmax (see _emit_manager).
    self.goal_soft_reuse_adapt = self.use_hrl and bool(
        getattr(config, 'goal_soft_reuse_adapt', False))
    if self.goal_soft_reuse_adapt and self.goal_delta_mode:
      raise ValueError(
          'goal_soft_reuse_adapt and goal_delta_mode are mutually exclusive '
          '-- both are reuse mechanisms for the same conditioning channel; '
          'goal_delta_mode recombines the SAMPLE, goal_soft_reuse_adapt only '
          'adds a loss and a soft input on top of plain (direct) sampling.')
    if self.goal_soft_reuse_adapt:
      # Needs to SEE the previous decision's soft distribution, same reasoning
      # as goal_delta_mode above.
      self.mgr_cond_goalcode = True
    # Either reuse mechanism needs the soft ``skill_probs`` side-channel
    # threaded through the manager-skill carry (see _emit_manager/_mgr_input
    # and every init_policy/_unpack_carry/_imagine_with_manager carry init).
    self._mgr_needs_skill_probs = self.goal_delta_mode or self.goal_soft_reuse_adapt
    # Differentiable continuous-goal-space reuse loss (2026-07-16). Code-level
    # reuse (delta mode / implicit_sparsity_block) is a PROXY: an unchanged code
    # doesn't guarantee an unchanged DECODED goal (the decoder needn't be
    # locally smooth there), and a changed code doesn't guarantee a changed one
    # -- while worker success is entirely a function of the decoded goal
    # (goal_reward_cosine_max(goal_deter, feat)), never the code directly.
    # goal_struct_weight targets a GLOBAL code<->deter geometry correlation as
    # an indirect proxy for this; goal_reuse targets the SPECIFIC quantity
    # that matters -- this decision's decoded goal vs. the previous one --
    # directly, with a real gradient path INTO THE MANAGER ONLY (decode the
    # sampled code fresh through goal_dec with the decoder's OWN parameters
    # stop-gradiented, see ``_decode_goal_no_decoder_grad`` -- the goal
    # autoencoder is trained by its own reconstruction/struct losses, not by
    # an incidental manager-shaping objective), compared via
    # goal_reward_cosine_max to the previous decision's (stop-gradiented,
    # fixed) decoded goal. See the loss body (``goal/reuse_sim_mean``) for the
    # exact computation. Two mutually exclusive modes (mirrors
    # goal_struct_weight/goal_struct_adapt): ``goal_reuse_weight`` (fixed,
    # pushes toward similarity=1 unconditionally) or ``goal_reuse_adapt``
    # (recommended: dual-ascent Lagrange multiplier holding mean similarity AT
    # ``goal_reuse_target`` rather than driving it to 1 unconditionally).
    # Both default off (0.0 / False) -- byte-for-byte prior behavior.
    self.goal_reuse_weight = float(getattr(config, 'goal_reuse_weight', 0.0))
    self.goal_reuse_adapt = bool(getattr(config, 'goal_reuse_adapt', False))
    if self.goal_reuse_weight > 0.0 and self.goal_reuse_adapt:
      raise ValueError(
          'goal_reuse_weight (fixed) and goal_reuse_adapt (adaptive Lagrange) '
          'are mutually exclusive modes for the same loss, like '
          'goal_struct_weight/goal_struct_adapt -- set exactly one.')
    if self.goal_reuse_weight > 0.0 or self.goal_reuse_adapt:
      # The manager needs to SEE the previous decoded goal to make an informed
      # decision about how far it's moving it -- force the channel on rather
      # than silently no-op'ing if the user forgot to also set this.
      self.mgr_cond_decgoal = True
    self._mgr_cond_any = (
        self.mgr_cond_goalcode or self.mgr_cond_achieve or self.mgr_cond_decgoal)

    self.skill_shape = config.skill_shape
    skill_shape_t = tuple(int(x) for x in self.skill_shape)
    skill_classes = int(getattr(config, 'skill_classes', skill_shape_t[-1]))
    # Variable goal length (HiPPO-style): manager additionally emits a discrete
    # duration p in {min..max}; the goal is held for p steps before the next
    # decision. ``n_duration_classes`` categorical classes map to p = min + index.
    self.variable_goal_length = bool(getattr(config, 'variable_goal_length', False))
    self.goal_duration_min = int(getattr(config, 'goal_duration_min', 1))
    self.goal_duration_max = int(getattr(config, 'goal_duration_max', 16))
    # HiTS-style timed subgoals (Guertler et al., NeurIPS 2021): condition the
    # worker policy AND value on the countdown until the next manager decision
    # (steps left including the current one, normalized to [-1, 1] by the
    # duration budget). Makes the worker's per-goal horizon observable, so
    # V(s, goal, countdown) is well-posed under variable goal lengths instead
    # of facing a random unobservable deadline.
    self.worker_timed_goals = bool(getattr(config, 'worker_timed_goals', False))
    self.goal_switch_cost = float(getattr(config, 'goal_switch_cost', 0.0))
    # Fixed per-edited-block cost (sparsity analog of goal_switch_cost): subtracted
    # from the manager reward per edited block so editing is priced, not targeted.
    self.goal_edit_cost = float(getattr(config, 'goal_edit_cost', 0.0))
    # B.3 achievability-gated edit cost: when >0, the per-command edit cost is scaled
    # by how reached the *standing* goal already is (achievement cosine in [0,1]), so
    # churning an already-satisfied goal is expensive while re-planning an unmet goal
    # is cheap. No sparsity target -> the interior edit fraction emerges from the state.
    # Cost applied = goal_edit_cost_ach * mask_frac * achievement. Needs mgr_cond_achieve
    # plumbing OFF is fine; the achievement cosine is computed locally at the cost site.
    self.goal_edit_cost_ach = float(getattr(config, 'goal_edit_cost_ach', 0.0))
    # C.5 sparsemax edit gate: replace the per-block independent Bernoulli mask with a
    # sparsemax projection over the L block logits (simplex -> exact zeros), so the edit
    # set is sparse *by construction* with no sparsity target. NOTE: sparsemax sums to 1,
    # i.e. it imposes a soft ~1-block edit budget (a structural choice, not a tuned prior).
    self.mask_sparsemax = bool(getattr(config, 'mask_sparsemax', False)) and self.use_masked_goals
    # Tier 1 (prob_ach): fixed weight on the analytic, achievement-gated expected-edit
    # penalty lambda * mean_i[achievement * sigma(l_i)] (zero-variance twin of B.3, no
    # target). Only used when mask_sparsity_mode == 'prob_ach'.
    self.mask_sparsity_prob_ach_weight = float(
        getattr(config, 'mask_sparsity_prob_ach_weight', 1.0))
    # Tier 2 (mask_perblock_credit): per-block leave-one-out advantage for the edit mask
    # via a goal-conditioned Q-head Q_g(s, Z). Each edited block is credited by how much
    # its edit raised Q_g vs reverting that block. A tiny edit cost breaks ties toward
    # fewer edits; the credit decides WHICH blocks survive. See per_block_credit_plan.md.
    self.mask_perblock_credit = bool(
        getattr(config, 'mask_perblock_credit', False)) and self.use_masked_goals
    # Tier 2 tie-break: small fixed cost subtracted from each edited block's leave-one-out
    # advantage, so a block survives only if its task credit beats the cost (-> sparsity).
    self.perblock_edit_cost = float(getattr(config, 'perblock_edit_cost', 0.0))
    if self.mask_perblock_credit and not self._mgr_cond_any:
      raise ValueError(
          'mask_perblock_credit requires mgr_cond_goalcode or mgr_cond_achieve '
          'or mgr_cond_decgoal so pre-edit goal codes are available.')
    # Single-head ("joint") masked manager: merge the separate ``skill`` (content,
    # L x C onehot) and ``mask`` (per-block Bernoulli edit bit) heads into ONE
    # categorical per block over ``C + 1`` classes -- class 0 = abstain (keep the
    # running block), classes 1..C = overwrite with content class c. One sample,
    # one log-prob per block, so the discarded-content REINFORCE confound of the
    # dual-head path (the content proposal for a masked-off block still gets
    # credited) vanishes by construction. Free sparsity: the abstain rate emerges
    # from REINFORCE alone (no init bias, no rate penalty). Opt-in, A/B'd against
    # the dual-head path; mutually exclusive with the mask-specific machinery below.
    self.mask_joint_edit = self.use_masked_goals and bool(
        getattr(config, 'mask_joint_edit', False))
    if self.mask_joint_edit and (
        self.mask_topk > 0 or self.mask_sparsemax or self.mask_perblock_credit):
      raise ValueError(
          'mask_joint_edit is mutually exclusive with mask_topk, mask_sparsemax, '
          'and mask_perblock_credit (there is no separate mask head to operate on).')
    # Manager extrinsic-reward block aggregation: 'mean' (per-step average, the
    # original; under per-step discounting this under-credits long blocks -> short-K
    # bias) or 'sum' (continuation-weighted SMDP option return, K-neutral).
    self.mgr_reward_agg = str(getattr(config, 'mgr_reward_agg', 'mean'))
    self.variable_goal_block_rew = bool(getattr(
        config, 'variable_goal_block_rew', False))
    # Hindsight-relabel a block-pooled decision's duration class when its hold
    # was cut short by the imagination horizon (rather than a real switch), so
    # REINFORCE credits the duration actually realized, not the one sampled --
    # see relabel_truncated_last_duration. Only meaningful when
    # variable_goal_block_rew is also set; default True (the fix), off is an
    # explicit ablation switch.
    self.goal_duration_relabel_truncated = bool(getattr(
        config, 'goal_duration_relabel_truncated', True))
    self.goal_duration_fixed = int(getattr(config, 'goal_duration_fixed', 0))
    # Set unconditionally so the manager loss call site can read it even in fixed-K
    # mode; the adapter itself is only built under ``variable_goal_length`` below.
    self.goal_duration_adapt = bool(
        getattr(config, 'goal_duration_adapt', False)) and self.variable_goal_length
    # Mask-style Lagrangian duration prior: an AutoAdapt tracks the mean duration
    # itself (not a squared-error-magnitude proxy) directly against
    # ``goal_duration_target``, mirroring ``mask_sparsity_adapter``'s dual ascent on
    # the realized quantity of interest. Folded out into its own top-level loss
    # (``goal_duration_prior``, own ``loss_scales`` entry), unlike the fixed-weight
    # ``goal_duration_reg`` / squared-error-adaptive ``goal_duration_adapt`` paths,
    # which are added directly into ``mgr_policy``. Mutually exclusive with
    # ``goal_duration_adapt`` (this takes priority if both are set).
    self.goal_duration_lagrange = bool(
        getattr(config, 'goal_duration_lagrange', False)) and self.variable_goal_length
    self.n_duration_classes = max(
        1, self.goal_duration_max - self.goal_duration_min + 1)
    if len(skill_shape_t) > 1:
      assert skill_shape_t[-1] == skill_classes, (
          'skill_shape[-1] must equal skill_classes (classes per categorical)')
    # Director-style sparse skills: float one-hot matrix, not integer indices.
    self.skill_space = elements.Space(np.float32, skill_shape_t, 0.0, 1.0)
    # Single-head manager (``mask_joint_edit``): the manager ``skill`` head widens
    # by one class per block (class 0 = abstain), so its onehot sample is (L, C+1);
    # the running goal code ``Z`` and the goal autoencoder stay (L, C).
    self.joint_skill_shape = (skill_shape_t[0], skill_shape_t[-1] + 1)
    self.goal_shape = (self.config.dyn.rssm.deter,)

    # Encoder/decoder omit control/meta keys; dynamics still sees actions separately.
    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    # Goal autoencoder (Director): L×C logits, straight-through one-hot sample,
    # flatten to sparse L*C vector for the decoder. ``skill_shape`` is (L, C).
    # Only built when HRL is enabled — flat mode has no goals.
    if self.use_hrl:
      self.goal_code_space = elements.Space(np.float32, skill_shape_t, 0.0, 1.0)
      self.goal_enc = embodied.jax.MLPHead(
          self.goal_code_space, **config.goal_enc, name='goal_enc')
      # Plain Director goal decoder (skill code -> deter). Masked goals reuse this
      # unchanged: the mask now edits the running goal *code* (block-overwrite),
      # not the decoder, so reconstruction is the standard goal autoencoder.
      self.goal_dec = embodied.jax.MLPHead(self.goal_shape, **config.goal_dec, name='goal_dec')
      self.goal_autoencoder_beta = config.goal_autoencoder_beta
      # Weight on the code/deter geometry-preservation term (0 = off).
      self.goal_struct_weight = float(getattr(config, 'goal_struct_weight', 0.0))
      # Uniform prior metadata only: built inside ``loss`` with ``zeros_like`` encoder
      # logits so arrays stay on-device (``jnp.zeros`` here breaks sharded init).
      self._skill_prior_unimix = float(config.goal_enc.unimix)
      self._skill_factorized = len(skill_shape_t) > 1

    # Flat RSSM state for MLP heads: deterministic dim + flattened stochastic samples.
    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    self.feat2deter = lambda x: nn.cast(x['deter'])

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    # Reward and continue predictors on RSSM features (Gaussian / Bernoulli heads).
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    policy_outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, policy_outs, **config.policy, name='pol')

    if self.use_hrl:
      if self.use_masked_goals or self.variable_goal_length:
        # Joint manager head with a shared trunk and multiple output heads:
        #   skill    : discrete skill code (onehot L,C) -- always present. Under
        #              ``mask_joint_edit`` it widens to (L, C+1) and absorbs the
        #              mask (class 0 = abstain), so no separate ``mask`` head.
        #   mask     : discrete block mask m_t (Bernoulli per block) -- masked goals
        #              (dual-head path only; dropped under ``mask_joint_edit``)
        #   duration : categorical over {0..n-1} -> p = goal_duration_min + idx
        #              -- variable goal length
        mgr_cfg = {**config.manager_policy}
        skill_out = mgr_cfg.pop('output')
        if self.mask_joint_edit:
          skill_space = elements.Space(
              np.float32, self.joint_skill_shape, 0.0, 1.0)
        else:
          skill_space = self.goal_code_space
        mgr_space = {'skill': skill_space}
        mgr_out = {'skill': skill_out}
        if self.use_masked_goals and not self.mask_joint_edit:
          mgr_space['mask'] = elements.Space(bool, (skill_shape_t[0],), 0, 2)
          mgr_out['mask'] = 'binary'
        if self.variable_goal_length:
          mgr_space['duration'] = elements.Space(
              np.int32, (), 0, self.n_duration_classes)
          mgr_out['duration'] = 'categorical'
        self.manager_pol = embodied.jax.MLPHead(
            mgr_space, mgr_out, **mgr_cfg, name='manager_pol')
      else:
        self.manager_pol = embodied.jax.MLPHead(
            self.goal_code_space, **config.manager_policy, name='manager_pol')
      self.manager_sample_freq = config.manager_sample_freq

    if self.use_hrl:
      # Separate extrinsic and exploratory value heads (+ EMA targets).
      self.mgr_extr_val = embodied.jax.MLPHead(scalar, **config.value, name='mgr_extr_val')
      self.mgr_extr_slowval = embodied.jax.SlowModel(
          embodied.jax.MLPHead(scalar, **config.value, name='mgr_extr_slowval'),
          source=self.mgr_extr_val, **config.slowvalue)
      self.mgr_expl_val = embodied.jax.MLPHead(scalar, **config.value, name='mgr_expl_val')
      self.mgr_expl_slowval = embodied.jax.SlowModel(
          embodied.jax.MLPHead(scalar, **config.value, name='mgr_expl_slowval'),
          source=self.mgr_expl_val, **config.slowvalue)

      self.wkr_goal_val = embodied.jax.MLPHead(scalar, **config.value, name='wkr_goal_val')
      self.wkr_goal_slowval = embodied.jax.SlowModel(
          embodied.jax.MLPHead(scalar, **config.value, name='wkr_goal_slowval'),
          source=self.wkr_goal_val, **config.slowvalue)

      if self.mask_perblock_credit:
        # Tier 2: goal-conditioned manager Q-head Q_g(s, Z) for leave-one-out
        # counterfactual credit on the edit mask (see per_block_credit_plan.md).
        self.mgr_goal_q = embodied.jax.MLPHead(scalar, **config.value, name='mgr_goal_q')
        self.mgr_goal_q_slowval = embodied.jax.SlowModel(
            embodied.jax.MLPHead(scalar, **config.value, name='mgr_goal_q_slowval'),
            source=self.mgr_goal_q, **config.slowvalue)
        self.mgr_goal_q_valnorm = embodied.jax.Normalize(
            **config.valnorm, name='mgr_goal_q_valnorm')

      # DreamerV3 actor-critic normalization (``none`` valnorm / ``none`` advnorm).
      # The critics are symexp_twohot heads trained on RAW returns. The worker and
      # flat heads scale the advantage by the percentile return range (``perc``
      # retnorm); the manager instead uses ``mgr_retnorm`` (``meanstd``), i.e.
      # Director's std-based return scaling. Each critic owns its own valnorm
      # (no-op under ``none`` but kept so the unnorm/norm path mirrors flat v3).
      self.mgr_extr_retnorm = embodied.jax.Normalize(**config.mgr_retnorm, name='mgr_extr_retnorm')
      self.mgr_expl_retnorm = embodied.jax.Normalize(**config.mgr_retnorm, name='mgr_expl_retnorm')
      self.wkr_goal_retnorm = embodied.jax.Normalize(**config.retnorm, name='wkr_goal_retnorm')

      self.mgr_extr_valnorm = embodied.jax.Normalize(**config.valnorm, name='mgr_extr_valnorm')
      self.mgr_expl_valnorm = embodied.jax.Normalize(**config.valnorm, name='mgr_expl_valnorm')
      self.wkr_goal_valnorm = embodied.jax.Normalize(**config.valnorm, name='wkr_goal_valnorm')

      self.mgr_advnorm = embodied.jax.Normalize(**config.advnorm, name='mgr_advnorm')
      self.wkr_goal_advnorm = embodied.jax.Normalize(**config.advnorm, name='wkr_goal_advnorm')

      self.mgr_expl_weight = config.mgr_expl_weight

      # Director-style adaptive Lagrange multipliers (``tfutils.AutoAdapt``).
      self.manager_actent_perdim = bool(config.manager_actent_perdim)
      mgr_actent_shape = (skill_shape_t[0],) if self.manager_actent_perdim else ()
      self.mgr_actent = embodied.jax.AutoAdapt(
          shape=mgr_actent_shape,
          impl=config.manager_actent_impl,
          target=float(config.manager_actent_target),
          min=float(config.manager_actent_min),
          max=float(config.manager_actent_max),
          vel=float(config.manager_actent_vel),
          inverse=True,
          init=float(config.manager_actent_init),
          name='mgr_actent')
      if self.variable_goal_length:
        # Dedicated SCALAR entropy adapter for the duration head (the per-dim
        # ``mgr_actent`` is shaped for the L skill blocks and cannot also regulate
        # the single duration categorical). Limits as ``mgr_actent``; target may be
        # raised independently via ``manager_actent_duration_target`` (<0 inherits
        # ``manager_actent_target``) to keep the duration head exploratory and stop
        # the variable-K short-K collapse without a duration prior.
        _dur_target = float(getattr(config, 'manager_actent_duration_target', -1.0))
        if _dur_target < 0:
          _dur_target = float(config.manager_actent_target)
        self.mgr_dur_actent = embodied.jax.AutoAdapt(
            shape=(),
            impl=config.manager_actent_impl,
            target=_dur_target,
            min=float(config.manager_actent_min),
            max=float(config.manager_actent_max),
            vel=float(config.manager_actent_vel),
            inverse=True,
            init=float(config.manager_actent_init),
            name='mgr_dur_actent')
        # Optional adaptive duration prior: auto-tunes the weight on the squared
        # duration error toward ``goal_duration_adapt_setpoint``, capped at
        # ``goal_duration_adapt_max`` so it can never dominate the manager REINFORCE
        # objective (inverse=False: multiplier grows while the error sits above the
        # setpoint, shrinks below). Only built when enabled.
        if self.goal_duration_adapt:
          adapt_init = float(getattr(config, 'goal_duration_adapt_init', 0.001))
          self.mgr_dur_reg_adapter = embodied.jax.AutoAdapt(
              shape=(),
              impl='mult',
              target=float(getattr(config, 'goal_duration_adapt_setpoint', 0.5)),
              min=adapt_init,
              max=float(getattr(config, 'goal_duration_adapt_max', 0.05)),
              inverse=False,
              init=adapt_init,
              name='mgr_dur_reg_adapter')
        # Mask-style Lagrangian alternative: regulates the switch-weighted mean
        # ABSOLUTE deviation |E[dur] - goal_duration_target| against a small
        # tolerance (``goal_duration_lagrange_tol``), so the response is symmetric
        # in the deviation direction (the first version regulated the raw mean
        # duration one-sidedly -- it WEAKENED the prior when durations collapsed
        # short, and railed at max during the early long-duration phase; e152-e159).
        # The multiplier grows while the deviation exceeds the tolerance and
        # self-relaxes once within it; capped at ``goal_duration_lagrange_max`` so
        # it can never dominate the manager REINFORCE objective.
        if self.goal_duration_lagrange:
          self.mgr_dur_lagrange_adapter = embodied.jax.AutoAdapt(
              shape=(),
              impl=getattr(config, 'goal_duration_lagrange_impl', 'mult'),
              target=float(getattr(config, 'goal_duration_lagrange_tol', 0.1)),
              min=float(getattr(config, 'goal_duration_lagrange_min', 1e-5)),
              max=float(getattr(config, 'goal_duration_lagrange_max', 5.0)),
              vel=float(getattr(config, 'goal_duration_lagrange_vel', 0.1)),
              inverse=False,
              init=float(getattr(config, 'goal_duration_lagrange_init', 1.0)),
              name='mgr_dur_lagrange_adapter')
      self.goal_kl_adapter = embodied.jax.AutoAdapt(
          shape=(),
          impl=config.goal_kl_impl,
          target=float(config.goal_kl_target),
          min=float(config.goal_kl_min),
          max=float(config.goal_kl_max),
          vel=float(config.goal_kl_vel),
          inverse=False,
          init=float(config.goal_kl_init),
          name='goal_kl_adapter')
      if self.impl_sparsity_mode != 'none':
        # Dual-ascent Lagrange multiplier on the per-decision CHANGE fraction
        # (1 - implicit block sparsity): grows while the manager changes more
        # blocks than allowed by the annealed kept-target, subtracting a REINFORCE
        # cost from the manager reward so it re-emits blocks identically. Applies to
        # any HRL run (masked, joint, or plain Director) -- no mask head needed.
        # ``impl_sparsity_one_sided`` -> ratchet (scale only grows).
        self.impl_sparsity_adapter = embodied.jax.AutoAdapt(
            shape=(),
            impl=str(getattr(config, 'impl_sparsity_impl', 'mult')),
            target=1.0 - float(getattr(config, 'impl_sparsity_target', 0.5)),
            min=float(getattr(config, 'impl_sparsity_min', 1e-5)),
            max=float(getattr(config, 'impl_sparsity_max', 5.0)),
            vel=float(getattr(config, 'impl_sparsity_vel', 0.1)),
            inverse=False,
            one_sided=bool(getattr(config, 'impl_sparsity_one_sided', False)),
            init=float(getattr(config, 'impl_sparsity_init', 1.0)),
            name='impl_sparsity_adapter')
        # Open-loop ramp of the KEPT-fraction target (Ratchet): init == target is a
        # DIRECT target (no gradual descent); init below target ratchets sparsity in.
        self.impl_sparsity_target_sched = embodied.jax.Ratchet(
            shape=(),
            init=float(getattr(
                config, 'impl_sparsity_target_init',
                getattr(config, 'impl_sparsity_target', 0.5))),
            final=float(getattr(config, 'impl_sparsity_target', 0.5)),
            vel=float(getattr(config, 'impl_sparsity_target_vel', 0.01)),
            name='impl_sparsity_target_sched')
      if getattr(config, 'goal_struct_adapt', False):
        # Adaptive struct weight (dual-ascent, two-sided by default): grows
        # while the raw struct MSE/margin loss (``goal/struct_loss``) sits above
        # ``goal_struct_adapt_target``, shrinks while below, auto-tuning the
        # pressure instead of a fixed ``goal_struct_weight``. Optional
        # ``goal_struct_adapt_one_sided`` (off by default) drops the shrink
        # branch -- relevant if the target is later re-violated after being
        # cleared (BIG-scale struct loss was observed to drift back above a
        # fixed weight's own earlier level later in training: e171, 0.0068
        # @445k -> 0.0183 @3.1M) and premature relaxation is a concern.
        self.goal_struct_adapter = embodied.jax.AutoAdapt(
            shape=(),
            impl='mult',
            target=float(config.goal_struct_adapt_target),
            min=float(config.goal_struct_adapt_min),
            max=float(config.goal_struct_adapt_max),
            vel=float(config.goal_struct_adapt_vel),
            inverse=False,
            one_sided=bool(getattr(config, 'goal_struct_adapt_one_sided', False)),
            init=float(config.goal_struct_adapt_init),
            name='goal_struct_adapter')
      if self.goal_reuse_adapt:
        # Adaptive goal-reuse weight (dual-ascent Lagrange multiplier): grows
        # while mean realized similarity (``goal/reuse_sim_mean``, in [-1,1])
        # sits BELOW ``goal_reuse_target``, shrinks while above --
        # ``inverse=True`` is the same sense used for entropy regularizers
        # ("push up when below target"), since higher similarity is what we
        # want, unlike struct/mask_sparsity's raw losses which we want to
        # push DOWN. Self-tunes the pressure to hold similarity AT the
        # target rather than driving it to 1 unconditionally.
        self.goal_reuse_adapter = embodied.jax.AutoAdapt(
            shape=(),
            impl='mult',
            target=float(config.goal_reuse_target),
            min=float(config.goal_reuse_adapt_min),
            max=float(config.goal_reuse_adapt_max),
            vel=float(config.goal_reuse_adapt_vel),
            inverse=True,
            one_sided=bool(getattr(config, 'goal_reuse_adapt_one_sided', False)),
            init=float(config.goal_reuse_adapt_init),
            name='goal_reuse_adapter')
        # Open-loop anneal of the TARGET itself (agent.Ratchet, same mechanism
        # as impl_sparsity_target_sched / mask_sparsity_target's ratchet):
        # ramps from goal_reuse_target_init toward goal_reuse_target by at
        # most goal_reuse_target_vel per training call, giving the manager
        # time to establish basic task competence before the similarity
        # constraint tightens to its final value, rather than imposing the
        # full target from step 0. Default init == target -> no-op (constant
        # target, byte-for-byte prior behavior unless explicitly set apart).
        self.goal_reuse_target_sched = embodied.jax.Ratchet(
            shape=(),
            init=float(getattr(
                config, 'goal_reuse_target_init', config.goal_reuse_target)),
            final=float(config.goal_reuse_target),
            vel=float(getattr(config, 'goal_reuse_target_vel', 0.01)),
            name='goal_reuse_target_sched')
      if self.goal_soft_reuse_adapt:
        # Adaptive weight on the direct block-overlap loss (dual-ascent
        # Lagrange multiplier): grows while mean realized overlap sits BELOW
        # ``goal_soft_reuse_target``, shrinks while above -- ``inverse=True``,
        # same sense as ``goal_reuse_adapter`` (higher overlap is what we want).
        self.goal_soft_reuse_adapter = embodied.jax.AutoAdapt(
            shape=(),
            impl='mult',
            target=float(config.goal_soft_reuse_target),
            min=float(config.goal_soft_reuse_adapt_min),
            max=float(config.goal_soft_reuse_adapt_max),
            vel=float(config.goal_soft_reuse_adapt_vel),
            inverse=True,
            one_sided=bool(getattr(config, 'goal_soft_reuse_adapt_one_sided', False)),
            init=float(config.goal_soft_reuse_adapt_init),
            name='goal_soft_reuse_adapter')
        # Open-loop ramp of the TARGET itself (agent.Ratchet, same mechanism as
        # impl_sparsity_target_sched / goal_reuse_target_sched): ramps from
        # goal_soft_reuse_target_init toward goal_soft_reuse_target by at most
        # goal_soft_reuse_target_vel per training call. Default init == target
        # -> no-op (constant target) unless explicitly ratcheted.
        self.goal_soft_reuse_target_sched = embodied.jax.Ratchet(
            shape=(),
            init=float(getattr(
                config, 'goal_soft_reuse_target_init', config.goal_soft_reuse_target)),
            final=float(config.goal_soft_reuse_target),
            vel=float(getattr(config, 'goal_soft_reuse_target_vel', 0.01)),
            name='goal_soft_reuse_target_sched')
      if self.use_masked_goals and self.mask_topk <= 0 and (
          not self.mask_joint_edit or self.mask_sparsity_mode != 'none'):
        # Adaptive sparsity: drives the mean fraction of active manager-mask bits
        # toward ``mask_sparsity_target`` so the mask does not collapse to
        # all-ones (which would reduce masked goals to plain goals).
        self.mask_sparsity_adapter = embodied.jax.AutoAdapt(
            shape=(),
            impl=getattr(config, 'mask_sparsity_impl', 'mult'),
            target=float(getattr(config, 'mask_sparsity_target', 0.3)),
            min=float(getattr(config, 'mask_sparsity_min', 1e-5)),
            max=float(getattr(config, 'mask_sparsity_max', 1.0)),
            vel=float(getattr(config, 'mask_sparsity_vel', 0.1)),
            inverse=False,
            init=float(getattr(config, 'mask_sparsity_init', 1.0)),
            name='mask_sparsity_adapter')
        # Open-loop anneal of the sparsity TARGET itself (not the Lagrange scale
        # above): ramps ``mask_sparsity_target_init`` -> ``mask_sparsity_target``
        # by at most ``mask_sparsity_target_vel`` per training call. Defaults to
        # init == target, i.e. a no-op constant target -- opt in per experiment
        # by setting ``mask_sparsity_target_init`` above ``mask_sparsity_target``
        # (e.g. 1.0 -> 0.3, to allow full-goal edits early for exploration before
        # narrowing to the sparse regime).
        self.mask_sparsity_target_sched = embodied.jax.Ratchet(
            shape=(),
            init=float(getattr(
                config, 'mask_sparsity_target_init',
                getattr(config, 'mask_sparsity_target', 0.3))),
            final=float(getattr(config, 'mask_sparsity_target', 0.3)),
            vel=float(getattr(config, 'mask_sparsity_target_vel', 0.01)),
            name='mask_sparsity_target_sched')
        # Entropy-native mask controls (built when mode == 'entropy' or 'prob_entropy';
        # off otherwise). ``mask_actent`` (inverse=True, entropy-style) holds the
        # per-block mask entropy near a fraction-of-max setpoint so the mask stays
        # stochastic (anti-collapse); ``mask_kl_adapter`` (inverse=False, KL-style)
        # holds KL(mask || sparse prior) near a nats budget so the mask prefers not to
        # edit unless task advantage pays for it. Both targets are dimensionless ->
        # transfer across env / model size. ``prob_entropy`` combines this with the
        # rate-targeting ``prob`` adapter below: the rate adapter controls the MEAN
        # edit fraction, while this controls PER-BLOCK stochasticity, preventing the
        # "railing to a corner" collapse where individual blocks saturate to hard 0/1
        # even though the batch-mean sits on target.
        if self.mask_sparsity_mode in ('entropy', 'prob_entropy'):
          self.mask_sparse_prior = float(getattr(config, 'mask_sparse_prior', 0.1))
          self.mask_actent = embodied.jax.AutoAdapt(
              shape=(),
              impl=getattr(config, 'mask_actent_impl', 'mult'),
              target=float(getattr(config, 'mask_actent_target', 0.5)),
              min=float(getattr(config, 'mask_actent_min', 1e-5)),
              max=float(getattr(config, 'mask_actent_max', 1e2)),
              vel=float(getattr(config, 'mask_actent_vel', 0.1)),
              inverse=True,
              init=float(getattr(config, 'mask_actent_init', 1.0)),
              name='mask_actent')
          # KL(mask || sparse prior) adapter: only built when mask_kl_enable, so a
          # disabled run carries no unused adapter state. See mask_kl_enable above.
          if self.mask_kl_enable:
            self.mask_kl_adapter = embodied.jax.AutoAdapt(
                shape=(),
                impl=getattr(config, 'mask_kl_impl', 'mult'),
                target=float(getattr(config, 'mask_kl_target', 0.2)),
                min=float(getattr(config, 'mask_kl_min', 1e-5)),
                max=float(getattr(config, 'mask_kl_max', 1e2)),
                vel=float(getattr(config, 'mask_kl_vel', 0.1)),
                inverse=False,
                init=float(getattr(config, 'mask_kl_init', 1.0)),
                name='mask_kl_adapter')
    else:
      # Flat AC heads (v4-online): single value/critic over WM features.
      self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
      self.slowval = embodied.jax.SlowModel(
          embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
          source=self.val, **config.slowvalue)
      self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
      self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
      self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    # Three independent optimizers (Director-style). The world-model + heads, the
    # goal autoencoder (VAE), and the actor-critic each get their own optax state
    # (separate momentum / RMS / AGC / lr). ``MultiOptimizer`` does a single
    # backward pass over the union of modules, then routes each module group's
    # gradients to its own optimizer — gradients are identical to one combined
    # optimizer (stop-gradients already isolate the groups) but the optimization
    # dynamics decouple, so each component can use the default v3 settings or be
    # tuned independently.
    model_modules = [self.dyn, self.enc, self.dec, self.rew, self.con]
    if self.use_hrl:
      goal_modules = [self.goal_enc, self.goal_dec]
      ac_modules = [
          self.manager_pol, self.pol,
          self.mgr_extr_val, self.mgr_expl_val, self.wkr_goal_val,
      ]
      if self.mask_perblock_credit:
        ac_modules.append(self.mgr_goal_q)
      groups = {
          'model': (model_modules, self._make_opt(**config.opt)),
          'goal': (goal_modules, self._make_opt(**config.goal_opt)),
          'ac': (ac_modules, self._make_opt(**config.ac_opt)),
      }
    else:
      ac_modules = [self.pol, self.val]
      groups = {
          'model': (model_modules, self._make_opt(**config.opt)),
          'ac': (ac_modules, self._make_opt(**config.ac_opt)),
      }
    self.modules = [m for ms, _ in groups.values() for m in ms]
    self.opt = embodied.jax.MultiOptimizer(
        groups, summary_depth=1, name='opt')

    # One ``rec`` scale is expanded to every reconstruction key in ``dec_space``.
    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    if self.use_hrl:
      policy_scale = scales.pop('policy', 1.0)
      value_scale = scales.pop('value', 1.0)
      scales['mgr_policy'] = policy_scale
      scales['wkr_policy'] = policy_scale
      scales['mgr_extr_value'] = value_scale
      scales['mgr_expl_value'] = value_scale
      scales['wkr_goal_value'] = value_scale
      if self.mask_perblock_credit:
        scales['mgr_goal_q_value'] = value_scale
      if self.config.repval_loss and 'repval' in scales:
        # ``train`` only emits the replay-value losses when ``repval_loss`` is on, so
        # only register their scales then; otherwise drop ``repval`` entirely (else
        # the loss/scale key-set assertion in ``loss`` fails). Mirrors flat mode.
        repval_scale = scales.pop('repval')
        scales['repmgr_extr_value'] = repval_scale
        scales['repmgr_expl_value'] = repval_scale
        scales['repwkr_goal_value'] = repval_scale
      else:
        scales.pop('repval', None)
      if (self.use_masked_goals and self.mask_topk <= 0 and not self.mask_sparsemax
          and self.mask_sparsity_mode not in ('reinforce', 'none')):
        # Soft sparsity penalty as its own loss term ('prob'/'sample' modes).
        # With a hard top-k edit budget OR sparsemax gate there is no sparsity loss
        # (sparsity is by construction); in 'reinforce'
        # mode the penalty is folded into the manager reward; and in 'none' mode
        # there is no sparsity penalty at all (free/priced sparsity). In all those
        # cases the key is dropped so losses and scales stay in sync.
        scales['mask_sparsity'] = scales.pop(
            'mask_sparsity', getattr(self.config.loss_scales, 'mask_sparsity', 1.0))
      else:
        scales.pop('mask_sparsity', None)
      if self.goal_duration_lagrange:
        # Folded-out Lagrangian duration prior (own loss key, own scale) -- see
        # ``mgr_dur_lagrange_adapter`` / ``losses['goal_duration_prior']``.
        scales['goal_duration_prior'] = scales.pop(
            'goal_duration_prior',
            getattr(self.config.loss_scales, 'goal_duration_prior', 1.0))
      else:
        scales.pop('goal_duration_prior', None)
      if self.goal_reuse_weight > 0.0 or self.goal_reuse_adapt:
        # Differentiable continuous-goal-space reuse loss, own key -- see
        # ``losses['goal_reuse']`` in ``loss()``. The actual weight (fixed
        # ``goal_reuse_weight`` or the adapter's own dual-ascent scale) is
        # baked into the loss value; this outer scale stays at its configured
        # default (1.0), matching ``mask_sparsity``/``goal_struct``.
        scales['goal_reuse'] = scales.pop(
            'goal_reuse', getattr(self.config.loss_scales, 'goal_reuse', 1.0))
      else:
        scales.pop('goal_reuse', None)
      if self.goal_soft_reuse_adapt:
        # Direct block-overlap sparsity loss, own key -- see
        # ``losses['goal_soft_reuse']`` in ``loss()``. Same convention as
        # ``goal_reuse`` above: the adapter's own dual-ascent scale is baked
        # into the loss value, so this outer scale stays at its default (1.0).
        scales['goal_soft_reuse'] = scales.pop(
            'goal_soft_reuse',
            getattr(self.config.loss_scales, 'goal_soft_reuse', 1.0))
      else:
        scales.pop('goal_soft_reuse', None)
    else:
      # Flat mode: keep ``policy``/``value``/``repval`` (default keys), drop HRL-only.
      scales.pop('goal_autoencoder', None)
      scales.pop('mask_sparsity', None)
      scales.pop('goal_duration_prior', None)
      scales.pop('goal_reuse', None)
      scales.pop('goal_soft_reuse', None)
      if not self.config.repval_loss:
        scales.pop('repval', None)
    self.scales = scales

  @property
  def policy_keys(self):
    # Regex for checkpoint / param groups synced to the actor process.
    if self.use_masked_goals:
      # Overwrite goals: the policy also encodes the current state (goal_enc) to
      # seed the running goal code at episode boundaries.
      return '^(enc|dyn|dec|pol|manager_pol|goal_dec|goal_enc)/'
    if self.use_hrl:
      return '^(enc|dyn|dec|pol|manager_pol|goal_dec)/'
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    """Extra keys stored in replay beyond ``obs_space`` (chunk id, optional RNN entries)."""
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    """RNN carries for enc/dyn/dec, prev action, and (HRL) manager skill state."""
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    base = (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))
    if not self.use_hrl:
      return base
    skill_shape = tuple(int(x) for x in self.skill_shape)
    mgr_shape = self.joint_skill_shape if self.mask_joint_edit else skill_shape
    mgr_skill = {'skill': jnp.zeros((batch_size, *mgr_shape), f32)}
    if self._mgr_needs_skill_probs:
      # Soft pre-sample distribution behind the current ``skill`` one-hot, for
      # ``_mgr_input``'s next-step conditioning (see ``_delta_combine_skill``/
      # ``goal_soft_reuse_adapt``). Zero-init like ``skill`` itself; harmless
      # since ``update`` is always True on the very first (reset) decision,
      # discarding this placeholder value.
      mgr_skill['skill_probs'] = jnp.zeros((batch_size, *mgr_shape), f32)
    if self.use_masked_goals:
      # Persisted running goal code Z (always (L, C)); the emitted edit is a
      # per-block Bernoulli ``mask`` in the dual-head path, absorbed into the
      # widened ``skill`` head under ``mask_joint_edit`` (no separate mask entry).
      if not self.mask_joint_edit:
        mgr_skill['mask'] = jnp.zeros((batch_size, skill_shape[0]), f32)
      mgr_skill['goal_code'] = jnp.zeros((batch_size, *skill_shape), f32)
      # Sticky mask of the most recent NON-empty edit, for the mask_viz yellow
      # overlay (held across empty/held goals; reset at episode start).
      mgr_skill['last_edit_mask'] = jnp.zeros((batch_size, skill_shape[0]), f32)
    # Sticky mask of blocks CHANGED vs the previous goal (implicit sparsity), for the
    # mask_viz overlay; updated on each manager switch, held between. Tracked for ANY
    # HRL run (masked or plain Director), not just ``use_masked_goals`` -- plain
    # Director can also reproduce a block's previous class (optionally biased there
    # by ``mgr_cond_goalcode``/the implicit-sparsity controller), so "unchanged" is a
    # real, informative signal there too, not just under explicit masking.
    mgr_skill['last_change_mask'] = jnp.zeros((batch_size, skill_shape[0]), f32)
    if self.variable_goal_length:
      mgr_skill['duration'] = jnp.zeros((batch_size,), i32)
    # Decoded worker goal (deter); refreshed on manager switch / episode reset and
    # held constant between switches so rollout does not chase a moving decode.
    mgr_skill['goal_deter'] = jnp.zeros((batch_size, int(self.goal_shape[0])), f32)
    # Held decoded-goal image frame(s); refreshed on switch/reset in ``policy`` so
    # the goal/mask_viz panels stay pixel-stable as the decoder keeps training.
    mgr_skill.update(self._goal_img_cache(batch_size))
    # ``mgr_step`` is a per-element counter (fixed K: step index for ``% K``;
    # variable: remaining steps until the next manager decision, starts at 0 so
    # the first step switches).
    mgr_step = jnp.zeros((batch_size,), i32)
    return (*base, mgr_skill, mgr_step)

  def init_train(self, batch_size):
    """Same carry shape as policy (training reuses the same state layout)."""
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    """Same carry shape as policy for ``report`` rollouts."""
    return self.init_policy(batch_size)

  def _goal_img_cache(self, batch_size):
    """Zeros for the held decoded-goal *image* cache (one uint8 frame per imgkey).

    Holding the deter goal vector is not enough for a stable panel: the image
    decoder / dynamics prior keep training during online rollout, so a fresh
    decode of the same held goal drifts every step. We cache the rendered frame in
    the carry and refresh it only on a manager switch / episode reset. Empty unless
    the policy goal-image panels are active; gated identically to the render path in
    ``policy`` so the carry pytree matches what ``policy`` writes back."""
    cache = {}
    if self.dec.imgkeys and bool(getattr(self.config, 'policy_goal_image', True)):
      for k in self.dec.imgkeys:
        shp = tuple(int(x) for x in self.obs_space[k].shape)
        cache[f'goal_img_{k}'] = jnp.zeros((batch_size, *shp), jnp.uint8)
    return cache

  def _unpack_carry(self, carry):
    """``(enc, dyn, dec, prevact, mgr_skill, mgr_step)``; tolerate 4-tuples (flat mode)."""
    if len(carry) == 6:
      return carry
    enc, dyn, dec, prevact = carry[:4]
    # ``enc`` carry can be empty (stateless encoder), so derive B from any
    # non-empty carry leaf (dyn/prevact always have a leading batch dim).
    B = jax.tree.leaves((enc, dyn, dec, prevact))[0].shape[0]
    skill_shape = tuple(int(x) for x in self.skill_shape)
    mgr_shape = self.joint_skill_shape if self.mask_joint_edit else skill_shape
    mgr_skill = {'skill': jnp.zeros((B, *mgr_shape), f32)}
    if self._mgr_needs_skill_probs:
      mgr_skill['skill_probs'] = jnp.zeros((B, *mgr_shape), f32)
    if self.use_masked_goals:
      if not self.mask_joint_edit:
        mgr_skill['mask'] = jnp.zeros((B, skill_shape[0]), f32)
      mgr_skill['goal_code'] = jnp.zeros((B, *skill_shape), f32)
      mgr_skill['last_edit_mask'] = jnp.zeros((B, skill_shape[0]), f32)
    mgr_skill['last_change_mask'] = jnp.zeros((B, skill_shape[0]), f32)
    if self.variable_goal_length:
      mgr_skill['duration'] = jnp.zeros((B,), i32)
    mgr_skill['goal_deter'] = jnp.zeros((B, int(self.goal_shape[0])), f32)
    mgr_skill.update(self._goal_img_cache(B))
    mgr_step = jnp.zeros((B,), i32)
    return enc, dyn, dec, prevact, mgr_skill, mgr_step

  def _pack_carry(self, enc, dyn, dec, prevact, mgr_skill, mgr_step):
    if not self.use_hrl:
      return (enc, dyn, dec, prevact)
    return (enc, dyn, dec, prevact, mgr_skill, mgr_step)

  def _video(self, video_bthwc):
    """Apply config-driven time/space stride before flattening for video logs."""
    ts = int(getattr(self.config, 'report_video_time_stride', 1))
    ss = int(getattr(self.config, 'report_video_space_stride', 1))
    return _tb_video_grid(video_bthwc, time_stride=ts, space_stride=ss)

  def _mgr_goal_q_inp(self, feat_inp, goal_code):
    """Input to the goal-conditioned manager Q-head: concat WM feat with flat Z."""
    flat = nn.cast(goal_code).reshape(*goal_code.shape[:-2], -1)
    return jnp.concatenate([feat_inp, flat], -1)

  def _countdown_budget(self):
    """Duration budget normalizing the worker countdown (HiTS delta_t_max)."""
    if self.variable_goal_length:
      return float(self.goal_duration_max)
    return float(max(1, int(self.manager_sample_freq)))

  def _countdown_norm(self, cd_steps):
    """Map steps-left-including-current to [-1, 1] (HiTS convert_time)."""
    dmax = self._countdown_budget()
    cd = jnp.clip(f32(cd_steps), 0.0, dmax)
    return (2.0 * cd / dmax - 1.0)[..., None]

  def _feat_goal2tensor(self, x, y, countdown=None):
    """Concatenate WM features with goal; supports ``(B, D)`` and ``(B, T, D)`` goals.

    With ``worker_timed_goals``, also appends the normalized countdown until the
    next manager decision ((..., 1), in [-1, 1]). ``countdown=None`` falls back
    to the full budget (+1) — only report/viz rollouts with a held goal use this;
    every training path passes the real countdown."""
    deter = nn.cast(x['deter'])
    stoch = nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))
    if y.ndim == deter.ndim:
      goal = nn.cast(y)
    else:
      goal = nn.cast(y.reshape((*y.shape[:-2], -1)))
    parts = [deter, stoch, goal]
    if self.worker_timed_goals:
      if countdown is None:
        countdown = jnp.ones(deter.shape[:-1] + (1,), f32)
      parts.append(nn.cast(sg(countdown)))
    return jnp.concatenate(parts, -1)

  def _running_goal_code(self, skills):
    """The skill code that decodes to the worker goal.

    Masked goals carry a persisted running goal code ``goal_code`` (Z) that the
    manager edits block-by-block; plain goals decode the emitted skill directly."""
    if isinstance(skills, dict) and 'goal_code' in skills:
      return skills['goal_code']
    return skills['skill'] if isinstance(skills, dict) else skills

  def _encode_goal_code(self, deter, bdims):
    """Encode a ``deter`` vector into a sampled skill code (goal-VAE encoder)."""
    enc = sample(self.goal_enc(sg(deter), bdims))
    return sg(enc['skill'] if isinstance(enc, dict) else enc)

  def _mgr_input(self, feat, mgr_skill):
    """Manager-policy input: ``feat2tensor(feat)`` plus optional pre-edit conditioning.

    ``mgr_skill['goal_code']`` MUST be the PRE-edit running code Z the manager
    conditions on at this step (the carry *before* ``_advance_mgr_skill`` overwrites
    it). Channels are appended in the fixed order goalcode, decgoal, achieve and are
    all stop-gradient'd, so this adds no gradient path into ``goal_enc``/``goal_dec``
    -- only ``manager_pol``'s own (lazily-sized) first layer consumes them. With every
    flag off this returns exactly ``feat2tensor(feat)`` (bit-identical to before).

    Under ``goal_delta_mode`` OR ``goal_soft_reuse_adapt``, the goalcode channel
    is the SOFT distribution (``mgr_skill['skill_probs']`` -- the combined
    distribution under delta mode, §``_delta_combine_skill``; the manager's own
    plain softmax under soft-reuse, §``_emit_manager``) rather than the collapsed
    one-hot -- a strictly richer signal (the one-hot is a lossy argmax of it)
    that also tells the manager how *contested* the last decision was, not just
    which class won. Only this input-side view changes; under
    ``goal_soft_reuse_adapt`` sampling itself is untouched (direct prediction,
    no recombination)."""
    parts = [self.feat2tensor(feat)]
    if self._mgr_cond_any:
      if self._mgr_needs_skill_probs and 'skill_probs' in mgr_skill:
        code = sg(f32(mgr_skill['skill_probs']))                 # (..., L, C) soft
      else:
        code = sg(self._running_goal_code(mgr_skill))            # (..., L, C) pre-edit Z
      if self.mgr_cond_goalcode:
        parts.append(code.reshape((*code.shape[:-2], -1)))     # (..., L*C)
      if self.mgr_cond_achieve or self.mgr_cond_decgoal:
        bdims = self.feat2deter(feat).ndim - 1                 # 1 rollout, 2 train
        flat = code.reshape((-1, *code.shape[-2:]))
        goal = self.goal_dec(flat, 1).pred()
        prev_goal = sg(goal.reshape((*code.shape[:bdims], goal.shape[-1])))  # (..., D)
        if self.mgr_cond_decgoal:
          parts.append(prev_goal)                              # (..., D)
        if self.mgr_cond_achieve:
          deter = sg(self.feat2deter(feat))                    # (..., D)
          cos = goal_reward_cosine_max(prev_goal, deter)       # (...,)
          parts.append(cos[..., None])                         # (..., 1)
    return jnp.concatenate(parts, -1)

  def _edit_running_goal(self, code_old, emit):
    """Block-overwrite: replace blocks where the manager mask is on with ``z_new``.

    ``Z_next[i] = m[i] ? z_new[i] : Z[i]`` over the ``L`` skill blocks. The mask is
    the manager's *edit selector* (a value of 1 means "change this block"), so the
    running goal accumulates the manager's edits across resamples.

    Under ``mask_joint_edit`` there is no separate mask: the ``skill`` sample is a
    per-block onehot over ``C+1`` classes (class 0 = abstain / keep the block,
    classes 1..C = overwrite with content class c-1), so the edit selector and the
    proposed content come from the same draw."""
    if self.mask_joint_edit:
      code = nn.cast(emit['skill'])                   # (..., L, C+1) onehot
      keep = code[..., :1] > 0.5                       # (..., L, 1) class 0 == abstain
      return jnp.where(keep, code_old, code[..., 1:])  # else content onehot (..., L, C)
    gate = (nn.cast(emit['mask']) > 0.5)[..., None]   # (..., L, 1)
    return jnp.where(gate, emit['skill'], code_old)

  def _select_topk_mask(self, dist, deterministic):
    """Force exactly ``mask_topk`` edited blocks via (Gumbel-)top-k on the mask head.

    Replaces the per-block Bernoulli *sample* with a hard exactly-k selection over
    the ``L`` blocks. Stochastic (Gumbel) selection during rollout keeps REINFORCE
    exploration on *which* blocks to edit; the realized 0/1 mask is reinforced by
    the same Binary ``logp`` as before (no sparsity penalty needed)."""
    inner = dist                                         # unwrap Agg(Binary)
    while not hasattr(inner, 'logit') and hasattr(inner, 'output'):
      inner = inner.output
    logit = f32(inner.logit)                             # (..., L)
    k = max(1, min(int(self.mask_topk), logit.shape[-1]))
    if deterministic:
      scores = logit
    else:
      u = jax.random.uniform(nj.seed(), logit.shape, f32, 1e-6, 1.0)
      scores = logit + (-jnp.log(-jnp.log(u)))           # Gumbel perturbation
    kth = jax.lax.top_k(scores, k)[0][..., -1:]          # k-th largest score
    return (scores >= kth).astype(f32)                   # exactly k ones

  def _select_sparsemax_mask(self, dist, deterministic):
    """C.5: sparse edit gate *by construction* via sparsemax over the L block logits.

    Sparsemax (Martins & Astudillo 2016) projects the logits onto the probability
    simplex, producing *exact zeros* for low-scoring blocks -- so the edit set is sparse
    with no sparsity target/penalty. The simplex sum-to-1 means a *decisive* edit must
    concentrate mass on a few blocks (zeroing the rest): a soft ~1-block edit budget that
    is a structural choice, not a tuned fraction. The realized 0/1 gate is reinforced by
    the same Binary ``logp`` as the Bernoulli mask (no sparsity penalty needed)."""
    logit = self._mask_logit(dist)                       # (..., L)
    if not deterministic:
      u = jax.random.uniform(nj.seed(), logit.shape, f32, 1e-6, 1.0)
      logit = logit + (-jnp.log(-jnp.log(u)))            # Gumbel: explore which blocks
    L = logit.shape[-1]
    z_sorted = jnp.flip(jnp.sort(logit, axis=-1), axis=-1)   # descending
    rng = jnp.arange(1, L + 1, dtype=f32)                    # (L,)
    cssv = jnp.cumsum(z_sorted, axis=-1) - 1.0               # (..., L)
    support = (1.0 + rng * z_sorted) > cssv                  # (..., L)
    k = jnp.sum(support.astype(f32), axis=-1, keepdims=True) # (..., 1), always >= 1
    idx = jnp.maximum(k.astype(jnp.int32) - 1, 0)
    tau = jnp.take_along_axis(cssv, idx, axis=-1) / k        # (..., 1) threshold
    w = jnp.maximum(logit - tau, 0.0)                        # sparse simplex weights
    return (w > 0.0).astype(f32)                             # hard edit gate

  def _mask_logit(self, dist):
    """Unwrap a (possibly Agg-wrapped) Binary mask dist to its per-block logits."""
    inner = dist
    while not hasattr(inner, 'logit') and hasattr(inner, 'output'):
      inner = inner.output
    return f32(inner.logit)                                # (..., L)

  def _mask_prob(self, dist):
    """Differentiable per-block edit probabilities ``sigmoid(logit)`` of the mask
    head. Used by the ``prob`` (B2) sparsity penalty so the gradient reaches the
    mask logits (the hard Bernoulli *sample* carries no gradient)."""
    return jax.nn.sigmoid(self._mask_logit(dist))          # (..., L) in (0, 1)

  def _delta_combine_skill(self, skill_out, prev_mgr_skill):
    """``goal_delta_mode``: add the manager's per-class SIGMOID votes (independent,
    unnormalized, can be driven to all-~0) to the previous goal code's SOFT
    pre-sample distribution before the categorical sample, instead of predicting
    a goal code from scratch.

    The additive base is ``prev_mgr_skill['skill_probs']`` when available, NOT
    the collapsed one-hot: a one-hot is a lossy view of the previous decision (a
    55%-confident pick and a 99%-confident pick both collapse to an identical
    onehot once sampled), and basing "how much to trust the standing choice" on
    that lossy view would silently max out trust at 1.0 regardless of how
    contested the decision actually was. Using the real pre-sample probabilities
    means a block's "stickiness" scales with how confidently it was last chosen:
    a decisively-won class (prob ~1.0) still caps a single fresh vote at a tie
    (``goal_delta_clip``'s ceiling can't exceed sigmoid's own [0,1] range), but a
    contested class (e.g. 0.55) can be outright overtaken by one strong vote on
    the alternative (0.45 + ~1.0 clearly beats 0.55 + ~0) -- a genuinely
    decisive switch, not just a coin flip. Falls back to the one-hot
    (``_running_goal_code``) when no ``skill_probs`` exists yet (a synthetic
    seed dict, e.g. the current-state encoding at episode start / report-time
    proposals -- those aren't a manager decision at all, so treating them as a
    confident commitment is the right default).

    All-~0 votes for a block -> the sum is (numerically) just the previous
    distribution, unchanged -> the same class stays most likely: a clean,
    explicit, DIFFERENTIABLE "reuse" action distinct from a stop-gradiented
    post-hoc argmax==argmax comparison (F18's root cause -- there was no
    gradient-carrying way for the manager to express "don't change this
    block"). Actively re-voting the same class still shifts its margin over the
    alternatives and so still carries gradient, so silent reuse and active
    re-confirmation stay distinguishable to any downstream loss reading the
    pre-sample probabilities.

    Preserves whatever ``outs.Agg`` wrapping (block-sum reduction for
    entropy/kl/logp) the original head output had. Returns ``(dist, probs)`` --
    ``probs`` is the pre-sample soft distribution, threaded into the carry as
    ``skill_probs`` so the manager's NEXT input (and NEXT combination, per
    above) can condition on it directly (see ``_mgr_input``) instead of only
    the collapsed one-hot."""
    is_agg = isinstance(skill_out, outs.Agg)
    inner = skill_out
    while not hasattr(inner, 'dist') and hasattr(inner, 'output'):
      inner = inner.output
    logits = f32(inner.dist.logits)                            # (..., L, C) or (..., C)
    if 'skill_probs' in prev_mgr_skill:
      prev_code = sg(f32(prev_mgr_skill['skill_probs']))        # soft, confidence-aware
    else:
      prev_code = sg(f32(self._running_goal_code(prev_mgr_skill)))  # onehot fallback
    votes = jnp.clip(jax.nn.sigmoid(logits), 0.0, self.goal_delta_clip)
    # ``prev_code`` sums to 1.0/block whether it's a onehot or a genuine
    # distribution (both normalized), so this is still a valid combination base.
    combined = prev_code + votes
    # Normalize to a valid categorical distribution, then floor + renormalize so the
    # subsequent ``log`` never sees an exact 0 (sigmoid saturates to 0.0 in float32
    # for very negative logits) -- the "clipping so it doesn't derail" safeguard.
    probs = combined / jnp.clip(combined.sum(-1, keepdims=True), 1e-6, None)
    probs = jnp.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum(-1, keepdims=True)
    new_inner = outs.OneHot(jnp.log(probs), 0.0)
    # ``Head.onehot`` stamps ``minent``/``maxent`` onto the OneHot instance it
    # builds (used by the adaptive entropy regularizer, ``agent.py``'s
    # ``mgr_actent_adapter`` loop); a freshly-constructed OneHot has neither, which
    # would silently drop the skill head out of that loop. Shape/class-count are
    # unchanged by delta-combination, so the original bounds still apply.
    for attr in ('minent', 'maxent'):
      if hasattr(inner, attr):
        setattr(new_inner, attr, getattr(inner, attr))
    new_dist = (
        outs.Agg(new_inner, len(skill_out.axes), skill_out.agg)
        if is_agg else new_inner)
    return new_dist, probs

  def _emit_manager(
      self, tensor, bdims, deterministic=False, prev_mgr_skill=None):
    """Sample a manager command, applying the hard top-k edit budget to the mask
    and (``goal_delta_mode``) combining the skill logits with the previous goal
    code before sampling. ``prev_mgr_skill`` is any dict ``_running_goal_code``
    accepts (e.g. ``{'goal_code': ...}`` or an ``mgr_skill``-shaped dict); ignored
    unless ``goal_delta_mode`` is on. When active, the sampled result also carries
    ``skill_probs`` (the pre-sample soft distribution) for the next step's input
    conditioning. Under ``goal_soft_reuse_adapt`` (direct prediction, no
    recombination), ``skill_probs`` is instead the manager's own PLAIN softmax --
    exposed the same way, purely as an input/loss signal, with sampling itself
    left untouched (mutually exclusive with ``goal_delta_mode``)."""
    out = mgr_as_dict(self.manager_pol(tensor, bdims))
    skill_probs = None
    if self.goal_delta_mode and prev_mgr_skill is not None and 'skill' in out:
      new_skill, skill_probs = self._delta_combine_skill(out['skill'], prev_mgr_skill)
      out = {**out, 'skill': new_skill}
    elif self.goal_soft_reuse_adapt and 'skill' in out:
      inner = out['skill']
      while not hasattr(inner, 'dist') and hasattr(inner, 'output'):
        inner = inner.output
      skill_probs = jax.nn.softmax(f32(inner.dist.logits), -1)
    pick = mode if deterministic else sample
    if (self.use_masked_goals and 'mask' in out
        and (self.mask_topk > 0 or self.mask_sparsemax)):
      sel = (self._select_topk_mask if self.mask_topk > 0
             else self._select_sparsemax_mask)
      result = {kk: (sel(vv, deterministic) if kk == 'mask' else pick(vv))
                for kk, vv in out.items()}
    else:
      result = pick(out)
    if skill_probs is not None:
      result = {**result, 'skill_probs': skill_probs}
    return result

  def _advance_mgr_skill(self, mgr_skill, emit, update, base_code=None):
    """Switch in the freshly emitted manager skill, threading the running goal.

    For masked goals the carry's ``goal_code`` is block-overwritten by ``emit``
    (optionally re-based on ``base_code``, e.g. the current-state encoding at an
    episode boundary) before the Director window switch; plain goals just switch
    the emitted skill."""
    if self.use_masked_goals:
      old_code = mgr_skill['goal_code'] if base_code is None else base_code
      new_skill = {**emit, 'goal_code': self._edit_running_goal(old_code, emit)}
    else:
      new_skill = emit
    return skill_switch(update, new_skill, mgr_skill)

  def _duration_steps(self, skill):
    """Map the held duration class index to a step count ``p`` in [min, max].

    The manager ``duration`` head is a categorical over ``n_duration_classes``
    classes (index 0..n-1); ``p = goal_duration_min + index``. When
    ``goal_duration_fixed > 0`` the head is bypassed and every decision holds for
    that constant number of steps (control: variable training graph at a fixed K)."""
    idx = skill['duration']
    if self.goal_duration_fixed > 0:
      return jnp.full(jnp.shape(idx), self.goal_duration_fixed, i32)
    return (self.goal_duration_min + idx).astype(i32)

  def _goal_from_skill(self, skills, bdims=1):
    """Decode the running goal code to a ``deter`` goal vector (single batch dim)."""
    return self.goal_dec(self._running_goal_code(skills), bdims).pred()

  def _held_goal_deter(self, mgr_skill, update, reset, cached_goal):
    """Decode Z to deter; refresh on manager switch or episode reset, else hold."""
    fresh = sg(self._goal_from_skill(jax.tree.map(sg, mgr_skill), bdims=1))
    refresh = jnp.logical_or(update, reset)
    rshape = refresh.reshape(refresh.shape + (1,) * (fresh.ndim - refresh.ndim))
    return jnp.where(rshape, fresh, cached_goal)

  def _goals_from_skills(self, skills, bdims=2):
    """Batch-decode running goal codes to goal vectors (Director ``dec.mode()``)."""
    s = self._running_goal_code(skills)
    bshape = s.shape[:bdims]
    flat = s.reshape((-1, *s.shape[bdims:]))
    goals = self.goal_dec(flat, 1).pred()
    return goals.reshape(bshape + goals.shape[1:])

  def _decode_goal_no_decoder_grad(self, code, bdims):
    """Decode ``code`` through ``goal_dec`` with gradient into the DECODER'S
    OWN PARAMETERS blocked, while gradient into ``code`` (hence, via
    straight-through, into whatever produced it -- typically the manager)
    passes through unchanged. Used by ``goal_reuse_weight``/``goal_reuse_adapt``
    so that loss shapes the manager only; the goal autoencoder is trained by
    its own reconstruction/struct losses, not by an incidental
    manager-shaping objective (``MultiOptimizer`` does one combined backward
    pass and routes gradients to each module's own optimizer purely by which
    params the loss's forward computation touched -- there is no other
    isolation, so this has to be explicit).

    Implements ``g(params, x) = f(sg(params), x)``: temporarily swap
    ``goal_dec``'s live parameter values for stop-gradiented copies of the
    SAME values (identical forward computation, only the backward path
    changes), decode, then restore. Verified in isolation (test_freeze_grad.py,
    2026-07-16): parameter gradient exactly 0, input gradient exactly matches
    the unfrozen call's, forward values match exactly. Uses the same
    ``.values``/``.write()`` primitives ``embodied.jax.utils.SlowModel``
    already relies on, for a different purpose (EMA target networks)."""
    live = self.goal_dec.values
    for k, v in live.items():
      self.goal_dec.write(k, sg(v))
    out = self.goal_dec(code, bdims).pred()
    for k, v in live.items():
      self.goal_dec.write(k, v)
    return out

  def _code_diag(self, repfeat):
    """Offline diagnostics on the goal-code structure (report_code_diag).

    Tests whether the running-goal-code overwrite is on a meaningful manifold:
      (1) code volatility -- per-block change rate of deterministic codes along
          real trajectories (consecutive steps; host computes the K-step too);
      (2) on-manifold splice test -- take a real code, overwrite a random subset
          of k blocks with blocks from another real state, decode and re-encode,
          and measure the round-trip block mismatch + goal drift as k grows. A
          clean per-block code round-trips (mismatch~0); off-manifold splices do
          not. Returns raw arrays; the driver computes the summary tables host-side."""
    L = int(self.skill_shape[0])
    C = int(self.skill_shape[-1])
    deter = sg(self.feat2deter(repfeat))                  # (B, T, D)
    B, T, D = deter.shape

    def code_onehot(d, bdims):
      enc = self.goal_enc(sg(d), bdims)
      dist = enc['skill'] if isinstance(enc, dict) else enc
      return dist.pred()                                  # (..., L, C) one-hot
    def code_ids(d, bdims):
      return jnp.argmax(code_onehot(d, bdims), -1).astype(i32)

    # (1) deterministic code ids along real trajectories.
    real_ids = code_ids(deter, 2)                         # (B, T, L)

    # (2) splice test on the flattened pool of real codes.
    flat_oh = code_onehot(deter, 2).reshape((B * T, L, C))
    flat_ids = jnp.argmax(flat_oh, -1)                    # (N, L)
    partner = jnp.roll(flat_oh, 1, axis=0)                # a *different* real code
    rk = jax.random.uniform(nj.seed(), (B * T, L))
    rank = jnp.argsort(jnp.argsort(rk, -1), -1)           # random 0..L-1 per row

    real_goal = self.goal_dec(flat_oh, 1).pred()          # (N, D)
    real_norm = jnp.linalg.norm(real_goal, axis=-1) + 1e-8
    rt_real = (code_ids(real_goal, 1) != flat_ids).astype(f32).mean()

    mism, drift = [], []
    for k in range(1, L + 1):
      gate = (rank < k)[..., None]                        # overwrite k random blocks
      spliced = jnp.where(gate, partner, flat_oh)
      goal = self.goal_dec(spliced, 1).pred()             # (N, D)
      rt = code_ids(goal, 1)
      mism.append((rt != jnp.argmax(spliced, -1)).astype(f32).mean())
      drift.append((jnp.linalg.norm(goal - real_goal, axis=-1) / real_norm).mean())

    return {
        'codediag/real_ids': real_ids,                    # (B, T, L)
        'codediag/rt_mismatch_real': rt_real,             # scalar baseline
        'codediag/rt_mismatch_by_k': jnp.stack(mism),     # (L,)
        'codediag/goal_drift_by_k': jnp.stack(drift),     # (L,)
        'codediag/manager_freq': jnp.asarray(
            int(getattr(self, 'manager_sample_freq', 1)), i32),
    }

  def _manager_skill_step(self, feat, mgr_skill, mgr_step, reset):
    """Resample skill every ``manager_sample_freq`` steps (Director carry switch).

    Masked goals: the manager emits skill+mask; the running goal code is re-based
    on the current-state encoding at episode start (so the *first* command edits
    the current state) and block-overwritten by the emitted edit."""
    # Pull carry-only fields out before any switch tree-map (``emit``/``new_skill``
    # don't carry them, so they must not reach ``skill_switch``).
    sticky = mgr_skill.get('last_edit_mask') if self.use_masked_goals else None
    sticky_change = mgr_skill.get('last_change_mask')
    cached_goal = mgr_skill.get('goal_deter')
    # Carry-only render caches (held decoded-goal images) must not reach the switch
    # tree-maps either; pull them out and thread them through unchanged. ``policy``
    # refreshes them on a switch.
    img_cache = {k: v for k, v in mgr_skill.items() if k.startswith('goal_img_')}
    strip = {'goal_deter', 'last_change_mask', *img_cache}
    if sticky is not None:
      strip.add('last_edit_mask')
    # Pre-switch running code: the "previous goal" reference for the
    # implicit-sparsity CHANGED overlay below, captured before ``skill_switch``/
    # ``_advance_mgr_skill`` overwrite it. Needed for masked AND plain goals alike.
    prev_code = self._running_goal_code(mgr_skill)
    mgr_skill = {k: v for k, v in mgr_skill.items() if k not in strip}
    K = max(1, int(self.manager_sample_freq))
    mgr_step = jnp.where(reset, 0, mgr_step)
    if self.variable_goal_length:
      # ``mgr_step`` holds the steps remaining on the current goal; switch when it
      # runs out (and always on the first step after a reset, where it is 0).
      update = mgr_step <= 0
    else:
      update = jnp.equal(mgr_step % K, 0)
    if self.use_masked_goals:
      init_code = self._encode_goal_code(self.feat2deter(feat), 1)
      r = reset.reshape(reset.shape + (1,) * (init_code.ndim - reset.ndim))
      base_code = jnp.where(r, init_code, prev_code)
      # Condition the manager on the PRE-edit running code it is about to edit.
      emit = self._emit_manager(
          self._mgr_input(feat, {'goal_code': base_code}), 1,
          prev_mgr_skill={'goal_code': base_code})
      mgr_skill = self._advance_mgr_skill(mgr_skill, emit, update, base_code)
      prev_code = base_code   # reset-aware pre-edit code is the correct reference
    else:
      # Plain Director: optionally condition the manager on the PREVIOUS emitted
      # skill (held in ``mgr_skill`` before the switch) so it can deliberately
      # re-emit blocks; falls back to bare feat when no conditioning is enabled.
      mgr_inp = (self._mgr_input(feat, mgr_skill) if self._mgr_cond_any
                 else self.feat2tensor(feat))
      emit = self._emit_manager(mgr_inp, 1, prev_mgr_skill=mgr_skill)
      mgr_skill = skill_switch(update, emit, mgr_skill)
    # Hold the decoded deter across non-switch steps (Z is already held in carry).
    goal = self._held_goal_deter(
        mgr_skill, update, reset,
        cached_goal if cached_goal is not None else jnp.zeros(
            (mgr_skill['skill'].shape[0], int(self.goal_shape[0])), f32))
    mgr_skill['goal_deter'] = goal
    mgr_skill.update(img_cache)  # held frames; policy() refreshes them on switch
    if self.variable_goal_length:
      # On a switch, reload the countdown with the freshly emitted duration p;
      # otherwise tick the held goal down by one.
      p = self._duration_steps(mgr_skill)
      mgr_step = jnp.where(update, p, mgr_step) - 1
    else:
      mgr_step = mgr_step + 1
    if sticky is not None:
      # Update the sticky mask_viz mask: replace with the current goal's edit mask
      # only when it actually edits something (mask has any 1s), so an empty/held
      # goal leaves the previous yellow in place; clear at the episode boundary.
      # Joint mode has no ``mask`` head -> the per-block edit indicator is "did the
      # block NOT abstain" (class 0 of the widened skill sample).
      if self.mask_joint_edit:
        m = f32(mgr_skill['skill'][..., 0] < 0.5)   # (B, L) non-abstain
      else:
        m = mgr_skill['mask']
      has_edit = (m.sum(-1, keepdims=True) > 0)
      rre = reset.reshape(reset.shape + (1,) * (m.ndim - reset.ndim))
      sticky = jnp.where(rre, jnp.zeros_like(m), jnp.where(has_edit, m, sticky))
      mgr_skill['last_edit_mask'] = sticky
    # Implicit-sparsity overlay: mark blocks whose class actually CHANGED from the
    # previous goal (argmax(new code) != argmax(pre-switch code)), independent of
    # any edit/abstain selector -- regenerating (or, under plain Director,
    # re-emitting) a block with the same value is not highlighted. Tracked for
    # masked AND plain-Director goals alike: a plain manager can reproduce a
    # block's previous class too (baseline, or steered there by
    # ``mgr_cond_goalcode``/the implicit-sparsity controller), and that reuse is
    # exactly what this overlay exists to surface as "white" (unchanged) blocks.
    # Refresh only on a manager switch (held otherwise, like the goal itself);
    # cleared at the episode boundary.
    changed = f32(jnp.argmax(self._running_goal_code(mgr_skill), -1)
                  != jnp.argmax(prev_code, -1))          # (B, L)
    rr = reset.reshape(reset.shape + (1,) * (changed.ndim - reset.ndim))
    u = update.reshape(update.shape + (1,) * (changed.ndim - update.ndim))
    sticky_change = jnp.where(
        rr, jnp.zeros_like(changed),
        jnp.where(u, changed, sticky_change))
    mgr_skill['last_change_mask'] = sticky_change
    # ``refresh`` marks steps where the goal changed (manager switch or reset); the
    # policy goal-image cache re-renders only on these steps and holds otherwise.
    refresh = jnp.logical_or(update, reset)
    return mgr_skill, goal, mgr_step, refresh

  def _wkr_goal_reward(self, goals, imgfeat):
    """Worker goal reward: ``cosine_max`` between ``goal`` and ``deter`` (Director)."""
    feat = sg(self.feat2deter(imgfeat))
    goal = sg(goals)
    cos = goal_reward_cosine_max(goal, feat)
    return imag_reward_pad(cos[:, 1:])

  def _mgr_extr_rew(self, rew, con, without_zeros=False):
    """Manager extrinsic reward: WM reward pooled over ``manager_sample_freq`` steps."""
    return aggregate_mgr_extr_rew(
        rew, con, self.manager_sample_freq, without_zeros,
        agg_mode=self.mgr_reward_agg)

  def _mgr_cont(self, con, without_zeros=False):
    """Manager continuation: WM continuation pooled over ``manager_sample_freq`` steps."""
    return aggregate_mgr_cont(con, self.manager_sample_freq, without_zeros)

  def _feat_from_goal(self, goal):
    """RSSM feat for image decode from a proposed ``deter`` goal vector."""
    goal = nn.cast(goal)
    logit = self.dyn._prior(goal)
    stoch = nn.cast(self.dyn._dist(logit).pred())
    return dict(deter=goal, stoch=stoch, logit=logit)

  def _propose_goal(self, feat, impl):
    """Propose a goal vector from ``start`` state (Director ``propose_goal``).

    For masked goals the ``manager`` proposal edits the current-state encoding by
    the emitted block mask; ``prior``/``replay`` proposals decode their skill code
    directly (they carry no edit mask)."""
    B = feat['deter'].shape[0]
    if impl == 'manager':
      if self.use_masked_goals:
        base = self._encode_goal_code(self.feat2deter(feat), 1)
        emit = self._emit_manager(
            self._mgr_input(feat, {'goal_code': base}), 1,
            prev_mgr_skill={'goal_code': base})
        code = self._edit_running_goal(base, emit)
      elif self._mgr_cond_any:
        # Plain Director + conditioning: seed the manager's "previous code" channel
        # with the current-state encoding so the input width matches every other
        # manager_pol call site (the lazily-sized first layer is shared). Under
        # ``goal_delta_mode`` this also seeds the additive reference, so a fresh
        # proposal here reuses/edits relative to the current state, not a zero code.
        base = self._encode_goal_code(self.feat2deter(feat), 1)
        emit = self._emit_manager(
            self._mgr_input(feat, {'goal_code': base}), 1,
            prev_mgr_skill={'goal_code': base})
        code = emit['skill'] if isinstance(emit, dict) else emit
      else:
        emit = self._emit_manager(self.feat2tensor(feat), 1)
        code = emit['skill'] if isinstance(emit, dict) else emit
      return sg(self.goal_dec(code, 1).pred())
    if impl == 'prior':
      logits = jnp.zeros((B,) + tuple(int(x) for x in self.skill_shape), f32)
      prior = outs.OneHot(logits, self._skill_prior_unimix)
      code = prior.sample(nj.seed())
      return sg(self.goal_dec(code, 1).pred())
    if impl == 'replay':
      deter = self.feat2deter(feat)
      perm = jax.random.permutation(nj.seed(), jnp.arange(B))
      target = deter[perm]
      skill = sample(self.goal_enc(target, 1))
      code = skill['skill'] if isinstance(skill, dict) else skill
      return sg(self.goal_dec(code, 1).pred())
    raise NotImplementedError(impl)

  def _worker_policy_fixed_goal(self, goal):
    goal = sg(goal)
    return lambda feat: sample(
        self.pol(self._feat_goal2tensor(feat, goal), bdims=1))

  def _decode_images_uint8(self, dec_carry, feat, reset):
    _, _, recons = self.dec(dec_carry, feat, reset, training=False)
    return {
        k: jnp.clip(recons[k].pred() * 255, 0, 255).astype(jnp.uint8)
        for k in self.dec.imgkeys}

  def _report_impl_videos(self, repfeat, prevact, dec_carry, impl, RB, T):
    """Director ``report_worker``: [initial | proposed goal | worker rollout]."""
    metrics = {}
    if not self.dec.imgkeys:
      return metrics
    horizon = int(getattr(self.config, 'worker_report_horizon', 32))
    t0 = min(4, max(T - 1, 0))
    horizon = min(horizon, max(T - t0 - 1, 1))
    length = 1 + horizon

    start_feat = jax.tree.map(lambda x: x[:, t0], repfeat)
    goal = self._propose_goal(start_feat, impl)
    dyn_start = dict(
        deter=nn.cast(start_feat['deter']),
        stoch=nn.cast(start_feat['stoch']))
    _, imgfeat, _ = self.dyn.imagine(
        dyn_start, self._worker_policy_fixed_goal(goal), horizon, training=False)

    reset1 = jnp.zeros((RB, 1), bool)
    reset_len = jnp.zeros((RB, length), bool)
    start_dec = self._decode_images_uint8(dec_carry, start_feat, reset1)
    goal_dec = self._decode_images_uint8(
        dec_carry, self._feat_from_goal(goal), reset1)
    roll_feat = concat([
        jax.tree.map(lambda x: x[:, None], start_feat), imgfeat], 1)
    roll_dec = self._decode_images_uint8(dec_carry, roll_feat, reset_len)

    def tile_rb(u8, length):
      if u8.ndim == 5:
        return jnp.repeat(u8, length, axis=1)
      return jnp.repeat(u8[:, None], length, axis=1)

    for key in self.dec.imgkeys:
      init_u8 = tile_rb(start_dec[key], length)
      targ_u8 = tile_rb(goal_dec[key], length)
      roll_u8 = roll_dec[key]
      if roll_u8.ndim == 4:
        roll_u8 = jnp.repeat(roll_u8[:, None], length, axis=1)
      video = jnp.concatenate([init_u8, targ_u8, roll_u8], axis=3)
      metrics[f'impl_{impl}/{key}'] = self._video(video)
    return metrics

  def _mgr_expl_reward(self, imgfeat):
    """Manager exploration reward: ``elbo_reward`` with ``adver_impl=squared``.

    Uses per-step goal VAE recon ``((dec.mode() - feat)^2).mean(-1)``; our
    encoder/decoder do not take a separate ``context`` input like Director.
    Returns dense rewards of shape (B, T).
    """
    deter = sg(self.feat2deter(imgfeat))
    encoded = self.goal_enc(deter, 2)
    skill = sample(encoded)
    s = skill['skill'] if isinstance(skill, dict) else skill
    pred = self.goal_dec(s, 2).pred()
    sq = ((pred - deter) ** 2).mean(-1)
    return sq

  def _imagine_with_manager(self, starts, H, training):
    """Imagine with the manager skill resampled at the temporal-abstraction boundary.

    Fixed K: resample every ``manager_sample_freq`` steps. Variable goal length:
    resample when the per-element countdown (carry ``remaining``) hits 0, reloading
    it with the freshly emitted duration ``p``. Also returns ``switch_mask``
    (B, H+1) marking the steps where a manager decision was made (aligned with
    ``img_skills``); under fixed K this is the regular every-K boundary."""
    K = max(1, int(self.manager_sample_freq))
    B = jax.tree.leaves(starts)[0].shape[0]
    skill_shape = tuple(int(x) for x in self.skill_shape)
    mgr_shape = self.joint_skill_shape if self.mask_joint_edit else skill_shape
    # Start with a dummy skill that will be replaced in first step
    mgr_skill = {'skill': jnp.zeros((B, *mgr_shape), f32)}
    if self._mgr_needs_skill_probs:
      mgr_skill['skill_probs'] = jnp.zeros((B, *mgr_shape), f32)
    seed_code = None
    if self.use_masked_goals:
      if not self.mask_joint_edit:
        mgr_skill['mask'] = jnp.zeros((B, skill_shape[0]), f32)
      # Seed the running goal code with the start-state encoding so the first
      # manager command (step 0) edits the current state, not a zero code.
      seed_code = self._encode_goal_code(starts['deter'], 1)
      mgr_skill['goal_code'] = seed_code
    elif self._mgr_cond_any:
      # Plain Director + conditioning: the body's step-0 emit conditions on the
      # initial (zero) skill; return it as the seed so the train re-derivation
      # reconstructs the same step-0 pre-edit conditioning code (no leakage of the
      # skill the manager is about to emit).
      seed_code = mgr_skill['skill']
    if self.variable_goal_length:
      mgr_skill['duration'] = jnp.zeros((B,), i32)

    def body(carry, _):
      dyn_carry, mgr_skill, step_i, remaining = carry
      feat = dict(deter=dyn_carry['deter'], stoch=dyn_carry['stoch'])
      if self.variable_goal_length:
        update = remaining <= 0
      else:
        update = jnp.equal(step_i % K, 0)
      # The carry's goal_code is the PRE-edit running Z for this step.
      emit = self._emit_manager(
          self._mgr_input(feat, mgr_skill), 1, prev_mgr_skill=mgr_skill)
      mgr_skill = self._advance_mgr_skill(mgr_skill, emit, update)
      if self.variable_goal_length:
        p = self._duration_steps(mgr_skill)
        # Countdown = steps left on the current goal INCLUDING this one (= the
        # fresh duration p on a switch step); remaining carries countdown - 1.
        cd_steps = jnp.where(update, p, remaining)
        remaining = cd_steps - 1
      else:
        cd_steps = jnp.broadcast_to(K - (step_i % K), (B,))
      # Match skill to state: decode goal from mgr_skill *after* resampling.
      goal = sg(self._goal_from_skill(jax.tree.map(sg, mgr_skill), bdims=1))
      act = sample(self.pol(self._feat_goal2tensor(
          feat, goal, countdown=self._countdown_norm(cd_steps)), 1))
      dyn_carry, (feat_next, act_out) = self.dyn.imagine(
          dyn_carry, act, 1, training, single=True)
      # Fixed K: ``update`` is a scalar (shared step counter); broadcast to (B,) so
      # the scan stacks a batched switch flag. Variable: ``update`` is already (B,).
      switch = jnp.broadcast_to(f32(update), (B,))
      return ((dyn_carry, mgr_skill, step_i + 1, remaining),
              (feat_next, act_out, mgr_skill, switch, i32(cd_steps)))

    if H < 1:
      raise ValueError(f'imagination length must be >= 1, got {H}')
    unroll = H if self.dyn.unroll else 1
    # ``nj.scan(..., axis=1)`` requires ``xs`` with rank >= 2 (it swapaxes 0/1).
    # Match ``rssm.imagine``: empty ``xs``, explicit ``length``, step in carry.
    init_remaining = jnp.zeros((B,), i32)
    ((last_dyn, last_mgr_skill, _, last_remaining),
     (imgfeat, imgact, img_skills, img_switch, img_cd)) = nj.scan(
        body, (starts, mgr_skill, jnp.int32(0), init_remaining), (), H,
        unroll=unroll, axis=1)

    # imgfeat is [s1...sH]. imgact is [a0...aH-1]. img_skills is [skill0...skillH-1].
    # We need to sample one more skill at last_dyn (sH) to align with imgfeat prefix starts (s0).
    feat_last = dict(deter=last_dyn['deter'], stoch=last_dyn['stoch'])
    if self.variable_goal_length:
      update_last = last_remaining <= 0
    else:
      update_last = jnp.equal(H % K, 0)
    emit_last = self._emit_manager(
        self._mgr_input(feat_last, last_mgr_skill), 1,
        prev_mgr_skill=last_mgr_skill)
    last_mgr_skill = self._advance_mgr_skill(last_mgr_skill, emit_last, update_last)

    img_skills = concat([img_skills, jax.tree.map(lambda x: x[:, None], last_mgr_skill)], 1)
    # switch_mask aligns with img_skills: [switch0...switchH] over H+1 manager steps.
    last_switch = jnp.broadcast_to(f32(update_last), (B,))[:, None]
    switch_mask = jnp.concatenate([img_switch, last_switch], 1)
    # Countdown at position H mirrors the body: fresh duration on a switch, else
    # the carried remaining (which equals countdown_{H-1} - 1).
    if self.variable_goal_length:
      cd_last = jnp.where(
          update_last, self._duration_steps(last_mgr_skill), last_remaining)
    else:
      cd_last = jnp.broadcast_to(K - (H % K), (B,))
    countdowns = jnp.concatenate([img_cd, i32(cd_last)[:, None]], 1)
    # ``seed_code`` is the exact step-0 pre-edit code; the train re-derivation uses it
    # to reconstruct per-step pre-edit codes for the manager-policy conditioning input.
    return imgfeat, imgact, img_skills, seed_code, switch_mask, countdowns

  def _manager_skills_on_sequence(self, repfeat, downsample=False, deterministic=False):
    """K-step manager skills along a ``(B, T)`` feature sequence (replay tail).

    Args:
      repfeat: Feature dict with shape keys, including 'deter': (B, T, ...).
      downsample: If True, only return the skills at every K'th step (i.e., one per K steps),
        instead of repeating them for every timestep.
      deterministic: If True, take the manager's mode (argmax skill) instead of sampling.
        Used by the replay value path so the worker target is not fit against random goals.

    Returns:
      If downsample=False (default): Pytree of (B, T, ...) manager skills; skill held for K timesteps.
      If downsample=True:    Pytree of (B, T//K [+ 1 if T % K !=0], ...) manager skills,
        only at resample steps: timesteps t where t % K == 0.
    """
    K = max(1, int(self.manager_sample_freq))
    T = repfeat['deter'].shape[1]
    feat0 = jax.tree.map(lambda x: x[:, 0], repfeat)
    if self.use_masked_goals:
      # Seed running goal code with the first state's encoding; body's step-0
      # edit (update True) then applies the first manager command on top of it.
      seed_code = self._encode_goal_code(self.feat2deter(feat0), 1)
      mgr_skill = self._emit_manager(
          self._mgr_input(feat0, {'goal_code': seed_code}), 1,
          deterministic=deterministic, prev_mgr_skill={'goal_code': seed_code})
      mgr_skill['goal_code'] = seed_code
    elif self._mgr_cond_any:
      # Plain Director + conditioning: seed the manager's "previous code" channel
      # with the start-state encoding so the first manager_pol call has the same
      # input width as the body (which conditions on the held previous skill).
      # Under ``goal_delta_mode`` this also seeds the additive reference.
      seed_code = self._encode_goal_code(self.feat2deter(feat0), 1)
      mgr_skill = self._emit_manager(
          self._mgr_input(feat0, {'goal_code': seed_code}), 1,
          deterministic=deterministic, prev_mgr_skill={'goal_code': seed_code})
    else:
      mgr_skill = self._emit_manager(
          self.feat2tensor(feat0), 1, deterministic=deterministic)

    B = repfeat['deter'].shape[0]
    init_remaining = jnp.zeros((B,), i32)

    def body(carry, t):
      mgr_skill, remaining = carry
      feat = jax.tree.map(lambda x: x[:, t], repfeat)
      if self.variable_goal_length:
        update = remaining <= 0
      else:
        update = jnp.equal(t % K, 0)
      emit = self._emit_manager(
          self._mgr_input(feat, mgr_skill), 1, deterministic=deterministic,
          prev_mgr_skill=mgr_skill)
      mgr_skill = self._advance_mgr_skill(mgr_skill, emit, update)
      if self.variable_goal_length:
        p = self._duration_steps(mgr_skill)
        cd_steps = jnp.where(update, p, remaining)
        remaining = cd_steps - 1
      else:
        cd_steps = jnp.broadcast_to(K - (t % K), remaining.shape)
      # ``countdown`` is output-only (steps left incl. current); it must never
      # enter the carry skill dict, which other code tree-maps over.
      return (mgr_skill, remaining), {**mgr_skill, 'countdown': i32(cd_steps)}

    if T <= 1:
      _, skill = body((mgr_skill, init_remaining), 0)
      skills = jax.tree.map(lambda x: x[:, None], skill)
    else:
      _, skills = nj.scan(
          body, (mgr_skill, init_remaining), jnp.arange(T), axis=0)

      def orient_time(s):
        if s.shape[0] == B and s.shape[1] == T:
          return s
        if s.shape[0] == T and s.shape[1] == B:
          return jnp.swapaxes(s, 0, 1)
        return s

      skills = jax.tree.map(orient_time, skills)

    if downsample:
      # Only keep the skills at every K'th timestep (t where t % K == 0)
      skills = jax.tree.map(lambda s: s[:, ::K], skills)
    return skills

  def _switch_mask_from_skills(self, skills):
    """Decision mask ``(B, T)`` consistent with a ``_manager_skills_on_sequence`` trace."""
    T = skills['skill'].shape[1]
    B = skills['skill'].shape[0]
    if self.variable_goal_length:
      p = self._duration_steps(skills)  # (B, T); honors goal_duration_fixed

      def body(remaining, pt):
        update = remaining <= 0
        remaining = jnp.where(update, pt, remaining) - 1
        return remaining, f32(update)

      init = jnp.zeros((B,), i32)
      if T <= 1:
        _, switch = body(init, p[:, 0])
        return switch[:, None]
      _, switches = jax.lax.scan(body, init, jnp.swapaxes(p, 0, 1))
      return jnp.swapaxes(switches, 0, 1)

    K = max(1, int(self.manager_sample_freq))
    pos = jnp.arange(T)
    return jnp.broadcast_to(f32(jnp.equal(pos % K, 0)), (B, T))

  def policy(self, carry, obs, mode='train'):
    """One env step: encode obs, RSSM observe, sample policy action, update carry."""
    if self.use_hrl:
      (enc_carry, dyn_carry, dec_carry, prevact, mgr_skill, mgr_step) = carry
    else:
      (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)

    if self.use_hrl:
      # mask_viz applies to any HRL run with an image decoder: the running/emitted
      # code is shown with blocks CHANGED vs the previous goal in yellow and blocks
      # left unchanged (implicit sparsity, or explicit abstain under masked goals)
      # in white/grayscale -- tracked via the sticky ``last_change_mask`` for both
      # masked and plain-Director goals alike (see ``_manager_skill_step``).
      mask_viz_on = (bool(self.dec.imgkeys) and
                     bool(getattr(self.config, 'report_mask_viz', True)))
      mgr_skill, goal, mgr_step, goal_refresh = self._manager_skill_step(
          feat, mgr_skill, mgr_step, reset)
      # Countdown for THIS step from the post-step counter: var-K decrements
      # (returned mgr_step = countdown - 1); fixed-K counts up (returned
      # mgr_step = position + 1).
      K = max(1, int(self.manager_sample_freq))
      if self.variable_goal_length:
        act_cd = mgr_step + 1
      else:
        act_cd = K - ((mgr_step - 1) % K)
      policy = self.pol(self._feat_goal2tensor(
          feat, goal, countdown=self._countdown_norm(act_cd)), bdims=1)
    else:
      policy = self.pol(self.feat2tensor(feat), bdims=1)
    act = sample(policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    # Episode policy_image_with_goal: stack obs image with decoded goal image vertically.
    # Stored under log/ prefix so replay filters it out (avoids doubling replay memory).
    # Enabled by default only when image decoder keys exist; adds one decoder forward pass.
    if (self.use_hrl and self.dec.imgkeys and
        bool(getattr(self.config, 'policy_goal_image', True))):
      goal_feat = self._feat_from_goal(jax.lax.stop_gradient(goal))
      _, _, goal_recons = self.dec({}, goal_feat, reset, training=False)
      # Episode mask_viz panel (logged as epstats/policy_mask_viz_{k}): a 3-row
      # composite per step accumulated over the whole episode, each row upscaled
      # 10x for legibility. Build the yellow-overlaid running-code heatmap once,
      # then reuse per imgkey.
      if mask_viz_on:
        code = self._running_goal_code(mgr_skill).astype(f32)   # (B, L, C) = Z_t
        lo = code.min((-2, -1), keepdims=True)
        hi = code.max((-2, -1), keepdims=True)
        gcode = (code - lo) / (hi - lo + 1e-8)                  # (B, L, C) in [0,1]
        # CHANGED blocks render yellow (keep R,G = intensity, zero blue); UNCHANGED
        # blocks stay grayscale/white (R=G=B). ``last_change_mask`` marks the blocks
        # that actually differ from the PREVIOUS goal (implicit sparsity), not an
        # edit/abstain selector: a block re-emitted with the same value is not
        # highlighted, whether that reuse came from an explicit abstain (masked
        # goals) or the manager simply redrawing the same class (plain Director).
        # Held between switches, cleared at episode start.
        edit = f32(mgr_skill['last_change_mask'])                 # (B, L)
        keepblue = (1.0 - edit)[:, :, None]                     # (B, L, 1)
        code_rgb = jnp.stack([gcode, gcode, gcode * keepblue], -1)  # (B, L, C, 3)
        code_u8 = (code_rgb * 255).astype(jnp.uint8)
      for k in self.dec.imgkeys:
        if k not in obs:
          continue
        obs_u8 = obs[k]  # (B, H, W, C) uint8
        fresh_u8 = jnp.clip(goal_recons[k].pred() * 255, 0, 255).astype(jnp.uint8)
        # Hold the rendered goal frame between manager switches. The deter goal is
        # already held, but the image decoder / dynamics prior keep training during
        # online rollout, so a fresh decode of the same goal drifts every step.
        # Freeze the pixels until the next switch (refresh) so the panel is stable.
        ckey = f'goal_img_{k}'
        if ckey in mgr_skill:
          r = goal_refresh.reshape(
              goal_refresh.shape + (1,) * (fresh_u8.ndim - goal_refresh.ndim))
          goal_u8 = jnp.where(r, fresh_u8, mgr_skill[ckey])
          mgr_skill[ckey] = goal_u8
        else:
          goal_u8 = fresh_u8
        # Stack observation (top) and decoded goal (bottom) for episode composite.
        out[f'log/{k}_with_goal'] = jnp.concatenate([obs_u8, goal_u8], axis=1)
        if mask_viz_on:
          to3 = lambda x: jnp.repeat(x, 3, -1) if x.shape[-1] == 1 else x[..., :3]
          Hi, Wi = int(obs_u8.shape[1]), int(obs_u8.shape[2])
          # Upscale every row (nearest-neighbor) by report_mask_viz_scale. The panel
          # is stacked into an episode gif, so size grows ~scale^2; the default 4
          # keeps it legible without overwhelming the WandB media sync (scale=10 made
          # ~21 MB gifs that never synced online).
          s = max(1, int(getattr(self.config, 'report_mask_viz_scale', 4)))
          up = lambda x: _resize_frames(x[:, None], s * Hi, s * Wi)[:, 0]
          # 3 rows: obs | running code Z (edits in yellow) | active goal.
          out[f'log/mask_viz_{k}'] = jnp.concatenate(
              [up(to3(obs_u8)), up(code_u8), up(to3(goal_u8))], axis=1)
    if self.use_hrl:
      carry = (enc_carry, dyn_carry, dec_carry, act, mgr_skill, mgr_step)
    else:
      carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    """Optimizer step on ``loss``; may attach replay context writes for next batch."""
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    enc, dyn, dec, prevact, mgr_skill, mgr_step = self._unpack_carry(carry)
    metrics, ((enc, dyn, dec), entries, outs, mets) = self.opt(
        self.loss, (enc, dyn, dec), obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    if self.use_hrl:
      self.mgr_extr_slowval.update()
      self.mgr_expl_slowval.update()
      self.wkr_goal_slowval.update()
    else:
      self.slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    carry = self._pack_carry(
        enc, dyn, dec,
        {k: data[k][:, -1] for k in self.act_space},
        mgr_skill, mgr_step)
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    """Full objective: world-model ELBO + imagined actor-critic (+ optional replay value)."""
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # --- World model (sequence ELBO): enc -> dyn -> dec, rew, con ---
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    # Optional stop-gradient on features feeding reward head (stabilize WM vs AC).
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    # Continue target: 1 until terminal; optional finite-horizon downweighting.
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    if not self.use_hrl:
      # ---- Flat AC path (v4-online): single ``pol``/``val`` over WM features. ----
      shapes_bt = {k: v.shape for k, v in losses.items()}
      assert all(x == (B, T) for x in shapes_bt.values()), ((B, T), shapes_bt)
      K_imag = min(self.config.imag_last or T, T)
      K_repl = K_imag
      H = self.config.imag_length
      starts = self.dyn.starts(dyn_entries, dyn_carry, K_imag)
      policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
      _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
      first = jax.tree.map(
          lambda x: x[:, -K_imag:].reshape((B * K_imag, 1, *x.shape[2:])), repfeat)
      imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat, skip=self.config.ac_grads)], 1)
      lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
      lastact = jax.tree.map(lambda x: x[:, None], lastact)
      imgact = concat([imgprevact, lastact], 1)
      inp = self.feat2tensor(imgfeat)
      los_flat, imgloss_out, mets = imag_loss(
          imgact,
          self.rew(inp, 2).pred(),
          self.con(inp, 2).prob(1),
          self.pol(inp, 2),
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.retnorm, self.valnorm, self.advnorm,
          update=training,
          contdisc=self.config.contdisc,
          horizon=self.config.horizon,
          **self.config.imag_loss)
      losses.update({k: v.mean(1).reshape((B, K_imag)) for k, v in los_flat.items()})
      metrics.update(mets)
      if self.config.repval_loss:
        feat = sg(repfeat, skip=self.config.repval_grad)
        last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
        boot = imgloss_out['ret'][:, 0].reshape(B, K_imag)
        feat, last, term, rew, boot = jax.tree.map(
            lambda x: x[:, -K_repl:], (feat, last, term, rew, boot))
        inp = self.feat2tensor(feat)
        los_rep, _, mets = repl_loss(
            last, term, rew, boot,
            self.val(inp, 2),
            self.slowval(inp, 2),
            self.valnorm,
            update=training,
            horizon=self.config.horizon,
            value_head='val',
            **self.config.repl_loss)
        # ``repl_loss(value_head='val')`` emits key ``repval_value`` — keep that key,
        # but rename to ``repval`` so it matches the existing loss-scale entry.
        losses['repval'] = los_rep['repval_value']
        metrics.update({f'reploss/{k}': v for k, v in mets.items()})
      assert set(losses.keys()) == set(self.scales.keys()), (
          sorted(losses.keys()), sorted(self.scales.keys()))
      metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
      loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])
      carry = (enc_carry, dyn_carry, dec_carry)
      entries = (enc_entries, dyn_entries, dec_entries)
      aux_outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
      return loss, (carry, entries, aux_outs, metrics)

    # --- Goal Autoencoder ---
    deter_feat = sg(self.feat2deter(repfeat))
    encoded_goal = self.goal_enc(deter_feat, 2)
    skill = sample(encoded_goal)
    skill_code = skill['skill'] if isinstance(skill, dict) else skill
    # Reconstruction + KL vs uniform skill prior (Director: ``rec + kl_divergence(enc, prior)``).
    # ``MSEDist('sum')`` over ``deter``; masked goals reuse this standard autoencoder
    # (the mask edits the code, not the decoder).
    decoded_goal = self.goal_dec(skill_code, 2)
    goal_rec_sum = decoded_goal.loss(sg(deter_feat))
    goal_rec_agg = self.config.goal_rec_loss_agg
    if goal_rec_agg == 'sum':
      goal_rec_loss = goal_rec_sum
    elif goal_rec_agg == 'mean':
      # Per-dim mean rescaled by a fixed reference dim so the rec:kl balance is
      # invariant to ``deter``; ref = current ``deter`` reproduces the sum exactly.
      deter_dim = self.goal_shape[0]
      goal_rec_loss = goal_rec_sum / deter_dim * float(self.config.goal_rec_dim_ref)
    else:
      raise NotImplementedError(goal_rec_agg)
    goal_dist = _head_inner(encoded_goal)
    skill_prior = outs.OneHot(
        jnp.zeros_like(goal_dist.dist.logits), self._skill_prior_unimix)
    inner_kl = goal_dist.kl(skill_prior)
    # OneHot.kl on [..., L, C] logits sums classes -> [..., L]; reduce remaining L dims -> [B, T].
    goal_kl_bt = inner_kl
    while goal_kl_bt.ndim > 2:
      goal_kl_bt = goal_kl_bt.sum(-1)
    if self.config.goal_kl:
      # Director ``encdec_kl`` AutoAdapt: scales total summed KL toward a target.
      goal_kl_loss, goal_kl_mets = self.goal_kl_adapter(goal_kl_bt, update=training)
    else:
      goal_kl_loss = jnp.zeros((B, T), f32)
      goal_kl_mets = {}

    losses['goal_autoencoder'] = goal_rec_loss + self.config.goal_autoencoder_beta * goal_kl_loss

    # --- Geometry preservation (goal_struct_weight): make code-space distances
    # mirror deter-space distances. Pairs of states far apart in deter (low
    # cosine_max) should map to far-apart codes, and nearby states to nearby
    # codes, so the running-goal overwrite moves the goal proportionally and
    # predictably. The target Gram matrix is stop-gradient; the gradient flows
    # only into the encoder through the soft code probabilities. Variants:
    #   goal_struct_target: 'deter' (default) | 'feat' (match the full deter+stoch
    #     tensor the worker conditions on, not just deter).
    #   goal_struct_loss:   'mse' (default squared Gram match) | 'margin'
    #     (contrastive: pull similar-state codes together, push dissimilar-state
    #     codes apart past a margin -- stronger separation of far-apart pairs).
    #   goal_struct_adapt:  AutoAdapt the weight toward a struct-loss setpoint,
    #     dual-ascent (grows above, shrinks below) unless _one_sided=True. ---
    struct_on = (self.goal_struct_weight > 0.0) or bool(
        getattr(self.config, 'goal_struct_adapt', False))
    if struct_on:
      if getattr(self.config, 'goal_struct_target', 'deter') == 'feat':
        d = sg(self.feat2tensor(repfeat)).reshape((B * T, -1))  # full deter+stoch
      else:
        d = sg(deter_feat).reshape((B * T, -1))                 # (N, D) target geom
      probs = jax.nn.softmax(goal_dist.dist.logits, -1)       # (B, T, L, C)
      z = probs.reshape((B * T, -1))                          # (N, L*C) soft code
      sd = pairwise_cosmax(d)                                 # fixed target geometry
      sz = pairwise_cosmax(z)                                 # differentiable code geometry
      offdiag = 1.0 - jnp.eye(B * T, dtype=f32)               # ignore self-similarity
      denom = offdiag.sum() + 1e-12
      if getattr(self.config, 'goal_struct_loss', 'mse') == 'margin':
        # Contrastive: pull positive pairs (high sd) toward sd, push negative
        # pairs (low sd) apart -- penalize code similarity above the margin,
        # weighted by how dissimilar the states are (1 - sd emphasizes negatives).
        margin = float(getattr(self.config, 'goal_struct_margin', 0.1))
        pos_w = jnp.clip(sd, 0.0, 1.0)
        neg_w = jnp.clip(1.0 - sd, 0.0, 1.0)
        pull = pos_w * jnp.square(sd - sz)
        push = neg_w * jnp.square(jnp.maximum(sz - margin, 0.0))
        struct_err = pull + push
      else:
        struct_err = jnp.square(sz - sd)
      struct_loss = (struct_err * offdiag).sum() / denom       # scalar
      if getattr(self.config, 'goal_struct_adapt', False):
        struct_scaled, struct_mets = self.goal_struct_adapter(
            struct_loss, update=training)
        losses['goal_autoencoder'] = losses['goal_autoencoder'] + struct_scaled
        metrics['goal/struct_scale_mean'] = struct_mets['scale_mean']
      else:
        losses['goal_autoencoder'] = (
            losses['goal_autoencoder'] + self.goal_struct_weight * struct_loss)
      # Diagnostic: Pearson correlation of the two geometries (should rise to ~1).
      sdf = (sd * offdiag).reshape((-1,))
      szf = (sz * offdiag).reshape((-1,))
      sdc = sdf - sdf.mean()
      szc = szf - szf.mean()
      corr = (sdc * szc).sum() / (
          jnp.sqrt((sdc * sdc).sum()) * jnp.sqrt((szc * szc).sum()) + 1e-12)
      metrics['goal/struct_loss'] = struct_loss
      metrics['goal/struct_corr'] = corr
    # Logged as ``train/goal/*`` when the train loop aggregates with prefix ``train``.
    ent = encoded_goal.entropy()
    goal_ent_bt = ent
    while goal_ent_bt.ndim > 2:
      goal_ent_bt = goal_ent_bt.sum(-1)
    metrics.update({
        'goal/rec_mean': goal_rec_loss.mean(),
        'goal/rec_std': goal_rec_loss.std(),
        'goal/kl_mean': goal_kl_loss.mean(),
        'goal/kl_raw_mean': goal_kl_bt.mean(),
        'goal/kl_std': goal_kl_loss.std(),
        'goal/entropy_mean': goal_ent_bt.mean(),
        'goal/entropy_std': goal_ent_bt.std(),
    })
    metrics.update({f'goal/kl_adapt_{k}': v for k, v in goal_kl_mets.items()})

    shapes_bt = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes_bt.values()), ((B, T), shapes_bt)

    # --- Imagination length K_imag vs replay value window K_repl ---
    # B,T = batch and time from replay. ``imag_last`` upper-bounds how many start
    # states we slice from the end of the sequence for imagination (standard
    # DreamerV3 multi-branch imagination).
    K_imag = min(self.config.imag_last or T, T)
    K_repl = K_imag
    H = self.config.imag_length  # imagined steps after the start state (H+1 states).
    starts = self.dyn.starts(dyn_entries, dyn_carry, K_imag)
    imgfeat, imgprevact, img_skills, mgr_seed_code, switch_mask, img_countdowns = (
        self._imagine_with_manager(starts, H, training))
    # Prefix replay states to imagined chain so AC sees grounded first step.
    first = jax.tree.map(
        lambda x: x[:, -K_imag:].reshape((B * K_imag, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat, skip=self.config.ac_grads)], 1)
    mgr_skills = img_skills
    last_feat = jax.tree.map(lambda x: x[:, -1], imgfeat)
    last_mgr_skill = jax.tree.map(lambda x: x[:, -1], mgr_skills)
    last_goal = sg(self._goal_from_skill(jax.tree.map(sg, last_mgr_skill), bdims=1))
    lastact = sample(self.pol(
        self._feat_goal2tensor(
            last_feat, last_goal,
            countdown=self._countdown_norm(img_countdowns[:, -1])), 1))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K_imag, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K_imag, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    con = self.con(inp, 2).prob(1)
    # Detach manager-produced goals from worker actor/critic.
    goals = sg(self._goals_from_skills(jax.tree.map(sg, mgr_skills), bdims=2))
    # --- Implicit (effective) sparsity, logged for EVERY HRL run (masked, joint,
    # AND plain Director). Measures how much of the goal is KEPT from the previous
    # goal, independent of any edit/abstain mask -- a block regenerated with the
    # same value counts as unchanged (this is exactly what block masking was a
    # proxy for). Two views:
    #   * goal/implicit_sparsity_block -- fraction of goal-code blocks whose class
    #     is identical to the previous goal (1 = regenerated identical / fully
    #     sparse, 0 = every block changed).
    #   * goal/implicit_sparsity_cont  -- decoded-goal similarity to the previous
    #     decoded goal (cosine_max in [-1, 1] mapped to [0, 1]; 1 = goal unchanged).
    # Both reduce over manager-DECISION steps only: on held steps the running code
    # is unchanged by construction and would otherwise pin the metric at 1.
    post_code_impl = self._running_goal_code(mgr_skills)          # (M, H+1, [L,] C)
    seed_impl = (mgr_seed_code if mgr_seed_code is not None
                 else post_code_impl[:, 0])
    pre_code_impl = jnp.concatenate(
        [seed_impl[:, None], post_code_impl[:, :-1]], 1)
    kept = f32(jnp.argmax(post_code_impl, -1) == jnp.argmax(pre_code_impl, -1))
    block_kept = kept.mean(-1) if self._skill_factorized else kept    # (M, H+1)
    # Held decodes are identical, so the previous decoded goal is just ``goals``
    # shifted one step (avoids a second goal_dec pass).
    pre_goal_impl = jnp.concatenate([goals[:, :1], goals[:, :-1]], 1)
    cont_sim = 0.5 * (goal_reward_cosine_max(pre_goal_impl, goals) + 1.0)  # (M,H+1)
    if self.variable_goal_length and switch_mask is not None:
      dec_w = f32(switch_mask)                                    # per-step switch
    else:
      K_dec = max(1, int(self.manager_sample_freq))
      dec_w = jnp.zeros(block_kept.shape, f32).at[:, ::K_dec].set(1.0)
    # Exclude the first imagined decision: its "previous goal" is the (unavailable)
    # real goal from before imagination, so comparing to the zero/self seed is not a
    # valid decision-to-decision change. Dropping it keeps this metric consistent
    # with the controller's change_frac (same exclusion below).
    dec_w = dec_w.at[:, 0].set(0.0)
    dec_sum = jnp.maximum(dec_w.sum(), 1.0)
    metrics['goal/implicit_sparsity_block'] = (block_kept * dec_w).sum() / dec_sum
    metrics['goal/implicit_sparsity_cont'] = (cont_sim * dec_w).sum() / dec_sum
    if self.goal_reuse_weight > 0.0 or self.goal_reuse_adapt:
      # Differentiable continuous-goal-space reuse loss (see __init__ comment).
      # ``cont_sim`` above is a pure diagnostic: ``goals`` is stop-gradiented
      # (twice over -- ``sg(self._goals_from_skills(jax.tree.map(sg, ...)))``),
      # by design, so worker-conditioning/logging never leaks gradient into the
      # manager. Here we decode ``post_code_impl`` FRESH, with the decoder's
      # OWN parameters stop-gradiented but not its input (see
      # ``_decode_goal_no_decoder_grad``), so the straight-through gradient
      # the sampled code already carries reaches this loss and shapes the
      # MANAGER only, not the goal autoencoder -- ``pre_goal_impl`` (the fixed
      # target, this decision's PREVIOUS decoded goal) stays exactly the
      # already-stop-gradiented value above, on purpose: we want gradient
      # flowing from "how far did I move" back into the CURRENT decision, not
      # into a revision of the past.
      goal_now = self._decode_goal_no_decoder_grad(post_code_impl, 2)  # (M,H+1,D)
      reuse_sim = goal_reward_cosine_max(pre_goal_impl, goal_now)   # (M,H+1) in [-1,1]
      sim_bt = (reuse_sim * dec_w).sum(1) / jnp.maximum(dec_w.sum(1), 1.0)
      sim_bt = sim_bt.reshape((B, K_imag))               # valid-weighted mean similarity
      if self.goal_reuse_adapt:
        # Dual-ascent Lagrange multiplier holding mean similarity AT the
        # (possibly ratcheted) target (inverse=True: grows the pressure while
        # similarity sits below target, shrinks while above -- see adapter
        # construction). Annealed target (no-op unless goal_reuse_target_init
        # was set apart from goal_reuse_target -- see construction comment).
        reuse_target_now = self.goal_reuse_target_sched(update=training)
        reuse_loss, reuse_mets = self.goal_reuse_adapter(
            sim_bt, update=training, target=reuse_target_now)
        metrics['goal/reuse_target_now'] = reuse_target_now
        metrics.update({f'goal/reuse_adapt_{k}': v for k, v in reuse_mets.items()})
      else:
        # Fixed-weight ablation: pushes toward similarity=1 unconditionally,
        # no target, no adaptive scale. Kept for comparison/completeness.
        reuse_loss = self.goal_reuse_weight * (1.0 - sim_bt)
      losses['goal_reuse'] = reuse_loss
      metrics['goal/reuse_sim_mean'] = (reuse_sim * dec_w).sum() / dec_sum
    # Re-derive the manager policy with the SAME conditioning input the rollout used.
    # ``mgr_skills`` carry the POST-edit code Z_t; the manager conditioned on the
    # PRE-edit code (Z_{t-1}, with step 0 = the imagination seed).
    if self._mgr_cond_any:
      post_code = self._running_goal_code(mgr_skills)              # (M, H+1, L, C)
      seed = mgr_seed_code if mgr_seed_code is not None else post_code[:, 0]
      pre_code = jnp.concatenate([seed[:, None], post_code[:, :-1]], 1)
      if self.goal_soft_reuse_adapt:
        # Same pre-edit shift as ``pre_code`` above, but for the soft
        # distribution: mgr_skills['skill_probs'] is a genuine per-decision
        # stacked field (threaded through the scan carry like goal_code/skill),
        # so no onehot fallback is needed. Seed with zeros, matching the real
        # carry's own zero-init at episode/imagination start (init_policy /
        # _imagine_with_manager) -- decision 0 is excluded from the loss below,
        # so this placeholder only ever matters as a (harmless, "no signal
        # yet") conditioning input, never as a loss target.
        post_probs = mgr_skills['skill_probs']                     # (M, H+1, L, C)
        seed_probs = jnp.zeros_like(post_probs[:, 0])
        pre_probs = jnp.concatenate([seed_probs[:, None], post_probs[:, :-1]], 1)
    rew_step = sg(self.rew(inp, 2).pred())
    expl_step = sg(self._mgr_expl_reward(imgfeat))
    if self.variable_goal_length:
      # Full-resolution manager training: keep every step (no ::K downsample). The
      # ``switch_mask`` masks the REINFORCE policy loss to the per-step manager-
      # decision steps; the critic still trains on every step. Rewards are the raw
      # per-step WM rewards (lambda_return handles continuation via ``term=1-con``,
      # like the flat/worker AC) rather than the fixed-K block-pooled rewards.
      # Optional ``variable_goal_block_rew``: pool rewards over each realized
      # duration segment and downsample to switch steps (Director-style credit).
      if self.variable_goal_block_rew:
        mgr_skills_eff = downsample_at_switch_mask(mgr_skills, switch_mask)
        if self.goal_duration_relabel_truncated:
          mgr_skills_eff = relabel_truncated_last_duration(
              mgr_skills_eff, switch_mask, self.goal_duration_min,
              self.goal_duration_max)
        imgfeat_eff = downsample_at_switch_mask(imgfeat, switch_mask)
        inp_eff = self.feat2tensor(imgfeat_eff)
        n_mgr = imgfeat_eff['deter'].shape[1]
        mgr_extr_rew, mgr_expl_rew, mgr_cont, mgr_switch = (
            variable_block_director_tensors(
                rew_step, con, expl_step, switch_mask, n_mgr,
                agg_mode=self.mgr_reward_agg))
        if self._mgr_cond_any:
          preedit_eff = {'goal_code': downsample_at_switch_mask(
              {'goal_code': pre_code}, switch_mask)['goal_code']}
          if self.goal_soft_reuse_adapt:
            preedit_eff['skill_probs'] = downsample_at_switch_mask(
                {'skill_probs': pre_probs}, switch_mask)['skill_probs']
          mgr_pol_inp = self._mgr_input(imgfeat_eff, preedit_eff)
        else:
          preedit_eff = None
          mgr_pol_inp = inp_eff
      else:
        mgr_skills_eff = mgr_skills
        imgfeat_eff = imgfeat
        inp_eff = inp
        mgr_switch = switch_mask
        mgr_cont = con
        mgr_extr_rew = imag_reward_pad(rew_step[:, 1:])
        mgr_expl_rew = imag_reward_pad(expl_step[:, 1:])
        if self._mgr_cond_any:
          preedit_eff = {'goal_code': pre_code}
          if self.goal_soft_reuse_adapt:
            preedit_eff['skill_probs'] = pre_probs
          mgr_pol_inp = self._mgr_input(imgfeat, preedit_eff)
        else:
          preedit_eff = None
          mgr_pol_inp = inp
      if self.goal_switch_cost:
        # HiTS-style per-decision cost: subtract a fixed cost at each switch step.
        if self.variable_goal_block_rew:
          mgr_extr_rew = mgr_extr_rew - self.goal_switch_cost * mgr_switch
        else:
          mgr_extr_rew = mgr_extr_rew - self.goal_switch_cost * switch_mask
      # Per-decision duration distribution (switch-weighted over decision steps)
      # and realized switch rate. Logging the spread/extremes, not just the mean,
      # surfaces whether the duration head collapses to a delta or stays varied.
      dur = f32(self._duration_steps(mgr_skills))  # honors goal_duration_fixed
      sw = f32(switch_mask)
      wsum = jnp.maximum(sw.sum(), 1.0)
      dur_mean = (dur * sw).sum() / wsum
      dur_ex2 = (dur * dur * sw).sum() / wsum
      metrics['goal/mgr_duration_mean'] = dur_mean
      metrics['goal/mgr_duration_std'] = jnp.sqrt(
          jnp.maximum(dur_ex2 - dur_mean * dur_mean, 0.0))
      # Extremes over decision steps only (held-goal steps masked out).
      metrics['goal/mgr_duration_min'] = jnp.min(
          jnp.where(sw > 0.5, dur, jnp.inf))
      metrics['goal/mgr_duration_max'] = jnp.max(
          jnp.where(sw > 0.5, dur, -jnp.inf))
      # Coarse histogram: fraction of decisions per duration quartile of [1, 16].
      for lab, lo, hi in (('1_4', 0.5, 4.5), ('5_8', 4.5, 8.5),
                          ('9_12', 8.5, 12.5), ('13_16', 12.5, 99.0)):
        in_bin = f32((dur > lo) & (dur <= hi))
        metrics[f'goal/mgr_duration_hist_p{lab}'] = (in_bin * sw).sum() / wsum
      metrics['goal/mgr_switch_rate'] = switch_mask.mean()
    else:
      # Fixed K: downsample the rollout to one entry per K-step manager window and
      # pool rewards/continuation over each window (Director ``abstract_traj``).
      mgr_skills_eff = jax.tree.map(
          lambda s: s[:, ::self.manager_sample_freq], mgr_skills)
      imgfeat_eff = jax.tree.map(
          lambda x: x[:, ::self.manager_sample_freq], imgfeat)
      inp_eff = self.feat2tensor(imgfeat_eff)  # value heads stay on raw feat
      mgr_switch = None
      # Reconstruct PRE-edit code at full resolution then downsample (downsample-
      # then-shift would be off by a full K), so the goalcode channel is never the
      # code the manager just produced (leakage).
      if self._mgr_cond_any:
        preedit_eff = {'goal_code': pre_code[:, ::self.manager_sample_freq]}
        if self.goal_soft_reuse_adapt:
          preedit_eff['skill_probs'] = pre_probs[:, ::self.manager_sample_freq]
        assert preedit_eff['goal_code'].shape[1] == imgfeat_eff['deter'].shape[1]
        mgr_pol_inp = self._mgr_input(imgfeat_eff, preedit_eff)
      else:
        preedit_eff = None
        mgr_pol_inp = inp_eff
      mgr_cont = self._mgr_cont(con, without_zeros=True)
      mgr_extr_rew = imag_reward_pad(
          self._mgr_extr_rew(rew_step, con, without_zeros=True))
      mgr_expl_rew = imag_reward_pad(
          self._mgr_extr_rew(expl_step, con, without_zeros=True))
    mgr_policy = mgr_as_dict(self.manager_pol(mgr_pol_inp, 2))
    if self.goal_delta_mode and preedit_eff is not None and 'skill' in mgr_policy:
      # Re-derive the SAME delta-combined distribution ``_emit_manager`` sampled
      # from during imagination (``preedit_eff`` is the identical pre-edit code
      # used to build ``mgr_pol_inp`` above), so logp/entropy/kl below are taken
      # under the distribution the action actually came from, not the raw
      # pre-combination network output.
      mgr_skill_dist, _ = self._delta_combine_skill(mgr_policy['skill'], preedit_eff)
      mgr_policy = {**mgr_policy, 'skill': mgr_skill_dist}

    # --- Edit sparsity: drive the fraction of edited blocks per command toward
    # ``mask_sparsity_target``. ``mask`` is the emitted edit mask m_t (a 1 means
    # "overwrite this block of the running goal"). The legacy ``sample`` penalty
    # was a no-op (the hard Bernoulli sample has no gradient to the mask logits);
    # ``prob`` (B2) penalizes the differentiable head probabilities and ``reinforce``
    # (C) shapes the manager reward. Placed before the manager AC loss so C can
    # subtract its cost from ``mgr_extr_rew``.
    if self.mask_joint_edit:
      # Single-head manager: realized non-abstain fraction per decision, read from
      # the widened skill sample (class 0 == abstain). ``blk/step`` =
      # mask_frac_mean * L / mgr_duration_mean.
      edit_gate = f32(mgr_skills_eff['skill'][..., 0] < 0.5)   # (M, n, L)
      mask_frac = edit_gate.mean(-1)                            # (M, n)
      if self.variable_goal_length:
        mask_valid = switch_valid_mask(switch_mask, mask_frac.shape[1])
      else:
        mask_valid = jnp.ones_like(mask_frac)
      metrics['goal/mask_frac_mean'] = (
          (mask_frac * mask_valid).sum() / jnp.maximum(mask_valid.sum(), 1.0))
      if self.mask_sparsity_mode != 'none':
        # Ratchet the *edit* rate (non-abstain probability) toward the annealed
        # target via the same rate-Lagrange used by the dual-head 'prob' mode. With
        # mask_sparsity_target_init=1.0 -> target=0.3 the schedule pins P(abstain)
        # from ~0 (full Director-style goal edits early) to ~0.7 (sparse) over the
        # anneal horizon. 'none' = fully free (no penalty), handled above.
        def _slot_mean(x, v):
          return (x * v).sum(1) / jnp.maximum(v.sum(1), 1.0)
        tgt = self.mask_sparsity_target_sched(update=training)
        # OneHot wraps a Categorical (logits live at ``.dist.logits``); unwrap Agg first.
        logits = _head_inner(mgr_policy['skill']).dist.logits      # (M, n, L, C+1)
        edit_prob = (1.0 - jax.nn.softmax(logits, -1)[..., 0]).mean(-1)  # (M, n)
        metrics['goal/mask_prob_mean'] = (
            (edit_prob * mask_valid).sum() / jnp.maximum(mask_valid.sum(), 1.0))
        metric_bt = _slot_mean(edit_prob, mask_valid).reshape((B, K_imag))
        sp_loss, sp_mets = self.mask_sparsity_adapter(
            metric_bt, update=training, target=tgt)
        losses['mask_sparsity'] = sp_loss
        metrics['goal/mask_sparsity_target_now'] = tgt
        metrics.update({f'goal/mask_sparsity_{k}': v for k, v in sp_mets.items()})
    if self.use_masked_goals and not self.mask_joint_edit:
      mask_frac = mgr_skills_eff['mask'].mean(-1)           # (M, n_mgr) realized
      # Under variable-K the packed decision tensors are forward-filled to full
      # width, so the LAST real decision occupies every trailing slot (~75% of
      # columns at K~4, H=32). Weight every mask-sparsity statistic and loss by the
      # valid-slot mask so estimates aren't dominated by the final decision. In
      # fixed-K mode all slots are real and the weights are all-ones.
      if self.variable_goal_length:
        mask_valid = switch_valid_mask(switch_mask, mask_frac.shape[1])
      else:
        mask_valid = jnp.ones_like(mask_frac)
      def _slot_mean(x, v):
        # Valid-weighted mean over the packed decision-slot axis: (M, n) -> (M,).
        return (x * v).sum(1) / jnp.maximum(v.sum(1), 1.0)
      metrics['goal/mask_frac_mean'] = (
          (mask_frac * mask_valid).sum() / jnp.maximum(mask_valid.sum(), 1.0))
      if self.goal_edit_cost:
        # Fixed per-edited-block cost (sparsity analog of ``goal_switch_cost``):
        # subtract a cost proportional to the realized edit fraction from the
        # manager reward, so the manager only edits blocks whose task-return gain
        # beats the cost. A no-op edit (same value) earns no extra worker reward and
        # is strictly dominated. No target fraction -> sparsity is free, priced.
        n = min(mgr_extr_rew.shape[1], mask_frac.shape[1])
        edit_pen = jnp.zeros_like(mgr_extr_rew).at[:, :n].set(
            self.goal_edit_cost * sg(mask_frac[:, :n]))
        mgr_extr_rew = mgr_extr_rew - edit_pen
        metrics['goal/edit_cost_pen_mean'] = edit_pen.mean()
      if self.goal_edit_cost_ach and preedit_eff is not None:
        # B.3 achievability-gated edit cost: scale the per-command edit cost by how
        # reached the *standing* (pre-edit) goal already is, so churning an already-
        # satisfied goal is expensive while re-planning an unmet goal is cheap. The
        # interior edit fraction emerges from the state -- no sparsity target.
        prev_goal_b3 = sg(self._goals_from_skills(preedit_eff, bdims=2))
        ach_b3 = goal_reward_cosine_max(
            prev_goal_b3, sg(self.feat2deter(imgfeat_eff)))     # (M, n) in [-1, 1]
        ach_b3 = jnp.clip(ach_b3, 0.0, 1.0)
        n = min(mgr_extr_rew.shape[1], mask_frac.shape[1], ach_b3.shape[1])
        ach_pen = jnp.zeros_like(mgr_extr_rew).at[:, :n].set(
            self.goal_edit_cost_ach * sg(mask_frac[:, :n]) * sg(ach_b3[:, :n]))
        mgr_extr_rew = mgr_extr_rew - ach_pen
        metrics['goal/edit_cost_ach_pen_mean'] = ach_pen.mean()
        metrics['goal/achievement_mean'] = ach_b3.mean()
      if self.mgr_cond_achieve:
        # Mean of the achievement cosine fed to the manager (how reached the standing
        # goal was). Should rise as the manager learns to hold near-reached goals.
        prev_goal_d = sg(self._goals_from_skills(preedit_eff, bdims=2))
        ach = goal_reward_cosine_max(prev_goal_d, sg(self.feat2deter(imgfeat_eff)))
        metrics['goal/mgr_cond_achieve_mean'] = ach.mean()
      if self.mask_topk <= 0 and not self.mask_sparsemax:
        # Differentiable expected edit fraction from the in-tape manager mask head.
        # (Skipped under sparsemax: sparsity is by construction, no penalty.)
        mask_prob_frac = self._mask_prob(mgr_policy['mask']).mean(-1)  # (M, n_down)
        if self.variable_goal_length:
          prob_valid = switch_valid_mask(switch_mask, mask_prob_frac.shape[1])
        else:
          prob_valid = jnp.ones_like(mask_prob_frac)
        metrics['goal/mask_prob_mean'] = (
            (mask_prob_frac * prob_valid).sum() / jnp.maximum(prob_valid.sum(), 1.0))
        # Annealed sparsity target (no-op constant unless mask_sparsity_target_init
        # was set away from mask_sparsity_target -- see construction comment).
        mask_sparsity_target_now = self.mask_sparsity_target_sched(update=training)
        metrics['goal/mask_sparsity_target_now'] = mask_sparsity_target_now
        if self.mask_sparsity_mode == 'none':
          # Free sparsity: no target, no penalty. The optional ``goal_edit_cost``
          # (priced editing) is the only force shaping the edit fraction; otherwise
          # the manager's task-return REINFORCE alone decides which blocks to edit.
          pass
        elif self.mask_sparsity_mode == 'prob_ach':
          # Tier 1: analytic, state-dependent sparsity shaping. Penalize the *expected*
          # edit fraction sigma(l_i) weighted by how reached the standing goal already
          # is (achievement in [0,1]), with a FIXED weight (no target). Zero-variance
          # twin of B.3's reward cost: reached goal -> editing penalized; unmet -> free.
          if preedit_eff is not None:
            prev_goal_t1 = sg(self._goals_from_skills(preedit_eff, bdims=2))
            ach_t1 = jnp.clip(goal_reward_cosine_max(
                prev_goal_t1, sg(self.feat2deter(imgfeat_eff))), 0.0, 1.0)  # (M, n)
            mp = self._mask_prob(mgr_policy['mask']).mean(-1)               # (M, n) sigma
            n = min(mp.shape[1], ach_t1.shape[1])
            weighted_bt = _slot_mean(
                mp[:, :n] * sg(ach_t1[:, :n]), prob_valid[:, :n]).reshape((B, K_imag))
            losses['mask_sparsity'] = self.mask_sparsity_prob_ach_weight * weighted_bt
            metrics['goal/mask_sparsity_achweighted_mean'] = weighted_bt.mean()
            metrics['goal/achievement_mean'] = ach_t1.mean()
        elif self.mask_sparsity_mode == 'reinforce':
          # C: penalize the realized edit fraction through the manager *reward* so
          # REINFORCE pushes the mask logp toward sparser commands. The adapter is
          # still stepped (to track the target via ``scale``); its loss is dropped.
          _, mask_sp_mets = self.mask_sparsity_adapter(
              _slot_mean(sg(mask_frac), mask_valid).reshape((B, K_imag)),
              update=training, target=mask_sparsity_target_now)
          sp_scale = sg(self.mask_sparsity_adapter.scale())        # Lagrange mult
          # Align the per-command cost to the reward length (imgfeat carries a
          # prepended start state, so the two can differ by one command step).
          n = min(mgr_extr_rew.shape[1], mask_frac.shape[1])
          reward_pen = jnp.zeros_like(mgr_extr_rew).at[:, :n].set(
              sp_scale * sg(mask_frac[:, :n]))
          mgr_extr_rew = mgr_extr_rew - reward_pen
          metrics['goal/mask_sparsity_reward_pen_mean'] = reward_pen.mean()
          metrics.update({f'goal/mask_sparsity_{k}': v for k, v in mask_sp_mets.items()})
        elif self.mask_sparsity_mode == 'entropy':
          # ENTROPY-NATIVE mask (like the action policy). Two dimensionless, adaptive
          # targets drive sparsity instead of a fixed fraction or reward-unit cost:
          #   (a) mask_actent: hold per-block mask entropy (normalized by ln2) near a
          #       fraction-of-max setpoint -> keeps the mask stochastic (anti-collapse);
          #   (b) mask_kl:     hold KL(mask || Bernoulli(mask_sparse_prior)) near a nats
          #       budget -> the transferable "prefer not to edit" pull (task advantage
          #       spends the budget only where editing pays). Which blocks to edit is
          #       decided by the manager REINFORCE gradient through the mask logp.
          probs = self._mask_prob(mgr_policy['mask'])                   # (M, n, L)
          ent = bernoulli_entropy(probs).mean(-1) / jnp.log(2.0)        # (M, n) in [0,1]
          kl = bernoulli_kl(probs, self.mask_sparse_prior).mean(-1)     # (M, n) nats
          ent_bt = _slot_mean(ent, prob_valid).reshape((B, K_imag))
          kl_bt = _slot_mean(kl, prob_valid).reshape((B, K_imag))
          ent_loss, ent_mets = self.mask_actent(ent_bt, update=training)
          losses['mask_sparsity'] = ent_loss
          metrics['goal/mask_entropy_norm_mean'] = (
              (ent * prob_valid).sum() / jnp.maximum(prob_valid.sum(), 1.0))
          metrics['goal/mask_kl_prior_mean'] = (
              (kl * prob_valid).sum() / jnp.maximum(prob_valid.sum(), 1.0))
          metrics.update({f'goal/mask_actent_{k}': v for k, v in ent_mets.items()})
          if self.mask_kl_enable:
            kl_loss, kl_mets = self.mask_kl_adapter(kl_bt, update=training)
            losses['mask_sparsity'] = losses['mask_sparsity'] + kl_loss
            metrics.update({f'goal/mask_kl_{k}': v for k, v in kl_mets.items()})
        elif self.mask_sparsity_mode == 'prob_entropy':
          # Combined: rate-targeting Lagrange (mean edit-fraction -> target, e.g. 0.3)
          # PLUS per-block entropy Lagrange (keeps individual block probabilities away
          # from 0/1 even while the mean sits on target). See construction comment
          # above for why these are complementary rather than redundant.
          metric_bt = _slot_mean(mask_prob_frac, prob_valid).reshape((B, K_imag))
          rate_loss, rate_mets = self.mask_sparsity_adapter(
              metric_bt, update=training, target=mask_sparsity_target_now)
          metrics.update({f'goal/mask_sparsity_{k}': v for k, v in rate_mets.items()})
          probs = self._mask_prob(mgr_policy['mask'])                   # (M, n, L)
          ent = bernoulli_entropy(probs).mean(-1) / jnp.log(2.0)        # (M, n) in [0,1]
          ent_bt = _slot_mean(ent, prob_valid).reshape((B, K_imag))
          ent_loss, ent_mets = self.mask_actent(ent_bt, update=training)
          losses['mask_sparsity'] = rate_loss + ent_loss
          metrics['goal/mask_entropy_norm_mean'] = (
              (ent * prob_valid).sum() / jnp.maximum(prob_valid.sum(), 1.0))
          metrics.update({f'goal/mask_actent_{k}': v for k, v in ent_mets.items()})
          if self.mask_kl_enable:
            kl = bernoulli_kl(probs, self.mask_sparse_prior).mean(-1)   # (M, n) nats
            kl_bt = _slot_mean(kl, prob_valid).reshape((B, K_imag))
            kl_loss, kl_mets = self.mask_kl_adapter(kl_bt, update=training)
            losses['mask_sparsity'] = losses['mask_sparsity'] + kl_loss
            metrics['goal/mask_kl_prior_mean'] = kl.mean()
            metrics.update({f'goal/mask_kl_{k}': v for k, v in kl_mets.items()})
        else:
          # 'prob' (B2): differentiable; 'sample' (legacy): gradient-free no-op.
          if self.mask_sparsity_mode == 'prob':
            metric, metric_valid = mask_prob_frac, prob_valid
          else:
            metric, metric_valid = mask_frac, mask_valid
          metric_bt = _slot_mean(metric, metric_valid).reshape((B, K_imag))
          if self.mask_sparsity_fixed > 0.0:
            # Fixed-multiplier B2 ablation: constant weight, target 0 (drive the
            # edit fraction toward "no goals modified"). No adapter — the manager's
            # task-return REINFORCE is the only force keeping any block edited, so
            # only edits whose return benefit beats the fixed cost survive.
            mask_sparsity_loss = self.mask_sparsity_fixed * metric_bt
            metrics['goal/mask_sparsity_scale_mean'] = jnp.float32(self.mask_sparsity_fixed)
          else:
            mask_sparsity_loss, mask_sp_mets = self.mask_sparsity_adapter(
                metric_bt, update=training, target=mask_sparsity_target_now)
            metrics.update({f'goal/mask_sparsity_{k}': v for k, v in mask_sp_mets.items()})
          losses['mask_sparsity'] = mask_sparsity_loss

    if self.impl_sparsity_mode != 'none':
      # Implicit-sparsity REINFORCE cost (mask-free): per manager decision, measure
      # the CHANGE fraction vs the previous decision's goal code and subtract
      # sg(lambda) * change_frac from the manager reward, so REINFORCE pushes the
      # skill logp toward re-emitting the same blocks. lambda is dual-ascent toward
      # the (annealed) kept-fraction target. Works for plain Director (full-goal
      # regeneration) and masked/joint alike, since it reads the running goal code.
      post_eff = self._running_goal_code(mgr_skills_eff)         # (M, n, [L,] C)
      pre_eff = jnp.concatenate([post_eff[:, :1], post_eff[:, :-1]], 1)
      chg = f32(jnp.argmax(post_eff, -1) != jnp.argmax(pre_eff, -1))
      chg = chg.mean(-1) if self._skill_factorized else chg      # (M, n) change frac
      if (self.variable_goal_length and mgr_switch is not None
          and mgr_switch.shape == chg.shape):
        v = f32(mgr_switch)                                      # valid decisions
      else:
        v = jnp.ones_like(chg)
      # Exclude decision 0: pre_eff[0] is a self-compare (chg==0), which would
      # otherwise dilute the tracked change_frac and mis-set the Lagrangian. Matches
      # the goal/implicit_sparsity_block exclusion so the reported metric and the
      # controller optimise the same quantity.
      v = v.at[:, 0].set(0.0)
      kept_target = self.impl_sparsity_target_sched(update=training)
      chg_bt = ((chg * v).sum(1) / jnp.maximum(v.sum(1), 1.0)).reshape((B, K_imag))
      _, impl_mets = self.impl_sparsity_adapter(
          chg_bt, update=training, target=1.0 - kept_target)
      impl_scale = sg(self.impl_sparsity_adapter.scale())        # Lagrange mult
      n = min(mgr_extr_rew.shape[1], chg.shape[1])
      impl_pen = jnp.zeros_like(mgr_extr_rew).at[:, :n].set(
          impl_scale * sg(chg[:, :n]))
      mgr_extr_rew = mgr_extr_rew - impl_pen
      metrics['goal/impl_sparsity_target_now'] = kept_target
      metrics['goal/impl_sparsity_change_frac_mean'] = (
          (chg * v).sum() / jnp.maximum(v.sum(), 1.0))
      metrics['goal/impl_sparsity_reward_pen_mean'] = impl_pen.mean()
      metrics.update({f'goal/impl_sparsity_{k}': vv for k, vv in impl_mets.items()})

    if self.goal_soft_reuse_adapt:
      # Direct (non-REINFORCE) implicit sparsity: overlap between this
      # decision's own plain softmax and the previous decision's, per block,
      # averaged over blocks -- an ordinary differentiable quantity (no
      # sampling, no decoder pass); backprops straight into both steps' own
      # logits through their softmax. ``mgr_policy['skill']`` here is the RAW
      # (uncombined) manager output -- goal_delta_mode's recombination never
      # runs since the two modes are mutually exclusive (__init__ check).
      cur_inner = mgr_policy['skill']
      while not hasattr(cur_inner, 'dist') and hasattr(cur_inner, 'output'):
        cur_inner = cur_inner.output
      p_cur = jax.nn.softmax(f32(cur_inner.dist.logits), -1)          # (M, n, [L,] C)
      p_prev = sg(preedit_eff['skill_probs'])                        # fixed target: don't revise the past
      overlap = (p_cur * p_prev).sum(-1)                             # (M, n, [L])
      overlap = overlap.mean(-1) if self._skill_factorized else overlap  # (M, n)
      if (self.variable_goal_length and mgr_switch is not None
          and mgr_switch.shape == overlap.shape):
        v = f32(mgr_switch)                                          # valid decisions
      else:
        v = jnp.ones_like(overlap)
      # Exclude decision 0: its "previous" is the zero seed, not a real
      # decision (matches impl_sparsity/implicit_sparsity_block's exclusion).
      v = v.at[:, 0].set(0.0)
      soft_target_now = self.goal_soft_reuse_target_sched(update=training)
      overlap_bt = ((overlap * v).sum(1) / jnp.maximum(v.sum(1), 1.0)).reshape((B, K_imag))
      soft_loss, soft_mets = self.goal_soft_reuse_adapter(
          overlap_bt, update=training, target=soft_target_now)
      losses['goal_soft_reuse'] = soft_loss
      metrics['goal/soft_reuse_target_now'] = soft_target_now
      metrics['goal/soft_reuse_overlap_mean'] = (
          (overlap * v).sum() / jnp.maximum(v.sum(), 1.0))
      metrics.update({f'goal/soft_reuse_{k}': vv for k, vv in soft_mets.items()})

    kwargs_mgr = {**self.config.imag_loss}
    kwargs_mgr.update(
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        mgr_expl_weight=self.mgr_expl_weight,
        actent=self.config.manager_actent,
        slowtar=self.config.manager_slowtar)

    perblock_kwargs = {}
    if self.mask_perblock_credit:
      post_code = self._running_goal_code(mgr_skills_eff)
      pre_code_pb = preedit_eff['goal_code'] if preedit_eff is not None else post_code
      perblock_kwargs.update(
          mask_perblock_credit=True,
          perblock_edit_cost=self.perblock_edit_cost,
          feat_inp=inp_eff,
          goal_code=post_code,
          pre_goal_code=pre_code_pb,
          emit_skill=mgr_skills_eff['skill'],
          mgr_goal_q_module=self.mgr_goal_q,
          mgr_goal_q_slowmodule=self.mgr_goal_q_slowval,
          mgr_goal_q_valnorm=self.mgr_goal_q_valnorm,
      )

    los_mgr, imgloss_mgr_out, mets_mgr = imag_loss_mgr(
        mgr_skills_eff,
        mgr_extr_rew,
        mgr_expl_rew,
        mgr_cont,
        mgr_policy,
        self.mgr_extr_val(inp_eff, 2),
        self.mgr_extr_slowval(inp_eff, 2),
        self.mgr_expl_val(inp_eff, 2),
        self.mgr_expl_slowval(inp_eff, 2),
        self.mgr_extr_retnorm,
        self.mgr_expl_retnorm,
        self.mgr_extr_valnorm,
        self.mgr_expl_valnorm,
        self.mgr_advnorm,
        mgr_actent_adapter=self.mgr_actent,
        mgr_actent_perdim=self.manager_actent_perdim,
        mgr_dur_actent_adapter=(
            self.mgr_dur_actent if self.variable_goal_length else None),
        mgr_dur_reg_adapter=(
            self.mgr_dur_reg_adapter if self.goal_duration_adapt else None),
        mgr_dur_lagrange_adapter=(
            self.mgr_dur_lagrange_adapter if self.goal_duration_lagrange else None),
        switch_mask=mgr_switch,
        dur_reg_weight=float(getattr(self.config, 'goal_duration_reg', 0.0)),
        dur_reg_target=float(getattr(self.config, 'goal_duration_target', 8.0)),
        dur_min=self.goal_duration_min,
        **perblock_kwargs,
        **kwargs_mgr)
    losses.update({k: v.mean(1).reshape((B, K_imag)) for k, v in los_mgr.items()})
    metrics.update(mets_mgr)

    # --- Worker actor-critic (Director ``split_traj``: per-skill-window fixed goal) ---
    K = self.manager_sample_freq
    M = B * K_imag
    kwargs_wkr = {**self.config.imag_loss}
    kwargs_wkr.update(
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon)
    if (self.config.worker_split_traj and not self.variable_goal_length
        and H >= K and H % K == 0):
      # Reshape the rollout into overlapping windows of length K+1. Each window uses
      # the goal decoded at its *start* state for all K+1 steps (incl. the shared
      # boundary), so the worker reward, value, and lambda-return bootstrap stay
      # within a single goal — no leak across goal switches (Director ``split_traj``).
      n_win = H // K
      win_starts = jnp.arange(n_win) * K
      win_idx = win_starts[:, None] + jnp.arange(K + 1)        # (n_win, K+1)
      merge = lambda x: x.reshape((M * n_win,) + x.shape[2:])
      win = lambda x: merge(x[:, win_idx])                     # (M, H+1, ..) -> (M*n_win, K+1, ..)
      win_feat = jax.tree.map(win, imgfeat)
      win_act = jax.tree.map(win, imgact)
      win_con = win(con)
      win_goal = merge(jnp.broadcast_to(
          goals[:, win_starts][:, :, None],
          (M, n_win, K + 1) + goals.shape[2:]))             # window goal held constant
      # Within-window countdown K..0: the boundary state (pos K) keeps the OLD
      # window's goal for bootstrap, so its honest countdown is 0 (time's up).
      win_cd = jnp.broadcast_to(
          K - jnp.arange(K + 1, dtype=i32), (M * n_win, K + 1))
      win_feat_goal = self._feat_goal2tensor(
          win_feat, win_goal, countdown=self._countdown_norm(win_cd))
      win_goal_rew = self._wkr_goal_reward(win_goal, win_feat)
      kwargs_wkr.update(skill_window=0)                         # each window is its own segment
      los_wkr, imgloss_wkr_out, mets_wkr = imag_loss_wkr(
          win_act, win_goal_rew, win_con,
          self.pol(win_feat_goal, 2),
          self.wkr_goal_val(win_feat_goal, 2),
          self.wkr_goal_slowval(win_feat_goal, 2),
          self.wkr_goal_retnorm, self.wkr_goal_valnorm, self.wkr_goal_advnorm,
          **kwargs_wkr)
      losses.update({k: v.mean(1).reshape((B, -1)) for k, v in los_wkr.items()})
      # Repval bootstrap: first window's return at the imagination start, per start state.
      boot_goal_full = imgloss_wkr_out['wkr_goal_ret'].reshape(
          M, n_win, -1)[:, 0, 0].reshape(B, K_imag)
    else:
      # Dense rollout with the lambda-return reset at goal-window boundaries via the
      # ``skill_window`` argument. Variable goal length passes the per-step boundary
      # mask (``switch_mask``); the fixed-K fallback (e.g. H not a multiple of K)
      # passes the integer window length K.
      feat_goal = self._feat_goal2tensor(
          imgfeat, goals, countdown=self._countdown_norm(img_countdowns))
      wkr_goal_rew = self._wkr_goal_reward(goals, imgfeat)
      kwargs_wkr.update(
          skill_window=(switch_mask if self.variable_goal_length else K))
      los_wkr, imgloss_wkr_out, mets_wkr = imag_loss_wkr(
          imgact, wkr_goal_rew, con,
          self.pol(feat_goal, 2),
          self.wkr_goal_val(feat_goal, 2),
          self.wkr_goal_slowval(feat_goal, 2),
          self.wkr_goal_retnorm, self.wkr_goal_valnorm, self.wkr_goal_advnorm,
          **kwargs_wkr)
      losses.update({k: v.mean(1).reshape((B, K_imag)) for k, v in los_wkr.items()})
      boot_goal_full = imgloss_wkr_out['wkr_goal_ret'][:, 0].reshape(B, K_imag)
    metrics.update(mets_wkr)

    # --- Optional replay value loss (tail of real sequence + imag bootstrap) ---
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term = [obs[k] for k in ('is_last', 'is_terminal')]
      boot_extr = imgloss_mgr_out['mgr_extr_ret'][:, 0].reshape(B, K_imag)
      boot_expl = imgloss_mgr_out['mgr_expl_ret'][:, 0].reshape(B, K_imag)
      boot_total = imgloss_mgr_out['ret'][:, 0].reshape(B, K_imag)
      boot_goal = boot_goal_full
      if K_repl != K_imag:
        boot_extr = jnp.broadcast_to(boot_extr[:, -1:], (B, K_repl))
        boot_expl = jnp.broadcast_to(boot_expl[:, -1:], (B, K_repl))
        boot_total = jnp.broadcast_to(boot_total[:, -1:], (B, K_repl))
        boot_goal = jnp.broadcast_to(boot_goal[:, -1:], (B, K_repl))

      # --- 1. Replay sequence for Manager ---
      repl_con_full = self.con(self.feat2tensor(feat), 2).prob(1)
      repl_rew_full = self.rew(self.feat2tensor(feat), 2).pred()
      repl_expl_full = self._mgr_expl_reward(feat)

      if self.variable_goal_length:
        # Mirror the imagination manager path: the default critic trains full-
        # resolution per-step (not fixed-K ``downsample_manager_states``). Only the
        # block-rew variant needs the realized switch boundaries; reconstruct them
        # there. NOTE the switches are *counterfactual* — derived from the current
        # deterministic manager on replay states, not the behavior policy that
        # generated the data — which is acceptable for an on-policy critic target.
        if self.variable_goal_block_rew:
          repl_skills_full = self._manager_skills_on_sequence(
              feat, deterministic=True)
          repl_skills_full.pop('countdown', None)
          repl_switch = self._switch_mask_from_skills(repl_skills_full)
          feat_down = downsample_at_switch_mask(feat, repl_switch)
          inp_down = self.feat2tensor(feat_down)
          n_mgr = feat_down['deter'].shape[1]
          repl_mgr_extr_rew, repl_mgr_expl_rew, repl_mgr_cont, valid_mgr = (
              variable_block_director_tensors(
                  repl_rew_full, repl_con_full, repl_expl_full,
                  repl_switch, n_mgr, agg_mode=self.mgr_reward_agg))
          last_down = downsample_at_switch_mask(
              {'last': last.astype(f32)}, repl_switch)['last'].astype(last.dtype)
          term_down = (1.0 - repl_mgr_cont).astype(term.dtype)
        else:
          feat_down = feat
          inp_down = self.feat2tensor(feat_down)
          repl_mgr_extr_rew = imag_reward_pad(repl_rew_full[:, 1:])
          repl_mgr_expl_rew = imag_reward_pad(repl_expl_full[:, 1:])
          last_down = last
          term_down = term
      else:
        feat_down = downsample_manager_states(feat, self.manager_sample_freq)
        inp_down = self.feat2tensor(feat_down)

        # Downsample flags
        T_feat = jax.tree.leaves(feat)[0].shape[1]
        Tm = T_feat - 1
        n = Tm // self.manager_sample_freq
        rem = Tm - n * self.manager_sample_freq
        idx_down = [0] + [int(1 + i * self.manager_sample_freq - 1) for i in range(1, n + 1)]
        if rem > 0:
          idx_down.append(int(n * self.manager_sample_freq))
          idx_down.append(int(Tm))
        else:
          idx_down.append(int(Tm))
        idx_down = sorted(list(set(idx_down)))

        last_down = last[:, idx_down]
        term_down = term[:, idx_down]
        repl_mgr_extr_rew = imag_reward_pad(self._mgr_extr_rew(
            repl_rew_full, repl_con_full, without_zeros=True))
        repl_mgr_expl_rew = imag_reward_pad(self._mgr_extr_rew(
            repl_expl_full, repl_con_full, without_zeros=True))

      # --- 2. Dense Replay sequence for Worker ---
      feat_wkr, last_wkr, term_wkr, boot_goal_wkr = jax.tree.map(
          lambda x: x[:, -K_repl:],
          (feat, last, term, boot_goal))
      inp_wkr = self.feat2tensor(feat_wkr)
      repl_skills = jax.tree.map(
          lambda x: x[:, -K_repl:],
          self._manager_skills_on_sequence(feat, deterministic=True))
      repl_cd = repl_skills.pop('countdown')
      # Detach manager goals in replay value path.
      repl_goals = sg(self._goals_from_skills(jax.tree.map(sg, repl_skills), bdims=2))
      feat_goal_wkr = self._feat_goal2tensor(
          feat_wkr, repl_goals, countdown=self._countdown_norm(repl_cd))
      repl_wkr_goal_rew = self._wkr_goal_reward(repl_goals, feat_wkr)

      # --- 3. Compute Value Losses ---
      # Manager Replay Value Loss (predicts combined return)
      kwargs_repmgr = {**self.config.repl_loss}
      kwargs_repmgr.update(
          update=training,
          horizon=self.config.horizon,
          value_head='mgr')

      # Manager Trajectory is short
      weight_down = f32(~last_down)
      if self.variable_goal_block_rew:
        weight_down = weight_down * valid_mgr
      disc = 1 - 1 / self.config.horizon
      lam = 0.95 # matches repl_loss default
      boot_extr_down = jnp.broadcast_to(boot_extr[:, -1:], repl_mgr_extr_rew.shape)
      boot_expl_down = jnp.broadcast_to(boot_expl[:, -1:], repl_mgr_expl_rew.shape)
      voff_extr_prev, vscale_extr_prev = self.mgr_extr_valnorm.stats()
      voff_expl_prev, vscale_expl_prev = self.mgr_expl_valnorm.stats()
      tarval_extr = (
          self.mgr_extr_val(inp_down, 2).pred() * vscale_extr_prev + voff_extr_prev)
      tarval_expl = (
          self.mgr_expl_val(inp_down, 2).pred() * vscale_expl_prev + voff_expl_prev)
      ret_extr = lambda_return(
          last_down, term_down, repl_mgr_extr_rew, tarval_extr, boot_extr_down, disc, lam)
      ret_expl = lambda_return(
          last_down, term_down, repl_mgr_expl_rew, tarval_expl, boot_expl_down, disc, lam)

      # Train the critics on RAW returns (``valnorm: none`` -> offset 0, scale 1),
      # matching the imagination critic target and flat-v3 ``repl_loss``. The
      # symexp_twohot head handles the raw return scale internally.
      voff_extr, vscale_extr = self.mgr_extr_valnorm(ret_extr, update=training)
      voff_expl, vscale_expl = self.mgr_expl_valnorm(ret_expl, update=training)

      ret_extr_normed = (ret_extr - voff_extr) / vscale_extr
      ret_extr_padded = jnp.concatenate([ret_extr_normed, jnp.zeros_like(ret_extr_normed[:, -1:])], 1)
      losses['repmgr_extr_value'] = weight_down[:, :-1] * (
          self.mgr_extr_val(inp_down, 2).loss(sg(ret_extr_padded)) +
          1.0 * self.mgr_extr_val(inp_down, 2).loss(sg(self.mgr_extr_slowval(inp_down, 2).pred())))[:, :-1]
      ret_expl_normed = (ret_expl - voff_expl) / vscale_expl
      ret_expl_padded = jnp.concatenate([ret_expl_normed, jnp.zeros_like(ret_expl_normed[:, -1:])], 1)
      losses['repmgr_expl_value'] = weight_down[:, :-1] * (
          self.mgr_expl_val(inp_down, 2).loss(sg(ret_expl_padded)) +
          1.0 * self.mgr_expl_val(inp_down, 2).loss(sg(self.mgr_expl_slowval(inp_down, 2).pred())))[:, :-1]
      metrics.update(prefix({}, 'repmgr')) # TODO add metrics if needed

      # Worker Goal Replay Value Loss
      kwargs_repwkr_goal = {**self.config.repl_loss}
      kwargs_repwkr_goal.update(
          update=training,
          horizon=self.config.horizon,
          value_head='wkr_goal')
      los, reploss_out, mets = repl_loss(
          last_wkr,
          term_wkr,
          repl_wkr_goal_rew,
          jnp.broadcast_to(boot_goal_wkr[:, -1:], repl_wkr_goal_rew.shape),
          self.wkr_goal_val(feat_goal_wkr, 2),
          self.wkr_goal_slowval(feat_goal_wkr, 2),
          self.wkr_goal_valnorm,
          **kwargs_repwkr_goal)
      losses.update(los)
      metrics.update(prefix(mets, 'repwkr_goal'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    aux_outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, aux_outs, metrics)

  def report(self, carry, data):
    """Eval metrics, open-loop video, goal VAE panels, and Director goal-proposal videos."""
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    enc_carry, dyn_carry, dec_carry, _, mgr_skill, mgr_step = self._unpack_carry(
        carry)
    wm_carry = (enc_carry, dyn_carry, dec_carry)
    B, T = obs['is_first'].shape
    RB = min(int(getattr(self.config, 'report_max_rows', 6)), B)
    metrics = {}

    _, (new_carry, entries, outs, mets) = self.loss(
        wm_carry, obs, prevact, training=False)
    metrics.update(mets)
    if self.use_hrl and bool(getattr(self.config, 'report_code_diag', False)):
      metrics.update(self._code_diag(outs['repfeat']))
    rep = jax.tree.map(lambda x: x[:RB, :T], outs['repfeat'])
    reset_s = obs['is_first'][:RB, :T]
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)

    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, c: self.loss(
              c, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, wm_carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop: observe first half, imagine second half (baseline Dreamer report).
    obs_rb = jax.tree.map(lambda x: x[:RB], obs)
    prevact_rb = jax.tree.map(lambda x: x[:RB], prevact)
    tokens_rb = jax.tree.map(lambda x: x[:RB], outs['tokens'])
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:, T // 2:], xs)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(tokens_rb), firsthalf(prevact_rb),
        firsthalf(obs_rb['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact_rb), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs_rb['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs_rb['is_first'])),
        training=False)
    for key in self.dec.imgkeys:
      assert obs_rb[key].dtype == jnp.uint8
      true = obs_rb[key]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)
      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)
      metrics[f'openloop/{key}'] = self._video(video)

    if not self.use_hrl:
      enc_carry_n, dyn_carry_n, dec_carry_n = new_carry
      carry = self._pack_carry(
          enc_carry_n, dyn_carry_n, dec_carry_n,
          {k: data[k][:, -1] for k in self.act_space},
          mgr_skill, mgr_step)
      return carry, metrics

    # Goal VAE on replay features (train-time encoder path).
    deter_feat = sg(self.feat2deter(rep))
    encoded_goal = self.goal_enc(deter_feat, 2)
    skill_s = sample(encoded_goal)
    skill_code_s = skill_s['skill'] if isinstance(skill_s, dict) else skill_s
    pred_deter = nn.cast(self.goal_dec(skill_code_s, 2).pred())
    feat_goal = self._feat_from_goal(pred_deter)
    _, _, recons_goal = self.dec(dec_carry, feat_goal, reset_s, training=False)

    # Manager-proposed goals over the report sequence (K-step skill hold).
    want_mgr_recon = bool(getattr(self.config, 'report_mgr_recon', False))
    want_skill_viz = bool(getattr(self.config, 'report_skill_viz', True))
    want_goal_enc_viz = bool(getattr(self.config, 'report_goal_enc_viz', True))
    need_mgr = (want_mgr_recon or want_skill_viz or
                bool(getattr(self.config, 'report_vec_viz', False)))
    if need_mgr:
      mgr_skills = self._manager_skills_on_sequence(rep)
      mgr_skills.pop('countdown', None)
      mgr_goals = sg(self._goals_from_skills(mgr_skills, bdims=2))
      mgr_goal_feat = self._feat_from_goal(mgr_goals)
      _, _, recons_mgr = self.dec(dec_carry, mgr_goal_feat, reset_s, training=False)
    else:
      mgr_skills = mgr_goals = mgr_goal_feat = recons_mgr = None

    # Optional dense vec→RGB visualisations of latent vectors and skills.
    # Off by default — they are debug-grade and slow down the video pipeline.
    if bool(getattr(self.config, 'report_vec_viz', False)):
      metrics['goal/deter_feat'] = self._video(_vec_to_tb_rgb(deter_feat))
      metrics['goal/decoded_deter'] = self._video(_vec_to_tb_rgb(pred_deter))
      sk = skill_s['skill'] if isinstance(skill_s, dict) else skill_s
      if sk.ndim == 3:
        metrics['goal/skill_sampled'] = self._video(_vec_to_tb_rgb(sk))
      else:
        metrics['goal/skill_sampled'] = self._video(
            jnp.repeat((sk * 255).astype(jnp.uint8)[..., None], 3, axis=-1))
      m_sk = mgr_skills['skill']
      if m_sk.ndim == 3:
        metrics['goal/mgr_skill'] = self._video(_vec_to_tb_rgb(m_sk))
      else:
        metrics['goal/mgr_skill'] = self._video(
            jnp.repeat((m_sk * 255).astype(jnp.uint8)[..., None], 3, axis=-1))
      metrics['goal/mgr_proposed_deter'] = self._video(_vec_to_tb_rgb(mgr_goals))

    # VAE / manager goal reconstruction panels. ``goal/image_{key}`` (true frames
    # only) was a duplicate of the leftmost column here — dropped.
    for key in self.dec.imgkeys:
      true = obs[key][:RB, :T]
      pred_g = jnp.clip(recons_goal[key].pred() * 255, 0, 255).astype(jnp.uint8)
      metrics[f'goal/recon_{key}'] = self._video(
          jnp.concatenate([true, pred_g, ((i32(pred_g) - i32(true) + 255) // 2).astype(np.uint8)], 2))
      if want_mgr_recon and recons_mgr is not None:
        pred_m = jnp.clip(recons_mgr[key].pred() * 255, 0, 255).astype(jnp.uint8)
        metrics[f'goal/mgr_recon_{key}'] = self._video(
            jnp.concatenate([true, pred_m, ((i32(pred_m) - i32(true) + 255) // 2).astype(np.uint8)], 2))

    # 3-row manager skill video: skill heatmap | deter heatmap | decoded goal image.
    # With masked goals the skill row is the *running* goal code Z (the accumulated
    # command after block-overwrite); the decoded goal row is its decode.
    if want_skill_viz and need_mgr and self.dec.imgkeys:
      sk = self._running_goal_code(mgr_skills)
      sk_flat = sk.reshape(*sk.shape[:2], -1)  # (RB, T, L*C)
      skill_rgb = _vec_to_tb_rgb(sk_flat)  # (RB, T, h_sk, w_sk, 3) uint8
      deter_rgb = _vec_to_tb_rgb(deter_feat)  # (RB, T, h_d, w_d, 3) uint8
      for key in self.dec.imgkeys:
        H_img, W_img = int(obs[key].shape[2]), int(obs[key].shape[3])
        skill_row = _resize_frames(skill_rgb, H_img, W_img)   # (RB, T, H, W, 3)
        deter_row = _resize_frames(deter_rgb, H_img, W_img)   # (RB, T, H, W, 3)
        goal_row = jnp.clip(recons_mgr[key].pred() * 255, 0, 255).astype(jnp.uint8)
        C_img = int(obs[key].shape[4])
        if C_img == 1:
          goal_row = goal_row[..., :1]
          skill_row = skill_row[..., :1]
          deter_row = deter_row[..., :1]
        panel = jnp.concatenate([skill_row, deter_row, goal_row], axis=2)
        metrics[f'skill_viz/{key}'] = self._video(panel)

    # 5-row goal encoder video: og image | encoded deter | encoded goal | decoded deter | decoded image.
    if want_goal_enc_viz and self.dec.imgkeys:
      sk_enc = skill_s['skill'] if isinstance(skill_s, dict) else skill_s
      sk_flat_enc = (sk_enc.reshape(*sk_enc.shape[:2], -1)
                     if sk_enc.ndim == 4 else sk_enc)  # (RB, T, L*C) or (RB, T, D)
      enc_deter_rgb = _vec_to_tb_rgb(deter_feat)    # input to goal encoder
      enc_goal_rgb = _vec_to_tb_rgb(sk_flat_enc)    # encoded skill
      dec_deter_rgb = _vec_to_tb_rgb(pred_deter)    # decoded deter
      for key in self.dec.imgkeys:
        H_img, W_img = int(obs[key].shape[2]), int(obs[key].shape[3])
        C_img = int(obs[key].shape[4])
        og_u8 = obs[key][:RB, :T]
        enc_deter_row = _resize_frames(enc_deter_rgb, H_img, W_img)
        enc_goal_row = _resize_frames(enc_goal_rgb, H_img, W_img)
        dec_deter_row = _resize_frames(dec_deter_rgb, H_img, W_img)
        dec_img_row = jnp.clip(recons_goal[key].pred() * 255, 0, 255).astype(jnp.uint8)
        if C_img == 1:
          enc_deter_row = enc_deter_row[..., :1]
          enc_goal_row = enc_goal_row[..., :1]
          dec_deter_row = dec_deter_row[..., :1]
        panel = jnp.concatenate(
            [og_u8, enc_deter_row, enc_goal_row, dec_deter_row, dec_img_row], axis=2)
        metrics[f'goal_enc_viz/{key}'] = self._video(panel)

    # NOTE: the masked / overwrite-goal panel moved to the online policy() path so
    # it spans an entire episode (logged as epstats/policy_mask_viz_{key}), mirroring
    # epstats/policy_image_with_goal. Its 3 rows (each upscaled 10x) are obs | goal
    # code with blocks CHANGED vs the previous goal in yellow, unchanged blocks
    # white/grayscale | active goal image. Tracked via the sticky
    # ``last_change_mask`` for masked AND plain-Director goals alike (a plain
    # manager reproducing a block's previous class shows white there too). See
    # ``mask_viz_on`` in ``policy``. ``report_mask_viz`` gates it.

    # Director-style: [initial | proposed goal | worker rollout] per proposal
    # mode. Disabled by default (lowest-value-per-encoding-cost panel); set
    # ``report_impl_videos_enabled: True`` to render the impls listed in
    # ``report_impl_videos``.
    if bool(getattr(self.config, 'report_impl_videos_enabled', False)):
      impls = tuple(getattr(self.config, 'report_impl_videos', ['manager']))
    else:
      impls = ()
    for impl in impls:
      metrics.update(self._report_impl_videos(
          rep, prevact, dec_carry, impl, RB, T))

    enc_carry, dyn_carry, dec_carry = new_carry
    carry = self._pack_carry(
        enc_carry, dyn_carry, dec_carry,
        {k: data[k][:, -1] for k in self.act_space},
        mgr_skill, mgr_step)
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    """If replay_context: first K steps recompute carries from stored entries; else identity."""
    enc_carry, dyn_carry, dec_carry, prevact, mgr_skill, mgr_step = (
        self._unpack_carry(carry))
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    # prevact[t] aligns with action before obs[t]; prepend stored carry, shift sequence.
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return (
          self._pack_carry(
              enc_carry, dyn_carry, dec_carry, prevact, mgr_skill, mgr_step),
          obs, prevact, stepid)

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    # New trajectory chunk (consec==0): use replay-derived carry/obs; else online path.
    first_chunk = (data['consec'][:, 0] == 0)
    carry_wm, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    enc_carry, dyn_carry, dec_carry = carry_wm
    return (
        self._pack_carry(
            enc_carry, dyn_carry, dec_carry, prevact, mgr_skill, mgr_step),
        obs, prevact, stepid)

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    """Adam-like chain: AGC clip, RMS scale, momentum, optional WD mask, LR schedule."""
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


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


def _head_inner(head):
  """Unwrap ``outs.Agg`` so skill axes are not confused with time."""
  return head.output if isinstance(head, outs.Agg) else head


def _mask_logit_head(head):
  """Unwrap a (possibly Agg-wrapped) Binary mask dist to its per-block logits."""
  inner = _head_inner(head)
  while not hasattr(inner, 'logit') and hasattr(inner, 'output'):
    inner = inner.output
  return f32(inner.logit)


def mask_logp_perblock_time(head, event):
  """Per-block Bernoulli log-prob with leading axes ``(batch, time - 1, L)``."""
  lp = _head_inner(head).logp(sg(event))
  return lp[:, :-1]


def eval_mgr_goal_q(feat_inp, goal_code, mgr_goal_q_module, valnorm_stats):
  """Unnormalized Q_g(s, Z) prediction at each (batch, time) step."""
  flat = goal_code.reshape(*goal_code.shape[:-2], -1)
  inp = jnp.concatenate([feat_inp, flat], -1)
  voff, vscale = valnorm_stats
  return mgr_goal_q_module(inp, 2).pred() * vscale + voff


def mask_perblock_advantages(
    feat_inp, goal_code, pre_goal_code, emit_skill, mask_head,
    mgr_goal_q_module, mgr_goal_q_valnorm, rscale_extr, perblock_edit_cost):
  """Leave-one-out COMA-style per-block advantages for the edit mask."""
  voff, vscale = mgr_goal_q_valnorm.stats()
  q_actual = eval_mgr_goal_q(
      feat_inp, goal_code, mgr_goal_q_module, (voff, vscale))[:, :-1]
  logit = _mask_logit_head(mask_head)[:, :-1]
  L = goal_code.shape[-2]
  advs = []
  for i in range(L):
    code0 = goal_code.at[..., i, :].set(pre_goal_code[..., i, :])
    code1 = goal_code.at[..., i, :].set(emit_skill[..., i, :])
    q0 = eval_mgr_goal_q(feat_inp, code0, mgr_goal_q_module, (voff, vscale))[:, :-1]
    q1 = eval_mgr_goal_q(feat_inp, code1, mgr_goal_q_module, (voff, vscale))[:, :-1]
    pi = jax.nn.sigmoid(logit[..., i])
    baseline = pi * q1 + (1.0 - pi) * q0
    advs.append((q_actual - baseline - f32(perblock_edit_cost)) / rscale_extr)
  return jnp.stack(advs, -1)


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


def imag_loss_wkr(
    act,
    wkr_goal_rew,
    con,
    policy,
    wkr_goal_value,
    wkr_goal_slowvalue,
    wkr_goal_retnorm,
    wkr_goal_valnorm,
    wkr_goal_advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
    skill_window=0,
):
  """Worker actor-critic losses on imagined trajectories.

  Uses the standard DreamerV3 actor-critic normalization: the critic is a
  symexp_twohot head trained on RAW goal-returns (``wkr_goal_valnorm`` is
  ``none``), the advantage is scaled by the percentile return range
  (``wkr_goal_retnorm`` = ``perc``), and ``wkr_goal_advnorm`` is ``none``.
  """
  losses = {}
  metrics = {}

  # Unnormalize critic predictions for bootstrapping and advantage baseline.
  wkr_goal_voffset, wkr_goal_vscale = wkr_goal_valnorm.stats()

  wkr_goal_val = wkr_goal_value.pred() * wkr_goal_vscale + wkr_goal_voffset
  wkr_goal_slowval = wkr_goal_slowvalue.pred() * wkr_goal_vscale + wkr_goal_voffset
  wkr_goal_tarval = wkr_goal_slowval if slowtar else wkr_goal_val

  # Discount per step: either γ or finite-horizon (1 - 1/horizon) when not contdisc.
  disc = 1 if contdisc else 1 - 1 / horizon
  # Discounted continuation weights from predicted continue probs ``con``.
  weight = jnp.cumprod(disc * con, 1) / disc
  # Reset the lambda-return at goal-window boundaries so V(s, g) bootstraps within
  # its own skill window instead of across goal switches (Director ``split_traj``).
  # ``skill_window`` is either an int (fixed-K: boundary every K steps) or a (B, T)
  # boundary mask (variable goal length: 1 wherever a new goal begins). The step-0
  # boundary is dropped (it starts the first window, no reset).
  if hasattr(skill_window, 'ndim'):
    last = f32(skill_window).at[:, 0].set(0.0)
  elif skill_window and skill_window > 1:
    pos = jnp.arange(con.shape[1])
    boundary = (pos % skill_window == 0) & (pos > 0)
    last = jnp.broadcast_to(boundary.astype(f32), con.shape)
  else:
    last = jnp.zeros_like(con)
  term = 1 - con

  wkr_goal_ret = lambda_return(
      last, term, wkr_goal_rew, wkr_goal_tarval, wkr_goal_tarval, disc, lam)

  wkr_goal_roffset, wkr_goal_rscale = wkr_goal_retnorm(wkr_goal_ret, update)

  wkr_goal_adv = (wkr_goal_ret - wkr_goal_tarval[:, :-1]) / wkr_goal_rscale

  wkr_goal_aoffset, wkr_goal_ascale = wkr_goal_advnorm(wkr_goal_adv, update)

  wkr_goal_adv_normed = (wkr_goal_adv - wkr_goal_aoffset) / wkr_goal_ascale

  wkr_logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  wkr_ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}

  w = sg(weight[:, :-1])

  # REINFORCE with the percentile-scaled advantage and a fixed entropy bonus
  # (DreamerV3 actor loss).
  wkr_goal_policy_loss = w * -(
      wkr_logpi * sg(wkr_goal_adv_normed) + actent * sum(wkr_ents.values()))

  losses['wkr_policy'] = wkr_goal_policy_loss

  metrics['wkr_goal_policy_loss'] = wkr_goal_policy_loss.mean()
  metrics['wkr_goal_rew'] = wkr_goal_rew.mean()

  # NLL of value distribution against λ-returns (padded for length match to head API).
  wkr_goal_voffset, wkr_goal_vscale = wkr_goal_valnorm(wkr_goal_ret, update)

  wkr_goal_tar_normed = (wkr_goal_ret - wkr_goal_voffset) / wkr_goal_vscale

  wkr_goal_tar_padded = jnp.concatenate([wkr_goal_tar_normed, 0 * wkr_goal_tar_normed[:, -1:]], 1)

  losses['wkr_goal_value'] = sg(weight[:, :-1]) * (
      wkr_goal_value.loss(sg(wkr_goal_tar_padded)) +
      slowreg * wkr_goal_value.loss(sg(wkr_goal_slowvalue.pred())))[:, :-1]

  wkr_goal_ret_normed = (wkr_goal_ret - wkr_goal_roffset) / wkr_goal_rscale

  metrics['wkr_goal_adv'] = wkr_goal_adv.mean()

  metrics['wkr_goal_adv_std'] = wkr_goal_adv.std()

  metrics['wkr_goal_adv_mag'] = jnp.abs(wkr_goal_adv_normed).mean()

  metrics['wkr_goal_ret'] = wkr_goal_ret_normed.mean()
  metrics['wkr_goal_val'] = wkr_goal_val.mean()
  # Removed: ``wkr_goal_tar`` (== ret_normed), ``wkr_goal_slowval`` (≈ val),
  # ``wkr_weight``/``wkr_con`` (≈ 1 in non-terminal imagined rollouts).

  metrics['wkr_goal_ret_min'] = wkr_goal_ret_normed.min()
  metrics['wkr_goal_ret_max'] = wkr_goal_ret_normed.max()
  metrics['wkr_goal_ret_rate'] = (jnp.abs(wkr_goal_ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'wkr_ent/{k}'] = wkr_ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'wkr_rand/{k}'] = (wkr_ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['wkr_goal_ret'] = wkr_goal_ret
  return losses, outs, metrics

def imag_loss_mgr(
    skills,
    mgr_extr_rew,
    mgr_expl_rew,
    con,
    manager_policy,
    mgr_extr_value,
    mgr_extr_slowvalue,
    mgr_expl_value,
    mgr_expl_slowvalue,
    mgr_extr_retnorm,
    mgr_expl_retnorm,
    mgr_extr_valnorm,
    mgr_expl_valnorm,
    mgr_advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
    mgr_expl_weight=0.1,
    mgr_actent_adapter=None,
    mgr_actent_perdim=True,
    mgr_dur_actent_adapter=None,
    mgr_dur_reg_adapter=None,
    mgr_dur_lagrange_adapter=None,
    switch_mask=None,
    dur_reg_weight=0.0,
    dur_reg_target=0.0,
    dur_min=1,
    mask_perblock_credit=False,
    perblock_edit_cost=0.0,
    feat_inp=None,
    goal_code=None,
    pre_goal_code=None,
    emit_skill=None,
    mgr_goal_q_module=None,
    mgr_goal_q_slowmodule=None,
    mgr_goal_q_value=None,
    mgr_goal_q_slowvalue=None,
    mgr_goal_q_valnorm=None,
):
  """Manager actor-critic losses on imagined trajectories.

  Fixed K (``switch_mask=None``): the trajectory is the downsampled one-entry-per-
  window sequence and every entry is a manager decision. Variable goal length
  (``switch_mask`` given, shape ``(B, T)``): the trajectory is full-resolution and
  only the steps where ``switch_mask==1`` are real manager decisions, so the
  REINFORCE policy loss is masked to those steps (the critic still trains on every
  step, each having a valid return-to-go target).

  Mirrors the flat DreamerV3 actor-critic (``imag_loss``) per critic: each
  critic is a symexp_twohot head trained on RAW λ-returns (``valnorm: none``),
  and the per-critic advantage is scaled by its percentile return range
  (``retnorm: perc``). The extrinsic and exploratory advantages are combined
  with ``mgr_expl_weight`` and left at that scale (``advnorm: none``), so the
  default v3 actor hyperparameters apply directly.
  """
  losses = {}
  metrics = {}

  # Unnormalize the critic preds back to raw return scale. Under ``valnorm: none``
  # this is the identity (offset 0, scale 1), so the symexp_twohot critic's raw
  # prediction is used directly as the λ-return bootstrap (matches flat v3).
  voff_extr_prev, vscale_extr_prev = mgr_extr_valnorm.stats()
  voff_expl_prev, vscale_expl_prev = mgr_expl_valnorm.stats()

  mgr_extr_val = mgr_extr_value.pred() * vscale_extr_prev + voff_extr_prev
  mgr_extr_slowval = mgr_extr_slowvalue.pred() * vscale_extr_prev + voff_extr_prev
  mgr_extr_tarval = mgr_extr_slowval if slowtar else mgr_extr_val

  mgr_expl_val = mgr_expl_value.pred() * vscale_expl_prev + voff_expl_prev
  mgr_expl_slowval = mgr_expl_slowvalue.pred() * vscale_expl_prev + voff_expl_prev
  mgr_expl_tarval = mgr_expl_slowval if slowtar else mgr_expl_val

  # Discount per step: either γ or finite-horizon (1 - 1/horizon) when not contdisc.
  disc = 1 if contdisc else 1 - 1 / horizon
  # Discounted continuation weights from predicted continue probs ``con``.
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con

  # Raw λ-returns from raw rewards + raw tarval bootstraps (the critic bootstraps
  # off its own value, like flat v3 ``imag_loss``).
  mgr_extr_ret = lambda_return(
      last, term, mgr_extr_rew, mgr_extr_tarval, mgr_extr_tarval, disc, lam)
  mgr_expl_ret = lambda_return(
      last, term, mgr_expl_rew, mgr_expl_tarval, mgr_expl_tarval, disc, lam)

  mgr_total_ret = mgr_extr_ret + mgr_expl_weight * mgr_expl_ret

  # Advantage: per critic ``(ret - tarval) / rscale`` with ``rscale`` the
  # percentile return range (retnorm = perc), exactly as flat v3. Combine the two
  # already-scaled advantages with ``mgr_expl_weight`` and leave at that scale.
  roff_extr, rscale_extr = mgr_extr_retnorm(mgr_extr_ret, update)
  roff_expl, rscale_expl = mgr_expl_retnorm(mgr_expl_ret, update)
  mgr_extr_adv = (mgr_extr_ret - mgr_extr_tarval[:, :-1]) / rscale_extr
  mgr_expl_adv = (mgr_expl_ret - mgr_expl_tarval[:, :-1]) / rscale_expl
  mgr_adv = mgr_extr_adv + mgr_expl_weight * mgr_expl_adv
  mgr_aoffset, mgr_ascale = mgr_advnorm(mgr_adv, update)
  mgr_adv_normed = (mgr_adv - mgr_aoffset) / mgr_ascale

  skill_events = align_skill_events(skills, manager_policy)
  mask_logpi = None
  mask_adv_normed = None
  if mask_perblock_credit and 'mask' in manager_policy:
    mgr_logpi = sum([
        head_logp_time(v, skill_events[k])
        for k, v in manager_policy.items() if k != 'mask'])
    mask_adv_blk = mask_perblock_advantages(
        feat_inp, goal_code, pre_goal_code, emit_skill, manager_policy['mask'],
        mgr_goal_q_module, mgr_goal_q_valnorm, rscale_extr, perblock_edit_cost)
    mask_adv_normed = (mask_adv_blk - mgr_aoffset) / mgr_ascale
    mask_logpi = mask_logp_perblock_time(
        manager_policy['mask'], skill_events['mask'])
    metrics['goal/mask_perblock_adv_std'] = mask_adv_blk.std()
    metrics['goal/mask_perblock_adv_spread'] = jnp.abs(mask_adv_blk).mean()
  else:
    mgr_logpi = sum([
        head_logp_time(v, skill_events[k]) for k, v in manager_policy.items()])
  mgr_ents = {k: head_entropy_time(v) for k, v in manager_policy.items()}

  # Director-style adaptive normalized entropy regularizer (per-categorical
  # entropy held near ``manager_actent_target`` of the max via AutoAdapt). When
  # the adapter is absent the policy loss falls back to the fixed v3 ``actent``.
  mgr_ent_loss_bt = jnp.zeros_like(mgr_logpi)
  mgr_actent_mets = {}
  if mgr_actent_adapter is not None:
    ent_loss_terms = []
    for k, head in manager_policy.items():
      inner = _head_inner(head)
      if not hasattr(inner, 'minent') or not hasattr(inner, 'maxent'):
        continue
      # The duration head is a single scalar categorical -> route it to its own
      # scalar adapter (the per-dim ``mgr_actent`` is shaped for the L skill
      # blocks). Skip it if no duration adapter was provided.
      if k == 'duration':
        if mgr_dur_actent_adapter is None:
          continue
        adapter, perdim = mgr_dur_actent_adapter, False
      else:
        adapter, perdim = mgr_actent_adapter, mgr_actent_perdim
      ent_perdim = head_entropy_perdim_time(head)  # (B, T-1, ...)
      L = ent_perdim.shape[-1] if ent_perdim.ndim > 2 else 1
      lo = inner.minent / L
      hi = inner.maxent / L
      denom = jnp.maximum(hi - lo, 1e-8)
      ent_norm = (ent_perdim - lo) / denom
      if perdim and ent_perdim.ndim > 2:
        loss_perdim, mets = adapter(ent_norm, update=update)
        ent_loss_terms.append(loss_perdim.sum(-1))
      else:
        ent_scalar = ent_norm.mean(-1) if ent_norm.ndim > 2 else ent_norm
        loss_scalar, mets = adapter(ent_scalar, update=update)
        ent_loss_terms.append(loss_scalar)
      mgr_actent_mets.update(
          {f'mgr_actent_{k}_{mk}': mv for mk, mv in mets.items()})
      mgr_actent_mets[f'mgr_ent_norm_{k}_mean'] = ent_norm.mean()
    if ent_loss_terms:
      mgr_ent_loss_bt = sum(ent_loss_terms)

  w = sg(weight[:, :-1])
  vw = sg(weight[:, :-1])
  if switch_mask is not None:
    # Variable goal length: only apply the REINFORCE update at the steps where a
    # manager decision was actually made (the duration/skill/mask log-probs and
    # advantage are meaningless on held-goal steps). Also masks critic updates on
    # padded switch timelines (``variable_goal_block_rew``).
    m = sg(switch_mask[:, :-1])
    # Zero masked advantages before weighting: ``0 * inf`` is NaN on padded block-rew
    # slots where retnorm can divide by a near-zero scale.
    mgr_adv_normed = jnp.where(m > 0, mgr_adv_normed, 0.0)
    w = w * m
    vw = vw * m
    metrics['mgr_switch_rate'] = switch_mask.mean()

  # REINFORCE manager actor. With the adaptive entropy adapter the per-dim
  # normalized-entropy loss is already in ``mgr_ent_loss_bt``; otherwise fall
  # back to the fixed v3 ``actent`` on summed per-categorical entropy.
  if mask_logpi is not None:
    mgr_reinforce = (
        mgr_logpi * sg(mgr_adv_normed) +
        (mask_logpi * sg(mask_adv_normed)).sum(-1))
  else:
    mgr_reinforce = mgr_logpi * sg(mgr_adv_normed)
  if mgr_actent_adapter is not None:
    losses['mgr_policy'] = w * (-mgr_reinforce + mgr_ent_loss_bt)
  else:
    losses['mgr_policy'] = w * -(
        mgr_reinforce + actent * sum(mgr_ents.values()))

  # Soft duration prior: pull the manager's expected goal duration toward
  # ``dur_reg_target`` directly through the duration logits (not via REINFORCE),
  # applied at decision steps only (``w`` carries the switch mask). The weight is
  # either fixed (``goal_duration_reg``) or an auto-tuned, capped AutoAdapt
  # multiplier (``mgr_dur_reg_adapter``) -- the cap keeps the prior from swamping
  # the manager REINFORCE objective (the suspected fixed-reg=0.1 collapse mode).
  if (dur_reg_weight > 0.0 or mgr_dur_reg_adapter is not None
      or mgr_dur_lagrange_adapter is not None) and ('duration' in manager_policy):
    dur_inner = _head_inner(manager_policy['duration'])
    dur_probs = jax.nn.softmax(dur_inner.logits, -1)         # (B, T, n_classes)
    classes = jnp.arange(dur_probs.shape[-1], dtype=f32)
    exp_p = dur_min + (dur_probs * classes).sum(-1)          # (B, T) expected steps
    exp_p = policy_time_slice(exp_p)                         # (B, T-1)
    sq_err = jnp.square(exp_p - f32(dur_reg_target))
    if mgr_dur_lagrange_adapter is not None:
      # Mask-style Lagrangian: regulate the switch-weighted mean |E[dur] - target|
      # against a small tolerance (symmetric in deviation direction -- see the
      # adapter construction comment). Folded out into its own loss key (not added
      # into ``mgr_policy``) so it carries an independent scale, same as
      # ``mask_sparsity``.
      mean_abs_err = (
          (w * jnp.abs(exp_p - f32(dur_reg_target))).sum() /
          jnp.maximum(w.sum(), 1.0))
      _, dur_lagrange_mets = mgr_dur_lagrange_adapter(mean_abs_err, update=update)
      dur_reg_bt = mgr_dur_lagrange_adapter.scale() * sq_err
      metrics.update(
          {f'mgr_duration_lagrange_{k}': v for k, v in dur_lagrange_mets.items()})
      losses['goal_duration_prior'] = w * dur_reg_bt
      metrics['mgr_duration_reg_loss'] = (w * dur_reg_bt).mean()
    else:
      if mgr_dur_reg_adapter is not None:
        # Step the Lagrange multiplier on the switch-weighted mean error, then apply
        # the (stop-grad, capped) scale per decision step.
        mean_err = (w * sq_err).sum() / jnp.maximum(w.sum(), 1.0)
        _, dur_adapt_mets = mgr_dur_reg_adapter(mean_err, update=update)
        dur_reg_bt = mgr_dur_reg_adapter.scale() * sq_err
        metrics.update(
            {f'mgr_duration_adapt_{k}': v for k, v in dur_adapt_mets.items()})
      else:
        dur_reg_bt = dur_reg_weight * sq_err
      losses['mgr_policy'] = losses['mgr_policy'] + w * dur_reg_bt
      metrics['mgr_duration_reg_loss'] = (w * dur_reg_bt).mean()
    metrics['mgr_duration_exp_mean'] = exp_p.mean()
    metrics['mgr_duration_exp_std'] = exp_p.std()

  metrics['mgr_policy_loss'] = losses['mgr_policy'].mean()
  metrics['mgr_ent_loss'] = mgr_ent_loss_bt.mean()
  metrics.update(mgr_actent_mets)
  metrics['mgr_extr_rew'] = mgr_extr_rew.mean()
  nz = jnp.maximum((jnp.abs(mgr_extr_rew[:, 1:]) > 0).sum(), 1)
  metrics['mgr_extr_rew_block'] = mgr_extr_rew[:, 1:].sum() / nz
  metrics['mgr_expl_rew'] = mgr_expl_rew.mean()

  # Critic NLL against RAW λ-returns (``valnorm: none`` -> target == raw return).
  # The symexp_twohot head handles the return scale; plus a slow-value regression
  # term (DreamerV3 ``imag_loss``).
  voff_extr, vscale_extr = mgr_extr_valnorm(mgr_extr_ret, update)
  voff_expl, vscale_expl = mgr_expl_valnorm(mgr_expl_ret, update)
  mgr_extr_ret_normed = (mgr_extr_ret - voff_extr) / vscale_extr
  mgr_expl_ret_normed = (mgr_expl_ret - voff_expl) / vscale_expl

  mgr_extr_tar_padded = jnp.concatenate([mgr_extr_ret_normed, 0 * mgr_extr_ret_normed[:, -1:]], 1)
  losses['mgr_extr_value'] = vw * (
      mgr_extr_value.loss(sg(mgr_extr_tar_padded)) +
      slowreg * mgr_extr_value.loss(sg(mgr_extr_slowvalue.pred())))[:, :-1]

  mgr_expl_tar_padded = jnp.concatenate([mgr_expl_ret_normed, 0 * mgr_expl_ret_normed[:, -1:]], 1)
  losses['mgr_expl_value'] = vw * (
      mgr_expl_value.loss(sg(mgr_expl_tar_padded)) +
      slowreg * mgr_expl_value.loss(sg(mgr_expl_slowvalue.pred())))[:, :-1]

  if mask_perblock_credit and mgr_goal_q_module is not None:
    q_inp = jnp.concatenate([
        feat_inp, goal_code.reshape(*goal_code.shape[:-2], -1)], -1)
    mgr_goal_q_value = mgr_goal_q_module(q_inp, 2)
    mgr_goal_q_slowvalue = mgr_goal_q_slowmodule(q_inp, 2)
    voff_q, vscale_q = mgr_goal_q_valnorm(mgr_extr_ret, update)
    mgr_goal_q_ret_normed = (mgr_extr_ret - voff_q) / vscale_q
    mgr_goal_q_tar_padded = jnp.concatenate(
        [mgr_goal_q_ret_normed, 0 * mgr_goal_q_ret_normed[:, -1:]], 1)
    losses['mgr_goal_q_value'] = vw * (
        mgr_goal_q_value.loss(sg(mgr_goal_q_tar_padded)) +
        slowreg * mgr_goal_q_value.loss(
            sg(mgr_goal_q_slowvalue.pred())))[:, :-1]
    q_pred = eval_mgr_goal_q(
        feat_inp, goal_code, mgr_goal_q_module, mgr_goal_q_valnorm.stats())
    metrics['mgr_goal_q_val'] = q_pred.mean()

  metrics['mgr_adv'] = mgr_adv.mean()
  metrics['mgr_adv_std'] = mgr_adv.std()
  metrics['mgr_adv_mag'] = jnp.abs(mgr_adv_normed).mean()
  metrics['mgr_extr_adv'] = mgr_extr_adv.mean()
  metrics['mgr_expl_adv'] = mgr_expl_adv.mean()

  metrics['mgr_total_ret'] = mgr_total_ret.mean()
  metrics['mgr_extr_ret'] = mgr_extr_ret_normed.mean()
  metrics['mgr_expl_ret'] = mgr_expl_ret_normed.mean()
  metrics['mgr_extr_val'] = mgr_extr_val.mean()
  metrics['mgr_expl_val'] = mgr_expl_val.mean()
  # Removed: ``mgr_extr_tar``/``mgr_expl_tar`` (== ret_normed already logged),
  # ``mgr_extr_slowval``/``mgr_expl_slowval`` (≈ ``*_val`` up to EMA lag),
  # ``mgr_con``/``mgr_weight`` (≈ 1 in non-terminal imagined rollouts).

  # Iterate the policy heads (skill/mask), not the carried skills dict — the latter
  # also holds the running goal code ``goal_code``, which has no policy/entropy.
  for k in manager_policy:
    metrics[f'mgr_ent/{k}'] = mgr_ents[k].mean()
    if hasattr(manager_policy[k], 'minent'):
      lo, hi = manager_policy[k].minent, manager_policy[k].maxent
      metrics[f'mgr_rand/{k}'] = (mgr_ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = mgr_total_ret
  outs['mgr_extr_ret'] = mgr_extr_ret
  outs['mgr_expl_ret'] = mgr_expl_ret
  return losses, outs, metrics


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  """Flat DreamerV3 actor-critic loss, used when ``use_hrl=False``."""
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  w = sg(weight[:, :-1])
  policy_loss = w * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)
  outs = {'ret': ret}
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
    value_head='mgr_extr',
):
  """Value loss on real replay tail; ``boot`` is return from imagination at slice boundary.

  ``last`` masks episode boundaries; ``boot`` supplies bootstrap value at the
  window edge. Same λ-return and slow-value mix as imagination, but no policy term.
  """
  losses = {}
  if last.shape[1] < 2:
    losses[f'rep{value_head}_value'] = jnp.zeros_like(f32(last))
    outs = {f'rep{value_head}_ret': jnp.zeros((last.shape[0], 0), f32)}
    return losses, outs, {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)  # Zero loss on steps after episode end (``last``).
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses[f'rep{value_head}_value'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs[f'rep{value_head}_ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  """TD(λ)-style returns along time; ``boot`` is per-step bootstrap (often ``val``).

  Shapes are (batch, time). ``last`` flags last step of trajectory; ``term`` is
  terminal / non-continue. Iteration is backward from the final bootstrap slice.
  """
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
