"""Credit-assignment tensor math for the hierarchical (Director-style) agent.

Pure functions over rollout tensors -- no module state, no ninjax. Two regimes
share this file:

* **Fixed K** -- the manager decides every ``manager_sample_freq`` steps, so the
  block boundaries are a static grid (``aggregate_mgr_*``,
  ``downsample_manager_states``).
* **Variable K** -- the manager also emits a hold length, so boundaries are
  data-dependent and come from a ``switch_mask``. Pooled per-segment tensors are
  packed into a static, forward-filled buffer (``downsample_at_switch_mask``,
  ``variable_block_director_tensors``); ``switch_valid_mask`` /
  ``decision_mean_rescale`` are what keep reductions over that padded axis equal
  to fixed-K's exactly-sized ones.
"""
import jax
import jax.numpy as jnp

f32 = jnp.float32
i32 = jnp.int32
# Stop-gradient helper: optionally pass gradients through (e.g. reward head).
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)


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


def worker_split_window(
    split_traj, variable_goal_length, duration_fixed, manager_sample_freq, H):
  """Window length for Director ``split_traj`` worker credit, or 0 if N/A.

  Director trains the worker on non-overlapping windows of ``K+1`` states, each
  conditioned throughout on the goal decoded at the window's START state -- the
  boundary state keeps the OLD goal so the lambda-return bootstraps inside one
  goal. That requires a statically regular switch pattern.

  ``goal_duration_fixed > 0`` bypasses the duration head and holds every goal
  for exactly that many steps (see ``_duration_steps``), so the variable-K
  rollout switches on precisely the same static every-``K`` grid as fixed K and
  the same windowing applies. Without this, the duration-pinned "control"
  configuration silently trained its WORKER with a different algorithm (the
  dense fallback, which conditions the boundary state on the NEW goal and only
  resets the return there) than the Director baseline it is compared against --
  a second Director divergence beyond the block-pooled manager credit path.
  Genuinely variable holds (``duration_fixed == 0``) still take the dense path.
  """
  if not split_traj:
    return 0
  k = int(duration_fixed) if (variable_goal_length and duration_fixed > 0) else (
      int(manager_sample_freq))
  if variable_goal_length and duration_fixed <= 0:
    return 0
  k = max(1, k)
  if H >= k and H % k == 0:
    return k
  return 0


def decision_mean_rescale(valid):
  """Per-row factor turning a padded-axis ``mean`` into a per-decision mean.

  ``downsample_at_switch_mask`` packs a data-dependent number of manager
  decisions into a static, full-width buffer and forward-fills the tail, so a
  plain ``x.mean(1)`` over that axis divides by the buffer width instead of by
  the number of real decisions. Multiplying the masked weights by this factor
  restores fixed-K Director's normalization, where the axis is exactly the
  decision count. Returns ``(B, 1)``; identically 1.0 when every column is
  valid.
  """
  valid = f32(valid)
  n_cols = valid.shape[1]
  return n_cols / jnp.maximum(jnp.sum(valid, axis=1, keepdims=True), 1.0)


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
  # Decisions that actually OWN realized transitions. Rewards/continuations are
  # indexed by ``rew[:, 1:]`` / ``con[:, :-1]``, so a switch landing on the
  # sequence's final timestep starts a segment with zero steps in it: it is the
  # bootstrap state, exactly the trailing column fixed-K drops with ``[:, :-1]``
  # (``downsample_manager_states`` appends it for that purpose alone). Counting
  # it as trainable would credit the manager for a decision whose pooled reward
  # is identically zero and whose advantage is pure critic noise, an extra
  # decision fixed-K Director never trains on.
  n_credit = jnp.sum(sw[:, :-1], axis=-1, keepdims=True)
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
  mgr_switch = f32(jnp.arange(horizon)[None, :] < n_credit)
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


def patch_trailing_replay_state(x_down, x_full, switch_mask):
  """Overwrite the first padding slot (index ``n_sw``, right after the last
  real switch) of a ``downsample_at_switch_mask`` result with the window's
  TRUE final state/value, mirroring fixed-K's ``downsample_manager_states``
  convention of always appending the window's genuine final state as an
  extra bootstrap anchor even when it isn't a real decision boundary.

  Without this, that slot holds a stale, forward-filled repeat of the last
  real switch's own state instead -- see EXPERIMENTS.md/paper 'replay
  downsample trailing-state asymmetry' (2026-07-27) and
  ``test_downsample_at_switch_mask_omits_fixed_k_trailing_state_on_remainder``.
  Training batch windows (length ``batch_length``) essentially never end
  exactly on a decision boundary for a fixed-length hold (only 1 of ``hold``
  possible phases aligns), so this fires on most replay-side manager value
  updates. No-op (returns ``x_down`` unchanged at that slot) whenever the
  last switch already lands on the window's final timestep.

  ``x_down``/``x_full`` may be a single array or a matching pytree (e.g. the
  ``feat`` dict); ``jax.tree.map`` treats a plain array as its own trivial
  leaf, so this works uniformly for both.
  """
  sw = f32(switch_mask)
  B, T = sw.shape
  n_sw = jnp.sum(sw, axis=-1).astype(i32)              # (B,) real decisions/row
  idx = jnp.clip(n_sw, 0, T - 1)                        # first padding slot
  needs_patch = n_sw < T                                # a trailing remainder exists

  def patch(down, full):
    true_final = full[:, -1]
    old = jnp.take_along_axis(
        down, idx.reshape((B,) + (1,) * (down.ndim - 1)), axis=1)[:, 0]
    # needs_patch/idx already carry the batch dim -- only pad with as many
    # extra singleton dims as true_final/old have BEYOND that batch dim.
    m = needs_patch.reshape((B,) + (1,) * (true_final.ndim - 1))
    new_val = jnp.where(m, true_final, old)
    return down.at[jnp.arange(B), idx].set(new_val)

  return jax.tree.map(patch, x_down, x_full)


def _last_decision_truncation(mgr_skills_eff, switch_mask, dur_min, dur_max):
  """Shared derivation for BOTH truncation policies (relabel and drop).

  Returns ``(is_last_slot, truncated, relabel_idx, orig_idx_at_last)``:
    * ``is_last_slot`` (B, n_mgr) bool -- the final real decision's slot.
    * ``truncated``    (B,)      bool -- did the horizon cut its hold short?
    * ``relabel_idx``  (B,)      i32  -- class index of the REALIZED duration.
    * ``orig_idx_at_last`` (B,)  int  -- the class it actually sampled.

  Factored out so ``relabel_truncated_last_duration`` and
  ``truncated_last_decision_mask`` cannot drift apart on what "truncated"
  means -- two independently-maintained copies of this arithmetic is exactly
  how the off-by-one below survived as long as it did.
  """
  sw = f32(switch_mask)                                    # (B, T)
  B, T = sw.shape
  n_sw = jnp.sum(sw, axis=-1)                               # (B,) real decisions
  idx = jnp.arange(T)[None, :]                              # (1, T)
  last_switch_pos = jnp.max(jnp.where(sw > 0.5, idx, -1), axis=-1)  # (B,)
  # Transitions actually credited to this decision, matching the "duration p ==
  # p pooled transitions" convention the (untruncated) countdown recurrence and
  # variable_block_director_tensors both use: a switch at state s that holds
  # for its full sampled duration p reaches the next switch at state s+p,
  # crediting exactly p transitions (rew[s+1..s+p], indices s..s+p-1). The
  # transition axis has length T-1 (indices 0..T-2, since rew[:, 1:]/con[:, :-1]
  # both drop one end), so when the rollout ends before the next switch the
  # transitions actually available are indices s..T-2: (T-1)-s of them, NOT the
  # (T-s) STATES from s through the final index inclusive. Using T-s here
  # over-counted the truncated decision's credited length by exactly one --
  # e.g. a switch at s=16 in a T=20 window (states 16..19, but only transitions
  # 16..18 exist) was relabeled to duration 4 when only 3 transitions were ever
  # pooled into its reward -- see
  # test_relabel_avail_matches_transitions_credited_by_variable_block_director_tensors.
  avail = f32(T - 1) - f32(last_switch_pos)
  dur_idx = mgr_skills_eff['duration']                       # (B, n_mgr) int class index
  n_mgr = dur_idx.shape[1]
  is_last_slot = (jnp.arange(n_mgr)[None, :] == (n_sw - 1)[:, None])  # (B, n_mgr)
  orig_idx_at_last = jnp.sum(jnp.where(is_last_slot, dur_idx, 0), axis=-1)  # (B,)
  orig_dur = f32(dur_min) + f32(orig_idx_at_last)
  truncated = avail < orig_dur
  relabel_dur = jnp.clip(avail, dur_min, dur_max)
  relabel_idx = i32(relabel_dur - dur_min)
  return is_last_slot, truncated, relabel_idx, orig_idx_at_last


def relabel_truncated_last_duration(mgr_skills_eff, switch_mask, dur_min, dur_max):
  """Hindsight-relabel the LAST decision's duration class when its hold was cut
  short by the imagination horizon rather than by a real switch, so REINFORCE
  credits the manager for the duration it actually got to run, not the one it
  sampled. Only the final real decision in a rollout can be truncated this way
  -- every earlier one is, by construction, followed by a genuine switch, so it
  always ran its full sampled duration and is left untouched. Only the
  ``duration`` field changes; the goal-code choice itself isn't invalidated by
  running short, only how long it got to hold.

  NOTE this substitutes an action the policy did NOT sample into the REINFORCE
  log-prob, which biases an on-policy estimator (hindsight relabeling is sound
  for GOAL relabeling in off-policy algorithms, not for this). See
  ``truncated_last_decision_mask`` for the unbiased alternative.
  """
  if 'duration' not in mgr_skills_eff:
    return mgr_skills_eff
  is_last_slot, truncated, relabel_idx, orig_idx_at_last = (
      _last_decision_truncation(mgr_skills_eff, switch_mask, dur_min, dur_max))
  dur_idx = mgr_skills_eff['duration']
  new_last_idx = jnp.where(truncated, relabel_idx, orig_idx_at_last.astype(i32))
  new_dur = jnp.where(is_last_slot, new_last_idx[:, None], dur_idx)
  return {**mgr_skills_eff, 'duration': new_dur}


def truncated_last_decision_mask(mgr_skills_eff, switch_mask, dur_min, dur_max):
  """``(B, n_mgr)`` float mask, 1 on the last decision iff its hold was cut
  short by the imagination horizon.

  The unbiased alternative to relabeling: instead of substituting a different
  action into the policy gradient, simply withhold credit for the decision
  whose consequence was never observed. Intended to gate the POLICY term only:
    * the CRITIC target for that decision is legitimate either way -- pooled
      reward over the realized steps plus a bootstrap off the final state is a
      valid n-step TD target with variable n;
    * by the same argument the SKILL head's advantage is valid, since it is
      that same n-step return minus its baseline;
    * only the DURATION head's action provably failed to execute as sampled,
      and under ``mgr_reward_agg='sum'`` truncation systematically shortchanges
      longer sampled holds -- a direct route to duration collapse.
  Hence ``goal_duration_truncated_policy='drop_duration'`` (surgical) vs.
  ``'drop_decision'`` (withhold the whole decision's policy credit).
  """
  if 'duration' not in mgr_skills_eff:
    return jnp.zeros_like(f32(switch_mask[:, :mgr_skills_eff[
        next(iter(mgr_skills_eff))].shape[1]]))
  is_last_slot, truncated, _, _ = _last_decision_truncation(
      mgr_skills_eff, switch_mask, dur_min, dur_max)
  return f32(is_last_slot) * f32(truncated)[:, None]


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
