"""The manager: emitting goal codes, holding them, and rolling them out.

The manager re-decides either every ``manager_sample_freq`` steps (fixed K) or
whenever the hold length it emitted runs out (``variable_goal_length``). These
methods cover one decision (``_emit_manager`` / ``_manager_skill_step``), the
worker's view of a held goal (``_feat_goal2tensor`` and the HiTS countdown), and
the three places a whole trajectory of decisions is produced: imagination
(``_imagine_with_manager``), replay sequences (``_manager_skills_on_sequence``),
and report-time goal proposals (``_propose_goal``).
"""
import jax
import jax.numpy as jnp
import ninjax as nj

import embodied.jax.nets as nn
import embodied.jax.outs as outs

from .tensors import (
    aggregate_mgr_cont,
    aggregate_mgr_extr_rew,
    mgr_as_dict,
    skill_switch,
)

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
mode = lambda xs: jax.tree.map(lambda x: x.pred(), xs)
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)


class ManagerMixin:
  """Manager decision/rollout logic mixed into ``dreamerv3.agent.Agent``."""

  def _countdown_budget(self):
    """Duration budget normalizing the worker countdown (HiTS delta_t_max).

    ``goal_duration_fixed`` (pinned holds) caps the REALIZED countdown at that
    constant, not at ``goal_duration_max`` -- ``_duration_steps`` bypasses the
    duration head entirely and always returns ``goal_duration_fixed`` in that
    regime, so a countdown can never exceed it. Launch configs default
    ``goal_duration_max`` to 16 regardless of ``goal_duration_fixed`` (it is
    only ever raised to match ``goal_duration_target``, never lowered), so a
    pinned run at e.g. ``goal_duration_fixed=8`` would otherwise normalize by
    16: the countdown at a fresh switch (the true maximum, 8) would read 0.0
    instead of +1.0, and the worker would never see the top half of the
    intended [-1, 1] range. Currently dormant everywhere (``worker_timed_goals``
    is off in every launched config) but real the moment it's turned on
    together with a pinned hold -- see
    ``test_countdown_budget_uses_the_pinned_hold_not_goal_duration_max``.
    """
    if self.variable_goal_length:
      if self.goal_duration_fixed > 0:
        return float(self.goal_duration_fixed)
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

  def _mgr_input(self, feat, mgr_skill):
    """Manager-policy input: ``feat2tensor(feat)`` plus optional pre-edit conditioning.

    ``mgr_skill['goal_code']`` MUST be the PRE-edit running code Z the manager
    conditions on at this step (the carry *before* ``_advance_mgr_skill`` overwrites
    it). Channels are appended in the fixed order goalcode, decgoal, achieve and are
    all stop-gradient'd, so this adds no gradient path into ``goal_enc``/``goal_dec``
    -- only ``manager_pol``'s own (lazily-sized) first layer consumes them. With every
    flag off this returns exactly ``feat2tensor(feat)`` (bit-identical to before).

    Under ``goal_soft_reuse_adapt`` the goalcode channel is the SOFT
    distribution (``mgr_skill['skill_probs']`` -- the manager's own plain
    softmax, §``_emit_manager``) rather than the collapsed one-hot: a strictly
    richer signal (the one-hot is a lossy argmax of it) that also tells the
    manager how *contested* the last decision was, not just which class won.
    Only this input-side view changes; sampling itself is untouched (direct
    prediction, no recombination)."""
    parts = [self.feat2tensor(feat)]
    if self.mgr_cond_goalcode:
      if self._mgr_needs_skill_probs and 'skill_probs' in mgr_skill:
        code = sg(f32(mgr_skill['skill_probs']))               # (..., L, C) soft
      else:
        code = sg(self._running_goal_code(mgr_skill))          # (..., L, C) previous
      parts.append(code.reshape((*code.shape[:-2], -1)))       # (..., L*C)
    return jnp.concatenate(parts, -1)

  def _emit_manager(
      self, tensor, bdims, deterministic=False, prev_mgr_skill=None):
    """Sample a manager command from the policy head.

    Under ``goal_soft_reuse_adapt`` the result also carries ``skill_probs`` --
    the manager's own plain softmax over the skill logits -- exposed purely as
    an input/loss signal for the next decision (see ``_mgr_input`` and the
    ``goal_soft_reuse`` loss); sampling itself is untouched (direct
    prediction). ``prev_mgr_skill`` is accepted for call-site symmetry."""
    out = mgr_as_dict(self.manager_pol(tensor, bdims))
    skill_probs = None
    if self.goal_soft_reuse_adapt and 'skill' in out:
      inner = out['skill']
      while not hasattr(inner, 'dist') and hasattr(inner, 'output'):
        inner = inner.output
      skill_probs = jax.nn.softmax(f32(inner.dist.logits), -1)
    result = (mode if deterministic else sample)(out)
    if skill_probs is not None:
      result = {**result, 'skill_probs': skill_probs}
    return result

  def _advance_mgr_skill(self, mgr_skill, emit, update, base_code=None):
    """Switch in the freshly emitted manager skill (Director window switch)."""
    del base_code  # accepted for call-site symmetry; plain goals have no running Z
    return skill_switch(update, emit, mgr_skill)

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

  def _manager_skill_step(self, feat, mgr_skill, mgr_step, reset):
    """Resample the skill at each manager decision (Director carry switch).

    Fixed K: every ``manager_sample_freq`` steps. Variable goal length: when the
    per-element countdown carried in ``mgr_step`` runs out."""
    # Pull carry-only fields out before any switch tree-map (``emit``/``new_skill``
    # don't carry them, so they must not reach ``skill_switch``).
    sticky_change = mgr_skill.get('last_change_mask')
    cached_goal = mgr_skill.get('goal_deter')
    # Carry-only render caches (held decoded-goal images) must not reach the switch
    # tree-maps either; pull them out and thread them through unchanged. ``policy``
    # refreshes them on a switch.
    img_cache = {k: v for k, v in mgr_skill.items() if k.startswith('goal_img_')}
    strip = {'goal_deter', 'last_change_mask', *img_cache}
    # Pre-switch running code: the "previous goal" reference for the
    # implicit-sparsity CHANGED overlay below, captured before ``skill_switch``/
    # ``_advance_mgr_skill`` overwrite it.
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
    # Optionally condition the manager on the PREVIOUS emitted skill (held in
    # ``mgr_skill`` before the switch) so it can deliberately re-emit blocks;
    # falls back to bare feat when no conditioning is enabled.
    mgr_inp = (self._mgr_input(feat, mgr_skill) if self.mgr_cond_goalcode
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
    # Implicit-sparsity overlay: mark blocks whose class actually CHANGED from the
    # previous goal (argmax(new code) != argmax(pre-switch code)), independent of
    # re-emitting a block with the same value is not highlighted. That reuse is
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

  def _mgr_extr_rew(self, rew, con, without_zeros=False):
    """Manager extrinsic reward: WM reward pooled over ``manager_sample_freq`` steps."""
    return aggregate_mgr_extr_rew(
        rew, con, self.manager_sample_freq, without_zeros,
        agg_mode=self.mgr_reward_agg)

  def _mgr_cont(self, con, without_zeros=False):
    """Manager continuation: WM continuation pooled over ``manager_sample_freq`` steps."""
    return aggregate_mgr_cont(con, self.manager_sample_freq, without_zeros)

  def _propose_goal(self, feat, impl):
    """Propose a goal vector from ``start`` state (Director ``propose_goal``)."""
    B = feat['deter'].shape[0]
    if impl == 'manager':
      if self.mgr_cond_goalcode:
        # Conditioning on: seed the manager's "previous code" channel with the
        # current-state encoding so the input width matches every other
        # manager_pol call site (the lazily-sized first layer is shared).
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
    # Start with a dummy skill that will be replaced in first step
    mgr_skill = {'skill': jnp.zeros((B, *skill_shape), f32)}
    if self._mgr_needs_skill_probs:
      mgr_skill['skill_probs'] = jnp.zeros((B, *skill_shape), f32)
    seed_code = None
    if self.mgr_cond_goalcode:
      # The body's step-0 emit conditions on the initial (zero) skill; return it
      # as the seed so the train re-derivation reconstructs the same step-0
      # conditioning code (no leakage of the skill the manager is about to emit).
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
    if self.mgr_cond_goalcode:
      # Seed the manager's "previous code" channel with the start-state encoding
      # so the first manager_pol call has the same input width as the body
      # (which conditions on the held previous skill).
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
