"""Actor-critic losses on imagined rollouts and on replay tails.

``imag_loss_wkr`` / ``imag_loss_mgr`` are the two halves of the hierarchical
objective; ``imag_loss`` / ``repl_loss`` are the flat (non-HRL) equivalents.
All of them are plain functions of already-evaluated head outputs, so the Agent
stays responsible for *which* features each head sees and these stay testable in
isolation.
"""
import chex
import jax
import jax.numpy as jnp

from .heads import (
    align_skill_events,
    head_entropy_perdim_time,
    head_entropy_time,
    head_logp_time,
    manager_reinforce_policy,
    policy_time_slice,
    _head_inner,
)
from .tensors import decision_mean_rescale

f32 = jnp.float32
i32 = jnp.int32
# Stop-gradient helper: optionally pass gradients through (e.g. reward head).
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)


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
    mgr_dur_lagrange_adapter=None,
    duration_fixed=False,
    switch_mask=None,
    dur_reg_weight=0.0,
    dur_reg_target=0.0,
    dur_min=1,
    trunc_mask=None,
    trunc_policy='keep',
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

  # Validity mask over the decision axis. Fixed K (``switch_mask=None``): every
  # column is a real decision. Variable goal length: only ``switch_mask==1``
  # columns are, and under ``variable_goal_block_rew`` the rest of the static
  # buffer is ``downsample_at_switch_mask``'s forward-filled padding -- repeated
  # copies of the last real decision. Every reduction below (running normalizer
  # statistics, entropy adapters, logged means, and the final ``.mean(1)`` the
  # caller applies) must therefore be taken over the real decisions only, or the
  # padded tail both dilutes the manager loss by the padding ratio and drags the
  # running statistics toward one repeated decision.
  dec_mask = None if switch_mask is None else sg(f32(switch_mask[:, :-1]))

  def dec_mean(x, mask=None):
    """Mean over real decisions (plain mean under fixed K)."""
    mask = dec_mask if mask is None else mask
    if mask is None:
      return x.mean()
    mask = jnp.broadcast_to(mask.reshape(mask.shape + (1,) * (x.ndim - 2)),
                            x.shape)
    return (x * mask).sum() / jnp.maximum(mask.sum(), 1e-8)

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
  roff_extr, rscale_extr = mgr_extr_retnorm(
      mgr_extr_ret, update, weights=dec_mask)
  roff_expl, rscale_expl = mgr_expl_retnorm(
      mgr_expl_ret, update, weights=dec_mask)
  mgr_extr_adv = (mgr_extr_ret - mgr_extr_tarval[:, :-1]) / rscale_extr
  mgr_expl_adv = (mgr_expl_ret - mgr_expl_tarval[:, :-1]) / rscale_expl
  mgr_adv = mgr_extr_adv + mgr_expl_weight * mgr_expl_adv
  mgr_aoffset, mgr_ascale = mgr_advnorm(mgr_adv, update, weights=dec_mask)
  mgr_adv_normed = (mgr_adv - mgr_aoffset) / mgr_ascale

  skill_events = align_skill_events(skills, manager_policy)
  reinforce_policy = manager_reinforce_policy(manager_policy, duration_fixed)
  per_head_logp = {
      k: head_logp_time(v, skill_events[k]) for k, v in reinforce_policy.items()}
  # Truncated-hold policy handling. ``trunc_mask`` (B, n_mgr) marks the final
  # decision when the imagination horizon cut its hold short, i.e. when the
  # sampled duration provably did not execute. ``'drop_duration'`` withholds
  # only the DURATION head's log-prob there (the surgical, principled choice:
  # the critic target and the skill head's advantage are both valid n-step
  # quantities regardless of truncation, so there is no reason to discard
  # them); ``'drop_decision'`` withholds the whole decision's policy credit.
  # ``'keep'`` is the historical behavior. See
  # ``tensors.truncated_last_decision_mask``.
  keep_bt = None
  if trunc_mask is not None and trunc_policy in ('drop_duration', 'drop_decision'):
    keep_bt = sg(1.0 - f32(trunc_mask)[:, :-1])            # (B, n_mgr-1)
    if trunc_policy == 'drop_duration' and 'duration' in per_head_logp:
      per_head_logp['duration'] = per_head_logp['duration'] * keep_bt
  mgr_logpi = sum(per_head_logp.values())
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
        if mgr_dur_actent_adapter is None or duration_fixed:
          # ``goal_duration_fixed`` bypasses the head (``_duration_steps``), so
          # its entropy cannot affect the trajectory. Regularizing it anyway
          # pushes the shared manager trunk with a signal true fixed-K Director
          # never carries -- same argument as dropping it from the REINFORCE
          # sum (``manager_reinforce_policy``), which left this second channel
          # open.
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
      # Weight the adapter's tracked average by decision validity: on a packed
      # variable-K axis the padded tail repeats the last real decision's
      # entropy, so an unweighted mean makes the controller chase that one
      # decision instead of the rollout's decisions.
      if perdim and ent_perdim.ndim > 2:
        ent_w = None if dec_mask is None else dec_mask[..., None]
        loss_perdim, mets = adapter(ent_norm, update=update, weights=ent_w)
        ent_loss_terms.append(loss_perdim.sum(-1))
      else:
        ent_scalar = ent_norm.mean(-1) if ent_norm.ndim > 2 else ent_norm
        loss_scalar, mets = adapter(
            ent_scalar, update=update, weights=dec_mask)
        ent_loss_terms.append(loss_scalar)
      mgr_actent_mets.update(
          {f'mgr_actent_{k}_{mk}': mv for mk, mv in mets.items()})
      mgr_actent_mets[f'mgr_ent_norm_{k}_mean'] = dec_mean(ent_norm)
    if ent_loss_terms:
      mgr_ent_loss_bt = sum(ent_loss_terms)

  w = sg(weight[:, :-1])
  vw = sg(weight[:, :-1])
  if switch_mask is not None:
    # Variable goal length: only apply the REINFORCE update at the steps where a
    # manager decision was actually made (the duration/skill/mask log-probs and
    # advantage are meaningless on held-goal steps). Also masks critic updates on
    # padded switch timelines (``variable_goal_block_rew``).
    m = dec_mask
    # Zero masked advantages before weighting: ``0 * inf`` is NaN on padded block-rew
    # slots where retnorm can divide by a near-zero scale.
    mgr_adv_normed = jnp.where(m > 0, mgr_adv_normed, 0.0)
    # Rescale so the caller's ``loss.mean(1)`` over this axis equals the mean
    # over the REAL decisions. Without it the manager's policy and critic losses
    # come out smaller by exactly the valid fraction of the axis -- at H=16/K=8
    # that is 2 real decisions in a 16-wide buffer, an 8x silent down-weighting
    # of the whole manager against the world model and the worker, and under
    # true variable K a factor that drifts with the realized hold length.
    # ``rescale_to_decisions`` is the A/B switch for this normalization alone
    # (agent.mgr_decision_mean_rescale). False reproduces the pre-2026-07-27
    # behavior -- the caller's ``.mean(1)`` divides by the padded buffer width,
    # so the whole manager is down-weighted by the valid fraction of the axis --
    # while keeping every other block-pooled fix in place.
    # The CRITIC keeps every real decision: a truncated hold's target (pooled
    # reward over the realized steps + bootstrap off the final state) is a
    # valid variable-n TD target, so there is nothing to withhold there.
    vw = vw * m * decision_mean_rescale(m)
    m_pol = m
    if keep_bt is not None and trunc_policy == 'drop_decision':
      m_pol = m * keep_bt
    # Rescale the POLICY term by its OWN valid count -- otherwise dropping a
    # decision silently shrinks the policy loss by the dropped fraction rather
    # than just averaging over fewer samples.
    w = w * m_pol * decision_mean_rescale(m_pol)
    metrics['mgr_switch_rate'] = switch_mask.mean()
    metrics['mgr_valid_decisions'] = m.sum(1).mean()
    if keep_bt is not None:
      metrics['mgr_truncated_decisions'] = (
          f32(trunc_mask)[:, :-1] * m).sum(1).mean()

  # REINFORCE manager actor. With the adaptive entropy adapter the per-dim
  # normalized-entropy loss is already in ``mgr_ent_loss_bt``; otherwise fall
  # back to the fixed v3 ``actent`` on summed per-categorical entropy.
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
  if ((dur_reg_weight > 0.0 or mgr_dur_lagrange_adapter is not None)
      and ('duration' in manager_policy) and not duration_fixed):
    # ``not duration_fixed``: with the head bypassed, a prior pulling E[dur]
    # toward a target trains the trunk to predict a hold length nothing reads.
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
      # Uses ``dec_mean`` (the RAW 0/1 decision mask), not ``w``: ``w`` bakes in
      # ``decision_mean_rescale``'s per-ROW factor, which is correct for the
      # per-row ``.mean(1)`` the caller applies to the returned loss tensors but
      # WRONG for a single flat ratio computed here across the whole batch --
      # that factor is LARGER for rows with fewer real decisions (sparser
      # holds), so a plain ``(w*err).sum()/w.sum()`` systematically over-weights
      # them. On a batch of one 1-decision row (err=10) and one 8-decision row
      # (err=1 each), the true flat mean over all 9 decisions is 2.0; the
      # w-weighted ratio gave 5.5 (measured). Only meaningfully wrong once rows
      # have DIFFERENT real decision counts, i.e. under genuinely variable
      # holds -- every pinned (duration_fixed>0) run has identical per-row
      # counts, so this was invisible there. See
      # test_lagrange_mean_abs_err_is_not_biased_by_per_row_decision_count.
      mean_abs_err = dec_mean(jnp.abs(exp_p - f32(dur_reg_target)))
      _, dur_lagrange_mets = mgr_dur_lagrange_adapter(mean_abs_err, update=update)
      dur_reg_bt = mgr_dur_lagrange_adapter.scale() * sq_err
      metrics.update(
          {f'mgr_duration_lagrange_{k}': v for k, v in dur_lagrange_mets.items()})
      losses['goal_duration_prior'] = w * dur_reg_bt
      metrics['mgr_duration_reg_loss'] = (w * dur_reg_bt).mean()
    else:
      dur_reg_bt = dur_reg_weight * sq_err
      losses['mgr_policy'] = losses['mgr_policy'] + w * dur_reg_bt
      metrics['mgr_duration_reg_loss'] = (w * dur_reg_bt).mean()
    metrics['mgr_duration_exp_mean'] = exp_p.mean()
    metrics['mgr_duration_exp_std'] = exp_p.std()

  metrics['mgr_policy_loss'] = losses['mgr_policy'].mean()
  metrics['mgr_ent_loss'] = dec_mean(mgr_ent_loss_bt)
  metrics.update(mgr_actent_mets)
  # ``mgr_*_rew[:, i + 1]`` is the pooled reward credited to decision ``i``, so
  # the decision mask lines up with the reward tensor after dropping slot 0.
  metrics['mgr_extr_rew'] = dec_mean(mgr_extr_rew[:, 1:])
  nz = jnp.maximum((jnp.abs(mgr_extr_rew[:, 1:]) > 0).sum(), 1)
  metrics['mgr_extr_rew_block'] = mgr_extr_rew[:, 1:].sum() / nz
  metrics['mgr_expl_rew'] = dec_mean(mgr_expl_rew[:, 1:])

  # Critic NLL against RAW λ-returns (``valnorm: none`` -> target == raw return).
  # The symexp_twohot head handles the return scale; plus a slow-value regression
  # term (DreamerV3 ``imag_loss``).
  voff_extr, vscale_extr = mgr_extr_valnorm(
      mgr_extr_ret, update, weights=dec_mask)
  voff_expl, vscale_expl = mgr_expl_valnorm(
      mgr_expl_ret, update, weights=dec_mask)
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

  # Decision-weighted (see ``dec_mean``): identical to ``.mean()`` under fixed K,
  # but on a packed variable-K axis an unweighted mean reports mostly padding.
  metrics['mgr_adv'] = dec_mean(mgr_adv)
  metrics['mgr_adv_std'] = jnp.sqrt(jnp.maximum(
      dec_mean(jnp.square(mgr_adv)) - jnp.square(dec_mean(mgr_adv)), 0.0))
  metrics['mgr_adv_mag'] = dec_mean(jnp.abs(mgr_adv_normed))
  metrics['mgr_extr_adv'] = dec_mean(mgr_extr_adv)
  metrics['mgr_expl_adv'] = dec_mean(mgr_expl_adv)

  metrics['mgr_total_ret'] = dec_mean(mgr_total_ret)
  metrics['mgr_extr_ret'] = dec_mean(mgr_extr_ret_normed)
  metrics['mgr_expl_ret'] = dec_mean(mgr_expl_ret_normed)
  # Decision-weighted like the return/advantage metrics above: these tensors span
  # the full padded buffer, so an unweighted mean is ~87% copies of the last real
  # decision's value at H=16/K=8 and is NOT comparable to fixed-K's.
  metrics['mgr_extr_val'] = dec_mean(mgr_extr_val[:, :-1])
  metrics['mgr_expl_val'] = dec_mean(mgr_expl_val[:, :-1])
  # Removed: ``mgr_extr_tar``/``mgr_expl_tar`` (== ret_normed already logged),
  # ``mgr_extr_slowval``/``mgr_expl_slowval`` (≈ ``*_val`` up to EMA lag),
  # ``mgr_con``/``mgr_weight`` (≈ 1 in non-terminal imagined rollouts).

  # Iterate the policy heads (skill/mask), not the carried skills dict — the latter
  # also holds the running goal code ``goal_code``, which has no policy/entropy.
  for k in manager_policy:
    ent_mean = dec_mean(mgr_ents[k])
    metrics[f'mgr_ent/{k}'] = ent_mean
    if hasattr(manager_policy[k], 'minent'):
      lo, hi = manager_policy[k].minent, manager_policy[k].maxent
      metrics[f'mgr_rand/{k}'] = (ent_mean - lo) / (hi - lo)

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
