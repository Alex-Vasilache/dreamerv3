"""DreamerV3-style agent: RSSM world model + policy/value on imagined rollouts.

Trains encoder, dynamics (RSSM), decoder, reward/continue heads, policy, and
value in one step from replay sequences. Actor-critic losses use imagined
trajectories from ``dyn.imagine``; optional ``repval_loss`` fits the value on
real replay tails with bootstrap from imagination.
"""
import re

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
from .hrl.goals import GoalCodeMixin
from .hrl.manager import ManagerMixin
from .hrl.reporting import ReportMixin
from .hrl import (
    downsample_at_switch_mask,
    downsample_manager_states,
    goal_reward_cosine_max,
    imag_loss,
    imag_loss_mgr,
    imag_loss_wkr,
    imag_reward_pad,
    lambda_return,
    mgr_as_dict,
    pairwise_cosmax,
    patch_trailing_replay_state,
    decision_mean_rescale,
    relabel_truncated_last_duration,
    truncated_last_decision_mask,
    repl_loss,
    resize_frames,
    variable_block_director_tensors,
    worker_split_window,
)
from .hrl import goal_ae
from .hrl.heads import _head_inner

f32 = jnp.float32
i32 = jnp.int32
# Stop-gradient helper: optionally pass gradients through (e.g. reward head).
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
# Sample from a tree of distribution-like outputs (policy heads).
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
# Deterministic mode (argmax / mean) of a tree of distribution-like outputs.
mode = lambda xs: jax.tree.map(lambda x: x.pred(), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
# Concatenate pytrees along axis ``a`` (e.g. time) for feat/action sequences.
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class Agent(ManagerMixin, GoalCodeMixin, ReportMixin, embodied.jax.Agent):
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
    # Manager-policy input conditioning: appends a stop-gradient'd channel built
    # from the previous decision's goal code so the manager can see what it is
    # about to overwrite. A Director manager regenerates the whole code each
    # decision, so it must SEE the previous goal to deliberately re-emit (reuse)
    # blocks -- this is what makes implicit sparsity learnable rather than
    # incidental. Forced on by ``goal_soft_reuse_adapt`` below.
    self.mgr_cond_goalcode = bool(getattr(config, 'mgr_cond_goalcode', False))
    # Implicit sparsity (the goal-reuse mechanism; see configs.yaml). Plain
    # (direct) goal sampling is left untouched and an ordinary loss is added on
    # the OVERLAP between this decision's own softmax and the previous
    # decision's -- sum_c p_t[c]*p_{t-1}[c] per block, averaged over blocks --
    # plus the manager is fed the previous decision's soft distribution as an
    # extra input. No sampling change, no decoder pass, no reward shaping: it
    # backprops straight into both steps' own logits through their softmax (see
    # ``_emit_manager``). A dual-ascent Lagrange multiplier holds the mean
    # overlap AT ``goal_soft_reuse_target`` rather than driving it to 1.
    self.goal_soft_reuse_adapt = self.use_hrl and bool(
        getattr(config, 'goal_soft_reuse_adapt', False))
    if self.goal_soft_reuse_adapt:
      # Needs to SEE the previous decision's soft distribution.
      self.mgr_cond_goalcode = True
    # The reuse mechanism needs the soft ``skill_probs`` side-channel threaded
    # through the manager-skill carry (see _emit_manager/_mgr_input and every
    # init_policy/_unpack_carry/_imagine_with_manager carry init).
    self._mgr_needs_skill_probs = self.goal_soft_reuse_adapt

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
    # Manager extrinsic-reward block aggregation: 'mean' (per-step average, the
    # original; under per-step discounting this under-credits long blocks -> short-K
    # bias) or 'sum' (continuation-weighted SMDP option return, K-neutral).
    self.mgr_reward_agg = str(getattr(config, 'mgr_reward_agg', 'mean'))
    # Hindsight-relabel a block-pooled decision's duration class when its hold
    # was cut short by the imagination horizon (rather than a real switch), so
    # REINFORCE credits the duration actually realized, not the one sampled --
    # see relabel_truncated_last_duration. Default True (the fix), off is an
    # explicit ablation switch.
    self.goal_duration_relabel_truncated = bool(getattr(
        config, 'goal_duration_relabel_truncated', True))
    # How the manager's POLICY term treats a hold the imagination horizon cut
    # short: 'keep' (historical; pair with goal_duration_relabel_truncated),
    # 'drop_duration' (withhold only the duration head's log-prob there -- the
    # only action that provably did not execute), or 'drop_decision' (withhold
    # the whole decision's policy credit). The critic is never gated: its
    # truncated target is a valid variable-n TD target. See
    # ``hrl.tensors.truncated_last_decision_mask``.
    self.goal_duration_truncated_policy = str(getattr(
        config, 'goal_duration_truncated_policy', 'keep'))
    assert self.goal_duration_truncated_policy in (
        'keep', 'drop_duration', 'drop_decision'), (
            self.goal_duration_truncated_policy)
    self.goal_duration_fixed = int(getattr(config, 'goal_duration_fixed', 0))
    # Resolve the pinned-hold flag first so the duration controllers can be
    # switched off when the head is bypassed: a pinned hold makes every duration
    # regularizer causally inert (see ``imag_loss_mgr``), and keeping the
    # Lagrangian one live would leave ``scales['goal_duration_prior']`` without a
    # matching loss key, tripping the loss/scale key-set assert in ``loss()``.
    _dur_pinned = self.goal_duration_fixed > 0
    # Lagrangian duration prior: an AutoAdapt regulates the switch-weighted mean
    # ABSOLUTE deviation |E[dur] - goal_duration_target| against a small tolerance,
    # so the response is symmetric in the deviation direction. Folded out into its
    # own top-level loss (``goal_duration_prior``, own ``loss_scales`` entry),
    # unlike the fixed-weight ``goal_duration_reg`` path, which is added directly
    # into ``mgr_policy``.
    self.goal_duration_lagrange = bool(
        getattr(config, 'goal_duration_lagrange', False)) and (
            self.variable_goal_length and not _dur_pinned)
    self.n_duration_classes = max(
        1, self.goal_duration_max - self.goal_duration_min + 1)
    if len(skill_shape_t) > 1:
      assert skill_shape_t[-1] == skill_classes, (
          'skill_shape[-1] must equal skill_classes (classes per categorical)')
    # Director-style sparse skills: float one-hot matrix, not integer indices.
    self.skill_space = elements.Space(np.float32, skill_shape_t, 0.0, 1.0)
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
      self.goal_ae_impl = str(getattr(config, 'goal_ae_impl', 'director'))
      if self.goal_ae_impl == 'vq':
        self._build_goal_vq(config, skill_shape_t, skill_classes)
      elif self.goal_ae_impl == 'director':
        self.goal_enc = embodied.jax.MLPHead(
            self.goal_code_space, **config.goal_enc, name='goal_enc')
        # Director goal decoder (skill code -> deter).
        self.goal_dec = embodied.jax.MLPHead(self.goal_shape, **config.goal_dec, name='goal_dec')
        self._goal_vq_kw = None
      else:
        raise NotImplementedError(self.goal_ae_impl)
      self.goal_autoencoder_beta = config.goal_autoencoder_beta
      # Weight on the code/deter geometry-preservation term (0 = off).
      self.goal_struct_weight = float(getattr(config, 'goal_struct_weight', 0.0))
      # Uniform prior metadata only: built inside ``loss`` with ``zeros_like`` encoder
      # logits so arrays stay on-device (``jnp.zeros`` here breaks sharded init).
      self._skill_prior_unimix = float(config.goal_enc.unimix)
      self._skill_factorized = len(skill_shape_t) > 1
      # goal_kl adapts a KL toward the uniform prior over the *categorical
      # posterior* the quantizer removes. Left reachable (the soft code
      # softmax(-d^2) is a well-defined distribution) but off in every vq
      # config block; codebook usage is reported as goal/used_frac instead.
      if self.goal_ae_impl == 'vq' and self.config.goal_kl:
        print('[goal-ae] WARNING: goal_kl=True with goal_ae_impl=vq applies the '
              'adaptive KL to softmax(-distances), which is not the Director '
              'posterior it was tuned for. The goal_* config blocks set it False.')

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
      if self.variable_goal_length:
        # Manager head with a shared trunk and two output heads:
        #   skill    : discrete skill code (onehot L, C)
        #   duration : categorical over {0..n-1} -> p = goal_duration_min + idx
        mgr_cfg = {**config.manager_policy}
        skill_out = mgr_cfg.pop('output')
        mgr_space = {
            'skill': self.goal_code_space,
            'duration': elements.Space(np.int32, (), 0, self.n_duration_classes)}
        mgr_out = {'skill': skill_out, 'duration': 'categorical'}
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
      scales.pop('mask_sparsity', None)
      if self.goal_duration_lagrange:
        # Folded-out Lagrangian duration prior (own loss key, own scale) -- see
        # ``mgr_dur_lagrange_adapter`` / ``losses['goal_duration_prior']``.
        scales['goal_duration_prior'] = scales.pop(
            'goal_duration_prior',
            getattr(self.config.loss_scales, 'goal_duration_prior', 1.0))
      else:
        scales.pop('goal_duration_prior', None)
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
      scales.pop('goal_duration_prior', None)
      scales.pop('goal_soft_reuse', None)
      if not self.config.repval_loss:
        scales.pop('repval', None)
    self.scales = scales

  @property
  def policy_keys(self):
    # Regex for checkpoint / param groups synced to the actor process.
    if self.use_hrl:
      # goal_enc is normally train/report-only (the policy step only *decodes*
      # the manager's sampled code via goal_dec); policy_struct_diag additionally
      # *encodes* the current state for offline diagnostics, so it needs
      # goal_enc's params included in the policy-side param group too.
      if bool(getattr(self.config, 'policy_struct_diag', False)):
        return '^(enc|dyn|dec|pol|manager_pol|goal_dec|goal_enc)/'
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
    mgr_skill = {'skill': jnp.zeros((batch_size, *skill_shape), f32)}
    if self._mgr_needs_skill_probs:
      # Soft pre-sample distribution behind the current ``skill`` one-hot, for
      # ``_mgr_input``'s next-step conditioning (see ``goal_soft_reuse_adapt``).
      # Zero-init like ``skill`` itself; harmless
      # since ``update`` is always True on the very first (reset) decision,
      # discarding this placeholder value.
      mgr_skill['skill_probs'] = jnp.zeros((batch_size, *skill_shape), f32)
    # Sticky mask of blocks CHANGED vs the previous goal (implicit sparsity), for
    # the mask_viz overlay; updated on each manager switch, held between. The
    # manager can reproduce a block's previous class (baseline, or steered there by
    # ``mgr_cond_goalcode``/``goal_soft_reuse_adapt``), so "unchanged" is a real,
    # informative signal.
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


  def _unpack_carry(self, carry):
    """``(enc, dyn, dec, prevact, mgr_skill, mgr_step)``; tolerate 4-tuples (flat mode)."""
    if len(carry) == 6:
      return carry
    enc, dyn, dec, prevact = carry[:4]
    # ``enc`` carry can be empty (stateless encoder), so derive B from any
    # non-empty carry leaf (dyn/prevact always have a leading batch dim).
    B = jax.tree.leaves((enc, dyn, dec, prevact))[0].shape[0]
    skill_shape = tuple(int(x) for x in self.skill_shape)
    mgr_skill = {'skill': jnp.zeros((B, *skill_shape), f32)}
    if self._mgr_needs_skill_probs:
      mgr_skill['skill_probs'] = jnp.zeros((B, *skill_shape), f32)
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
      # left unchanged (implicit sparsity) in white/grayscale -- tracked via the
      # sticky ``last_change_mask`` (see ``_manager_skill_step``).
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
    # Offline diagnostic: encode the *current real state* (not a manager proposal)
    # through the goal VAE and emit both the deter vector and the soft code, so an
    # external driver can correlate goal-space vs. goal-code geometry over a live
    # on-policy rollout without needing archived replay. off by default; see
    # experiments/goal_code_struct_corr/diag_goal_struct_corr.py.
    if self.use_hrl and bool(getattr(self.config, 'policy_struct_diag', False)):
      diag_deter = self.feat2deter(feat)
      diag_code = self.goal_enc(diag_deter, bdims=1)
      diag_dist = diag_code['skill'] if isinstance(diag_code, dict) else diag_code
      diag_probs = jax.nn.softmax(_head_inner(diag_dist).dist.logits, -1)  # (B, L, C)
      out['log/struct_diag_deter'] = diag_deter
      out['log/struct_diag_probs'] = diag_probs.reshape(diag_probs.shape[:-2] + (-1,))
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
        # that actually differ from the PREVIOUS goal (implicit sparsity): a
        # block re-emitted with the same value is not highlighted. Held between
        # switches, cleared at episode start.
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
          up = lambda x: resize_frames(x[:, None], s * Hi, s * Wi)[:, 0]
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

  def _flat_ac_loss(
      self, obs, repfeat, tokens, dyn_entries, dyn_carry, losses, metrics,
      enc_carry, dec_carry, enc_entries, dec_entries, B, T, training):
    """Non-HRL actor-critic: a single ``pol``/``val`` over world-model features.

    Terminal phase of ``loss`` when ``use_hrl`` is False -- returns the finished
    ``(loss, aux)`` tuple rather than accumulating into the HRL path.
    """
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

  def _build_goal_vq(self, config, skill_shape_t, skill_classes):
    """Build the VQ / SOM / LipVQ goal autoencoder (``hrl/goal_ae.py``).

    Replaces the Director ``goal_enc``/``goal_dec`` MLPHeads with the blockwise
    quantizer. The decoder owns the codebook so ``policy_keys`` ships it to the
    actor unchanged; see the module docstring there.
    """
    cfg = config.goal_vq
    blocks = int(skill_shape_t[0])
    dim = int(cfg.dim)
    lip = bool(cfg.lip)
    apply_to = str(cfg.lip_apply)
    assert apply_to in ('enc', 'dec', 'enc_dec'), apply_to
    lipkw = dict(
        lip_out=bool(cfg.lip_out), per_row=bool(cfg.lip_per_row),
        cinit=float(cfg.lip_cinit), clamp=bool(cfg.lip_clamp),
        strict_bound=bool(cfg.strict_bound), dtype=str(cfg.dtype))
    self.goal_dec = goal_ae.GoalVQDecoder(
        self.goal_shape, blocks, int(skill_classes), dim,
        lip=lip and apply_to in ('dec', 'enc_dec'),
        stddev=float(cfg.stddev), include_self=bool(cfg.include_self),
        **lipkw, **config.goal_vq_dec, name='goal_dec')
    self.goal_enc = goal_ae.GoalVQEncoder(
        self.goal_dec.codebook, blocks, dim,
        lip=lip and apply_to in ('enc', 'enc_dec'),
        temp=float(cfg.temp), **lipkw, **config.goal_vq_enc, name='goal_enc')
    # YAML 1.1 turns unquoted on/off into booleans, so accept both spellings.
    _ste = cfg.ste
    if isinstance(_ste, bool):
      ste = _ste
    else:
      ste = {'auto': None, 'on': True, 'off': False, 'true': True,
             'false': False}[str(_ste).strip().lower()]
    self._goal_vq_kw = dict(
        som=bool(cfg.som), ste=ste, agg=str(cfg.agg),
        codebook_scale=float(cfg.codebook_scale),
        commit_scale=float(cfg.commit_scale),
        som_scale=float(cfg.som_scale),
        lip_scale=float(cfg.lip_scale), lip_impl=str(cfg.lip_impl))

  def _goal_autoencoder_loss(self, repfeat, losses, metrics, B, T, training):
    """Goal VAE: reconstruct ``deter`` from a discrete skill code.

    Accumulates ``losses['goal_autoencoder']`` (reconstruction + adaptive KL to
    the uniform skill prior, plus the optional code/deter geometry-preservation
    term) and its diagnostics into ``metrics``.
    """
    # --- Goal Autoencoder ---
    deter_feat = sg(self.feat2deter(repfeat))
    if self.goal_ae_impl == 'vq':
      # One encoder pass shared by the loss and the diagnostics below.
      z_e = self.goal_enc.latent(deter_feat, 2)
      encoded_goal = self.goal_enc.code_from_latent(z_e)
      goal_base_loss, vq_mets = goal_ae.vq_goal_loss(
          self.goal_enc, self.goal_dec, deter_feat, 2, z_e=z_e,
          **self._goal_vq_kw)
      # ``vq/x`` -> ``goal/x`` so everything lands under the same log prefix as
      # the Director-arm goal metrics.
      metrics.update({f'goal/{k.split("/", 1)[1]}': v for k, v in vq_mets.items()})
      metrics['goal/total'] = goal_base_loss.mean()
      # Reconstruction only: from z_q, plus from z_e on the SOM arms.
      goal_rec_loss = vq_mets['vq/rec_q'] + vq_mets.get(
          'vq/rec_e', jnp.zeros((), f32))
      # goal/rec_* must stay the RECONSTRUCTION for continuity with the
      # Director arm, not the total: the VQ objective carries codebook,
      # commitment, neighborhood and Lipschitz terms too, and reporting their
      # sum here made goal/rec_mean read ~6e4 in the Lipschitz arms while the
      # reconstruction itself was ~7. The total is logged as
      # goal/total (and as loss/goal_autoencoder). ``goal_rec_loss_agg``, a
      # rec:kl balance knob, does not apply: the VQ reduction is
      # ``goal_vq.agg``.
    else:
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
      goal_base_loss = goal_rec_loss
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

    losses['goal_autoencoder'] = (
        goal_base_loss + self.config.goal_autoencoder_beta * goal_kl_loss)

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
    struct_apply = (self.goal_struct_weight > 0.0) or bool(
        getattr(self.config, 'goal_struct_adapt', False))
    # goal_struct_diag: compute the soft/hard correlation diagnostics below
    # (logged as train/goal/struct_corr[_hard]) even when the struct loss
    # term itself is off (goal_struct_weight=0, no adapt) -- e.g. pure-
    # Director baselines, where we still want to TRACK this geometry-
    # preservation property across training without optimizing against it.
    # Reuses the batch already sampled for the goal-VAE loss this step (no
    # extra rollout) -- same quantity as
    # experiments/goal_code_struct_corr/diag_goal_struct_corr.py's offline
    # measurement. Off by default: extra O((B*T)^2) pairwise-similarity work
    # every train step, opt-in only where the trajectory is wanted.
    struct_diag = struct_apply or bool(getattr(self.config, 'goal_struct_diag', False))
    if struct_diag:
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
      if struct_apply:
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
        metrics['goal/struct_loss'] = struct_loss
      # Diagnostic: Pearson correlation of the two geometries (should rise to ~1).
      # Soft (differentiable code probs) -- unchanged from before.
      sdf = (sd * offdiag).reshape((-1,))
      szf = (sz * offdiag).reshape((-1,))
      sdc = sdf - sdf.mean()
      szc = szf - szf.mean()
      corr = (sdc * szc).sum() / (
          jnp.sqrt((sdc * sdc).sum()) * jnp.sqrt((szc * szc).sum()) + 1e-12)
      metrics['goal/struct_corr'] = corr
      # Hard (argmax code the manager actually samples via REINFORCE) --
      # matches diag_goal_struct_corr.py's batch_corr exactly: fraction of
      # matching blocks (Hamming similarity) between each pair's hard codes.
      L, C = int(self.skill_shape[0]), int(self.skill_shape[-1])
      ids = jnp.argmax(probs.reshape((B * T, L, C)), axis=-1)      # (N, L)
      hamming = jnp.mean(ids[:, None, :] != ids[None, :, :], axis=-1)  # (N, N)
      sh = 1.0 - hamming
      shf = (sh * offdiag).reshape((-1,))
      shc = shf - shf.mean()
      corr_hard = (sdc * shc).sum() / (
          jnp.sqrt((sdc * sdc).sum()) * jnp.sqrt((shc * shc).sum()) + 1e-12)
      metrics['goal/struct_corr_hard'] = corr_hard
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
      return self._flat_ac_loss(
          obs, repfeat, tokens, dyn_entries, dyn_carry, losses, metrics,
          enc_carry, dec_carry, enc_entries, dec_entries, B, T, training)

    self._goal_autoencoder_loss(repfeat, losses, metrics, B, T, training)

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
    # Re-derive the manager policy with the SAME conditioning input the rollout used.
    # ``mgr_skills`` carry the POST-edit code Z_t; the manager conditioned on the
    # PRE-edit code (Z_{t-1}, with step 0 = the imagination seed).
    if self.mgr_cond_goalcode:
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
      # Block-pooled manager credit (Director ``abstract_traj`` on adaptive
      # boundaries): pool rewards/continuation over each realized duration segment
      # and downsample the manager AC timeline to the switch steps, matching
      # fixed-K credit assignment. ``mgr_switch`` marks the real decisions inside
      # the resulting static, forward-filled buffer.
      mgr_skills_eff = downsample_at_switch_mask(mgr_skills, switch_mask)
      # Which decision (if any) had its hold cut short by the horizon. Computed
      # BEFORE any relabeling, since relabeling rewrites the very field the
      # truncation test reads.
      trunc_mask = None
      if self.goal_duration_truncated_policy != 'keep':
        trunc_mask = truncated_last_decision_mask(
            mgr_skills_eff, switch_mask, self.goal_duration_min,
            self.goal_duration_max)
      if (self.goal_duration_relabel_truncated
          and self.goal_duration_truncated_policy == 'keep'):
        # Relabeling only affects the duration log-prob, which the drop modes
        # withhold anyway -- so the two are mutually exclusive by construction.
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
      if self.mgr_cond_goalcode:
        preedit_eff = {'goal_code': downsample_at_switch_mask(
            {'goal_code': pre_code}, switch_mask)['goal_code']}
        if self.goal_soft_reuse_adapt:
          preedit_eff['skill_probs'] = downsample_at_switch_mask(
              {'skill_probs': pre_probs}, switch_mask)['skill_probs']
        mgr_pol_inp = self._mgr_input(imgfeat_eff, preedit_eff)
      else:
        preedit_eff = None
        mgr_pol_inp = inp_eff
      if self.goal_switch_cost:
        # HiTS-style per-decision cost: subtract a fixed cost at each switch step.
        mgr_extr_rew = mgr_extr_rew - self.goal_switch_cost * mgr_switch
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
      if self.mgr_cond_goalcode:
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
    if self.goal_soft_reuse_adapt:
      # Direct (non-REINFORCE) implicit sparsity: overlap between this
      # decision's own plain softmax and the previous decision's, per block,
      # averaged over blocks -- an ordinary differentiable quantity (no
      # sampling, no decoder pass); backprops straight into both steps' own
      # logits through their softmax.
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
      # decision (matches the goal/implicit_sparsity_block exclusion above).
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
        mgr_dur_lagrange_adapter=(
            self.mgr_dur_lagrange_adapter if self.goal_duration_lagrange else None),
        duration_fixed=self.goal_duration_fixed > 0,
        switch_mask=mgr_switch,
        dur_reg_weight=float(getattr(self.config, 'goal_duration_reg', 0.0)),
        dur_reg_target=float(getattr(self.config, 'goal_duration_target', 8.0)),
        dur_min=self.goal_duration_min,
        trunc_mask=trunc_mask if self.variable_goal_length else None,
        trunc_policy=self.goal_duration_truncated_policy,
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
    K_split = worker_split_window(
        self.config.worker_split_traj, self.variable_goal_length,
        self.goal_duration_fixed, self.manager_sample_freq, H)
    if K_split:
      # Reshape the rollout into overlapping windows of length K+1. Each window uses
      # the goal decoded at its *start* state for all K+1 steps (incl. the shared
      # boundary), so the worker reward, value, and lambda-return bootstrap stay
      # within a single goal — no leak across goal switches (Director ``split_traj``).
      K = K_split
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
        # Block-pooled manager credit on the realized switch boundaries, mirroring
        # the imagination path. NOTE the switches are *counterfactual* — derived
        # from the current deterministic manager on replay states, not the behavior
        # policy that generated the data — which is acceptable for an on-policy
        # critic target.
        repl_skills_full = self._manager_skills_on_sequence(
            feat, deterministic=True)
        repl_skills_full.pop('countdown', None)
        repl_switch = self._switch_mask_from_skills(repl_skills_full)
        feat_down = downsample_at_switch_mask(feat, repl_switch)
        # Patch the stale forward-filled trailing slot to the window's TRUE
        # final state (fixed 2026-07-27 -- see patch_trailing_replay_state's
        # docstring): without this, the manager's replay-side value target
        # for the last real decision in almost every batch window bootstraps
        # off a repeated old state instead of the actual final replayed one.
        feat_down = patch_trailing_replay_state(feat_down, feat, repl_switch)
        inp_down = self.feat2tensor(feat_down)
        n_mgr = feat_down['deter'].shape[1]
        repl_mgr_extr_rew, repl_mgr_expl_rew, repl_mgr_cont, valid_mgr = (
            variable_block_director_tensors(
                repl_rew_full, repl_con_full, repl_expl_full,
                repl_switch, n_mgr, agg_mode=self.mgr_reward_agg))
        last_down = downsample_at_switch_mask(
            {'last': last.astype(f32)}, repl_switch)['last']
        last_down = patch_trailing_replay_state(
            last_down, last.astype(f32), repl_switch).astype(last.dtype)
        # Terminal flags are downsampled the same way as ``last`` above, NOT
        # derived from the pooled continuation. ``term`` is a BOOL array, so
        # the previous ``(1.0 - repl_mgr_cont).astype(term.dtype)`` cast a
        # continuation-complement of ~0.003 to ``True`` at essentially every
        # manager decision. ``lambda_return`` computes ``live = (1 - term)``,
        # so every decision read as terminal and the manager's replay-side
        # value target collapsed to the immediate pooled block reward with no
        # bootstrap at all (~24x too small in a typical window). That starved
        # the manager critic on every block-pooled run, and is fatal on
        # sparse-reward tasks where the bootstrap carries all of the signal.
        # See test_replay_terminal_flags_are_not_derived_from_continuation.
        term_down = downsample_at_switch_mask(
            {'term': term.astype(f32)}, repl_switch)['term']
        term_down = patch_trailing_replay_state(
            term_down, term.astype(f32), repl_switch).astype(term.dtype)
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
      if self.variable_goal_length:
        # Same packing as the imagination path: ``valid_mgr`` marks the real
        # decisions inside a static ``T``-wide buffer, so the loss must be
        # rescaled to a per-decision mean or the replay-side manager critic is
        # down-weighted by the padding ratio (9 real decisions in a 63-wide
        # buffer at batch_length=64/K=8) relative to fixed-K's exactly-sized
        # ``downsample_manager_states`` timeline. See
        # ``test_block_pooled_replay_manager_value_loss_matches_fixed_k``.
        weight_down = weight_down * valid_mgr * decision_mean_rescale(
            valid_mgr[:, :-1])
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
      # ``lambda_return`` drops the bootstrap column, so the decision mask for
      # the return tensors is ``valid_mgr`` without its last slot.
      rep_dec_mask = valid_mgr[:, :-1] if self.variable_goal_length else None
      voff_extr, vscale_extr = self.mgr_extr_valnorm(
          ret_extr, update=training, weights=rep_dec_mask)
      voff_expl, vscale_expl = self.mgr_expl_valnorm(
          ret_expl, update=training, weights=rep_dec_mask)

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
