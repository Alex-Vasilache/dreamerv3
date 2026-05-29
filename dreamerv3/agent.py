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
from embodied.jax import internal as jaxinternal
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


def aggregate_mgr_extr_rew(rew, con, k, without_zeros=False):
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
  agg = (r_blk * weights).mean(axis=-1)  # shape (B, n)

  rem = Tm - n * k
  if rem > 0:
    r_rem = r[:, n * k:]
    c_rem = c[:, n * k:]
    w = jnp.cumprod(c_rem, axis=-1)
    agg_rem = (r_rem * w).mean(-1, keepdims=True)  # shape (B,1)
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


def _tb_video_grid(video_bthwc):
  """(batch, time, H, W, C) uint8 -> (time, H, batch*W, C) for TensorBoard video."""
  rb, t, h, w, c = video_bthwc.shape
  return video_bthwc.transpose(1, 2, 0, 3, 4).reshape(t, h, rb * w, c)


def _vec_to_tb_rgb(vec_bt_d):
  """(B, T, D) float -> (B, T, H, W, 3) uint8 via min-max norm on flattened vector."""
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

    self.skill_shape = config.skill_shape
    skill_shape_t = tuple(int(x) for x in self.skill_shape)
    skill_classes = int(getattr(config, 'skill_classes', skill_shape_t[-1]))
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
      self.goal_enc = embodied.jax.MLPHead(
          self.goal_code_space, **config.goal_enc, name='goal_enc')
      self.goal_dec = embodied.jax.MLPHead(self.goal_shape, **config.goal_dec, name='goal_dec')
      self.goal_autoencoder_beta = config.goal_autoencoder_beta
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
      self.manager_pol = embodied.jax.MLPHead(
          self.goal_code_space, **config.manager_policy, name='manager_pol')
      self.manager_sample_freq = config.manager_sample_freq

    # Slow (EMA) behavioral priors for the PMPO reverse-KL term. ``pol_slow`` is
    # built in both modes when PMPO is on; ``manager_pol_slow`` is HRL-only.
    self.use_pmpo_actor = config.imag_loss.use_pmpo_actor
    if self.use_pmpo_actor:
      self.pol_slow = embodied.jax.SlowModel(
          embodied.jax.MLPHead(act_space, policy_outs, **config.policy, name='pol_slow'),
          source=self.pol, **config.slowvalue)
      if self.use_hrl:
        self.manager_pol_slow = embodied.jax.SlowModel(
            embodied.jax.MLPHead(self.goal_code_space, **config.manager_policy, name='manager_pol_slow'),
            source=self.manager_pol, **config.slowvalue)

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

      self.mgr_extr_retnorm = embodied.jax.Normalize(**config.retnorm, name='mgr_extr_retnorm')
      self.mgr_expl_retnorm = embodied.jax.Normalize(**config.retnorm, name='mgr_expl_retnorm')
      self.wkr_goal_retnorm = embodied.jax.Normalize(**config.retnorm, name='wkr_goal_retnorm')

      self.mgr_valnorm = embodied.jax.Normalize(**config.valnorm, name='mgr_valnorm')
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
    else:
      # Flat AC heads (v4-online): single value/critic over WM features.
      self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
      self.slowval = embodied.jax.SlowModel(
          embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
          source=self.val, **config.slowvalue)
      self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
      self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
      self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    # Modules updated by the single ``self.opt`` step in ``train``.
    if self.use_hrl:
      self.modules = [
          self.dyn,
          self.enc,
          self.dec,
          self.goal_enc,
          self.goal_dec,
          self.rew,
          self.con,
          self.manager_pol,
          self.pol,
          self.mgr_extr_val,
          self.mgr_expl_val,
          self.wkr_goal_val,
      ]
    else:
      self.modules = [
          self.dyn, self.enc, self.dec,
          self.rew, self.con,
          self.pol, self.val,
      ]
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

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
      if 'repval' in scales:
        repval_scale = scales.pop('repval')
        scales['repmgr_extr_value'] = repval_scale
        scales['repmgr_expl_value'] = repval_scale
        scales['repwkr_goal_value'] = repval_scale
    else:
      # Flat mode: keep ``policy``/``value``/``repval`` (default keys), drop HRL-only.
      scales.pop('goal_autoencoder', None)
      if not self.config.repval_loss:
        scales.pop('repval', None)
    self.scales = scales

    if self.config.use_rms_loss_norm:
      self.lossrms = {
          k: embodied.jax.RmsTracker(
              rate=self.config.loss_rms_rate, name=f'lossrms_{k.replace("/", "_")}')
          for k in self.scales}
    else:
      self.lossrms = None

  @property
  def policy_keys(self):
    # Regex for checkpoint / param groups synced to the actor process.
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
    mgr_skill = {'skill': jnp.zeros((batch_size, *skill_shape), f32)}
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
    B = jax.tree.leaves(enc)[0].shape[0]
    skill_shape = tuple(int(x) for x in self.skill_shape)
    mgr_skill = {'skill': jnp.zeros((B, *skill_shape), f32)}
    mgr_step = jnp.zeros((B,), i32)
    return enc, dyn, dec, prevact, mgr_skill, mgr_step

  def _pack_carry(self, enc, dyn, dec, prevact, mgr_skill, mgr_step):
    if not self.use_hrl:
      return (enc, dyn, dec, prevact)
    return (enc, dyn, dec, prevact, mgr_skill, mgr_step)

  def _feat_goal2tensor(self, x, y):
    """Concatenate WM features with goal; supports ``(B, D)`` and ``(B, T, D)`` goals."""
    deter = nn.cast(x['deter'])
    stoch = nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))
    if y.ndim == deter.ndim:
      goal = nn.cast(y)
    else:
      goal = nn.cast(y.reshape((*y.shape[:-2], -1)))
    return jnp.concatenate([deter, stoch, goal], -1)

  def _goal_from_skill(self, skill, bdims=1):
    """Decoder mode (MSE ``pred``) from a sampled manager / VAE skill."""
    s = skill['skill'] if isinstance(skill, dict) else skill
    return self.goal_dec(s, bdims).pred()

  def _manager_skill_step(self, feat, mgr_skill, mgr_step, reset):
    """Resample skill every ``manager_sample_freq`` steps (Director carry switch)."""
    K = max(1, int(self.manager_sample_freq))
    mgr_step = jnp.where(reset, 0, mgr_step)
    update = jnp.equal(mgr_step % K, 0)
    new_skill = sample(mgr_as_dict(
        self.manager_pol(self.feat2tensor(feat), bdims=1)))
    mgr_skill = skill_switch(update, new_skill, mgr_skill)
    # Manager skill must be stopped before decoding goal for worker.
    goal = sg(self._goal_from_skill(jax.tree.map(sg, mgr_skill)))
    mgr_step = mgr_step + 1
    return mgr_skill, goal, mgr_step

  def _goals_from_skills(self, skills, bdims=2):
    """Batch-decode skills to goal vectors (Director ``dec.mode()``)."""
    s = skills['skill'] if isinstance(skills, dict) else skills
    bshape = s.shape[:bdims]
    flat = s.reshape((-1, *s.shape[bdims:]))
    goals = self.goal_dec(flat, 1).pred()
    return goals.reshape(bshape + goals.shape[1:])

  def _wkr_goal_reward(self, goals, imgfeat):
    """Worker goal reward: ``cosine_max`` between ``goal`` and ``deter`` (Director)."""
    feat = sg(self.feat2deter(imgfeat))
    goal = sg(goals)
    cos = goal_reward_cosine_max(goal, feat)
    return imag_reward_pad(cos[:, 1:])

  def _mgr_extr_rew(self, rew, con, without_zeros=False):
    """Manager extrinsic reward: WM reward pooled over ``manager_sample_freq`` steps."""
    return aggregate_mgr_extr_rew(rew, con, self.manager_sample_freq, without_zeros)

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
    """Propose a goal vector from ``start`` state (Director ``propose_goal``)."""
    B = feat['deter'].shape[0]
    if impl == 'manager':
      skill = sample(mgr_as_dict(
          self.manager_pol(self.feat2tensor(feat), bdims=1)))
      return sg(self._goal_from_skill(skill))
    if impl == 'prior':
      logits = jnp.zeros((B,) + tuple(int(x) for x in self.skill_shape), f32)
      prior = outs.OneHot(logits, self._skill_prior_unimix)
      skill = {'skill': prior.sample(nj.seed())}
      return sg(self._goal_from_skill(skill))
    if impl == 'replay':
      deter = self.feat2deter(feat)
      perm = jax.random.permutation(nj.seed(), jnp.arange(B))
      target = deter[perm]
      encoded = self.goal_enc(target, 1)
      skill = sample(encoded)
      s = skill['skill'] if isinstance(skill, dict) else skill
      return sg(self.goal_dec(s, 1).pred())
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
      metrics[f'impl_{impl}/{key}'] = _tb_video_grid(video)
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
    """Imagine with manager skill resampled every ``manager_sample_freq`` steps."""
    K = max(1, int(self.manager_sample_freq))
    B = jax.tree.leaves(starts)[0].shape[0]
    skill_shape = tuple(int(x) for x in self.skill_shape)
    # Start with a dummy skill that will be replaced in first step
    mgr_skill = {'skill': jnp.zeros((B, *skill_shape), f32)}

    def body(carry, _):
      dyn_carry, mgr_skill, step_i = carry
      feat = dict(deter=dyn_carry['deter'], stoch=dyn_carry['stoch'])
      update = jnp.equal(step_i % K, 0)
      new_skill = sample(mgr_as_dict(
          self.manager_pol(self.feat2tensor(feat), 1)))
      mgr_skill = skill_switch(update, new_skill, mgr_skill)
      # Match skill to state: apply goal from mgr_skill *after* resampling.
      goal = sg(self._goal_from_skill(jax.tree.map(sg, mgr_skill)))
      act = sample(self.pol(self._feat_goal2tensor(feat, goal), 1))
      dyn_carry, (feat_next, act_out) = self.dyn.imagine(
          dyn_carry, act, 1, training, single=True)
      return (dyn_carry, mgr_skill, step_i + 1), (feat_next, act_out, mgr_skill)

    if H < 1:
      raise ValueError(f'imagination length must be >= 1, got {H}')
    unroll = H if self.dyn.unroll else 1
    # ``nj.scan(..., axis=1)`` requires ``xs`` with rank >= 2 (it swapaxes 0/1).
    # Match ``rssm.imagine``: empty ``xs``, explicit ``length``, step in carry.
    (last_dyn, last_mgr_skill, _), (imgfeat, imgact, img_skills) = nj.scan(
        body, (starts, mgr_skill, jnp.int32(0)), (), H,
        unroll=unroll, axis=1)

    # imgfeat is [s1...sH]. imgact is [a0...aH-1]. img_skills is [skill0...skillH-1].
    # We need to sample one more skill at last_dyn (sH) to align with imgfeat prefix starts (s0).
    feat_last = dict(deter=last_dyn['deter'], stoch=last_dyn['stoch'])
    update_last = jnp.equal(H % K, 0)
    new_skill_last = sample(mgr_as_dict(
        self.manager_pol(self.feat2tensor(feat_last), 1)))
    last_mgr_skill = skill_switch(update_last, new_skill_last, last_mgr_skill)

    img_skills = concat([img_skills, jax.tree.map(lambda x: x[:, None], last_mgr_skill)], 1)
    return imgfeat, imgact, img_skills

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
    pick = mode if deterministic else sample
    feat0 = jax.tree.map(lambda x: x[:, 0], repfeat)
    mgr_skill = pick(mgr_as_dict(
        self.manager_pol(self.feat2tensor(feat0), 1)))

    def body(mgr_skill, t):
      feat = jax.tree.map(lambda x: x[:, t], repfeat)
      update = jnp.equal(t % K, 0)
      new_skill = pick(mgr_as_dict(
          self.manager_pol(self.feat2tensor(feat), 1)))
      mgr_skill = skill_switch(update, new_skill, mgr_skill)
      return mgr_skill, mgr_skill

    B = repfeat['deter'].shape[0]
    if T <= 1:
      _, skill = body(mgr_skill, 0)
      skills = jax.tree.map(lambda x: x[:, None], skill)
    else:
      _, skills = nj.scan(body, mgr_skill, jnp.arange(T), axis=0)

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
      mgr_skill, goal, mgr_step = self._manager_skill_step(
          feat, mgr_skill, mgr_step, reset)
      policy = self.pol(self._feat_goal2tensor(feat, goal), bdims=1)
    else:
      policy = self.pol(self.feat2tensor(feat), bdims=1)
    act = sample(policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
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
    if self.use_pmpo_actor:
      self.pol_slow.update()
      if self.use_hrl:
        self.manager_pol_slow.update()
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
      K_cap = min(self.config.imag_last or T, T)
      if self.config.use_single_rollout:
        K_imag = 1
        K_repl = max(K_cap, 2) if self.config.repval_loss else K_cap
        K_repl = min(K_repl, T)
      else:
        K_imag = K_cap
        K_repl = K_cap
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
      policy_prior = (
          self.pol_slow(inp, 2) if self.use_pmpo_actor else None)
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
          policy_prior=policy_prior,
          **self.config.imag_loss)
      losses.update({k: v.mean(1).reshape((B, K_imag)) for k, v in los_flat.items()})
      metrics.update(mets)
      if self.config.repval_loss:
        feat = sg(repfeat, skip=self.config.repval_grad)
        last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
        boot = imgloss_out['ret'][:, 0].reshape(B, K_imag)
        if K_repl != K_imag:
          boot = jnp.broadcast_to(boot[:, -1:], (B, K_repl))
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
      if self.config.use_rms_loss_norm:
        losses = {
            k: v / sg(self.lossrms[k](v, training))
            for k, v in losses.items()}
        metrics.update({f'loss_rms/{k}': v.mean() for k, v in losses.items()})
      loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])
      carry = (enc_carry, dyn_carry, dec_carry)
      entries = (enc_entries, dyn_entries, dec_entries)
      aux_outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
      return loss, (carry, entries, aux_outs, metrics)

    # --- Goal Autoencoder ---
    deter_feat = sg(self.feat2deter(repfeat))
    encoded_goal = self.goal_enc(deter_feat, 2)
    skill = sample(encoded_goal)
    decoded_goal = self.goal_dec(
        skill['skill'] if isinstance(skill, dict) else skill, 2)
    # Reconstruction + KL vs uniform skill prior (Director: ``rec + kl_divergence(enc, prior)``).
    goal_rec_loss = decoded_goal.loss(sg(deter_feat))
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

    losses['goal_autoencoder'] = goal_rec_loss + goal_kl_loss
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
    # B,T = batch and time from replay. K_cap upper-bounds how many start states
    # we slice from the end of the sequence for imagination.
    # Single-rollout uses K_imag=1 but K_repl>=2 when repval_loss is on, else
    # lambda_return gets an empty term[:, 1:] slice and jnp.stack fails.
    K_cap = min(self.config.imag_last or T, T)
    if self.config.use_single_rollout:
      K_imag = 1
      K_repl = max(K_cap, 2) if self.config.repval_loss else K_cap
      K_repl = min(K_repl, T)
    else:
      K_imag = K_cap
      K_repl = K_cap
    H = self.config.imag_length  # imagined steps after the start state (H+1 states).
    starts = self.dyn.starts(dyn_entries, dyn_carry, K_imag)
    imgfeat, imgprevact, img_skills = self._imagine_with_manager(starts, H, training)
    # Prefix replay states to imagined chain so AC sees grounded first step.
    first = jax.tree.map(
        lambda x: x[:, -K_imag:].reshape((B * K_imag, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat, skip=self.config.ac_grads)], 1)
    mgr_skills = img_skills
    mgr_skills_downsampled = jax.tree.map(lambda s: s[:, ::self.manager_sample_freq], mgr_skills)
    last_feat = jax.tree.map(lambda x: x[:, -1], imgfeat)
    last_mgr_skill = jax.tree.map(lambda x: x[:, -1], mgr_skills)
    last_goal = sg(self._goal_from_skill(jax.tree.map(sg, last_mgr_skill)))
    lastact = sample(self.pol(self._feat_goal2tensor(last_feat, last_goal), 1))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K_imag, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K_imag, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    inp_downsampled = self.feat2tensor(jax.tree.map(lambda x: x[:, ::self.manager_sample_freq], imgfeat))
    # Detach manager-produced goals from worker actor/critic.
    goals = sg(self._goals_from_skills(jax.tree.map(sg, mgr_skills), bdims=2))
    mgr_policy = mgr_as_dict(self.manager_pol(inp_downsampled, 2))
    mgr_policy_prior = (
        mgr_as_dict(self.manager_pol_slow(inp_downsampled, 2))
        if self.use_pmpo_actor else None)
    con = self.con(inp, 2).prob(1)
    mgr_cont = self._mgr_cont(con, without_zeros=True)
    rew_step = sg(self.rew(inp, 2).pred())
    mgr_extr_rew = self._mgr_extr_rew(rew_step, con, without_zeros=True)
    mgr_extr_rew = imag_reward_pad(mgr_extr_rew)
    expl_step = sg(self._mgr_expl_reward(imgfeat))
    mgr_expl_rew = self._mgr_extr_rew(expl_step, con, without_zeros=True)
    mgr_expl_rew = imag_reward_pad(mgr_expl_rew)

    kwargs_mgr = {**self.config.imag_loss}
    kwargs_mgr.update(
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        mgr_expl_weight=self.mgr_expl_weight,
        actent=self.config.manager_actent,
        slowtar=self.config.manager_slowtar)

    los_mgr, imgloss_mgr_out, mets_mgr = imag_loss_mgr(
        mgr_skills_downsampled,
        mgr_extr_rew,
        mgr_expl_rew,
        mgr_cont,
        mgr_policy,
        self.mgr_extr_val(inp_downsampled, 2),
        self.mgr_extr_slowval(inp_downsampled, 2),
        self.mgr_expl_val(inp_downsampled, 2),
        self.mgr_expl_slowval(inp_downsampled, 2),
        self.mgr_extr_retnorm,
        self.mgr_expl_retnorm,
        self.mgr_valnorm,
        self.mgr_advnorm,
        manager_policy_prior=mgr_policy_prior,
        mgr_actent_adapter=self.mgr_actent,
        mgr_actent_perdim=self.manager_actent_perdim,
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
    if self.config.worker_split_traj and H >= K and H % K == 0:
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
      win_feat_goal = self._feat_goal2tensor(win_feat, win_goal)
      win_goal_rew = self._wkr_goal_reward(win_goal, win_feat)
      kwargs_wkr.update(skill_window=0)                         # each window is its own segment
      win_policy_prior = (
          self.pol_slow(win_feat_goal, 2) if self.use_pmpo_actor else None)
      los_wkr, imgloss_wkr_out, mets_wkr = imag_loss_wkr(
          win_act, win_goal_rew, win_con,
          self.pol(win_feat_goal, 2),
          self.wkr_goal_val(win_feat_goal, 2),
          self.wkr_goal_slowval(win_feat_goal, 2),
          self.wkr_goal_retnorm, self.wkr_goal_valnorm, self.wkr_goal_advnorm,
          policy_prior=win_policy_prior,
          **kwargs_wkr)
      losses.update({k: v.mean(1).reshape((B, -1)) for k, v in los_wkr.items()})
      # Repval bootstrap: first window's return at the imagination start, per start state.
      boot_goal_full = imgloss_wkr_out['wkr_goal_ret'].reshape(
          M, n_win, -1)[:, 0, 0].reshape(B, K_imag)
    else:
      # Fallback (e.g. H not a multiple of K): dense rollout with the lambda-return
      # reset at window boundaries via the ``skill_window`` mask.
      feat_goal = self._feat_goal2tensor(imgfeat, goals)
      wkr_goal_rew = self._wkr_goal_reward(goals, imgfeat)
      kwargs_wkr.update(skill_window=K)
      feat_policy_prior = (
          self.pol_slow(feat_goal, 2) if self.use_pmpo_actor else None)
      los_wkr, imgloss_wkr_out, mets_wkr = imag_loss_wkr(
          imgact, wkr_goal_rew, con,
          self.pol(feat_goal, 2),
          self.wkr_goal_val(feat_goal, 2),
          self.wkr_goal_slowval(feat_goal, 2),
          self.wkr_goal_retnorm, self.wkr_goal_valnorm, self.wkr_goal_advnorm,
          policy_prior=feat_policy_prior,
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

      # --- 1. Downsampled Replay sequence for Manager ---
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

      # Compute aggregated continuation and rewards on full sequence
      repl_con_full = self.con(self.feat2tensor(feat), 2).prob(1)
      repl_mgr_cont = self._mgr_cont(repl_con_full, without_zeros=True)
      repl_mgr_extr_rew = imag_reward_pad(self._mgr_extr_rew(self.rew(self.feat2tensor(feat), 2).pred(), repl_con_full, without_zeros=True))
      repl_expl_step = self._mgr_expl_reward(feat)
      repl_mgr_expl_rew = imag_reward_pad(self._mgr_extr_rew(repl_expl_step, repl_con_full, without_zeros=True))

      # --- 2. Dense Replay sequence for Worker ---
      feat_wkr, last_wkr, term_wkr, boot_goal_wkr = jax.tree.map(
          lambda x: x[:, -K_repl:],
          (feat, last, term, boot_goal))
      inp_wkr = self.feat2tensor(feat_wkr)
      repl_skills = jax.tree.map(
          lambda x: x[:, -K_repl:],
          self._manager_skills_on_sequence(feat, deterministic=True))
      # Detach manager goals in replay value path.
      repl_goals = sg(self._goals_from_skills(jax.tree.map(sg, repl_skills), bdims=2))
      feat_goal_wkr = self._feat_goal2tensor(feat_wkr, repl_goals)
      repl_wkr_goal_rew = self._wkr_goal_reward(repl_goals, feat_wkr)

      # --- 3. Compute Value Losses ---
      # Manager Replay Value Loss (predicts combined return)
      kwargs_repmgr = {**self.config.repl_loss}
      kwargs_repmgr.update(
          update=training,
          horizon=self.config.horizon,
          value_head='mgr')

      # For manager replay, we need to compute combined return of normalized signals
      voff_extr, vscale_extr = self.mgr_extr_retnorm.stats()
      voff_expl, vscale_expl = self.mgr_expl_retnorm.stats()

      # Manager Trajectory is short
      weight_down = f32(~last_down)
      disc = 1 - 1 / self.config.horizon
      lam = 0.95 # matches repl_loss default
      boot_extr_down = jnp.broadcast_to(boot_extr[:, -1:], repl_mgr_extr_rew.shape)
      boot_expl_down = jnp.broadcast_to(boot_expl[:, -1:], repl_mgr_expl_rew.shape)
      ret_extr = lambda_return(last_down, term_down, repl_mgr_extr_rew, jnp.zeros_like(repl_mgr_extr_rew), boot_extr_down, disc, lam)
      ret_expl = lambda_return(last_down, term_down, repl_mgr_expl_rew, jnp.zeros_like(repl_mgr_expl_rew), boot_expl_down, disc, lam)

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
    if self.config.use_rms_loss_norm:
      losses = {
          k: v / sg(self.lossrms[k](v, training))
          for k, v in losses.items()}
      metrics.update({f'loss_rms/{k}': v.mean() for k, v in losses.items()})
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
      metrics[f'openloop/{key}'] = _tb_video_grid(video)

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
    pred_deter = nn.cast(self.goal_dec(
        skill_s['skill'] if isinstance(skill_s, dict) else skill_s, 2).pred())
    feat_goal = self._feat_from_goal(pred_deter)
    _, _, recons_goal = self.dec(dec_carry, feat_goal, reset_s, training=False)

    # Manager-proposed goals over the report sequence (K-step skill hold).
    mgr_skills = self._manager_skills_on_sequence(rep)
    mgr_goals = sg(self._goals_from_skills(mgr_skills, bdims=2))
    mgr_goal_feat = self._feat_from_goal(mgr_goals)
    _, _, recons_mgr = self.dec(dec_carry, mgr_goal_feat, reset_s, training=False)

    # Optional dense vec→RGB visualisations of latent vectors and skills.
    # Off by default — they are debug-grade and slow down the video pipeline.
    if bool(getattr(self.config, 'report_vec_viz', False)):
      metrics['goal/deter_feat'] = _tb_video_grid(_vec_to_tb_rgb(deter_feat))
      metrics['goal/decoded_deter'] = _tb_video_grid(_vec_to_tb_rgb(pred_deter))
      sk = skill_s['skill'] if isinstance(skill_s, dict) else skill_s
      if sk.ndim == 3:
        metrics['goal/skill_sampled'] = _tb_video_grid(_vec_to_tb_rgb(sk))
      else:
        metrics['goal/skill_sampled'] = _tb_video_grid(
            jnp.repeat((sk * 255).astype(jnp.uint8)[..., None], 3, axis=-1))
      m_sk = mgr_skills['skill']
      if m_sk.ndim == 3:
        metrics['goal/mgr_skill'] = _tb_video_grid(_vec_to_tb_rgb(m_sk))
      else:
        metrics['goal/mgr_skill'] = _tb_video_grid(
            jnp.repeat((m_sk * 255).astype(jnp.uint8)[..., None], 3, axis=-1))
      metrics['goal/mgr_proposed_deter'] = _tb_video_grid(_vec_to_tb_rgb(mgr_goals))

    # VAE / manager goal reconstruction panels. ``goal/image_{key}`` (true frames
    # only) was a duplicate of the leftmost column here — dropped.
    for key in self.dec.imgkeys:
      true = obs[key][:RB, :T]
      pred_g = jnp.clip(recons_goal[key].pred() * 255, 0, 255).astype(jnp.uint8)
      pred_m = jnp.clip(recons_mgr[key].pred() * 255, 0, 255).astype(jnp.uint8)
      metrics[f'goal/recon_{key}'] = _tb_video_grid(
          jnp.concatenate([true, pred_g, ((i32(pred_g) - i32(true) + 255) // 2).astype(np.uint8)], 2))
      metrics[f'goal/mgr_recon_{key}'] = _tb_video_grid(
          jnp.concatenate([true, pred_m, ((i32(pred_m) - i32(true) + 255) // 2).astype(np.uint8)], 2))

    # Director-style: [initial | proposed goal | worker rollout] per proposal mode.
    # Defaults to just ``manager`` (config ``report_impl_videos``) — dropping
    # ``prior`` and ``replay`` halves the GIF count per report by default.
    impls = getattr(self.config, 'report_impl_videos', ['manager'])
    for impl in tuple(impls):
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


def pmpo_global_sum(x):
  """Sum over all local (batch x time) elements, all-reduced across data axes.

  PMPO partitions imagined states into D+/D- over the *full* (batch x time)
  population (DreamerV4 eq. 11), so the |D+|/|D-| counts must pool globally, not
  per trajectory row. For multi-device, all-reduce over the data axes.
  """
  s = jnp.sum(x)
  axes = jaxinternal.get_data_axes()
  if axes:
    s = jax.lax.psum(s, axes)
  return s


def policy_behavior_kl(policy, prior=None):
  """KL(policy || prior) reduced over skill axes to ``(batch, time)``.

  ``prior`` is a separate stop-grad behavioral policy (e.g. a slow/EMA copy).
  If ``prior`` is None, falls back to the same-forward-pass copy (KL == 0).
  """
  total = None
  for k, v in policy.items():
    inner = _head_inner(v)
    pri = _head_inner(prior[k]) if prior is not None else inner
    if isinstance(inner, (outs.OneHot, outs.Categorical)):
      logits = inner.dist.logits if isinstance(inner, outs.OneHot) else inner.logits
      prilogits = pri.dist.logits if isinstance(pri, outs.OneHot) else pri.logits
      logp = jax.nn.log_softmax(logits, -1)
      p = jax.nn.softmax(logits, -1)
      logpref = jax.nn.log_softmax(sg(prilogits), -1)
      kl = (p * (logp - logpref)).sum(-1)
    elif isinstance(inner, outs.Normal):
      ref = outs.Normal(sg(pri.mean), sg(pri.stddev))
      kl = inner.kl(ref)
    else:
      raise NotImplementedError(type(inner))
    # Reduce skill axes only; do not use ``Agg.axes`` on tensors that include time.
    while kl.ndim > 2:
      kl = kl.sum(-1)
    total = kl if total is None else total + kl
  return total


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
    use_pmpo_actor=False,
    pmpo_beta=0.3,
    pmpo_alpha=0.5,
    skill_window=0,
    policy_prior=None,
):
  """Worker actor-critic losses on imagined trajectories."""
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
  # Reset the lambda-return at goal-window boundaries (every ``skill_window`` steps)
  # so V(s, g) bootstraps within its own skill window instead of across goal switches
  # (Director ``split_traj``). ``skill_window <= 1`` keeps the full-horizon return.
  if skill_window and skill_window > 1:
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

  if use_pmpo_actor:
    # PMPO (DreamerV4 eq. 11) as a global mean over all imagined states (Bug C):
    #   (1-α) mean_{D-} ln π - α mean_{D+} ln π + β mean KL(π || prior).
    wkr_goal_adv_raw = wkr_goal_ret - wkr_goal_tarval[:, :-1]
    wkr_goal_pos = (wkr_goal_adv_raw >= 0).astype(f32)
    wkr_goal_neg = (wkr_goal_adv_raw < 0).astype(f32)
    wkr_n_tot = jnp.maximum(pmpo_global_sum(jnp.ones_like(wkr_goal_pos)), 1.0)
    wkr_goal_den_p = jnp.maximum(pmpo_global_sum(wkr_goal_pos), 1.0)
    wkr_goal_den_n = jnp.maximum(pmpo_global_sum(wkr_goal_neg), 1.0)

    wkr_goal_pos_coeff = pmpo_alpha * (wkr_n_tot / wkr_goal_den_p) * wkr_goal_pos
    wkr_goal_neg_coeff = (1.0 - pmpo_alpha) * (wkr_n_tot / wkr_goal_den_n) * wkr_goal_neg

    wkr_kl_t = policy_time_slice(policy_behavior_kl(policy, policy_prior))
    wkr_goal_policy_loss = (wkr_goal_neg_coeff - wkr_goal_pos_coeff) * wkr_logpi + pmpo_beta * wkr_kl_t

    metrics['wkr_goal_kl_behavior'] = wkr_kl_t.mean()
    metrics['wkr_pmpo_frac_pos'] = wkr_goal_den_p / wkr_n_tot
  else:
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
    mgr_valnorm,
    mgr_advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
    use_pmpo_actor=False,
    pmpo_beta=0.3,
    pmpo_alpha=0.5,
    mgr_expl_weight=0.1,
    manager_policy_prior=None,
    mgr_actent_adapter=None,
    mgr_actent_perdim=True,
):
  """Manager actor-critic losses on imagined trajectories."""
  losses = {}
  metrics = {}

  # Separate extrinsic and exploratory critic predictions.
  mgr_voffset, mgr_vscale = mgr_valnorm.stats()

  mgr_extr_val = mgr_extr_value.pred() * mgr_vscale + mgr_voffset
  mgr_extr_slowval = mgr_extr_slowvalue.pred() * mgr_vscale + mgr_voffset
  mgr_extr_tarval = mgr_extr_slowval if slowtar else mgr_extr_val

  mgr_expl_val = mgr_expl_value.pred() * mgr_vscale + mgr_voffset
  mgr_expl_slowval = mgr_expl_slowvalue.pred() * mgr_vscale + mgr_voffset
  mgr_expl_tarval = mgr_expl_slowval if slowtar else mgr_expl_val

  # Discount per step: either γ or finite-horizon (1 - 1/horizon) when not contdisc.
  disc = 1 if contdisc else 1 - 1 / horizon
  # Discounted continuation weights from predicted continue probs ``con``.
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con

  mgr_extr_ret = lambda_return(
      last, term, mgr_extr_rew, mgr_extr_tarval, jnp.zeros_like(mgr_extr_rew), disc, lam)
  mgr_expl_ret = lambda_return(
      last, term, mgr_expl_rew, mgr_expl_tarval, jnp.zeros_like(mgr_expl_rew), disc, lam)

  voff_extr, vscale_extr = mgr_extr_retnorm(mgr_extr_ret, update)
  voff_expl, vscale_expl = mgr_expl_retnorm(mgr_expl_ret, update)

  mgr_extr_ret_normed = (mgr_extr_ret - voff_extr) / vscale_extr
  mgr_expl_ret_normed = (mgr_expl_ret - voff_expl) / vscale_expl

  mgr_total_ret = mgr_extr_ret_normed + mgr_expl_weight * mgr_expl_ret_normed

  # Normalize tarvals to return space for consistent advantage baseline.
  mgr_extr_tarval_normed = (mgr_extr_tarval - voff_extr) / vscale_extr
  mgr_expl_tarval_normed = (mgr_expl_tarval - voff_expl) / vscale_expl
  mgr_extr_adv = mgr_extr_ret_normed - mgr_extr_tarval_normed[:, :-1]
  mgr_expl_adv = mgr_expl_ret_normed - mgr_expl_tarval_normed[:, :-1]
  mgr_adv = mgr_extr_adv + mgr_expl_weight * mgr_expl_adv
  mgr_aoffset, mgr_ascale = mgr_advnorm(mgr_adv, update)
  mgr_adv_normed = (mgr_adv - mgr_aoffset) / mgr_ascale

  skill_events = align_skill_events(skills, manager_policy)
  mgr_logpi = sum([
      head_logp_time(v, skill_events[k]) for k, v in manager_policy.items()])
  mgr_ents = {k: head_entropy_time(v) for k, v in manager_policy.items()}

  # Director-style adaptive normalized entropy regularizer. Computed for both
  # PMPO and REINFORCE branches so the manager always has an entropy floor.
  mgr_ent_loss_bt = jnp.zeros_like(mgr_logpi)
  mgr_actent_mets = {}
  if mgr_actent_adapter is not None:
    ent_loss_terms = []
    for k, head in manager_policy.items():
      inner = _head_inner(head)
      if not hasattr(inner, 'minent') or not hasattr(inner, 'maxent'):
        continue
      ent_perdim = head_entropy_perdim_time(head)  # (B, T-1, ...)
      L = ent_perdim.shape[-1] if ent_perdim.ndim > 2 else 1
      lo = inner.minent / L
      hi = inner.maxent / L
      denom = jnp.maximum(hi - lo, 1e-8)
      ent_norm = (ent_perdim - lo) / denom
      if mgr_actent_perdim and ent_perdim.ndim > 2:
        loss_perdim, mets = mgr_actent_adapter(ent_norm, update=update)
        ent_loss_terms.append(loss_perdim.sum(-1))
      else:
        ent_scalar = ent_norm.mean(-1) if ent_norm.ndim > 2 else ent_norm
        loss_scalar, mets = mgr_actent_adapter(ent_scalar, update=update)
        ent_loss_terms.append(loss_scalar)
      mgr_actent_mets.update(
          {f'mgr_actent_{k}_{mk}': mv for mk, mv in mets.items()})
      mgr_actent_mets[f'mgr_ent_norm_{k}_mean'] = ent_norm.mean()
    if ent_loss_terms:
      mgr_ent_loss_bt = sum(ent_loss_terms)

  w = sg(weight[:, :-1])

  if use_pmpo_actor:
    # PMPO target formula:
    # (1-α)/|D-| * Σ_{D-} ln π - α/|D+| * Σ_{D+} ln π + β * mean(KL(π||prior)).
    mgr_adv_raw = mgr_total_ret - (mgr_extr_tarval_normed + mgr_expl_weight * mgr_expl_tarval_normed)[:, :-1]

    mgr_pos = (mgr_adv_raw >= 0).astype(f32)
    mgr_neg = (mgr_adv_raw < 0).astype(f32)

    mgr_n_tot = jnp.maximum(pmpo_global_sum(jnp.ones_like(mgr_pos)), 1.0)
    mgr_den_p = jnp.maximum(pmpo_global_sum(mgr_pos), 1.0)
    mgr_den_n = jnp.maximum(pmpo_global_sum(mgr_neg), 1.0)

    mgr_pos_coeff = pmpo_alpha * (mgr_n_tot / mgr_den_p) * mgr_pos
    mgr_neg_coeff = (1.0 - pmpo_alpha) * (mgr_n_tot / mgr_den_n) * mgr_neg

    mgr_kl_t = policy_time_slice(policy_behavior_kl(manager_policy, manager_policy_prior))
    losses['mgr_policy'] = (
        (mgr_neg_coeff - mgr_pos_coeff) * mgr_logpi
        + pmpo_beta * mgr_kl_t
        + w * mgr_ent_loss_bt)

    metrics['mgr_kl_behavior'] = mgr_kl_t.mean()
    metrics['mgr_pmpo_frac_pos'] = mgr_den_p / mgr_n_tot
  else:
    if mgr_actent_adapter is not None:
      # Adaptive normalized actent already in mgr_ent_loss_bt; drop fixed scalar.
      losses['mgr_policy'] = w * (-mgr_logpi * sg(mgr_adv_normed) + mgr_ent_loss_bt)
    else:
      losses['mgr_policy'] = w * -(
          mgr_logpi * sg(mgr_adv_normed) + actent * sum(mgr_ents.values()))

  metrics['mgr_policy_loss'] = losses['mgr_policy'].mean()
  metrics['mgr_ent_loss'] = mgr_ent_loss_bt.mean()
  metrics.update(mgr_actent_mets)
  metrics['mgr_extr_rew'] = mgr_extr_rew.mean()
  nz = jnp.maximum((jnp.abs(mgr_extr_rew[:, 1:]) > 0).sum(), 1)
  metrics['mgr_extr_rew_block'] = mgr_extr_rew[:, 1:].sum() / nz
  metrics['mgr_expl_rew'] = mgr_expl_rew.mean()

  # Separate NLL losses for each critic.
  mgr_extr_tar_padded = jnp.concatenate([mgr_extr_ret_normed, 0 * mgr_extr_ret_normed[:, -1:]], 1)
  losses['mgr_extr_value'] = sg(weight[:, :-1]) * (
      mgr_extr_value.loss(sg(mgr_extr_tar_padded)) +
      slowreg * mgr_extr_value.loss(sg(mgr_extr_slowvalue.pred())))[:, :-1]

  mgr_expl_tar_padded = jnp.concatenate([mgr_expl_ret_normed, 0 * mgr_expl_ret_normed[:, -1:]], 1)
  losses['mgr_expl_value'] = sg(weight[:, :-1]) * (
      mgr_expl_value.loss(sg(mgr_expl_tar_padded)) +
      slowreg * mgr_expl_value.loss(sg(mgr_expl_slowvalue.pred())))[:, :-1]

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

  for k in skills:
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
    use_pmpo_actor=False,
    pmpo_beta=0.3,
    pmpo_alpha=0.5,
    policy_prior=None,
):
  """Flat actor-critic loss (v4-online), used when ``use_hrl=False``."""
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
  if use_pmpo_actor:
    adv_raw = ret - tarval[:, :-1]
    pos = (adv_raw >= 0).astype(f32)
    neg = (adv_raw < 0).astype(f32)
    n_tot = jnp.maximum(pmpo_global_sum(jnp.ones_like(pos)), 1.0)
    den_p = jnp.maximum(pmpo_global_sum(pos), 1.0)
    den_n = jnp.maximum(pmpo_global_sum(neg), 1.0)
    pos_coeff = pmpo_alpha * (n_tot / den_p) * pos
    neg_coeff = (1.0 - pmpo_alpha) * (n_tot / den_n) * neg
    kl_t = policy_time_slice(policy_behavior_kl(policy, policy_prior))
    policy_loss = (neg_coeff - pos_coeff) * logpi + pmpo_beta * kl_t
    metrics['kl_behavior'] = kl_t.mean()
    metrics['pmpo_frac_pos'] = den_p / n_tot
  else:
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
