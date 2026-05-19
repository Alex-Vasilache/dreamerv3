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
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
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

    # x is WM feat, y is goal
    self.feat_goal2tensor = lambda x, y: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1))),
        nn.cast(y.reshape((*y.shape[:-2], -1))),
    ], -1)

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

    self.manager_pol = embodied.jax.MLPHead(
        self.goal_code_space, **config.manager_policy, name='manager_pol')
    self.manager_sample_freq = config.manager_sample_freq

    # Value and EMA target for bootstrapping / slow regularizer in ``imag_loss``.
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

    self.mgr_extr_valnorm = embodied.jax.Normalize(**config.valnorm, name='mgr_extr_valnorm')
    self.mgr_expl_valnorm = embodied.jax.Normalize(**config.valnorm, name='mgr_expl_valnorm')
    self.wkr_goal_valnorm = embodied.jax.Normalize(**config.valnorm, name='wkr_goal_valnorm')

    self.mgr_extr_advnorm = embodied.jax.Normalize(**config.advnorm, name='mgr_extr_advnorm')
    self.mgr_expl_advnorm = embodied.jax.Normalize(**config.advnorm, name='mgr_expl_advnorm')
    self.wkr_goal_advnorm = embodied.jax.Normalize(**config.advnorm, name='wkr_goal_advnorm')

    self.mgr_expl_weight = config.mgr_expl_weight

    # Modules updated by the single ``self.opt`` step in ``train``.
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
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    # One ``rec`` scale is expanded to every reconstruction key in ``dec_space``.
    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
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
    # Regex for checkpoint / param groups treated as policy (not value-only).
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
    """RNN carries for enc/dyn/dec plus zero initial previous action."""
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    """Same carry shape as policy (training reuses the same state layout)."""
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    """Same carry shape as policy for ``report`` rollouts."""
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    """One env step: encode obs, RSSM observe, sample policy action, update carry."""
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)

    manager_policy = self.manager_pol(self.feat2tensor(feat), bdims=1)
    manager_act = sample(manager_policy) # TODO change to only sample every K steps, otherwise keep previous sampled manager_act
    goal = self.goal_dec(manager_act, 2)

    # Policy reads representation only (decoder used for logging / replay context).
    policy = self.pol(self.feat_goal2tensor(feat, sg(goal)), bdims=1)
    act = sample(policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    """Optimizer step on ``loss``; may attach replay context writes for next batch."""
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    self.mgr_extr_slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
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

    # --- Goal Autoencoder ---
    deter_feat = sg(self.feat2deter(repfeat))
    encoded_goal = self.goal_enc(deter_feat, 2)
    skill = sample(encoded_goal)
    decoded_goal = self.goal_dec(skill, 2)
    # Reconstruction + KL vs uniform skill prior (Director: ``rec + kl_divergence(enc, prior)``).
    goal_rec_loss = decoded_goal.loss(sg(deter_feat))
    goal_dist = encoded_goal.output
    skill_prior = outs.OneHot(
        jnp.zeros_like(goal_dist.dist.logits), self._skill_prior_unimix)
    inner_kl = goal_dist.kl(skill_prior)
    # OneHot.kl on [..., L, C] logits already sums classes -> [..., L]; sum L -> [B, T].
    goal_kl_bt = inner_kl.sum(-1) if self._skill_factorized else inner_kl
    goal_kl_loss = (
        f32(self.config.goal_autoencoder_beta) * goal_kl_bt
        if self.config.goal_kl
        else jnp.zeros((B, T), f32))

    losses['goal_autoencoder'] = goal_rec_loss + goal_kl_loss
    # Logged as ``train/goal/*`` when the train loop aggregates with prefix ``train``.
    ent = goal_dist.dist.entropy()
    goal_ent_bt = ent.sum(-1) if self._skill_factorized else ent
    metrics.update({
        'goal/rec_mean': goal_rec_loss.mean(),
        'goal/rec_std': goal_rec_loss.std(),
        'goal/kl_mean': goal_kl_loss.mean(),
        'goal/kl_raw_mean': goal_kl_bt.mean(),
        'goal/kl_std': goal_kl_loss.std(),
        'goal/entropy_mean': goal_ent_bt.mean(),
        'goal/entropy_std': goal_ent_bt.std(),
    })

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
    # TODO check in director if we sample here or use mode
    manager_policyfn = lambda feat: self.goal_dec(sample(self.manager_pol(self.feat2tensor(feat), 1)), 2) # TODO change to only sample every K steps, otherwise keep previous sampled manager_act
    
    policyfn = lambda feat: sample(self.pol(self.feat_goal2tensor(feat, sg(manager_policyfn(feat))), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    # Prefix replay states to imagined chain so AC sees grounded first step.
    first = jax.tree.map(
        lambda x: x[:, -K_imag:].reshape((B * K_imag, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K_imag, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K_imag, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    feat_goal = self.feat_goal2tensor(imgfeat, sg(self.goal_dec(sample(self.manager_pol(inp, 1)), 2))) # TODO check in director if we sample here or use mode
    los, imgloss_out, mets = imag_loss(
        imgact,
        # TODO add mgr policy actions (skills)
        self.rew(inp, 2).pred(),
        self.con(inp, 2).prob(1),
        self.manager_pol(inp, 2),
        self.pol(feat_goal, 2),
        self.mgr_extr_val(inp, 2),
        self.mgr_extr_slowval(inp, 2),
        self.mgr_expl_val(inp, 2),
        self.wkr_goal_val(feat_goal, 2),
        self.wkr_goal_slowval(feat_goal, 2),
        self.mgr_extr_retnorm, self.mgr_expl_retnorm, self.wkr_goal_retnorm, 
        self.mgr_extr_valnorm, self.mgr_expl_valnorm, self.wkr_goal_valnorm, 
        self.mgr_extr_advnorm, self.mgr_expl_advnorm, self.wkr_goal_advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        mgr_expl_weight=self.mgr_expl_weight,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K_imag)) for k, v in los.items()})
    metrics.update(mets)

    # --- Optional replay value loss (tail of real sequence + imag bootstrap) ---
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K_imag)
      if K_repl != K_imag:
        boot = jnp.broadcast_to(boot[:, -1:], (B, K_repl))
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K_repl:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      feat_goal = self.feat_goal2tensor(feat, sg(self.goal_dec(sample(self.manager_pol(inp, 1)), 2))) # TODO check in director if we sample here or use mode
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.mgr_extr_val(inp, 2),
          self.mgr_extr_slowval(inp, 2),
          self.mgr_extr_valnorm,
          update=training,
          horizon=self.config.horizon,
          value_head='mgr_extr',
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.mgr_expl_val(inp, 2),
          self.mgr_expl_slowval(inp, 2),
          self.mgr_expl_valnorm,
          update=training,
          horizon=self.config.horizon,
          value_head='mgr_expl',
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.wkr_goal_val(feat_goal, 2),
          self.wkr_goal_slowval(feat_goal, 2),
          self.wkr_goal_valnorm,
          update=training,
          horizon=self.config.horizon,
          value_head='wkr_goal',
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

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
    """Eval-style forward, optional grad norms, open-loop video, and goal panels.

    ``report/goal/*`` TensorBoard videos: ``deter_feat``, ``decoded_deter``, sampled
    skill grid, per-image truth, and ``[truth | decoder(goal) | error]`` where the
    decoder uses ``deter=decoded_goal`` and prior ``stoch`` from ``deter`` (Director
    ``get_stoch`` pattern in ``hierarchy.py``).
    """
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    metrics.update(mets)

    # Per-loss-key gradient norm (expensive: extra backward per key).
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop: first half from real tokens; second half imagined from that state.
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Stack [truth | pred | abs error] with a time border for TensorBoard video.
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
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

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    # Goal autoencoder + image decode (Director: ``decoder({deter: goal, stoch: rssm.get_stoch(goal)})``).
    rep = jax.tree.map(lambda x: x[:RB, :T], outs['repfeat'])
    reset_s = obs['is_first'][:RB, :T]
    deter_feat = sg(self.feat2deter(rep))
    encoded_goal = self.goal_enc(deter_feat, 2)
    skill_s = sample(encoded_goal)
    dec_goal = self.goal_dec(skill_s, 2)
    pred_deter = nn.cast(dec_goal.pred())
    logit_p = self.dyn._prior(pred_deter)
    stoch_p = nn.cast(self.dyn._dist(logit_p).pred())
    feat_goal = {**rep, 'deter': pred_deter, 'stoch': stoch_p, 'logit': logit_p}
    _, _, recons_goal = self.dec(dec_carry, feat_goal, reset_s, training=False)

    metrics['goal/deter_feat'] = _tb_video_grid(_vec_to_tb_rgb(deter_feat))
    metrics['goal/decoded_deter'] = _tb_video_grid(_vec_to_tb_rgb(pred_deter))
    # (RB, T, L, C) sparse one-hot matrix -> 8×8 RGB panel per frame.
    sk_u8 = (skill_s * 255).astype(jnp.uint8)
    sk_rgb = jnp.repeat(sk_u8[..., None], 3, axis=-1)
    metrics['goal/skill_sampled'] = _tb_video_grid(sk_rgb)

    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB, :T]
      metrics[f'goal/image_{key}'] = _tb_video_grid(true)
      pred_g = jnp.clip(recons_goal[key].pred() * 255, 0, 255).astype(jnp.uint8)
      err = ((i32(pred_g) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred_g, err], 2)
      metrics[f'goal/recon_{key}'] = _tb_video_grid(video)

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    """If replay_context: first K steps recompute carries from stored entries; else identity."""
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    # prevact[t] aligns with action before obs[t]; prepend stored carry, shift sequence.
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

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
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

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


def policy_behavior_kl(policy):
  """KL(policy || prior) with prior = stop-grad behavioral copy."""
  total = None
  for v in policy.values():
    inner = v.output
    if isinstance(inner, outs.OneHot):
      d = inner.dist
      logits = d.logits
      logp = jax.nn.log_softmax(logits, -1)
      p = jax.nn.softmax(logits, -1)
      logpref = jax.nn.log_softmax(sg(logits), -1)
      kl = (p * (logp - logpref)).sum(-1)
    elif isinstance(inner, outs.Categorical):
      logits = inner.logits
      logp = jax.nn.log_softmax(logits, -1)
      p = jax.nn.softmax(logits, -1)
      logpref = jax.nn.log_softmax(sg(logits), -1)
      kl = (p * (logp - logpref)).sum(-1)
    elif isinstance(inner, outs.Normal):
      ref = outs.Normal(sg(inner.mean), sg(inner.stddev))
      kl = inner.kl(ref)
    else:
      raise NotImplementedError(type(inner))
    kl = v.agg(kl, v.axes)
    total = kl if total is None else total + kl
  return total


def imag_loss(
    act,
    skills,
    rew,
    con,
    manager_policy,
    policy,
    mgr_extr_value,
    mgr_extr_slowvalue,
    mgr_expl_value,
    mgr_expl_slowvalue,
    wkr_goal_value,
    wkr_goal_slowvalue,
    mgr_extr_retnorm,
    mgr_expl_retnorm,
    wkr_goal_retnorm,
    mgr_extr_valnorm,
    mgr_expl_valnorm,
    wkr_goal_valnorm,
    mgr_extr_advnorm,
    mgr_expl_advnorm,
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
    mgr_expl_weight=0.1,
):
  """Actor-critic losses on imagined trajectories.

  Shapes use replay batch ``B``, imagination starts per batch ``K_imag``, and
  imagination depth ``H`` (``imag_length``). Write ``B_imag = B * K_imag``.

  Args:
    act: Imagined actions (``imgact``): pytree aligned with ``act_space``; each
      leaf ``(B_imag, H + 1, *action_dims)``.
    rew: Predicted rewards from the world-model head on imagined feats; scalar
      per step, shape ``(B_imag, H + 1)``.
    con: Predicted continuation probabilities (Bernoulli mean); shape
      ``(B_imag, H + 1)``.
    manager_policy: Manager policy head outputs (``self.manager_pol(...)``): dict keyed like ``act``;
      ``.logp`` / ``.entropy`` reduce to leading axes ``(B_imag, H + 1)``.
    policy: Policy head outputs (``self.pol(...)``): dict keyed like ``act``;
      ``.logp`` / ``.entropy`` reduce to leading axes ``(B_imag, H + 1)``.
    value: Online value head outputs; ``.pred()`` shape ``(B_imag, H + 1)``.
    slowvalue: Slow / target value head (EMA of ``value``); same batch axes.
    retnorm: Return running normalizer (``Normalize``); maps ``ret`` to offset
      and scale scalars (no leading batch axes).
    valnorm: Value / return target normalizer (``Normalize``); same.
    advnorm: Advantage normalizer (``Normalize``); same.
    update: If true, update running stats inside the ``*norm`` modules (train).
    contdisc: If true, per-step discount factor is 1; else ``1 - 1/horizon``.
    slowtar: If true, bootstrap target uses ``slowvalue``; else ``value``.
    horizon: Episode horizon used when ``contdisc`` is false (scalar int).
    lam: TD(λ) parameter λ (scalar float).
    actent: Entropy bonus coefficient on summed action entropies (scalar float).
    slowreg: Weight on auxiliary NLL toward ``slowvalue.pred()`` (scalar float).
    use_pmpo_actor: If true, use PMPO-style policy loss; else REINFORCE + entropy.
    pmpo_beta: PMPO KL term coefficient β (scalar float).
    pmpo_alpha: PMPO weight α on positive-advantage mass (scalar float).

  Returns:
    ``losses`` (per-step policy/value tensors), ``outs`` (e.g. ``ret`` for replay
    bootstrap), and ``metrics`` scalars for logging.
  """
  losses = {}
  metrics = {}

  # Unnormalize critic predictions for bootstrapping and advantage baseline.
  mgr_extr_voffset, mgr_extr_vscale = mgr_extr_valnorm.stats()
  mgr_expl_voffset, mgr_expl_vscale = mgr_expl_valnorm.stats()
  wkr_goal_voffset, wkr_goal_vscale = wkr_goal_valnorm.stats()

  mgr_extr_val = mgr_extr_value.pred() * mgr_extr_vscale + mgr_extr_voffset
  mgr_expl_val = mgr_expl_value.pred() * mgr_expl_vscale + mgr_expl_voffset
  wkr_goal_val = wkr_goal_value.pred() * wkr_goal_vscale + wkr_goal_voffset

  mgr_extr_slowval = mgr_extr_slowvalue.pred() * mgr_extr_vscale + mgr_extr_voffset
  mgr_expl_slowval = mgr_expl_slowvalue.pred() * mgr_expl_vscale + mgr_expl_voffset
  wkr_goal_slowval = wkr_goal_slowvalue.pred() * wkr_goal_vscale + wkr_goal_voffset

  mgr_extr_tarval = mgr_extr_slowval if slowtar else mgr_extr_val
  mgr_expl_tarval = mgr_expl_slowval if slowtar else mgr_expl_val
  wkr_goal_tarval = wkr_goal_slowval if slowtar else wkr_goal_val

  # Discount per step: either γ or finite-horizon (1 - 1/horizon) when not contdisc.
  disc = 1 if contdisc else 1 - 1 / horizon
  # Discounted continuation weights from predicted continue probs ``con``.
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con

  mgr_extr_ret = lambda_return(last, term, rew, mgr_extr_tarval, mgr_extr_tarval, disc, lam)
  mgr_expl_ret = lambda_return(last, term, rew, mgr_expl_tarval, mgr_expl_tarval, disc, lam)
  wkr_goal_ret = lambda_return(last, term, rew, wkr_goal_tarval, wkr_goal_tarval, disc, lam)

  mgr_extr_roffset, mgr_extr_rscale = mgr_extr_retnorm(mgr_extr_ret, update)
  mgr_expl_roffset, mgr_expl_rscale = mgr_expl_retnorm(mgr_expl_ret, update)
  wkr_goal_roffset, wkr_goal_rscale = wkr_goal_retnorm(wkr_goal_ret, update)

  mgr_extr_adv = (mgr_extr_ret - mgr_extr_tarval[:, :-1]) / mgr_extr_rscale
  mgr_expl_adv = (mgr_expl_ret - mgr_expl_tarval[:, :-1]) / mgr_expl_rscale
  wkr_goal_adv = (wkr_goal_ret - wkr_goal_tarval[:, :-1]) / wkr_goal_rscale

  mgr_extr_aoffset, mgr_extr_ascale = mgr_extr_advnorm(mgr_extr_adv, update)
  mgr_expl_aoffset, mgr_expl_ascale = mgr_expl_advnorm(mgr_expl_adv, update)
  wkr_goal_aoffset, wkr_goal_ascale = wkr_goal_advnorm(wkr_goal_adv, update)

  mgr_extr_adv_normed = (mgr_extr_adv - mgr_extr_aoffset) / mgr_extr_ascale
  mgr_expl_adv_normed = (mgr_expl_adv - mgr_expl_aoffset) / mgr_expl_ascale
  wkr_goal_adv_normed = (wkr_goal_adv - wkr_goal_aoffset) / wkr_goal_ascale

  wkr_logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  wkr_ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}

  mgr_logpi = sum([v.logp(sg(skills[k]))[:, :-1] for k, v in manager_policy.items()])
  mgr_ents = {k: v.entropy()[:, :-1] for k, v in manager_policy.items()}

  w = sg(weight[:, :-1])

  if use_pmpo_actor:
    # PMPO target formula:
    # (1-α)/|D-| * Σ_{D-} ln π - α/|D+| * Σ_{D+} ln π + β * mean(KL(π||prior)).
    mgr_extr_adv_raw = mgr_extr_adv - mgr_extr_tarval[:, :-1]
    mgr_expl_adv_raw = mgr_expl_adv - mgr_expl_tarval[:, :-1]
    wkr_goal_adv_raw = wkr_goal_adv - wkr_goal_tarval[:, :-1]

    mgr_extr_pos = (mgr_extr_adv_raw >= 0).astype(f32)
    mgr_extr_neg = (mgr_extr_adv_raw < 0).astype(f32)
    mgr_expl_pos = (mgr_expl_adv_raw >= 0).astype(f32)
    mgr_expl_neg = (mgr_expl_adv_raw < 0).astype(f32)
    wkr_goal_pos = (wkr_goal_adv_raw >= 0).astype(f32)
    wkr_goal_neg = (wkr_goal_adv_raw < 0).astype(f32)

    mgr_extr_den_p = jnp.maximum(jnp.sum(mgr_extr_pos, axis=-1, keepdims=True), 1.0)
    mgr_extr_den_n = jnp.maximum(jnp.sum(mgr_extr_neg, axis=-1, keepdims=True), 1.0)
    mgr_expl_den_p = jnp.maximum(jnp.sum(mgr_expl_pos, axis=-1, keepdims=True), 1.0)
    mgr_expl_den_n = jnp.maximum(jnp.sum(mgr_expl_neg, axis=-1, keepdims=True), 1.0)
    wkr_goal_den_p = jnp.maximum(jnp.sum(wkr_goal_pos, axis=-1, keepdims=True), 1.0)
    wkr_goal_den_n = jnp.maximum(jnp.sum(wkr_goal_neg, axis=-1, keepdims=True), 1.0)

    mgr_extr_pos_coeff = pmpo_alpha * mgr_extr_pos / mgr_extr_den_p
    mgr_extr_neg_coeff = (1.0 - pmpo_alpha) * mgr_extr_neg / mgr_extr_den_n
    mgr_expl_pos_coeff = pmpo_alpha * mgr_expl_pos / mgr_expl_den_p
    mgr_expl_neg_coeff = (1.0 - pmpo_alpha) * mgr_expl_neg / mgr_expl_den_n
    wkr_goal_pos_coeff = pmpo_alpha * wkr_goal_pos / wkr_goal_den_p
    wkr_goal_neg_coeff = (1.0 - pmpo_alpha) * wkr_goal_neg / wkr_goal_den_n

    wkr_kl_t = policy_behavior_kl(policy)[:, :-1]
    mgr_kl_t = policy_behavior_kl(manager_policy)[:, :-1]

    mgr_extr_policy_loss = (mgr_extr_neg_coeff - mgr_extr_pos_coeff) * mgr_logpi + pmpo_beta * mgr_kl_t
    mgr_expl_policy_loss = (mgr_expl_neg_coeff - mgr_expl_pos_coeff) * mgr_logpi + pmpo_beta * mgr_kl_t

    wkr_goal_policy_loss = (wkr_goal_neg_coeff - wkr_goal_pos_coeff) * wkr_logpi + pmpo_beta * wkr_kl_t

    metrics['mgr_extr_kl_behavior'] = mgr_kl_t.mean()
    metrics['mgr_expl_kl_behavior'] = mgr_kl_t.mean()
    metrics['wkr_goal_kl_behavior'] = wkr_kl_t.mean()
  else:
    mgr_extr_policy_loss = w * -(
        mgr_logpi * sg(mgr_extr_adv_normed) + actent * sum(mgr_ents.values()))
    mgr_expl_policy_loss = w * -(
        mgr_logpi * sg(mgr_expl_adv_normed) + actent * sum(mgr_ents.values()))
    wkr_policy_loss = w * -(
        wkr_logpi * sg(wkr_adv_normed) + actent * sum(wkr_ents.values()))

  losses['mgr_policy'] = mgr_extr_policy_loss + mgr_expl_weight * mgr_expl_policy_loss
  losses['wkr_policy'] = wkr_goal_policy_loss

  metrics['mgr_extr_policy_loss'] = mgr_extr_policy_loss.mean()
  metrics['mgr_expl_policy_loss'] = mgr_expl_policy_loss.mean()
  metrics['wkr_goal_policy_loss'] = wkr_goal_policy_loss.mean()

  # NLL of value distribution against λ-returns (padded for length match to head API).
  mgr_extr_voffset, mgr_extr_vscale = mgr_extr_valnorm(mgr_extr_ret, update)
  mgr_expl_voffset, mgr_expl_vscale = mgr_expl_valnorm(mgr_expl_ret, update)
  wkr_goal_voffset, wkr_goal_vscale = wkr_goal_valnorm(wkr_goal_ret, update)

  mgr_extr_tar_normed = (mgr_extr_ret - mgr_extr_voffset) / mgr_extr_vscale
  mgr_expl_tar_normed = (mgr_expl_ret - mgr_expl_voffset) / mgr_expl_vscale
  wkr_goal_tar_normed = (wkr_goal_ret - wkr_goal_voffset) / wkr_goal_vscale

  mgr_extr_tar_padded = jnp.concatenate([mgr_extr_tar_normed, 0 * mgr_extr_tar_normed[:, -1:]], 1)
  mgr_expl_tar_padded = jnp.concatenate([mgr_expl_tar_normed, 0 * mgr_expl_tar_normed[:, -1:]], 1)
  wkr_goal_tar_padded = jnp.concatenate([wkr_goal_tar_normed, 0 * wkr_goal_tar_normed[:, -1:]], 1)

  losses['mgr_extr_value'] = sg(weight[:, :-1]) * (
      mgr_extr_val.loss(sg(mgr_extr_tar_padded)) +
      slowreg * mgr_extr_val.loss(sg(mgr_extr_slowval.pred())))[:, :-1]

  losses['mgr_expl_value'] = sg(weight[:, :-1]) * (
      mgr_expl_val.loss(sg(mgr_expl_tar_padded)) +
      slowreg * mgr_expl_val.loss(sg(mgr_expl_slowval.pred())))[:, :-1]

  losses['wkr_goal_value'] = sg(weight[:, :-1]) * (
      wkr_goal_val.loss(sg(wkr_goal_tar_padded)) +
      slowreg * wkr_goal_val.loss(sg(wkr_goal_slowval.pred())))[:, :-1]

  mgr_extr_ret_normed = (mgr_extr_ret - mgr_extr_roffset) / mgr_extr_rscale
  mgr_expl_ret_normed = (mgr_expl_ret - mgr_expl_roffset) / mgr_expl_rscale
  wkr_goal_ret_normed = (wkr_goal_ret - wkr_goal_roffset) / wkr_goal_rscale

  metrics['mgr_extr_adv'] = mgr_extr_adv.mean()
  metrics['mgr_expl_adv'] = mgr_expl_adv.mean()
  metrics['wkr_goal_adv'] = wkr_goal_adv.mean()

  metrics['mgr_extr_adv_std'] = mgr_extr_adv.std()
  metrics['mgr_expl_adv_std'] = mgr_expl_adv.std()
  metrics['wkr_goal_adv_std'] = wkr_goal_adv.std()

  metrics['mgr_extr_adv_mag'] = jnp.abs(mgr_extr_adv_normed).mean()
  metrics['mgr_expl_adv_mag'] = jnp.abs(mgr_expl_adv_normed).mean()
  metrics['wkr_goal_adv_mag'] = jnp.abs(wkr_goal_adv_normed).mean()

  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['mgr_extr_ret'] = mgr_extr_ret_normed.mean()
  metrics['mgr_expl_ret'] = mgr_expl_ret_normed.mean()
  metrics['wkr_goal_ret'] = wkr_goal_ret_normed.mean()
  metrics['mgr_extr_val'] = mgr_extr_val.mean()
  metrics['mgr_expl_val'] = mgr_expl_val.mean()
  metrics['wkr_goal_val'] = wkr_goal_val.mean()
  metrics['mgr_extr_tar'] = mgr_extr_tar_normed.mean()
  metrics['mgr_expl_tar'] = mgr_expl_tar_normed.mean()
  metrics['wkr_goal_tar'] = wkr_goal_tar_normed.mean()
  metrics['weight'] = weight.mean()

  metrics['mgr_extr_slowval'] = mgr_extr_slowval.mean()
  metrics['mgr_expl_slowval'] = mgr_expl_slowval.mean()
  metrics['wkr_goal_slowval'] = wkr_goal_slowval.mean()

  metrics['mgr_extr_ret_min'] = mgr_extr_ret_normed.min()
  metrics['mgr_expl_ret_min'] = mgr_expl_ret_normed.min()
  metrics['wkr_goal_ret_min'] = wkr_goal_ret_normed.min()
  metrics['mgr_extr_ret_max'] = mgr_extr_ret_normed.max()
  metrics['mgr_expl_ret_max'] = mgr_expl_ret_normed.max()
  metrics['wkr_goal_ret_max'] = wkr_goal_ret_normed.max()
  metrics['mgr_extr_ret_rate'] = (jnp.abs(mgr_extr_ret_normed) >= 1.0).mean()
  metrics['mgr_expl_ret_rate'] = (jnp.abs(mgr_expl_ret_normed) >= 1.0).mean()
  metrics['wkr_goal_ret_rate'] = (jnp.abs(wkr_goal_ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'wkr_ent/{k}'] = wkr_ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'wkr_rand/{k}'] = (wkr_ents[k].mean() - lo) / (hi - lo)

  for k in skills:
    metrics[f'mgr_ent/{k}'] = mgr_ents[k].mean()
    if hasattr(manager_policy[k], 'minent'):
      lo, hi = manager_policy[k].minent, manager_policy[k].maxent
      metrics[f'mgr_rand/{k}'] = (mgr_ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret  # Used as bootstrap for optional replay value loss.
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
