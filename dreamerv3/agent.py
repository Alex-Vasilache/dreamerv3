"""DreamerV3-style agent: RSSM world model + policy/value on imagined rollouts.

Trains encoder, dynamics (RSSM), decoder, reward/continue heads, policy, and
value in one step from replay sequences. Actor-critic losses use imagined
trajectories from ``dyn.imagine``; optional ``repval_loss`` fits the value on
real replay tails with bootstrap from imagination.
"""
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
    skill_classes = int(getattr(config, 'skill_classes', skill_shape_t[0]))
    self.skill_space = elements.Space(np.int32, skill_shape_t, 0, skill_classes)
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

    # Goal Autoencoder: takes in deterministic hidden state of world model as input
    # and outputs a categorical distribution as goal code (onehot head).
    self.goal_code_space = elements.Space(
        np.int32, skill_shape_t, 0, skill_classes)
    self.goal_enc = embodied.jax.MLPHead(
        self.goal_code_space, **config.goal_enc, name='goal_enc')
    self.goal_dec = embodied.jax.MLPHead(self.goal_shape, **config.goal_dec, name='goal_dec')
    self.goal_autoencoder_beta = config.goal_autoencoder_beta
    # Uniform prior metadata only: built inside ``loss`` with ``zeros_like`` encoder
    # logits so arrays stay on-device (``jnp.zeros`` here breaks sharded init).
    self._skill_prior_unimix = float(config.goal_enc.unimix)
    self._skill_prior_agg_dims = len(skill_shape_t) + 1

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

    # Value and EMA target for bootstrapping / slow regularizer in ``imag_loss``.
    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    # Modules updated by the single ``self.opt`` step in ``train``.
    self.modules = [
        self.dyn,
        self.enc,
        self.dec,
        self.goal_enc,
        self.goal_dec,
        self.rew,
        self.con,
        self.pol,
        self.val,
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
    # Policy reads representation only (decoder used for logging / replay context).
    policy = self.pol(self.feat2tensor(feat), bdims=1)
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
    self.slowval.update()
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
    losses['goal_rec'] = decoded_goal.loss(sg(deter_feat))
    prior_inner = outs.OneHot(
        jnp.zeros_like(encoded_goal.output.dist.logits),
        self._skill_prior_unimix)
    skill_prior = outs.Agg(
        prior_inner, self._skill_prior_agg_dims, jnp.sum)
    inner_kl = encoded_goal.output.kl(skill_prior.output)
    kd = len(self.skill_shape)
    goal_kl_bt = jnp.sum(inner_kl, axis=tuple(range(-kd, 0)))
    losses['goal_kl'] = (
        f32(self.config.goal_autoencoder_beta) * goal_kl_bt
        if self.config.goal_kl
        else jnp.zeros((B, T), f32))
    # Logged as ``train/goal/*`` when the train loop aggregates with prefix ``train``.
    ent_inner = encoded_goal.output.dist.entropy()
    goal_ent_bt = jnp.sum(ent_inner, axis=tuple(range(-kd, 0)))
    metrics.update({
        'goal/rec_mean': losses['goal_rec'].mean(),
        'goal/rec_std': losses['goal_rec'].std(),
        'goal/kl_mean': losses['goal_kl'].mean(),
        'goal/kl_raw_mean': goal_kl_bt.mean(),
        'goal/kl_std': losses['goal_kl'].std(),
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
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
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
    los, imgloss_out, mets = imag_loss(
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
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
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
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    aux_outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, aux_outs, metrics)

  def report(self, carry, data):
    """Eval-style forward, optional grad norms, and open-loop reconstruction video."""
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
    mets.update(mets)

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
    rew,
    con,
    policy,
    value,
    slowvalue,
    retnorm,
    valnorm,
    advnorm,
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
  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  # Discount per step: either γ or finite-horizon (1 - 1/horizon) when not contdisc.
  disc = 1 if contdisc else 1 - 1 / horizon
  # Discounted continuation weights from predicted continue probs ``con``.
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
    # PMPO target formula provided by user:
    # (1-α)/|D-| * Σ_{D-} ln π - α/|D+| * Σ_{D+} ln π + β * mean(KL(π||prior)).
    adv_raw = ret - tarval[:, :-1]
    pos = (adv_raw >= 0).astype(f32)
    neg = (adv_raw < 0).astype(f32)
    den_p = jnp.maximum(jnp.sum(pos, axis=-1, keepdims=True), 1.0)
    den_n = jnp.maximum(jnp.sum(neg, axis=-1, keepdims=True), 1.0)
    pos_coeff = pmpo_alpha * pos / den_p
    neg_coeff = (1.0 - pmpo_alpha) * neg / den_n
    kl_t = policy_behavior_kl(policy)[:, :-1]
    policy_loss = (neg_coeff - pos_coeff) * logpi + pmpo_beta * kl_t
    metrics['kl_behavior'] = kl_t.mean()
  else:
    policy_loss = w * -(
        logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  # NLL of value distribution against λ-returns (padded for length match to head API).
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
):
  """Value loss on real replay tail; ``boot`` is return from imagination at slice boundary.

  ``last`` masks episode boundaries; ``boot`` supplies bootstrap value at the
  window edge. Same λ-return and slow-value mix as imagination, but no policy term.
  """
  losses = {}
  if last.shape[1] < 2:
    losses['repval'] = jnp.zeros_like(f32(last))
    outs = {'ret': jnp.zeros((last.shape[0], 0), f32)}
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
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
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
