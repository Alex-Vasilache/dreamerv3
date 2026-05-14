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

from . import director_full as dfull
from . import director_hrl as dch
from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class Agent(embodied.jax.Agent):

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

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    # Flat policy is unused in train loss when Director HRL is on; omit from
    # ``self.modules`` so Ninjax state matches ``nj.grad`` targets.
    core_modules = [self.dyn, self.enc, self.dec, self.rew, self.con, self.val]
    if not self.config.use_director_hrl:
      core_modules.insert(-1, self.pol)
    self.modules = core_modules
    self.goal_enc = None
    self.goal_dec = None
    self.man_pol = None
    self.worker_pol = None
    if self.config.use_director_hrl:
      dh = self.config.director_hrl
      deter = self.config.dyn.rssm.deter
      stoch_flat = self.config.dyn.rssm.stoch * self.config.dyn.rssm.classes
      skill_space = elements.Space(
          np.int32, (dh.skill_syms,), 0, dh.skill_classes)
      goal_space = elements.Space(np.float32, (deter,))
      self.goal_enc = embodied.jax.MLPHead(
          {'skill': skill_space},
          {'skill': 'categorical'},
          **dh.goal_encoder, name='goal_enc')
      self.goal_dec = embodied.jax.MLPHead(
          goal_space, 'mse', **dh.goal_decoder, name='goal_dec')
      self.man_pol = embodied.jax.MLPHead(
          {'skill': skill_space},
          {'skill': 'categorical'},
          **dh.manager_policy, name='man_pol')
      self.worker_pol = embodied.jax.MLPHead(
          act_space, outs, **dh.worker_policy, name='worker_pol')
      self.disag_heads = []
      if str(dh.get('expl_rew', 'adver')) == 'disag':
        dhead = dict(getattr(dh, 'disag_head', None) or {})
        if not dhead:
          dhead = {
              'layers': 2, 'units': 512, 'act': 'silu', 'norm': 'rms',
              'outscale': 0.01, 'winit': 'trunc_normal_in'}
        for i in range(int(dh.get('disag_models', 8))):
          h = embodied.jax.MLPHead(
              elements.Space(np.float32, (stoch_flat,)),
              'mse',
              **dhead,
              name=f'disag{i}')
          self.disag_heads.append(h)
      self.modules += [
          self.goal_enc, self.goal_dec, self.man_pol, self.worker_pol,
          *self.disag_heads]

    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    scales.setdefault('director_disag', 0.0)
    if self.config.use_director_hrl:
      scales.update(self.config.director_hrl.loss_scales)
      scales['policy'] = 0.0
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
    if self.config.use_director_hrl:
      return (
          '^(enc|dyn|dec|goal_enc|goal_dec|man_pol|worker_pol|disag)/')
    return '^(enc|dyn|dec|pol)/'

  def _director_decode_goal(self, skill_idx, ctx_deter):
    dh = self.config.director_hrl
    oh = jax.nn.one_hot(skill_idx, dh.skill_classes, dtype=f32)
    flat = oh.reshape(*oh.shape[:-2], -1)
    dec_in = jnp.concatenate([flat, ctx_deter], -1)
    pred = self.goal_dec(dec_in, 1).pred()
    if dh.manager_delta:
      return ctx_deter + pred
    return pred

  @property
  def ext_space(self):
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
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    prev = jax.tree.map(zeros, self.act_space)
    carry = (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        prev)
    if not self.config.use_director_hrl:
      return carry
    dh = self.config.director_hrl
    hrl = dch.initial_hrl_carry(
        batch_size, self.config.dyn.rssm.deter, dh.skill_syms)
    return (*carry, hrl)

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    if self.config.use_director_hrl:
      (enc_carry, dyn_carry, dec_carry, prevact, hrl) = carry
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
    if self.config.use_director_hrl:
      dh = self.config.director_hrl
      K = int(dh.get('env_skill_duration', dh.manager_step_K))
      deter = feat['deter']
      renew = ((hrl['step'] % K) == 0).astype(f32)
      man_out = self.man_pol(deter, bdims=1)
      new_s = jnp.where(
          renew[:, None], man_out['skill'].sample(nj.seed()), hrl['skill'])
      g_raw = self._director_decode_goal(new_s, deter)
      goal = jnp.where(renew[:, None], g_raw, hrl['goal'])
      xw = jnp.concatenate([self.feat2tensor(feat), goal], -1)
      pol_out = self.worker_pol(xw, bdims=1)
      act = sample(pol_out)
      hrl = dict(
          step=hrl['step'] + 1,
          skill=new_s,
          goal=goal,
      )
    else:
      policy = self.pol(self.feat2tensor(feat), bdims=1)
      act = sample(policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.use_director_hrl:
      carry = (*carry, hrl)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
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
    act_last = {k: data[k][:, -1] for k in self.act_space}
    if self.config.use_director_hrl:
      enc_carry, dyn_carry, dec_carry, hrl_carry = carry
      carry = (enc_carry, dyn_carry, dec_carry, act_last, hrl_carry)
    else:
      enc_carry, dyn_carry, dec_carry = carry
      carry = (enc_carry, dyn_carry, dec_carry, act_last)
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    if self.config.use_director_hrl:
      enc_carry, dyn_carry, dec_carry, hrl_carry = carry
    else:
      enc_carry, dyn_carry, dec_carry = carry
      hrl_carry = None
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination (K_imag) vs replay repval window (K_repl). Single-rollout uses
    # K_imag=1 but K_repl>=2 when repval_loss is on, else lambda_return gets an
    # empty term[:, 1:] slice and jnp.stack fails.
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
    if self.config.use_director_hrl:
      dh = self.config.director_hrl
      K = int(dh.get('train_skill_duration', dh.manager_step_K))
      jointly = str(dh.get('jointly', 'new'))
      assert jointly == 'new', (
          f'director_hrl.jointly={jointly!r} is not supported; use jointly: new')
      assert (H + 1) % K == 1, (
          f'Director split/abstract requires (imag_length+1) % train_skill_duration == 1; '
          f'got imag_length={H}, train_skill_duration={K}. Example: imag_length=15 -> 16 '
          f'steps is invalid for K=8; use imag_length=16 -> 17 steps.')
      if dh.get('vae_replay', True):
        ae_loss, ae_mets = dch.goal_autoencoder_replay_loss(
            self.goal_enc, self.goal_dec, repfeat['deter'], K, dh.skill_classes)
        losses['director_ae'] = ae_loss
        metrics.update(ae_mets)
      else:
        losses['director_ae'] = 0 * losses['rew']
      if self.disag_heads:
        losses['director_disag'] = dch.disag_replay_loss(
            self.disag_heads, repfeat['deter'], repfeat['stoch'])
      else:
        losses['director_disag'] = 0 * losses['rew']
      Bsz = B * K_imag
      zskill = jnp.zeros((Bsz, dh.skill_syms), i32)
      zgoal = jnp.zeros((Bsz, self.config.dyn.rssm.deter), f32)
      zt = jnp.array(0, jnp.int32)

      def hrl_step(pack, _):
        carry_r, skill_idx, goal, t = pack
        renew = ((t % K) == 0).astype(f32)
        deter = carry_r['deter']
        man_out = self.man_pol(deter, bdims=1)
        new_s = jnp.where(
            renew.astype(jnp.int32), man_out['skill'].sample(nj.seed()), skill_idx)
        new_g = self._director_decode_goal(new_s, deter)
        # renew is scalar in time (shared across batch); jnp.where broadcasts.
        goal_n = jnp.where(renew > 0, new_g, goal)
        xw = jnp.concatenate([
            self.feat2tensor({'deter': deter, 'stoch': carry_r['stoch']}),
            goal_n], -1)
        w_act = sample(self.worker_pol(xw, 1))
        carry_n, (feat, _) = self.dyn.imagine(
            carry_r, w_act, 1, training, single=True)
        t_n = t + 1
        return (carry_n, new_s, goal_n, t_n), (feat, w_act, goal_n, new_s)

      pack0 = (starts, zskill, zgoal, zt)
      (carry_f, _, _, _), stacked = nj.scan(
          hrl_step, pack0, (), H, unroll=self.dyn.unroll, axis=1)
      feat_seq, act_seq, goal_seq, skill_seq = stacked
      first = jax.tree.map(
          lambda x: x[:, -K_imag:].reshape((B * K_imag, 1, *x.shape[2:])), repfeat)
      imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(feat_seq)], 1)
      goals_full = jnp.concatenate([zgoal[:, None], goal_seq], axis=1)
      last_feat = jax.tree.map(lambda x: x[:, -1], imgfeat)
      lx = jnp.concatenate([
          self.feat2tensor(last_feat), goals_full[:, -1]], -1)
      lastact = sample(self.worker_pol(lx, 1))
      lastact = jax.tree.map(lambda x: x[:, None], lastact)
      imgact = concat([act_seq, lastact], 1)
      assert all(x.shape[:2] == (B * K_imag, H + 1) for x in jax.tree.leaves(imgfeat))
      assert all(x.shape[:2] == (B * K_imag, H + 1) for x in jax.tree.leaves(imgact))
      inp = self.feat2tensor(imgfeat)
      deter_bt = imgfeat['deter']
      cont_bt = self.con(inp, 2).prob(1)
      rew_wm = self.rew(inp, 2).pred()
      expl_source = str(dh.get('expl_rew', 'adver'))
      if expl_source == 'disag':
        assert self.disag_heads, (
            'director_hrl.expl_rew=disag requires disag_models and disag_head in config')
        expl_bt = dch.disag_reward(deter_bt, self.disag_heads)
      else:
        expl_bt = dch.elbo_adver_reward(
            self.goal_enc, self.goal_dec, deter_bt, dh.skill_classes,
            str(dh.get('adver_impl', 'squared')))
      skill_full = jnp.concatenate([zskill[:, None], skill_seq], axis=1)
      goal_reward_name = str(dh.get('goal_reward', 'cosine_max'))
      traj = dfull.build_imag_traj_for_hierarchy(
          deter_bt=deter_bt,
          stoch_bt=imgfeat['stoch'],
          act_primitives=imgact,
          skill_idx=skill_full,
          goal_bt=goals_full,
          cont_bt=cont_bt,
          rew_wm_pred=rew_wm,
          expl_rew_bt=expl_bt,
          goal_reward_name=goal_reward_name,
          enc_fwd=None,
      )
      discount = (
          1.0 if self.config.contdisc
          else (1.0 - 1.0 / float(self.config.horizon)))
      wtraj, mtraj = dfull.split_and_abstract(traj, K, discount)
      wr = dict(dh.get('worker_rews', {}))
      we = float(wr.get('extr', 0.0))
      wx = float(wr.get('expl', 0.0))
      wg = float(wr.get('goal', 1.0))
      dw = wtraj
      fe_w = self.feat2tensor({'deter': dw['deter'], 'stoch': dw['stoch']})
      inp_w_t = jnp.concatenate([fe_w, dw['goal']], axis=-1)
      con_w = dw['cont']
      rew_w = (
          we * dw['reward_extr'] + wx * dw['reward_expl'] + wg * dw['reward_goal'])
      rew_w = jnp.concatenate(
          [rew_w, jnp.zeros((rew_w.shape[0], 1), f32)], axis=1)
      los, imgloss_out, mets = imag_loss(
          dw['action'],
          rew_w,
          con_w,
          self.worker_pol(inp_w_t, 2),
          self.val(fe_w, 2),
          self.slowval(fe_w, 2),
          self.retnorm, self.valnorm, self.advnorm,
          update=training,
          contdisc=self.config.contdisc,
          horizon=self.config.horizon,
          **self.config.imag_loss)
      los_d = {k: v.mean(1).reshape((B, K_imag)) for k, v in los.items() if k != 'policy'}
      los_d['director_wrk'] = los['policy'].mean(1).reshape((B, K_imag))
      losses.update(los_d)
      metrics.update(mets)
      mr = dict(dh.get('manager_rews', {}))
      me = float(mr.get('extr', 1.0))
      mx = float(mr.get('expl', 0.1))
      mg = float(mr.get('goal', 0.0))
      mm = mtraj
      fe_m = self.feat2tensor({'deter': mm['deter'], 'stoch': mm['stoch']})
      rew_m = (
          me * mm['reward_extr'] + mx * mm['reward_expl'] + mg * mm['reward_goal'])
      rew_m = jnp.concatenate(
          [rew_m, jnp.zeros((rew_m.shape[0], 1), f32)], axis=1)
      mgr_act = {'skill': mm['action']}
      los_m, _, mets_m = imag_loss(
          mgr_act,
          rew_m,
          mm['cont'],
          self.man_pol(mm['deter'], 2),
          self.val(fe_m, 2),
          self.slowval(fe_m, 2),
          self.retnorm, self.valnorm, self.advnorm,
          update=training,
          contdisc=self.config.contdisc,
          horizon=self.config.horizon,
          **self.config.imag_loss)
      losses['director_mgr'] = los_m['policy'].mean(1).reshape((B, K_imag))
      losses['value'] = losses['value'] + los_m['value'].mean(1).reshape((B, K_imag))
      metrics.update(prefix(mets_m, 'director/mgr'))
    else:
      policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
      _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
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
      losses['director_ae'] = 0 * losses['rew']
      losses['director_mgr'] = 0 * losses['rew']
      losses['director_wrk'] = 0 * losses['rew']
      losses['director_disag'] = 0 * losses['rew']

    # Replay
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

    for k in list(losses.keys()):
      assert k in self.scales, (k, sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    if self.config.use_rms_loss_norm:
      losses = {
          k: v / sg(self.lossrms[k](v, training))
          for k, v in losses.items()}
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    if hrl_carry is not None:
      carry = (*carry, hrl_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    if self.config.use_director_hrl:
      enc_carry, dyn_carry, dec_carry, hrl_carry = carry
    else:
      enc_carry, dyn_carry, dec_carry = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    if not self.config.use_director_hrl:
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

      # Video preds
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
    if self.config.use_director_hrl:
      (enc_carry, dyn_carry, dec_carry, prevact, hrl_carry) = carry
    else:
      (enc_carry, dyn_carry, dec_carry, prevact) = carry
      hrl_carry = None
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      if hrl_carry is not None:
        hrl_carry = dch.reset_hrl_on_episode(
            hrl_carry, obs['is_first'][:, 0],
            deter_dim=self.config.dyn.rssm.deter,
            skill_syms=self.config.director_hrl.skill_syms)
      if hrl_carry is None:
        return carry, obs, prevact, stepid
      return (*carry, hrl_carry), obs, prevact, stepid

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

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    if hrl_carry is not None:
      hrl_carry = dch.reset_hrl_on_episode(
          hrl_carry, obs['is_first'][:, 0],
          deter_dim=self.config.dyn.rssm.deter,
          skill_syms=self.config.director_hrl.skill_syms)
    if hrl_carry is None:
      return carry, obs, prevact, stepid
    return (*carry, hrl_carry), obs, prevact, stepid

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
):
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
  outs['ret'] = ret
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
  weight = f32(~last)
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
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
