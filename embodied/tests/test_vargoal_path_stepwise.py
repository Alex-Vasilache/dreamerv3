"""Step-by-step checks of the whole variable-goal-length path on the REAL agent.

Builds ``dreamerv3.agent.Agent`` from the real config blocks (V + goal hold),
takes one real train step and runs the real env policy, and captures the
intermediate tensors at every stage with ``jax.debug.callback``:

  1. env step     ``_manager_skill_step``: who switches, and when;
  2. imagination  ``_imagine_with_manager``: switch mask, held skills, reach
                  flags, countdowns, start states;
  3. manager      what reaches ``imag_loss_mgr``: decision skills, pooled
                  rewards / continuations, credited-decision mask;
  4. worker       what reaches ``imag_loss_wkr``: reward vs the old goal,
                  boundary mask, old-goal bootstrap;
  5. replay       ``_manager_skills_on_sequence`` switch rule, and the pooled
                  tensors / discount the replay manager return is built from.

Every stage is recomputed longhand in numpy from the captured inputs and
compared to what the code produced. With the reached test off and a cap of K
the manager tensors must also equal the fixed-K Director helpers.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dreamerv3.agent as agent_mod
from dreamerv3.agent import Agent
from dreamerv3.hrl import (
    aggregate_mgr_cont, aggregate_mgr_extr_rew, imag_reward_pad)

from test_goal_ae_agent import make_agent, make_batch

f32 = jnp.float32
CAP = 8
H = 16
BL = 32
BLOCKS = ('director', 'director_og', 'wkr_ent_fixed', 'wkr_v3',
          'goal_hold_reach', 'goal_hold_cap8', 'debug')

STORE = {}


def _put(name, tree, append=False):
  def cb(t):
    t = jax.tree.map(np.asarray, t)
    if append:
      STORE.setdefault(name, []).append(t)
    else:
      STORE[name] = t
  jax.debug.callback(cb, tree)


def _cosmax(g, x):
  g = np.asarray(g, np.float64); x = np.asarray(x, np.float64)
  n = np.maximum(np.linalg.norm(g, axis=-1), np.linalg.norm(x, axis=-1)) + 1e-12
  return (g * x).sum(-1) / (n * n)


# --------------------------------------------------------------------------
# Instrumentation (installed once, before any agent is traced)
# --------------------------------------------------------------------------

def _install():
  if getattr(agent_mod, '_vargoal_probe', False):
    return
  agent_mod._vargoal_probe = True

  orig_step = Agent._manager_skill_step
  def step(self, feat, mgr_skill, mgr_step, reset, deterministic=False):
    out = orig_step(self, feat, mgr_skill, mgr_step, reset, deterministic)
    if self.goal_reach:
      _put('env', dict(
          deter=f32(self.feat2deter(feat)), cached=f32(mgr_skill['goal_deter']),
          step_in=mgr_step, reset=reset, step_out=out[2], goal_out=f32(out[1]),
          tau=self._goal_reach_tau()), append=True)
    return out
  Agent._manager_skill_step = step

  orig_imag = Agent._imagine_with_manager
  def imag(self, starts, H_, training):
    out = orig_imag(self, starts, H_, training)
    imgfeat, imgact, skills, seed, sw, cd = out
    if self.goal_reach:
      full = jax.tree.map(lambda a, b: jnp.concatenate([a[:, None], b], 1),
                          {k: starts[k] for k in ('deter', 'stoch')},
                          {k: imgfeat[k] for k in ('deter', 'stoch')})
      inp = self.feat2tensor(full)
      _put('imag', dict(
          starts=f32(starts['deter']), deter=f32(self.feat2deter(full)),
          skill=f32(skills['skill']), sw=sw, cd=cd,
          goals=f32(self._goals_from_skills(skills, bdims=2)),
          reached=f32(self._imag_reached), tau=self._goal_reach_tau(),
          rew=f32(self.rew(inp, 2).pred()), con=f32(self.con(inp, 2).prob(1)),
          expl=f32(self._mgr_expl_reward(full))))
    return out
  Agent._imagine_with_manager = imag

  orig_ae = Agent._goal_autoencoder_loss
  def ae(self, repfeat, *a, **k):
    _put('repfeat', f32(self.feat2deter(repfeat)))
    return orig_ae(self, repfeat, *a, **k)
  Agent._goal_autoencoder_loss = ae

  orig_seq = Agent._manager_skills_on_sequence
  def seq(self, repfeat, downsample=False, deterministic=False):
    out = orig_seq(self, repfeat, downsample, deterministic)
    if self.goal_reach and deterministic and not downsample:
      _put('seq', dict(
          deter=f32(self.feat2deter(repfeat)), skill=f32(out['skill']),
          sw=out['switch'], cd=out['countdown'], tau=self._goal_reach_tau(),
          det=jnp.asarray(bool(deterministic)),
          down=jnp.asarray(bool(downsample)),
          goals=f32(self._goals_from_skills(
              {'skill': out['skill']}, bdims=2))), append=True)
    return out
  Agent._manager_skills_on_sequence = seq

  orig_vbd = agent_mod.variable_block_director_tensors
  def vbd(rew, con, expl, switch_mask, horizon, agg_mode='mean'):
    out = orig_vbd(rew, con, expl, switch_mask, horizon, agg_mode=agg_mode)
    _put('vbd', dict(rew=f32(rew), con=f32(con), expl=f32(expl),
                     sw=f32(switch_mask), out=[f32(x) for x in out]),
         append=True)
    return out
  agent_mod.variable_block_director_tensors = vbd

  orig_mgr = agent_mod.imag_loss_mgr
  def mgr(skills, extr, expl, cont, policy, *a, **k):
    _put('mgr', dict(skill=f32(skills['skill']), extr=f32(extr),
                     expl=f32(expl), cont=f32(cont),
                     switch=f32(k['switch_mask'])))
    return orig_mgr(skills, extr, expl, cont, policy, *a, **k)
  agent_mod.imag_loss_mgr = mgr

  orig_wkr = agent_mod.imag_loss_wkr
  def wkr(act, rew, con, policy, value, slowvalue, *a, **k):
    sw = k.get('skill_window')
    d = dict(rew=f32(rew), con=f32(con), val=f32(value.pred()),
             slowval=f32(slowvalue.pred()))
    if hasattr(sw, 'ndim'):
      d['sw'] = f32(sw)
    if k.get('boot_value') is not None:
      d['boot'] = f32(k['boot_value'].pred())
      d['bootslow'] = f32(k['boot_slowvalue'].pred())
    _put('wkr', d)
    return orig_wkr(act, rew, con, policy, value, slowvalue, *a, **k)
  agent_mod.imag_loss_wkr = wkr

  orig_lr = agent_mod.lambda_return
  def lr(last, term, rew, val, boot, disc, lam):
    _put('lret', dict(last=f32(last), term=f32(term), rew=f32(rew),
                      disc=jnp.asarray(disc, f32)), append=True)
    return orig_lr(last, term, rew, val, boot, disc, lam)
  agent_mod.lambda_return = lr


# --------------------------------------------------------------------------
# Run one agent: env steps + one train step, return the captures
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def run(frac, agg):
  _install()
  agent = make_agent(
      *BLOCKS, batch_size=4, batch_length=BL,
      agent=dict(goal_reach_frac=frac, mgr_reward_agg=agg, imag_length=H,
                 goal_duration_fixed=CAP))
  from embodied.jax import internal
  STORE.clear()
  carry = agent.init_train(agent.config.batch_size)
  batch = make_batch(agent)
  data = {**internal.device_put(batch, agent.train_sharded),
          'seed': agent._seeds(0, agent.train_mirrored)}
  STORE.clear()
  agent.train(carry, data)
  jax.effects_barrier()
  train = {k: v for k, v in STORE.items()}
  # Env rollout (40 steps, one mid-episode reset).
  STORE.clear()
  B = 3
  pc = agent.init_policy(B)
  rng = np.random.default_rng(1)
  for t in range(40):
    obs = {}
    for k, sp in agent.obs_space.items():
      if sp.dtype == bool:
        obs[k] = np.zeros((B,) + sp.shape, bool)
      elif np.issubdtype(sp.dtype, np.integer):
        obs[k] = rng.integers(0, 2, (B,) + sp.shape).astype(sp.dtype)
      else:
        obs[k] = rng.normal(0, 1, (B,) + sp.shape).astype(sp.dtype)
    obs['is_first'] = np.array([t == 0, t == 0 or t == 21, t == 0])
    pc, _, _ = agent.policy(pc, obs, mode='train')
  jax.effects_barrier()
  env = STORE.get('env', [])
  return train, env, batch


CASES = [(50.0, 'mean'), (0.3, 'sum'), (0.15, 'sum'), (-50.0, 'sum')]


def _close_or_tie(got, want, cos, tau, what):
  """Boolean rule check; disagreement only allowed within rounding of tau."""
  bad = (np.asarray(got) > 0.5) != np.asarray(want)
  tie = np.abs(cos - tau) < 1e-4
  assert not (bad & ~tie).any(), (what, np.argwhere(bad & ~tie)[:5])


def _longhand_switch(deter, goals, tau, T):
  """Reached-or-cap rule: switch at t=0, at the cap, or when the goal held
  coming into t (goals[t-1]) matches deter[t]."""
  B = deter.shape[0]
  sw = np.zeros((B, T), bool); cos = np.zeros((B, T)); cd = np.zeros((B, T), int)
  rem = np.zeros(B, int)
  for t in range(T):
    c = _cosmax(goals[:, t - 1], deter[:, t]) if t > 0 else np.full(B, -9.0)
    cos[:, t] = c
    reached = (c >= tau) & (t > 0)
    upd = (rem <= 0) | reached
    sw[:, t] = upd
    cd[:, t] = np.where(upd, CAP, rem)
    rem = cd[:, t] - 1
  return sw, cos, cd


def _pool_longhand(rew, con, sw, agg):
  """Director abstract_traj on arbitrary boundaries, written out by hand."""
  B, T = rew.shape
  R = np.zeros((B, T)); C = np.ones((B, T)); n_sw = sw.sum(1).astype(int)
  for b in range(B):
    starts = list(np.where(sw[b] > 0.5)[0])
    for j, s in enumerate(starts):
      e = starts[j + 1] if j + 1 < len(starts) else T
      trans = [t for t in range(s, e) if t <= T - 2]
      w, tot, cp = 1.0, 0.0, 1.0
      for t in trans:
        w *= con[b, t]
        tot += rew[b, t + 1] * w
        cp *= con[b, t + 1]
      if agg == 'mean' and trans:
        tot /= len(trans)
      R[b, j] = tot
      C[b, j] = cp
  return R, C, n_sw


# --------------------------------------------------------------------------
# 1. Env step
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frac,agg', CASES)
def test_1_env_switch_rule(frac, agg):
  _, env, _ = run(frac, agg)
  assert len(env) >= 40, len(env)
  env = env[-40:]
  for t, e in enumerate(env):
    reset = e['reset'].astype(bool)
    step_in = np.where(reset, 0, e['step_in'])
    cos = _cosmax(e['cached'], e['deter'])
    want = (step_in <= 0) | (cos >= e['tau'])
    got = e['step_out'] == CAP - 1
    _close_or_tie(got, want, cos, e['tau'], f'env t={t}')
    # Held goal is untouched on non-switch steps.
    hold = ~got
    np.testing.assert_array_equal(e['goal_out'][hold], e['cached'][hold])
    # Countdown never exceeds the cap and ticks by one.
    assert (e['step_out'] >= 0).all() and (e['step_out'] <= CAP - 1).all()
    np.testing.assert_array_equal(e['step_out'][hold], step_in[hold] - 1)
  if frac == 50.0:
    # Never reached: switches exactly every CAP steps from each reset.
    sw = np.stack([e['step_out'] == CAP - 1 for e in env], 1)
    for b, resets in enumerate([[0], [0, 21], [0]]):
      want = np.zeros(40, bool)
      for r0 in resets:
        nxt = [r for r in resets if r > r0]
        end = nxt[0] if nxt else 40
        want[r0:end:CAP] = True
      np.testing.assert_array_equal(sw[b], want, err_msg=f'row {b}')


# --------------------------------------------------------------------------
# 2. Imagination
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frac,agg', CASES)
def test_2_imagination_rollout(frac, agg):
  train, _, _ = run(frac, agg)
  im = train['imag']
  sw = im['sw'] > 0.5
  M, T = sw.shape
  assert T == H + 1
  # Start states are the replay states the loss prefixes (same alignment).
  rep = train['repfeat']
  Kimag = M // rep.shape[0]
  np.testing.assert_allclose(
      im['starts'], rep[:, -Kimag:].reshape(M, -1), rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(im['deter'][:, 0], im['starts'], atol=1e-6)
  # Switch rule, recomputed from the captured states and decoded goals.
  want, cos, cd = _longhand_switch(im['deter'], im['goals'], im['tau'], T)
  _close_or_tie(sw, want, cos, im['tau'], 'imag switch')
  assert sw[:, 0].all()
  # Reach flags: exactly the reached part of the rule (t >= 1).
  _close_or_tie(im['reached'][:, 1:], (cos >= im['tau'])[:, 1:],
                cos[:, 1:], im['tau'], 'imag reached')
  # Skills and goals constant within a hold.
  for t in range(1, T):
    keep = ~sw[:, t]
    np.testing.assert_array_equal(im['skill'][keep, t], im['skill'][keep, t - 1])
    np.testing.assert_allclose(im['goals'][keep, t], im['goals'][keep, t - 1],
                               atol=1e-6)
  # Hold lengths within the cap; countdown matches the longhand one wherever
  # the switch pattern agrees.
  ok = (sw == want).all(1)
  np.testing.assert_array_equal(im['cd'][ok], cd[ok])
  for b in range(M):
    starts = np.where(sw[b])[0]
    assert (np.diff(np.append(starts, T)) <= CAP).all(), starts
  if frac == 50.0:
    np.testing.assert_array_equal(
        sw, np.broadcast_to(np.arange(T) % CAP == 0, sw.shape))
  if frac == -50.0:
    assert sw.all()
  print(f'frac={frac}: imag switch rate {sw.mean():.3f}, '
        f'reach share {im["reached"][:, 1:].mean():.3f}')


# --------------------------------------------------------------------------
# 3. Manager tensors that reach imag_loss_mgr
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frac,agg', CASES)
def test_3_manager_tensors(frac, agg):
  train, _, _ = run(frac, agg)
  im, mg = train['imag'], train['mgr']
  sw = im['sw'] > 0.5
  M, T = sw.shape
  R, C, n_sw = _pool_longhand(im['rew'], im['con'], sw, agg)
  # The exploration reward samples a code, so pool the tensor the loss used.
  vb = [v for v in train['vbd'] if v['sw'].shape == sw.shape
        and np.array_equal(v['sw'] > 0.5, sw)]
  assert vb, 'imagination pooling call not found'
  # nj.grad traces the loss twice (a discovery pass that is dead code without
  # these probes); appended captures hold both passes, so use the last one --
  # the pass the single-slot captures (imag, mgr, wkr) come from.
  np.testing.assert_allclose(vb[-1]['rew'], im['rew'], rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(vb[-1]['con'], im['con'], rtol=1e-5, atol=1e-6)
  E, _, _ = _pool_longhand(vb[-1]['expl'], im['con'], sw, agg)
  for b in range(M):
    n = n_sw[b]
    starts = np.where(sw[b])[0]
    # Decision j acts at the j-th switch state, with that state's skill.
    np.testing.assert_array_equal(mg['skill'][b, :n], im['skill'][b, starts])
    # Reward column j+1 = pooled reward of decision j; column 0 is padding.
    assert mg['extr'][b, 0] == 0
    np.testing.assert_allclose(mg['extr'][b, 1:n + 1], R[b, :n][:T - 1],
                               rtol=1e-4, atol=1e-5, err_msg=f'extr row {b}')
    np.testing.assert_allclose(mg['expl'][b, 1:n + 1], E[b, :n][:T - 1],
                               rtol=1e-4, atol=1e-5, err_msg=f'expl row {b}')
    # Continuation: con of the start state, then the product over each hold.
    np.testing.assert_allclose(mg['cont'][b, 0], im['con'][b, 0], rtol=1e-6)
    np.testing.assert_allclose(mg['cont'][b, 1:n + 1], C[b, :n][:T - 1],
                               rtol=1e-4, err_msg=f'cont row {b}')
    # Only decisions that own at least one transition are credited.
    n_credit = int(sw[b, :-1].sum())
    np.testing.assert_array_equal(
        mg['switch'][b], (np.arange(T) < n_credit).astype(np.float32))
  if frac == 50.0:
    # Never reached + cap K: identical to the fixed-K Director helpers.
    rew, con, expl = (jnp.asarray(im[k]) for k in ('rew', 'con', 'expl'))
    n = T // CAP + 1
    fx_r = np.asarray(imag_reward_pad(aggregate_mgr_extr_rew(
        rew, con, CAP, without_zeros=True, agg_mode=agg)))
    fx_e = np.asarray(imag_reward_pad(aggregate_mgr_extr_rew(
        jnp.asarray(vb[-1]['expl']), con, CAP, without_zeros=True, agg_mode=agg)))
    fx_c = np.asarray(aggregate_mgr_cont(con, CAP, without_zeros=True))
    np.testing.assert_allclose(mg['extr'][:, :n], fx_r[:, :n], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(mg['expl'][:, :n], fx_e[:, :n], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(mg['cont'][:, :n], fx_c[:, :n], rtol=1e-4)
    np.testing.assert_array_equal(
        mg['skill'][:, :n], im['skill'][:, ::CAP][:, :n])


# --------------------------------------------------------------------------
# 4. Worker tensors that reach imag_loss_wkr
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frac,agg', CASES)
def test_4_worker_tensors(frac, agg):
  train, _, _ = run(frac, agg)
  im, wk = train['imag'], train['wkr']
  sw = im['sw'] > 0.5
  # Boundary mask is the imagination switch mask.
  np.testing.assert_array_equal(wk['sw'] > 0.5, sw)
  # Reward for arriving in state t is the match to the goal held INTO t.
  want = np.zeros_like(wk['rew'])
  want[:, 1:] = _cosmax(im['goals'][:, :-1], im['deter'][:, 1:])
  np.testing.assert_allclose(wk['rew'], want, rtol=1e-4, atol=1e-5)
  np.testing.assert_allclose(wk['con'], im['con'], rtol=1e-6)
  # Old-goal bootstrap differs from the critic only where the goal changed.
  assert 'boot' in wk
  same = np.ones_like(sw); same[:, 1:] = (
      np.abs(im['goals'][:, 1:] - im['goals'][:, :-1]).max(-1) == 0)
  same[:, 0] = False
  np.testing.assert_allclose(wk['boot'][same], wk['val'][same], rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(wk['bootslow'][same], wk['slowval'][same],
                             rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------
# 5. Replay
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frac,agg', CASES)
def test_5_replay_path(frac, agg):
  train, _, batch = run(frac, agg)
  # The replay value path traces the sequence twice (manager, worker), both
  # with the manager's mode; other callers (count tables) sample.
  seqs = [q for q in train['seq'] if q['det'] and not q['down']]
  assert len(seqs) >= 2 and len(seqs) % 2 == 0, len(seqs)
  seqs = seqs[-2:]   # last pass: nj.grad traces the loss twice under probes
  for k in ('skill', 'sw', 'cd'):
    np.testing.assert_array_equal(seqs[0][k], seqs[1][k], err_msg=k)
  sq = seqs[0]
  sw = sq['sw'] > 0.5
  T = sw.shape[1]
  want, cos, _ = _longhand_switch(sq['deter'], sq['goals'], sq['tau'], T)
  _close_or_tie(sw, want, cos, sq['tau'], 'replay switch')
  # The replay pooling call: real rewards of the last T steps, this mask.
  vb = [v for v in train['vbd'] if v['sw'].shape == sw.shape and
        np.array_equal(v['sw'] > 0.5, sw)]
  assert vb, 'replay pooling call not found'
  vb = vb[-1]
  np.testing.assert_allclose(vb['rew'], batch['reward'][:, -T:], rtol=1e-6)
  R, C, _ = _pool_longhand(vb['rew'], vb['con'], sw, agg)
  r_out, _, c_out, valid = vb['out']
  for b in range(sw.shape[0]):
    n = int(sw[b].sum())
    np.testing.assert_allclose(r_out[b, 1:n + 1], R[b, :n][:T - 1], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(c_out[b, 1:n + 1], C[b, :n][:T - 1], rtol=1e-4)
  # Replay manager return: per-env-step discount (term = 1 - pooled cont,
  # disc = 1), not a flat 0.99 per decision.
  rets = [l for l in train['lret'] if l['rew'].shape == r_out.shape
          and np.allclose(l['rew'], r_out)]
  assert rets, 'replay manager lambda_return call not found'
  np.testing.assert_allclose(rets[-1]['term'], 1.0 - c_out, rtol=1e-6, atol=1e-7)
  assert float(rets[-1]['disc']) == 1.0
