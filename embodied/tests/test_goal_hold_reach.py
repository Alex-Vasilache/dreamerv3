"""Reached-or-cap goal holding (``goal_reach_frac``, config block ``goal_hold_reach``).

The manager keeps a goal until the worker has reached it -- cos_max(goal,
deter) >= frac x a running reconstruction ceiling -- or a cap runs out. These
tests pin down the pieces that are easy to get subtly wrong:

  * the switch fires on reach and at the cap, in the env step, in imagination
    and on replay, and the three agree on the rule;
  * with reach switched off (frac so high it never fires) and cap 8 the
    training targets are Director's fixed-K ones;
  * the worker's return for a goal ends at the switch and bootstraps from the
    value under the OLD goal;
  * the replay-side manager return is discounted by hold length.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import embodied.jax.outs as outs
from dreamerv3.hrl.losses import imag_loss_wkr, lambda_return, repl_loss
from dreamerv3.hrl.tensors import (
    goal_reward_cosine_max, variable_block_director_tensors)

from test_goal_ae_agent import make_agent, train_steps

f32 = jnp.float32


class _StubNorm:
  def stats(self):
    return 0.0, 1.0

  def __call__(self, x, update=True, weights=None):
    return 0.0, 1.0


# --------------------------------------------------------------------------
# Worker credit: boundary state belongs to the old goal
# --------------------------------------------------------------------------

def _wkr_ret(rew, con, val, boot, sw, lam=0.95):
  B, T = rew.shape
  act = {'action': jax.nn.one_hot(jnp.zeros((B, T), jnp.int32), 3, dtype=f32)}
  policy = {'action': outs.OneHot(jnp.zeros((B, T, 3), f32))}
  kw = dict(update=True, contdisc=True, slowtar=False, horizon=333, lam=lam,
            actent=3e-4, skill_window=sw)
  if boot is not None:
    kw.update(boot_value=outs.MSE(boot), boot_slowvalue=outs.MSE(boot))
  _, out, _ = imag_loss_wkr(
      act, rew, con, policy, outs.MSE(val), outs.MSE(val),
      _StubNorm(), _StubNorm(), _StubNorm(), **kw)
  return np.asarray(out['wkr_goal_ret'])


def test_worker_return_bootstraps_from_old_goal_value_at_a_switch():
  """One hold of 4 steps ending in a switch at t=4. The return of every step
  of that hold must end at state 4 and bootstrap from ``boot[4]`` (the value
  of state 4 under the OLD goal), never from ``val[4]`` (new goal)."""
  T = 9
  rew = jnp.zeros((1, T), f32).at[0, 4].set(1.0)       # arrival reward
  con = jnp.ones((1, T), f32)
  val = jnp.full((1, T), 100.0)                        # value under the goal in force
  boot = val.at[0, 4].set(7.0)                         # state 4 under the old goal
  sw = jnp.array([[1, 0, 0, 0, 1, 0, 0, 0, 0]], f32)
  ret = _wkr_ret(rew, con, val, boot, sw, lam=1.0)[0]
  np.testing.assert_allclose(ret[:4], 1.0 + 7.0, rtol=1e-6)
  # Without the old-goal bootstrap the same steps would read the new goal.
  ret_old = _wkr_ret(rew, con, val, None, sw, lam=1.0)[0]
  np.testing.assert_allclose(ret_old[:4], 1.0 + 100.0, rtol=1e-6)


def test_worker_dense_path_with_old_goal_boot_equals_split_traj_windows():
  """Regression to Director: on a K-periodic switch pattern, the dense path
  with previous-goal rewards and bootstrap must give exactly the per-window
  (split_traj) returns, where each window of K+1 states keeps ONE goal --
  including the boundary state, whose value is taken under the old goal."""
  B, K, n_win = 2, 8, 2
  T = n_win * K + 1
  rng = np.random.RandomState(0)
  con = jnp.array(rng.uniform(0.95, 0.999, (B, T)).astype(np.float32))
  # Per-window quantities: rew_w[b, w, i] / val_w[b, w, i] for i in 0..K, i.e.
  # the reward for arriving in, and the value of, state w*K+i under goal w.
  rew_w = rng.randn(B, n_win, K + 1).astype(np.float32)
  val_w = rng.randn(B, n_win, K + 1).astype(np.float32)
  # Dense timeline. Goal in force at state t is goal t//K (new goal at the
  # boundary); the goal in force coming INTO state t is goal (t-1)//K.
  val = np.zeros((B, T), np.float32); boot = np.zeros((B, T), np.float32)
  rew = np.zeros((B, T), np.float32)
  for t in range(T):
    w_cur, w_prev = min(t // K, n_win - 1), max(t - 1, 0) // K
    val[:, t] = val_w[:, w_cur, t - w_cur * K]
    boot[:, t] = val_w[:, w_prev, t - w_prev * K]
    rew[:, t] = rew_w[:, w_prev, t - w_prev * K]
  sw = jnp.array([[1.0 if t % K == 0 else 0.0 for t in range(T)]] * B, f32)
  dense = _wkr_ret(jnp.array(rew), con, jnp.array(val), jnp.array(boot), sw)
  for w in range(n_win):
    s = w * K
    seg = lambda x: jnp.array(x[:, s:s + K + 1])
    v = jnp.array(val_w[:, w])
    ref = lambda_return(
        jnp.zeros((B, K + 1), f32), 1 - seg(np.asarray(con)),
        jnp.array(rew_w[:, w]), v, v, 1.0, 0.95)
    np.testing.assert_allclose(
        dense[:, s:s + K], np.asarray(ref), rtol=1e-5, atol=1e-5,
        err_msg=f'window {w}')


def test_repl_loss_reset_ends_the_return_but_keeps_the_loss_weight():
  B, T = 1, 7
  rew = jnp.array([[0, 1, 2, 3, 4, 5, 6]], f32)
  boot = jnp.array([[0, 0, 0, 50, 0, 0, 9]], f32)
  last = jnp.zeros((B, T), bool); term = jnp.zeros((B, T), bool)
  reset = jnp.array([[0, 0, 0, 1, 0, 0, 0]], f32)
  val = outs.MSE(jnp.zeros((B, T), f32))
  kw = dict(update=True, slowreg=0.0, slowtar=True, horizon=100, lam=1.0,
            value_head='wkr_goal')
  los, out, _ = repl_loss(last, term, rew, boot, val, val, _StubNorm(),
                          reset=reset, **kw)
  ret = np.asarray(out['repwkr_goal_ret'])[0]
  d = 0.99
  # Steps 0..2 belong to the hold that ends in state 3: their return stops
  # there and takes boot[3] in full.
  np.testing.assert_allclose(ret[2], 3 + d * 50, rtol=1e-6)
  np.testing.assert_allclose(ret[1], 2 + d * (3 + d * 50), rtol=1e-6)
  # Step 3 starts the next hold and runs to the window end.
  np.testing.assert_allclose(ret[3], 4 + d * (5 + d * (6 + d * 9)), rtol=1e-6)
  # The switch state still trains the critic (weight 1), unlike a real `last`.
  assert float(los['repwkr_goal_value'][0, 3]) > 0
  los2, _, _ = repl_loss(last.at[0, 3].set(True), term, rew, boot, val, val,
                         _StubNorm(), **kw)
  assert float(los2['repwkr_goal_value'][0, 3]) == 0


# --------------------------------------------------------------------------
# Manager return: discount by hold length
# --------------------------------------------------------------------------

def test_replay_manager_return_is_discounted_per_env_step_of_the_hold():
  """Holds of 3, 1 and 5 steps with a constant per-step continuation g: the
  return the replay path builds (``term = 1 - pooled cont``, ``disc = 1``)
  must weigh the n-th decision's reward by g^(steps before it), not g^n."""
  g, T = 0.9, 10
  sw = jnp.array([[1, 0, 0, 1, 1, 0, 0, 0, 0, 1]], f32)   # holds 3, 1, 5
  con = jnp.full((1, T), g)
  rew = jnp.ones((1, T), f32)
  expl = jnp.zeros((1, T), f32)
  r, _, cont, valid = variable_block_director_tensors(
      rew, con, expl, sw, T, agg_mode='sum')
  np.testing.assert_allclose(np.asarray(cont)[0, 1:4], [g**3, g**1, g**5], rtol=1e-5)
  zeros = jnp.zeros_like(r)
  ret = lambda_return(
      jnp.zeros(r.shape, bool), 1.0 - cont, r, zeros, zeros, 1.0, 1.0)
  ret = np.asarray(ret)[0]
  # Pooled SMDP rewards. The pooling weighs the i-th reward of a hold by the
  # continuation of the states up to and including the one it was earned
  # from (``rew[1:] * cumprod(con[:-1])``, Director's abstract_traj), so each
  # hold's sum carries one extra factor g. That is a uniform scale on every
  # reward, the same in imagination and replay.
  R = [g * sum(g**i for i in range(n)) for n in (3, 1, 5)]
  want0 = R[0] + g**3 * (R[1] + g**1 * R[2])
  np.testing.assert_allclose(ret[0], want0, rtol=1e-5)
  np.testing.assert_allclose(ret[1], R[1] + g * R[2], rtol=1e-5)
  # And that is exactly g x the plain per-step discounted return of the raw
  # stream: the discount depends on elapsed env steps, not on decisions.
  np.testing.assert_allclose(want0, g * sum(g**i for i in range(9)), rtol=1e-5)
  # A per-DECISION discount (the old replay path) would instead give:
  per_decision = R[0] + g * (R[1] + g * R[2])
  assert abs(per_decision - want0) > 0.5
  np.testing.assert_allclose(np.asarray(valid)[0, :4], [1, 1, 1, 0])


# --------------------------------------------------------------------------
# Agent level: the switch rule in env, imagination and replay
# --------------------------------------------------------------------------

def _agent(frac, cap=4, **kw):
  ag = dict(variable_goal_length=True, goal_duration_fixed=cap,
            goal_reach_frac=frac, mgr_reward_agg='sum', imag_length=8)
  ag.update(kw)
  return make_agent(agent=ag)


@pytest.fixture(scope='module')
def trained():
  """Agents after one real train step, keyed by reach fraction."""
  cache = {}
  def get(frac, cap=4):
    key = (frac, cap)
    if key not in cache:
      agent = _agent(frac, cap)
      mets = train_steps(agent, steps=3)
      cache[key] = (agent, {k: float(np.asarray(v)) for k, v in mets.items()
                            if np.asarray(v).size == 1})
    return cache[key]
  return get


def _policy_switch_steps(agent, steps=13):
  B = 2
  carry = agent.init_policy(B)
  rng = np.random.default_rng(0)
  goals = []
  for t in range(steps):
    obs = {}
    for k, sp in agent.obs_space.items():
      if sp.dtype == bool:
        obs[k] = np.zeros((B,) + sp.shape, bool)
      elif np.issubdtype(sp.dtype, np.integer):
        obs[k] = rng.integers(0, 2, (B,) + sp.shape).astype(sp.dtype)
      else:
        obs[k] = rng.normal(0, 1, (B,) + sp.shape).astype(sp.dtype)
    obs['is_first'] = np.full((B,), t == 0)
    carry, act, out = agent.policy(carry, obs, mode='train')
    g = np.concatenate([np.asarray(x) for x in jax.tree.leaves(
        carry[4]['goal_deter'])], 0)
    goals.append(g.reshape(B, -1))
  goals = np.stack(goals, 1)                               # (B, steps, D)
  changed = np.abs(np.diff(goals, axis=1)).max(-1) > 1e-6  # (B, steps-1)
  return changed


def test_metrics_and_training_run_with_reach(trained):
  agent, mets = trained(0.8)
  for key in ('goal/reach_ceil', 'goal/reach_tau', 'goal/hold_reach_frac',
              'goal/hold_len_imag', 'goal/hold_len_repl'):
    hits = [k for k in mets if k.endswith(key)]
    assert hits, (key, sorted(mets)[:40])
    assert np.isfinite(mets[hits[0]]), key
  ceil = mets[[k for k in mets if k.endswith('goal/reach_ceil')][0]]
  tau = mets[[k for k in mets if k.endswith('goal/reach_tau')][0]]
  assert 0 < ceil <= 1.0 and abs(tau - 0.8 * ceil) < 1e-5


def test_cap_only_when_reach_never_fires(trained):
  """frac far above any possible match: the hold always runs to the cap."""
  agent, mets = trained(50.0)
  get = lambda key: mets[[k for k in mets if k.endswith(key)][0]]
  assert get('goal/hold_reach_frac') == 0.0
  # imag_length 8, cap 4 -> decisions at 0, 4, 8 of 9 states.
  np.testing.assert_allclose(get('goal/hold_len_imag'), 9 / 3, rtol=1e-5)
  changed = _policy_switch_steps(agent)
  # Goal changes exactly entering steps 4, 8, 12 (diff index t-1).
  # (A resample can land on the same code, so only "never elsewhere" is exact.)
  at_cap = np.zeros(changed.shape[1], bool); at_cap[[3, 7, 11]] = True
  assert not changed[:, ~at_cap].any(), changed.astype(int)
  assert changed[:, at_cap].any(), changed.astype(int)


def test_switch_every_step_when_always_reached(trained):
  """frac < 0 makes every state count as reached (tau below any cos_max):
  the manager re-decides at every step, in imagination, replay and the env."""
  agent, mets = trained(-50.0)
  get = lambda key: mets[[k for k in mets if k.endswith(key)][0]]
  np.testing.assert_allclose(get('goal/hold_len_imag'), 1.0, rtol=1e-5)
  np.testing.assert_allclose(get('goal/hold_len_repl'), 1.0, rtol=1e-5)
  np.testing.assert_allclose(get('goal/hold_reach_frac'), 1.0, rtol=1e-5)
  changed = _policy_switch_steps(agent)
  assert changed.mean() > 0.5, changed.astype(int)
