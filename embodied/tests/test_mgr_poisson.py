"""The unimodal Poisson manager policy (Zhu et al. 2024, Eqs. 8-10).

Three layers, because the ways this can fail live at three different levels:

  * the distribution itself -- is it actually unimodal, does the mode track the
    rate, does the temperature span the entropy range the adapter needs;
  * the head -- does it produce the same event shape and the same
    ``minent``/``maxent`` contract as ``onehot``, so it is a true drop-in;
  * the real agent -- does the config block reach the manager, do gradients
    reach the new parameters, and is ``mgr_ent_loss`` actually alive.

That last one is not paranoia. ``ManagerRingHead`` (e478) shipped with
``train/mgr_ent_loss`` identically 0 for its whole run because ``losses.py``
silently skips any head without ``minent``/``maxent``. The experiment measured
nothing. ``test_entropy_regularizer_is_alive`` is the assertion that would have
caught it.
"""
import functools

import elements
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ruamel.yaml as yaml

import embodied.jax.outs as outs
from dreamerv3.agent import Agent

CONFIGS = elements.Path(__file__).parent.parent.parent / 'dreamerv3/configs.yaml'
L, C = 8, 8


def dist(rate, logtau=0.0, classes=C, **kw):
  rate = jnp.asarray(rate, jnp.float32)
  logtau = jnp.broadcast_to(jnp.asarray(logtau, jnp.float32), rate.shape)
  return outs.PoissonOnehot(rate, logtau, classes, **kw)


def probs(d):
  return np.asarray(jax.nn.softmax(d.dist.logits, -1))


# --------------------------------------------------------------------------
# 1. the distribution
# --------------------------------------------------------------------------

def unimodal(p):
  """No ascent after a descent, i.e. mass decays away from the mode."""
  d = np.sign(np.diff(p))
  d = d[d != 0]
  return not any(d[i] < 0 < d[i + 1] for i in range(len(d) - 1))


def test_is_unimodal_everywhere():
  # Span the whole usable range of both parameters, including the extremes the
  # network can reach after the softplus offset.
  for r in np.linspace(-8.0, 8.0, 41):
    for lt in np.linspace(-4.0, 4.0, 17):
      p = probs(dist(np.full((1,), r), np.full((1,), lt)))[0]
      assert unimodal(p), (r, lt, p)


def test_mode_is_monotone_in_the_rate():
  """The scalar IS the position on the SOM line: raising it never moves the
  mode backwards, and the full 0..C-1 range is reachable."""
  rates = np.linspace(-6.0, 8.0, 400)
  modes = [int(probs(dist(np.full((1,), r)))[0].argmax()) for r in rates]
  assert modes == sorted(modes), 'mode is not monotone in the rate'
  assert set(modes) == set(range(C)), 'unreachable classes: %s' % (
      set(range(C)) - set(modes))


def test_probability_decays_with_distance_from_the_mode():
  """The property the SOM line is supposed to exploit: credit for class j
  spreads to j-1/j+1 more than to distant classes."""
  for r in (-2.0, 0.0, 2.0, 4.0):
    p = probs(dist(np.full((1,), r)))[0]
    m = int(p.argmax())
    for lo, hi in ((m - 1, m - 2), (m + 1, m + 2)):
      if 0 <= hi < C and 0 <= lo < C:
        assert p[lo] >= p[hi], (r, m, lo, hi, p)


def test_temperature_is_a_monotone_entropy_dial():
  """Why the head emits tau at all. At FIXED tau the entropy is non-monotone in
  the rate (it peaks near the middle class), so an entropy constraint would
  quietly become a constraint on WHICH class the manager may prefer. With tau
  free, entropy is monotone in tau at every rate."""
  taus = np.linspace(-6.0, 6.0, 60)
  for r in (-4.0, -1.0, 0.0, 2.0, 5.0):
    ent = np.array([float(dist(np.full((1,), r), np.full((1,), t)).entropy()[0])
                    for t in taus])
    assert np.all(np.diff(ent) > -1e-5), 'entropy not monotone in tau at r=%s' % r


def test_entropy_spans_the_normalized_range_the_adapter_regulates():
  """``mgr_actent`` drives normalized entropy to ``manager_actent_target``
  (0.5). That is only meaningful if the target sits strictly inside the range
  the head can actually reach."""
  target = 0.5 * np.log(C)
  hi = float(dist(jnp.zeros((1,)), jnp.full((1,), 8.0)).entropy()[0])
  lo = float(dist(jnp.zeros((1,)), jnp.full((1,), -8.0)).entropy()[0])
  assert hi / np.log(C) > 0.99, hi
  assert lo < target < hi, (lo, target, hi)


def test_entropy_floor_stays_below_the_actent_target():
  """The e479 hazard, and the reason ``tau_min`` is 0.02 rather than 0.1.

  At integer ``lam`` two adjacent classes are exactly tied, so cooling the head
  yields a 50/50 split rather than a one-hot and the normalized entropy cannot
  drop below log(2)/log(C) = 0.333. If ``manager_actent_target`` (0.5) were to
  fall BELOW the worst-case floor, the controller would be chasing an
  unreachable entropy and its multiplier would rail -- which is exactly how the
  smoothed-credit run e479 was lost. Check the floor over the whole rate range,
  not just at the ties.
  """
  rates = np.linspace(-8.0, 8.0, 400)
  cold = np.full_like(rates, -12.0)
  ent = np.asarray(dist(rates.astype(np.float32),
                        cold.astype(np.float32)).entropy()) / np.log(C)
  assert ent.max() < 0.45, 'entropy floor %.3f leaves no room under the 0.5 target' % ent.max()
  # and the floor really is the tie value, not something smaller by accident
  assert abs(ent.max() - np.log(2) / np.log(C)) < 0.02, ent.max()


def test_entropy_at_init_is_high():
  """A fresh head must explore, like a fresh categorical head does. The rate
  offset exists for this: without it softplus(0)=0.69 pins every block's mode
  to class 0 before the first gradient step."""
  d = dist(jnp.zeros((L,)), jnp.zeros((L,)))
  ent = np.asarray(d.entropy())
  assert (ent / np.log(C) > 0.85).all(), ent
  assert (np.asarray(probs(d)).argmax(-1) == (C - 1) // 2).all()


def test_logp_is_a_normalized_log_probability():
  d = dist(jnp.array([-1.0, 0.0, 3.0]))
  onehots = jnp.eye(C)[None].repeat(3, 0)          # (3, C, C)
  lp = jnp.stack([d.logp(onehots[:, i]) for i in range(C)], -1)
  assert np.allclose(np.exp(np.asarray(lp)).sum(-1), 1.0, atol=1e-5)


def test_sample_is_onehot_and_matches_the_shape_of_a_plain_head():
  d = dist(jnp.zeros((4, L)), jnp.zeros((4, L)))
  ref = outs.OneHot(jnp.zeros((4, L, C)))
  s = d.sample(jax.random.PRNGKey(0))
  assert s.shape == ref.sample(jax.random.PRNGKey(0)).shape == (4, L, C)
  assert d.pred().shape == ref.pred().shape
  assert d.entropy().shape == ref.entropy().shape == (4, L)
  hard = np.asarray(jax.lax.stop_gradient(s))
  assert np.allclose(hard.sum(-1), 1.0)
  assert np.isin(hard, (0.0, 1.0)).all()


def test_kl_against_itself_is_zero_and_otherwise_positive():
  a = dist(jnp.array([0.0, 2.0]))
  b = dist(jnp.array([1.0, -1.0]))
  assert np.allclose(np.asarray(a.kl(a)), 0.0, atol=1e-6)
  assert (np.asarray(a.kl(b)) > 0).all()


def test_no_nans_at_extreme_parameters():
  """log(lam) is the one place this can blow up; the rate floor guards it."""
  for r in (-1e4, -50.0, 50.0, 1e4):
    for t in (-1e4, 1e4):
      d = dist(jnp.full((2,), r), jnp.full((2,), t))
      for x in (d.dist.logits, d.entropy(), d.pred()):
        assert np.isfinite(np.asarray(x)).all(), (r, t, x)


def test_gradients_flow_to_both_parameters():
  def loss(r, t):
    return dist(r, t).logp(jnp.eye(C)[5][None]).sum()
  g_r, g_t = jax.grad(loss, (0, 1))(jnp.zeros((1,)), jnp.zeros((1,)))
  assert np.isfinite(g_r).all() and np.isfinite(g_t).all()
  assert abs(float(g_r[0])) > 1e-6, 'rate gets no gradient'
  assert abs(float(g_t[0])) > 1e-6, 'temperature gets no gradient'


# --------------------------------------------------------------------------
# 2. the head
# --------------------------------------------------------------------------

def build_head(impl):
  import ninjax as nj
  from embodied.jax import heads, nets
  space = elements.Space(np.float32, (L, C), 0.0, 1.0)
  head = heads.Head(space, impl, name='h')
  fn = lambda x: head(x)
  x = jnp.asarray(
      np.random.default_rng(0).normal(size=(2, 16)), nets.COMPUTE_DTYPE)
  params = nj.init(fn)({}, x, seed=0)
  return params, nj.pure(fn)(params, x, seed=0)[1]


def test_head_is_a_drop_in_for_onehot():
  p_pois, o_pois = build_head('poisson')
  p_hot, o_hot = build_head('onehot')
  assert o_pois.pred().shape == o_hot.pred().shape
  inner = lambda o: o.output if isinstance(o, outs.Agg) else o
  assert inner(o_pois).minent == inner(o_hot).minent
  assert inner(o_pois).maxent == inner(o_hot).maxent
  # 2 scalars per block instead of C logits: strictly fewer output parameters.
  size = lambda p: sum(np.prod(v.shape) for k, v in p.items() if 'kernel' in k)
  assert size(p_pois) < size(p_hot)


def test_head_exposes_rate_and_logtau_not_logits():
  params, _ = build_head('poisson')
  keys = ' '.join(params)
  assert 'rate' in keys and 'logtau' in keys
  assert '/logits/' not in keys, keys


# --------------------------------------------------------------------------
# 3. the real agent
# --------------------------------------------------------------------------

def make_config(*blocks, **overrides):
  raw = yaml.YAML(typ='safe').load(CONFIGS.read())
  config = elements.Config(raw['defaults'])
  for block in ('debug',) + blocks:
    config = config.update(raw[block])
  config = config.update(overrides) if overrides else config
  return elements.Config(
      **config.agent, logdir='/tmp/mgr_poisson_test', seed=0, jax=config.jax,
      batch_size=config.batch_size, batch_length=config.batch_length,
      replay_context=config.replay_context, report_length=config.report_length,
      replica=0, replicas=1)


def make_agent(*blocks, **overrides):
  from embodied.envs import dummy
  env = dummy.Dummy('disc', size=(64, 64), length=20)
  keep = ('image', 'vector', 'reward', 'is_first', 'is_last', 'is_terminal')
  obs_space = {k: v for k, v in env.obs_space.items() if k in keep}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  return Agent(obs_space, act_space, make_config(*blocks, **overrides))


def make_batch(agent, seed=0):
  batch = agent.config.batch_size
  length = agent.config.batch_length + agent.config.replay_context
  rng = np.random.default_rng(seed)
  data = {}
  for key, space in agent.spaces.items():
    shape = (batch, length, *space.shape)
    if space.dtype == bool:
      data[key] = np.zeros(shape, bool)
    elif np.issubdtype(space.dtype, np.integer):
      hi = 2 if space.classes is None else int(
          np.asarray(space.classes).flatten()[0])
      data[key] = rng.integers(0, max(hi, 2), shape).astype(space.dtype)
    else:
      data[key] = rng.normal(0, 1, shape).astype(space.dtype)
  data['is_first'] = np.zeros((batch, length), bool)
  data['is_first'][:, 0] = True
  data['is_last'] = np.zeros((batch, length), bool)
  data['is_terminal'] = np.zeros((batch, length), bool)
  return data


@functools.lru_cache(maxsize=None)
def trained(*blocks):
  """Build one agent per arm, train it, and cache. Returns the parameters as
  they were BEFORE training too, so the gradient test has a baseline."""
  from embodied.jax import internal
  agent = make_agent(*blocks)
  before = {k: np.asarray(v).copy() for k, v in agent.params.items()}
  carry = agent.init_train(agent.config.batch_size)
  base = internal.device_put(make_batch(agent), agent.train_sharded)
  last = {}
  for i in range(4):
    data = {**base, 'seed': agent._seeds(i, agent.train_mirrored)}
    carry, _, mets = agent.train(carry, data)
    if mets:
      last = mets
  assert last, 'no metrics'
  return agent, before, last


ARM = ('goal_som_line', 'mgr_poisson')


def mgr_head_params(params):
  """Manager-head weights only: no optimizer moments, which live under the same
  leaf names below ``opt/`` and would otherwise be mistaken for weights."""
  out = {k: v for k, v in params.items()
         if 'manager_pol/head/' in k and '/opt/' not in k}
  assert out, 'no manager_pol head params among %s' % sorted(
      k for k in params if 'manager_pol' in k)
  return out


def test_agent_trains_with_the_poisson_manager():
  _, _, mets = trained(*ARM)
  bad = {}
  for k, v in mets.items():
    v = np.asarray(v)
    if v.dtype.kind == 'f' and not np.isfinite(v).all():
      bad[k] = v
  assert not bad, 'non-finite metrics: %s' % sorted(bad)


def test_manager_head_really_swapped():
  agent, _, _ = trained(*ARM)
  head = mgr_head_params(agent.params)
  assert head, 'no manager_pol head params'
  assert any('rate' in k for k in head), sorted(head)
  assert any('logtau' in k for k in head), sorted(head)
  assert not any('/logits/' in k for k in head), sorted(head)


def test_entropy_regularizer_is_alive():
  """The e478 regression test: a head without minent/maxent is silently skipped
  by ``losses.py`` and the entropy loss is identically 0 for the whole run."""
  _, _, mets = trained(*ARM)
  key = [k for k in mets if k.endswith('mgr_ent_norm_skill_mean')]
  assert key, 'no normalized manager entropy metric: %s' % sorted(mets)
  ent = float(np.asarray(mets[key[0]]))
  assert 0.0 < ent <= 1.0, ent
  assert [k for k in mets if 'mgr_actent_skill' in k], 'entropy adapter never ran'


def test_gradients_reach_the_new_manager_parameters():
  agent, before, _ = trained(*ARM)
  after = mgr_head_params(agent.params)
  moved = [k for k, v in after.items()
           if not np.allclose(before[k], np.asarray(v))]
  assert any('rate' in k for k in moved), 'rate never updated: %s' % sorted(after)
  assert any('logtau' in k for k in moved), 'logtau never updated: %s' % sorted(after)


def test_som_line_baseline_still_builds():
  """Guard against the config edit breaking the arm it layers on."""
  _, _, mets = trained('goal_som_line')
  assert mets


# --------------------------------------------------------------------------
# 4. the discretized Gaussian alternative
# --------------------------------------------------------------------------

def gdist(loc, logscale=0.0, classes=C, **kw):
  loc = jnp.asarray(loc, jnp.float32)
  logscale = jnp.broadcast_to(jnp.asarray(logscale, jnp.float32), loc.shape)
  return outs.GaussianOnehot(loc, logscale, classes, **kw)


def test_gaussian_is_unimodal_everywhere():
  for m in np.linspace(-8.0, 8.0, 41):
    for s in np.linspace(-4.0, 4.0, 17):
      p = probs(gdist(np.full((1,), m), np.full((1,), s)))[0]
      assert unimodal(p), (m, s, p)


def test_gaussian_mode_is_monotone_and_covers_every_class():
  locs = np.linspace(-6.0, 6.0, 400)
  modes = [int(probs(gdist(np.full((1,), m)))[0].argmax()) for m in locs]
  assert modes == sorted(modes), 'mode not monotone in loc'
  assert set(modes) == set(range(C)), set(range(C)) - set(modes)


def test_gaussian_scale_is_a_monotone_entropy_dial():
  scales = np.linspace(-6.0, 6.0, 60)
  for m in (-3.0, 0.0, 3.0):
    ent = np.array([float(gdist(np.full((1,), m), np.full((1,), s)).entropy()[0])
                    for s in scales])
    assert np.all(np.diff(ent) > -1e-5), 'entropy not monotone in scale at %s' % m


def test_gaussian_entropy_floor_stays_below_the_actent_target():
  """Same tie hazard as the Poisson head: at loc exactly between two classes
  they are tied, so the floor is log2/logC however small the scale gets."""
  locs = np.linspace(-8.0, 8.0, 400)
  cold = np.full_like(locs, -12.0)
  ent = np.asarray(gdist(locs.astype(np.float32),
                         cold.astype(np.float32)).entropy()) / np.log(C)
  assert ent.max() < 0.45, 'entropy floor %.3f leaves no room under 0.5' % ent.max()
  assert abs(ent.max() - np.log(2) / np.log(C)) < 0.02, ent.max()


def test_gaussian_credit_is_symmetric_at_an_integer_class():
  """The property the Poisson cannot provide: on an ordered code with no
  preferred direction, both neighbours of the mode get equal credit."""
  for c in range(1, C - 1):
    p = probs(gdist(np.full((1,), float(c) - 0.5 * (C - 1))))[0]
    assert int(p.argmax()) == c, (c, p.argmax())
    assert abs(np.log(p[c + 1] / p[c - 1])) < 1e-4, (c, p[c - 1], p[c + 1])


def test_gaussian_is_more_symmetric_than_the_poisson():
  """The measured claim in the mgr_gaussian config block."""
  def skew(fn):
    worst = 0.0
    for c in (2, 3, 4, 5):
      best = np.inf
      for a in np.linspace(-6, 6, 60):
        for b in np.linspace(-4, 4, 40):
          p = probs(fn(np.full((1,), a), np.full((1,), b)))[0]
          if int(p.argmax()) != c:
            continue
          h = -(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(C)
          if abs(h - 0.5) > 0.03:
            continue
          best = min(best, abs(np.log(p[c + 1] / p[c - 1])))
      if np.isfinite(best):
        worst = max(worst, best)
    return worst
  assert skew(gdist) < skew(dist), 'gaussian should spread credit more evenly'


def test_gaussian_head_is_a_drop_in_for_onehot():
  p_g, o_g = build_head('gaussian')
  p_h, o_h = build_head('onehot')
  assert o_g.pred().shape == o_h.pred().shape
  inner = lambda o: o.output if isinstance(o, outs.Agg) else o
  assert inner(o_g).minent == inner(o_h).minent
  assert inner(o_g).maxent == inner(o_h).maxent
  keys = ' '.join(p_g)
  assert 'loc' in keys and 'logscale' in keys
  assert '/logits/' not in keys, keys


GARM = ('goal_som_lipvq_line_prod', 'mgr_gaussian')


def test_gaussian_agent_trains_and_the_regularizer_is_alive():
  _, _, mets = trained(*GARM)
  bad = {}
  for k, v in mets.items():
    v = np.asarray(v)
    if v.dtype.kind == 'f' and not np.isfinite(v).all():
      bad[k] = v
  assert not bad, 'non-finite metrics: %s' % sorted(bad)
  key = [k for k in mets if k.endswith('mgr_ent_norm_skill_mean')]
  assert key, 'entropy regularizer skipped the head (the e478 failure)'
  assert 0.0 < float(np.asarray(mets[key[0]])) <= 1.0


def test_gaussian_manager_head_really_swapped_and_learns():
  agent, before, _ = trained(*GARM)
  head = mgr_head_params(agent.params)
  assert any('loc' in k for k in head), sorted(head)
  assert any('logscale' in k for k in head), sorted(head)
  assert not any('/logits/' in k for k in head), sorted(head)
  moved = [k for k, v in head.items()
           if not np.allclose(before[k], np.asarray(v))]
  assert any('loc' in k for k in moved), 'loc never updated'
  assert any('logscale' in k for k in moved), 'logscale never updated'


# --------------------------------------------------------------------------
# 5. the Student-t / Cauchy alternative
# --------------------------------------------------------------------------

def tdist(loc, logscale=0.0, classes=C, nu=1.0, **kw):
  loc = jnp.asarray(loc, jnp.float32)
  logscale = jnp.broadcast_to(jnp.asarray(logscale, jnp.float32), loc.shape)
  return outs.StudentTOnehot(loc, logscale, classes, nu=nu, **kw)


# loc defaults to -0.5 because loc_init shifts by (C-1)/2 = 3.5, so -0.5 puts
# the mode ON class 3 rather than at 3.5, where two classes tie exactly.
def _scale_for_entropy(fn, target=0.5, lo=-8.0, hi=8.0, loc=-0.5):
  """Bisect logscale until normalized entropy hits target (entropy is monotone
  in the scale, which is what test_studentt_scale_is_a_monotone_entropy_dial
  independently asserts)."""
  ent = lambda s: float(fn(np.full((1,), loc, np.float32),
                          np.full((1,), s, np.float32)).entropy()[0]) / np.log(C)
  for _ in range(80):
    mid = 0.5 * (lo + hi)
    if ent(mid) < target:
      lo = mid
    else:
      hi = mid
  return 0.5 * (lo + hi)


def test_studentt_is_unimodal_everywhere():
  for nu in (0.5, 1.0, 3.0, 30.0):
    for m in np.linspace(-8.0, 8.0, 21):
      for s in np.linspace(-4.0, 4.0, 9):
        p = probs(tdist(np.full((1,), m), np.full((1,), s), nu=nu))[0]
        assert unimodal(p), (nu, m, s, p)


def test_studentt_mode_is_monotone_and_covers_every_class():
  locs = np.linspace(-6.0, 6.0, 400)
  modes = [int(probs(tdist(np.full((1,), m)))[0].argmax()) for m in locs]
  assert modes == sorted(modes), 'mode not monotone in loc'
  assert set(modes) == set(range(C)), set(range(C)) - set(modes)


def test_studentt_scale_is_a_monotone_entropy_dial():
  scales = np.linspace(-6.0, 6.0, 60)
  for m in (-3.0, 0.0, 3.0):
    ent = np.array([float(tdist(np.full((1,), m), np.full((1,), s)).entropy()[0])
                    for s in scales])
    assert np.all(np.diff(ent) > -1e-5), 'entropy not monotone in scale at %s' % m


def test_studentt_credit_is_symmetric_at_an_integer_class():
  for c in range(1, C - 1):
    p = probs(tdist(np.full((1,), float(c) - 0.5 * (C - 1))))[0]
    assert int(p.argmax()) == c, (c, p.argmax())
    assert abs(np.log(p[c + 1] / p[c - 1])) < 1e-4, (c, p[c - 1], p[c + 1])


def test_large_nu_reproduces_the_gaussian():
  """nu -> inf IS GaussianOnehot, so mgr_studentt with a large nu is the
  control arm for mgr_gaussian. Anything else means the two heads are not on
  the same parameterization and an A/B between them would not be single-factor.
  """
  for m in np.linspace(-5.0, 5.0, 21):
    for s in (-2.0, 0.0, 2.0):
      loc, ls = np.full((1,), m, np.float32), np.full((1,), s, np.float32)
      pt = probs(tdist(loc, ls, nu=1e7))[0]
      pg = probs(gdist(loc, ls))[0]
      assert np.allclose(pt, pg, atol=2e-4), (m, s, pt, pg)


def test_studentt_tail_keeps_decaying_where_the_gaussian_underflows():
  """The property the whole head exists for: mass in the far region that is
  still ORDERED, so a distant draw says which direction is better.

  A uniform mixture also puts mass out there but flattens it -- that is the
  contrast asserted at the end.
  """
  at = lambda fn: probs(fn(np.full((1,), -0.5, np.float32),
                           np.full((1,), _scale_for_entropy(fn), np.float32)))[0]
  p = at(tdist)
  tail = p[int(p.argmax()):]
  assert np.all(np.diff(tail) < 0), 'tail is not strictly decreasing: %s' % tail
  # Reach: the far class is sampled at a rate that is actually observable,
  # unlike the Gaussian's ~2e-08 (~0.01 draws in a 4M-step run).
  assert p[-1] > 1e-3, p[-1]
  assert at(gdist)[-1] < 1e-5, at(gdist)[-1]
  # And with a uniform floor the far classes go FLAT, which is what makes them
  # directionless: consecutive far classes sit at the same u/C. Measured at the
  # same entropy, the student-t still falls by ~1.8x per step out there.
  pm = at(functools.partial(gdist, unimix=0.075))
  assert pm[-2] / pm[-1] < 1.1, 'unimix far tail should be flat: %s' % pm[-3:]
  assert p[-2] / p[-1] > 1.3, 'student-t far tail should stay ordered: %s' % p[-3:]


def test_studentt_far_class_actually_moves_the_mean():
  """d p_j / d loc at the far class -- the quantity that decides whether a
  distant sample can steer the policy, since the REINFORCE gradient on loc is
  sum_j A_j * d p_j / d loc.

  Gaussian: ~2e-07. Gaussian + unimix: ~0 (the draw is attributed to the
  uniform component, which does not depend on loc). Student-t: ~3e-03.
  """
  def dp_far(fn):
    ls = _scale_for_entropy(fn)
    e, m = 1e-3, -0.5   # mode on class 3, as everywhere else here
    hi = probs(fn(np.full((1,), m + e, np.float32), np.full((1,), ls, np.float32)))[0]
    lo = probs(fn(np.full((1,), m - e, np.float32), np.full((1,), ls, np.float32)))[0]
    return abs(float((hi[-1] - lo[-1]) / (2 * e)))
  t = dp_far(tdist)
  g = dp_far(gdist)
  u = dp_far(functools.partial(gdist, unimix=0.075))
  assert t > 100 * g, 'student-t should give the far class real leverage (%g vs %g)' % (t, g)
  assert t > 100 * u, 'unimix far mass should be directionless (%g vs %g)' % (t, u)


def test_studentt_entropy_floor_stays_below_its_actent_target():
  """THE constraint that ties nu to the entropy target.

  At a tie -- loc exactly between two classes -- shrinking the scale does not
  give a one-hot. For this family the limit is p_j/p_mode -> (d_mode/d_j)^(nu+1)
  and it does NOT depend on scale_min, so it is a bound on nu alone. It has to
  sit under manager_actent_target or the adapter rails (e479). The shipped arm
  is nu=1 at target 0.7. Scanned over the CLASS RANGE, since outside it the
  mode is clamped to an edge class.
  """
  target = 0.7                       # mgr_studentt raises it from the default
  locs = np.linspace(0.0, C - 1.0, 801) - 0.5 * (C - 1)   # mode over classes 0..7
  cold = np.full_like(locs, -12.0)
  ent = np.asarray(tdist(locs.astype(np.float32), cold.astype(np.float32),
                         scale_min=0.1).entropy()) / np.log(C)
  assert ent.max() < target, (
      'tie floor %.3f is at or above the %.2f target -- the adapter will rail'
      % (ent.max(), target))
  assert ent.max() < 0.65, 'floor %.3f leaves too little headroom' % ent.max()
  # A confident state on an integer class can still commit, at scale_min 0.1.
  assert ent.min() < 0.10, ent.min()


def test_nu1_needs_the_raised_target_and_the_bounds_are_where_we_think():
  """Guards the two numbers the mgr_studentt block depends on: nu=1 floors at
  0.601, so it needs a target above 0.6, and at the DEFAULT target of 0.5 it
  would instead need nu > 1.558.

  This is not hypothetical -- nu=1 at target 0.5 was the first thing tried, and
  22.7% of the loc range sits above target there.
  """
  def tie_floor(nu, scale_min=0.1):
    # loc 0 -> mode at 3.5, a tie. scale_min is what the head can actually
    # reach, so this is the real floor rather than the s -> 0 limit (0.595).
    return float(np.asarray(tdist(
        np.zeros((1,), np.float32), np.full((1,), -12.0, np.float32),
        nu=nu, scale_min=scale_min).entropy())[0]) / np.log(C)
  # nu=1 clears 0.7 but not 0.5, and not 0.6 either -- 0.6 lands just under it.
  assert abs(tie_floor(1.0) - 0.601) < 0.01, tie_floor(1.0)
  assert 0.6 < tie_floor(1.0) < 0.7, tie_floor(1.0)
  # The alternative route: keep target 0.5 and raise nu past ~1.558.
  assert tie_floor(1.5) > 0.5 > tie_floor(1.6), (tie_floor(1.5), tie_floor(1.6))
  # Monotone in nu, so a single bound is meaningful.
  fl = [tie_floor(n) for n in (1.0, 1.5, 2.0, 2.5, 3.0, 5.0)]
  assert all(a > b for a, b in zip(fl, fl[1:])), fl


def test_cauchy_beats_the_gaussian_at_a_matched_entropy():
  """The claim the arm rests on, and the reason for the raised target: at the
  SAME entropy the Cauchy is more decisive, keeps more ordering, AND reaches
  further. If this ever inverts, the design argument is gone."""
  gs = _scale_for_entropy(gdist, target=0.7)
  ts = _scale_for_entropy(tdist, target=0.7)
  at = lambda fn, s: probs(fn(np.full((1,), -0.5, np.float32),
                              np.full((1,), s, np.float32)))[0]
  g, t = at(gdist, gs), at(tdist, ts)
  mode = int(g.argmax())
  assert t[mode] > g[mode], ('cauchy should be more decisive', t[mode], g[mode])
  assert (t[mode] / t[mode + 1]) > 2 * (g[mode] / g[mode + 1]), (
      'cauchy should keep more ordering', t[mode] / t[mode + 1],
      g[mode] / g[mode + 1])
  assert t[-1] > 20 * g[-1], ('cauchy should reach further', t[-1], g[-1])


def test_studentt_scale_min_must_be_lower_than_the_gaussians():
  """A separate floor from the tie bound: the heavy tail leaks mass, so at the
  Gaussian's scale_min=0.3 a CONFIDENT state cannot commit. This one scale_min
  does fix, which is why the config block ships 0.1."""
  cold = np.full((1,), -12.0, np.float32)
  on_class = np.full((1,), -0.5, np.float32)     # mode exactly on class 3
  ent = lambda sm: float(np.asarray(
      tdist(on_class, cold, scale_min=sm).entropy())[0]) / np.log(C)
  assert ent(0.3) > 0.15, 'expected a hard floor at scale_min 0.3: %.3f' % ent(0.3)
  # What matters is headroom under the arm's own target (0.7), not parity with
  # the Gaussian: a heavier tail always leaks more, and at nu=1 this floor is
  # 0.079 against the Gaussian's 0.012. Both leave the adapter plenty of room.
  assert ent(0.1) < 0.15, ent(0.1)
  assert ent(0.1) < 0.5 * ent(0.3), (ent(0.1), ent(0.3))


def test_studentt_head_is_a_drop_in_for_onehot():
  p_t, o_t = build_head('studentt')
  p_h, o_h = build_head('onehot')
  assert o_t.pred().shape == o_h.pred().shape
  inner = lambda o: o.output if isinstance(o, outs.Agg) else o
  assert inner(o_t).minent == inner(o_h).minent
  assert inner(o_t).maxent == inner(o_h).maxent
  keys = ' '.join(p_t)
  assert 'loc' in keys and 'logscale' in keys
  assert '/logits/' not in keys, keys


TARM = ('goal_som_lipvq_line_prod', 'mgr_studentt')


def test_studentt_agent_trains_and_the_regularizer_is_alive():
  _, _, mets = trained(*TARM)
  bad = {}
  for k, v in mets.items():
    v = np.asarray(v)
    if v.dtype.kind == 'f' and not np.isfinite(v).all():
      bad[k] = v
  assert not bad, 'non-finite metrics: %s' % sorted(bad)
  key = [k for k in mets if k.endswith('mgr_ent_norm_skill_mean')]
  assert key, 'entropy regularizer skipped the head (the e478 failure)'
  assert 0.0 < float(np.asarray(mets[key[0]])) <= 1.0


def test_studentt_manager_head_really_swapped_and_learns():
  agent, before, _ = trained(*TARM)
  head = mgr_head_params(agent.params)
  assert any('loc' in k for k in head), sorted(head)
  assert any('logscale' in k for k in head), sorted(head)
  assert not any('/logits/' in k for k in head), sorted(head)
  moved = [k for k, v in head.items()
           if not np.allclose(before[k], np.asarray(v))]
  assert any('loc' in k for k in moved), 'loc never updated'
  assert any('logscale' in k for k in moved), 'logscale never updated'


def test_studentt_config_block_sets_scale_min_low():
  """The block must not silently inherit the Gaussian's scale_min."""
  configs = yaml.YAML(typ='safe').load(CONFIGS.read())
  mp = configs['mgr_studentt']['agent']['manager_policy']
  assert mp['output'] == 'studentt', mp
  assert mp['scale_min'] <= 0.15, mp
  # nu and the entropy target are one choice: nu=1's tie floor is 0.601, so the
  # block MUST raise manager_actent_target above it or the adapter rails. The
  # only other admissible combination is nu > 1.558 at the default 0.5.
  tgt = configs['mgr_studentt']['agent'].get(
      'manager_actent_target',
      configs['defaults']['agent']['manager_actent_target'])
  if mp['nu'] <= 1.558:
    assert tgt > 0.61, (mp['nu'], tgt)
  else:
    assert tgt >= 0.5, (mp['nu'], tgt)
  assert 'nu' in configs['defaults']['agent']['manager_policy'], (
      'nu must exist in defaults; elements.Config cannot introduce new keys')
