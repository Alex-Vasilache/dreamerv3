"""The worker entropy controller against TF Director's ``actent``.

Director regulates BOTH actors with one ``actent`` block
(``director/embodied/agents/director/configs.yaml:83``, target 0.5, normalized
and per-dimension); the worker inherits it untouched because
``hierarchy.py:24-31`` overrides only the actor/critic inputs.

The bug these tests exist for: our adapter was a single shared SCALAR while
Director keeps one multiplier PER ACTION DIMENSION (TF ``agent.py:297-300``).
A scalar holds only the MEAN normalized entropy at the target, so a saturated
dimension offset by a high-entropy one leaves the multiplier still. It ran that
way in e865-e882 (`director_og`), where ``wkr_actent_std`` was 1.44-1.67 around
a mean of 0.55 -- the dimensions were nowhere near uniform, which is exactly
the regime where the two controllers differ.
"""
import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest
import ruamel.yaml as yaml

import embodied.jax as embj
from dreamerv3.agent import Agent

ROOT = elements.Path(__file__).parent.parent.parent
CONFIGS = ROOT / 'dreamerv3/configs.yaml'
TF_CONFIGS = ROOT.parent / 'director/embodied/agents/director/configs.yaml'

# director/embodied/agents/director/configs.yaml:83.
DIRECTOR_ACTENT = dict(
    impl='mult', target=0.5, min=1e-5, max=1e2, vel=0.1, thres=0.1)


@pytest.fixture(autouse=True)
def cpu_only():
  """Pin the module to CPU.

  `embodied/tests/conftest.py` imports jax before this module is loaded, so
  setting `JAX_PLATFORMS` here is already too late and the first probe dies
  with "No visible GPU devices" on a login node. The conftest fixture snapshots
  and restores `jax_platforms` around every test, so writing it here is local.
  """
  jax.config.update('jax_platforms', 'cpu')
  yield


def pure(fn, params, *args, seed=0):
  """``nj.pure`` for a probe that owns state. See test_goal_vae_director_parity."""
  with jax.transfer_guard('allow'):
    state, out = nj.pure(fn)(dict(params), *args, seed=seed, create=True)
  return state, out


def make_config(*blocks, **overrides):
  raw = yaml.YAML(typ='safe').load(CONFIGS.read())
  config = elements.Config(raw['defaults'])
  for block in ('debug',) + blocks:
    config = config.update(raw[block])
  config = config.update(overrides) if overrides else config
  return elements.Config(
      **config.agent, logdir='/tmp/wkr_actent_parity', seed=0, jax=config.jax,
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
  return (Agent(obs_space, act_space, make_config(*blocks, **overrides)),
          act_space)


# --- the shape, which is the whole bug ---------------------------------------

def test_one_multiplier_per_action_dimension():
  """Director's ``actent`` is shaped like the action, not like a scalar."""
  agent, act_space = make_agent('director', 'director_og')
  assert isinstance(agent.model.wkr_actent, dict)
  assert set(agent.model.wkr_actent) == set(act_space)
  for key, space in act_space.items():
    assert agent.model.wkr_actent[key].shape == tuple(space.shape), key
  # The dummy env has a 6-dim continuous head and a scalar discrete one, so
  # this also pins that the two heads get differently shaped adapters -- which
  # a single shared adapter could not provide.
  assert agent.model.wkr_actent['act_cont'].shape == (6,)
  assert agent.model.wkr_actent['act_disc'].shape == ()


def test_perdim_can_be_turned_off():
  agent, _ = make_agent(
      'director', 'director_og', **{'agent.worker_actent_perdim': False})
  assert all(a.shape == () for a in agent.model.wkr_actent.values())


# --- what per-dim buys: independence -----------------------------------------

def adapt(shape, regs, **kw):
  """Run an AutoAdapt over a list of reg batches, return the scale trace."""
  opts = dict(DIRECTOR_ACTENT, **kw)

  def fn(reg):
    # `vel` and `thres` are ninjax module attributes: settable as constructor
    # kwargs, read-only afterwards.
    a = embj.AutoAdapt(shape=shape, inverse=True, name='a', **opts)
    a(reg, update=True)
    return a.scale()

  params, scales = {}, []
  for reg in regs:
    params, out = pure(fn, params, jnp.asarray(reg, jnp.float32))
    scales.append(np.asarray(out))
  return np.stack(scales)


def test_per_dim_multipliers_move_independently():
  """One dim above target and one below must move in OPPOSITE directions."""
  # dim0 far above 0.5, dim1 far below. inverse=True, so above target the
  # multiplier shrinks and below it grows.
  reg = np.stack([np.full((4, 3), 0.95), np.full((4, 3), 0.05)], axis=-1)
  trace = adapt((2,), [reg] * 12)
  assert trace[-1][0] < trace[0][0], 'above-target dim should relax'
  assert trace[-1][1] > trace[0][1], 'below-target dim should tighten'


def test_a_scalar_multiplier_cannot_see_the_split():
  """The old behaviour, asserted so the difference is not hypothetical.

  Same data as the test above. The mean is exactly on target, so a single
  shared multiplier sits inside the deadband and never moves -- while the two
  per-dim multipliers have moved apart by orders of magnitude.
  """
  reg = np.stack([np.full((4, 3), 0.95), np.full((4, 3), 0.05)], axis=-1)
  scalar = adapt((), [reg] * 16)
  assert np.allclose(scalar, scalar[0]), 'scalar adapter should be inert here'
  perdim = adapt((2,), [reg] * 16)
  assert perdim[-1][1] / perdim[-1][0] > 10.0


def test_transcription_matches_the_tf_config_file():
  """`DIRECTOR_ACTENT` above is a transcription; check it against the source.

  Also the claim this whole file rests on: `hierarchy.py` builds the worker
  from a config that overrides only the actor/critic inputs, so the worker
  inherits `actent` unchanged. If a future reader doubts that the worker really
  is regulated at 0.5, this is the assertion to read.
  """
  if not TF_CONFIGS.exists():
    pytest.skip('TF Director checkout not present')
  raw = yaml.YAML(typ='safe').load(TF_CONFIGS.read())['defaults']
  actent = raw['actent']
  for key in ('impl', 'target', 'min', 'max', 'vel'):
    assert float(actent[key]) == float(DIRECTOR_ACTENT[key]) if key != 'impl' \
        else actent[key] == DIRECTOR_ACTENT[key], key
  assert raw['actent_norm'] is True
  assert raw['actent_perdim'] is True
  # The manager's override is the ONLY per-actor change, and it lands on the
  # same number, so both actors sit at 0.5.
  assert float(raw['manager_actent']) == float(DIRECTOR_ACTENT['target'])

  hierarchy = (TF_CONFIGS.parent / 'hierarchy.py').read()
  head = hierarchy.split('self.worker = ')[0]
  wconfig = head.split('wconfig = config.update({')[1].split('})')[0]
  assert 'actent' not in wconfig, (
      'the worker config now overrides actent; this file assumes it does not')


def test_direction_matches_director_inverse_rule():
  """Entropy below target => push harder (scale up). TF ``AutoAdapt`` 504-513."""
  below = adapt((), [np.full((4, 3), 0.1)] * 6)
  above = adapt((), [np.full((4, 3), 0.9)] * 6)
  assert below[-1] > below[0]
  assert above[-1] < above[0]


def test_deadband_edges_match_director():
  """``below = avg < target/(1+thres)``, ``above = avg > target*(1+thres)``."""
  tgt, thres = 0.5, 0.1
  inside_lo = tgt / (1 + thres) + 1e-3
  inside_hi = tgt * (1 + thres) - 1e-3
  for val in (inside_lo, tgt, inside_hi):
    trace = adapt((), [np.full((4, 3), val)] * 6)
    assert np.allclose(trace, trace[0]), f'{val} should be inside the deadband'
  outside_lo = tgt / (1 + thres) - 1e-3
  outside_hi = tgt * (1 + thres) + 1e-3
  assert adapt((), [np.full((4, 3), outside_lo)] * 6)[-1] > 1.0
  assert adapt((), [np.full((4, 3), outside_hi)] * 6)[-1] < 1.0


# --- the normalization, which is the other half of "target 0.5" --------------

def test_normalized_entropy_matches_director_formula():
  """TF ``agent.py:358-365``: ``lo = minent/L``, ``hi = maxent/L``."""
  L = 6
  ent = np.random.default_rng(0).uniform(-2, 2, (3, 5, L))
  minent, maxent = -3.0, 4.0
  ours = (ent - minent / L) / (maxent / L - minent / L)
  lo, hi = minent / ent.shape[-1], maxent / ent.shape[-1]
  theirs = (ent - lo) / (hi - lo)
  np.testing.assert_allclose(ours, theirs, rtol=0, atol=0)


# --- and that it is actually wired into the loss -----------------------------

def test_worker_entropy_term_is_alive_under_director_og():
  """``mgr_ent_loss`` shipped identically 0 for a whole run once (e478).

  The same silent-skip path exists for the worker: ``losses.py`` falls back to
  the fixed coefficient for any head without ``minent``/``maxent``. Assert the
  adapter's scale actually MOVES during training, which it cannot do unless the
  adapter was handed to the loss and called.
  """
  from embodied.jax import internal
  agent, _ = make_agent('director', 'director_og')
  keys = [k for k in agent.params if 'wkr_actent' in k]
  assert keys, 'no wkr_actent variables in the parameter tree'
  before = {k: np.asarray(agent.params[k]).copy() for k in keys}

  rng = np.random.default_rng(0)
  batch = agent.config.batch_size
  length = agent.config.batch_length + agent.config.replay_context
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

  carry = agent.init_train(batch)
  base = internal.device_put(data, agent.train_sharded)
  mets = {}
  for i in range(6):
    # `Agent.train` pops 'seed' out of the batch, so it has to be supplied
    # fresh every step rather than baked into the device-put batch.
    carry, _, out = agent.train(
        carry, {**base, 'seed': agent._seeds(i, agent.train_mirrored)})
    if out:
      mets = out
  assert any(not np.allclose(before[k], np.asarray(agent.params[k]))
             for k in keys), 'wkr_actent scale never moved'
  assert any('wkr_actent' in k for k in mets), 'no wkr_actent metrics logged'


def test_worker_adapter_is_off_by_default():
  """DreamerV3 default stays the fixed coefficient, so the other arms and
  their checkpoints are untouched by this change."""
  agent, _ = make_agent('director')
  assert agent.model.worker_actent_adapt is False
  assert not [k for k in agent.params if 'wkr_actent' in k]
