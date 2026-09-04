"""Goal-VAE numerics and wiring, asserted against the TF Director reference.

`docs/VERIFICATION.md` (*Goal autoencoder*) records a 2026-09-01 line-by-line
read of both sides. A read is not an assertion: it cannot catch a reduction that
silently changed axis, a straight-through estimator that stopped passing
gradient, or a controller whose clip makes its target unreachable. This file is
the test-backed version, in the spirit of `test_tensors_vs_director.py` for
block pooling.

Reference: ``code/director/embodied/agents/director/hierarchy.py`` and its
``configs.yaml``. The path being asserted is ``train_vae_replay``::

    feat  = Input(['deter'])(data)             # posterior deter, [B, T, D]
    enc   = self.enc({'goal': feat})           # MLP -> onehot [8, 8]
    dec   = self.dec({'skill': enc.sample()})  # MLP -> MSEDist(dims=1, 'sum')
    rec   = -dec.log_prob(sg(feat))            # [B, T], summed over D
    kl    = tfd.kl_divergence(enc, prior)      # [B, T], summed over 8 blocks
    kl, _ = self.kl(kl)                        # AutoAdapt 'mult', clipped to 1.0
    loss  = (rec + kl).mean()                  # scalar; NO beta coefficient
    self.opt(tape, loss, [self.enc, self.dec]) # own optimizer, enc + dec only

Deliberate divergences are asserted too (``test_known_divergences_*``) so they
cannot drift back silently: they are choices, and the tests name them as such.
"""
import functools
import re

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest
import ruamel.yaml as yaml

import embodied.jax.outs as outs
from dreamerv3.agent import Agent
from dreamerv3.hrl.heads import _head_inner

f32 = jnp.float32
ROOT = elements.Path(__file__).parent.parent.parent
CONFIGS = ROOT / 'dreamerv3/configs.yaml'
AGENT_PY = (ROOT / 'dreamerv3/agent.py').read()

# Director defaults, from director/embodied/agents/director/configs.yaml.
TF_SKILL_SHAPE = (8, 8)
TF_MAXKL = TF_SKILL_SHAPE[0] * np.log(TF_SKILL_SHAPE[1])  # 8*ln8 = 16.6355
TF_KL_TARGET, TF_KL_MIN, TF_KL_MAX = 10.0, 1e-5, 1.0
TF_KL_VEL, TF_KL_THRES = 0.1, 0.1


def pure(fn, params, *args, seed=0):
  """``nj.pure`` for a read-only probe.

  Three things bite here and each cost a debugging round:
  ``nj.pure`` returns ``(state, out)`` and not ``(out, state)``; embodied sets
  ``jax_transfer_guard='disallow'`` at agent construction, which rejects the
  host-side probe inputs; and modules that own state (``AutoAdapt``) need
  ``create=True`` or they raise on their first write.
  """
  with jax.transfer_guard('allow'):
    _, out = nj.pure(fn)(dict(params), *args, seed=seed, create=True)
  return out


def make_config(*blocks, **overrides):
  raw = yaml.YAML(typ='safe').load(CONFIGS.read())
  config = elements.Config(raw['defaults'])
  for block in ('debug',) + blocks:
    config = config.update(raw[block])
  config = config.update(overrides) if overrides else config
  return elements.Config(
      **config.agent, logdir='/tmp/goal_vae_parity', seed=0, jax=config.jax,
      batch_size=config.batch_size, batch_length=config.batch_length,
      replay_context=config.replay_context, report_length=config.report_length,
      replica=0, replicas=1)


@functools.lru_cache(maxsize=None)
def agent():
  """One Director-arm agent, shared by every read-only check in this file."""
  from embodied.envs import dummy
  # 64x64: the simple CNN encoder downsamples 4x (mults [2, 3, 4, 4]), so a
  # smaller image reshapes to a zero-sized spatial grid.
  env = dummy.Dummy('disc', size=(64, 64), length=20)
  keep = ('image', 'vector', 'reward', 'is_first', 'is_last', 'is_terminal')
  obs_space = {k: v for k, v in env.obs_space.items() if k in keep}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  return Agent(obs_space, act_space, make_config())


def make_batch(ag, seed=0):
  rng = np.random.default_rng(seed)
  batch = ag.config.batch_size
  length = ag.config.batch_length + ag.config.replay_context
  data = {}
  for key, space in ag.spaces.items():
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
def trained_metrics():
  """Metrics from real train steps; ``train`` reports one call behind."""
  from embodied.jax import internal
  ag = agent()
  carry = ag.init_train(ag.config.batch_size)
  base = internal.device_put(make_batch(ag), ag.train_sharded)
  last = {}
  for i in range(3):
    data = {**base, 'seed': ag._seeds(i, ag.train_mirrored)}
    carry, _, mets = ag.train(carry, data)
    if mets:
      last = mets
  assert last, 'no metrics after 3 train steps'
  return last


def probe(B=3, T=5, seed=0):
  """Run encoder -> sample -> decoder once and return every intermediate."""
  ag = agent()
  model = ag.model
  dim = model.goal_shape[0]
  rng = np.random.default_rng(seed)
  deter = jnp.asarray(rng.normal(0, 0.3, (B, T, dim)), f32)

  def fn(deter):
    enc = model.goal_enc(deter, 2)
    dist = _head_inner(enc)
    code = enc.sample(nj.seed())
    dec = model.goal_dec(code, 2)
    prior = outs.OneHot(jnp.zeros_like(dist.dist.logits), 0.0)
    return dict(logits=dist.dist.logits, code=code, pred=dec.pred(),
                rec=dec.loss(deter), kl=dist.kl(prior))

  return ag, np.asarray(deter), pure(fn, ag.params, deter), dim


def ref_categorical_kl(logits):
  """KL(q || uniform) per block, in numpy, from the logits alone."""
  logits = np.asarray(logits, np.float64)
  logp = logits - np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(
      -1, keepdims=True)) - logits.max(-1, keepdims=True)
  probs = np.exp(logp)
  return (probs * (logp - np.log(1.0 / logits.shape[-1]))).sum(-1)


class TestConfigParity:
  """Every constant the Director VAE path depends on."""

  def test_skill_shape(self):
    assert tuple(agent().config.skill_shape) == TF_SKILL_SHAPE

  def test_reconstruction_target_is_deter(self):
    """TF: ``self.feat = Input(['deter'])``, ``goal_shape = (rssm.deter,)``."""
    ag = agent()
    assert ag.model.goal_shape == (ag.config.dyn.rssm.deter,)

  def test_encoder_and_decoder_inputs(self):
    """TF: ``goal_encoder.inputs: [goal]``, ``goal_decoder.inputs: [skill]``.

    Director's MLPs are handed a ``context`` entry that its config does not
    list, so it is dropped. Ours pass one tensor each, which is the same thing.
    """
    assert 'self.goal_enc(deter_feat, 2)' in AGENT_PY
    assert 'self.goal_dec(skill_code, 2)' in AGENT_PY

  def test_reconstruction_reduction_is_sum(self):
    """TF: ``DistLayer(dist='mse')`` -> ``MSEDist(out, dims=1, agg='sum')``."""
    assert agent().config.goal_rec_loss_agg == 'sum'

  def test_kl_reduction_in_agent_is_sum(self):
    assert 'goal_kl_bt = goal_kl_bt.sum(-1)' in AGENT_PY

  def test_kl_controller_constants(self):
    """TF: ``encdec_kl: {impl: mult, target: 10.0, min: 1e-5, max: 1.0}``."""
    c = agent().config
    assert c.goal_kl is True
    assert c.goal_kl_impl == 'mult'
    assert float(c.goal_kl_target) == TF_KL_TARGET
    assert float(c.goal_kl_min) == TF_KL_MIN
    assert float(c.goal_kl_max) == TF_KL_MAX
    assert float(c.goal_kl_vel) == TF_KL_VEL

  def test_kl_scale_starts_at_one(self):
    """Trap in Director's config: it reads ``scale: 0.0``, which looks like the
    multiplier starts at zero. It does not -- ``AutoAdapt.__init__`` ignores
    ``scale`` unless ``impl == 'fixed'`` and always inits to ``tf.ones``.
    """
    assert float(agent().config.goal_kl_init) == 1.0

  def test_beta_matches_director(self):
    """TF: ``loss = (rec + kl).mean()`` -- no coefficient on the KL."""
    assert float(agent().config.goal_autoencoder_beta) == 1.0

  def test_trains_on_replay_only(self):
    """TF: ``vae_replay: True``, ``vae_imag: False``."""
    assert ('self._goal_autoencoder_loss('
            'repfeat, losses, metrics, B, T, training)') in AGENT_PY
    assert AGENT_PY.count('_goal_autoencoder_loss(') == 2  # def + one call

  def test_goal_optimizer_group_is_encoder_and_decoder_only(self):
    """TF: ``self.opt(tape, loss, [self.enc, self.dec])``."""
    assert 'goal_modules = [self.goal_enc, self.goal_dec]' in AGENT_PY
    names = sorted({k.split('/')[0] for k in agent().params
                    if k.split('/')[0] in ('goal_enc', 'goal_dec')})
    assert names == ['goal_dec', 'goal_enc'], names

  def test_weight_decay_is_kernel_only(self):
    """TF: ``encdec_opt.wd_pattern: 'kernel'``.

    ``wdregex`` is not a `configs.yaml` key -- it is an ``Agent._make_opt``
    default that every optimizer group inherits, so assert it there.
    """
    import inspect
    default = inspect.signature(Agent._make_opt).parameters['wdregex'].default
    assert default == r'/kernel$'
    assert 'wdregex' not in agent().config.goal_opt, (
        'goal_opt now overrides wdregex; check it still matches TF')

  def test_optimizer_lr_and_wd_are_dreamerv3_not_director(self):
    """Deliberate divergence since 2026-09-04.

    Director's ``encdec_opt`` is ``{lr: 1e-4, wd: 1e-2}``; DreamerV3's ``opt``
    is ``{lr: 4e-5, wd: 0.0}``. We take DreamerV3's, because the rest of this
    optimizer is already DreamerV3's -- it is LaProp with ``eps: 1e-20`` and
    AGC 0.3, not Director's Adam with ``eps: 1e-6`` and a global-norm clip --
    and because both DreamerV3 papers state in their hyperparameter tables that
    "we do not use any hyperparameter annealing, prioritized replay, weight
    decay, or dropout".
    """
    opt = agent().config.goal_opt
    assert float(opt.lr) == 4e-5, 'DreamerV3 lr; Director uses 1e-4'
    assert float(opt.wd) == 0.0, 'DreamerV3 uses no weight decay'

  def test_known_divergences_are_still_the_ones_we_chose(self):
    """Deliberate, documented differences. Failing here is a signal to update
    ``docs/VERIFICATION.md``, not necessarily a bug.
    """
    ag = agent()
    opt = ag.config.goal_opt
    # eps 1e-20 belongs to LaProp, which is what DreamerV3 (and we) use;
    # Director's 1e-6 is an Adam epsilon and does not transfer.
    assert float(opt.eps) == 1e-20, 'TF uses Adam eps 1e-6'
    assert float(opt.agc) == 0.3, 'TF clips global grad norm at 100.0'
    assert ag.config.goal_enc.act == 'silu', 'TF goal nets use elu'
    assert ag.config.goal_enc.norm == 'rms', 'TF goal nets use layer norm'


class TestShapesAndReductions:
  """Follow one batch through the path and check every axis."""

  def test_encoder_emits_per_block_logits(self):
    _, _, out, _ = probe()
    assert out['logits'].shape == (3, 5, *TF_SKILL_SHAPE)

  def test_sampled_code_is_a_hard_onehot_per_block(self):
    _, _, out, _ = probe()
    code = np.asarray(out['code'])
    assert code.shape == (3, 5, *TF_SKILL_SHAPE)
    np.testing.assert_allclose(code.sum(-1), 1.0, atol=1e-5)
    assert np.isclose(code.max(-1), 1.0, atol=1e-5).all()

  def test_decoder_reconstructs_the_whole_deter_vector(self):
    _, _, out, dim = probe()
    assert out['pred'].shape == (3, 5, dim)

  def test_reconstruction_is_summed_over_deter_not_averaged(self):
    """TF ``MSEDist('sum')``: ``rec[b,t] = sum_d (pred - target)**2``."""
    _, deter, out, dim = probe()
    manual = np.square(np.asarray(out['pred']) - deter).sum(-1)
    assert out['rec'].shape == (3, 5)
    np.testing.assert_allclose(np.asarray(out['rec']), manual, rtol=1e-4)
    # A per-dim mean would differ by exactly the deter dimension.
    assert not np.allclose(np.asarray(out['rec']), manual / dim, rtol=1e-2)

  def test_kl_is_per_block_then_summed_over_blocks(self):
    """TF: ``Independent(OneHotDist, len(shape) - 1)`` sums the L categoricals."""
    _, _, out, _ = probe()
    raw = np.asarray(out['kl'])
    ref = ref_categorical_kl(out['logits'])
    assert raw.shape == (3, 5, TF_SKILL_SHAPE[0])
    # ``nets.COMPUTE_DTYPE`` is bfloat16, so the logits reaching the reference
    # have already been rounded; the tolerance is float width, not disagreement
    # (observed max absolute difference 2.2e-7 on values of order 1e-3).
    np.testing.assert_allclose(raw, ref, rtol=1e-3, atol=1e-6)
    reduced = raw
    while reduced.ndim > 2:  # the reduction agent.py performs
      reduced = reduced.sum(-1)
    assert reduced.shape == (3, 5)
    np.testing.assert_allclose(reduced, ref.sum(-1), rtol=1e-3, atol=1e-6)


class TestKLIdentities:
  """What the logged ``goal/kl_raw_mean`` actually measures."""

  def _kl_and_entropy(self, logits):
    def fn(logits):
      dist = outs.OneHot(logits, 0.0)
      prior = outs.OneHot(jnp.zeros_like(logits), 0.0)
      return dict(kl=dist.kl(prior).sum(-1), ent=dist.entropy().sum(-1))
    return pure(fn, {}, logits)

  def test_uniform_code_carries_zero_information(self):
    out = self._kl_and_entropy(jnp.zeros((2, *TF_SKILL_SHAPE), f32))
    np.testing.assert_allclose(np.asarray(out['kl']), 0.0, atol=1e-5)

  def test_deterministic_code_hits_the_ceiling(self):
    logits = np.full((2, *TF_SKILL_SHAPE), -1e4, np.float32)
    logits[..., 0] = 1e4
    out = self._kl_and_entropy(jnp.asarray(logits))
    np.testing.assert_allclose(np.asarray(out['kl']), TF_MAXKL, rtol=1e-4)

  def test_kl_is_exactly_the_ceiling_minus_entropy(self):
    """So ``goal/kl_raw_mean`` and ``goal/entropy_mean`` are one signal, not two:
    KL = 8*ln8 - H. A run reporting KL 11.0 has a code entropy of 5.6 nats.
    """
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(0, 2.0, (4, *TF_SKILL_SHAPE)), f32)
    out = self._kl_and_entropy(logits)
    np.testing.assert_allclose(
        np.asarray(out['kl']) + np.asarray(out['ent']), TF_MAXKL, rtol=1e-5)


class TestStraightThrough:

  def test_sample_gradient_equals_the_softmax_gradient(self):
    """TF ``OneHotDist.sample``: ``sample = sg(onehot) + (probs - sg(probs))``,
    so the backward pass sees ``d(softmax)/d(logits)`` exactly.
    """
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(0, 1.5, (2, *TF_SKILL_SHAPE)), f32)
    weight = jnp.asarray(rng.normal(0, 1.0, (2, *TF_SKILL_SHAPE)), f32)
    via_sample = jax.grad(lambda x: pure(
        lambda x: (outs.OneHot(x, 0.0).sample(nj.seed()) * weight).sum(),
        {}, x))(logits)
    via_probs = jax.grad(
        lambda x: (jax.nn.softmax(x, -1) * weight).sum())(logits)
    np.testing.assert_allclose(
        np.asarray(via_sample), np.asarray(via_probs), rtol=1e-4, atol=1e-6)


class TestGradientRouting:
  """TF tapes only ``[self.enc, self.dec]`` and stop-grads the target."""

  @staticmethod
  @functools.lru_cache(maxsize=None)
  def _grad_magnitudes():
    ag = agent()
    model = ag.model
    B, T = 2, 4
    rssm = model.config.dyn.rssm
    rng = np.random.default_rng(0)
    deter = jnp.asarray(rng.normal(0, 0.3, (B, T, model.goal_shape[0])), f32)

    def loss(deter):
      losses, metrics = {}, {}
      feat = {'deter': deter,
              'stoch': jnp.zeros((B, T, rssm.stoch, rssm.classes), f32)}
      model._goal_autoencoder_loss(feat, losses, metrics, B, T, True)
      return losses['goal_autoencoder'].mean()

    # jax.grad rejects integer leaves; counters and RNG state are not params.
    floats = {k: v for k, v in ag.params.items()
              if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)}
    rest = {k: v for k, v in ag.params.items() if k not in floats}
    grads = jax.grad(
        lambda p, d: pure(loss, {**p, **rest}, d))(floats, deter)
    return {k: float(jnp.abs(v).sum()) for k, v in grads.items()}

  def test_encoder_and_decoder_are_trained(self):
    mags = self._grad_magnitudes()
    assert sum(v for k, v in mags.items() if k.startswith('goal_enc/')) > 0
    assert sum(v for k, v in mags.items() if k.startswith('goal_dec/')) > 0

  def test_nothing_leaks_into_the_world_model(self):
    mags = self._grad_magnitudes()
    leaked = {k: v for k, v in mags.items()
              if re.match(r'^(enc|dyn|dec|rew|con)/', k) and v > 0}
    assert not leaked, leaked

  def test_nothing_leaks_into_the_actor_critic(self):
    mags = self._grad_magnitudes()
    leaked = {k: v for k, v in mags.items()
              if re.match(r'^(pol|manager_pol|mgr_|wkr_)', k) and v > 0}
    assert not leaked, leaked


class TestAutoAdapt:
  """Reproduce Director's ``AutoAdapt.update`` for ``impl='mult'`` step by step.

  The deadband is +-``thres`` around the target, so with target 10.0 the
  multiplier grows above 11.0, shrinks below 9.0909, and holds between.
  """

  def _ours(self, scale, kl):
    from embodied.jax.utils import AutoAdapt

    def fn(kl):
      adapter = AutoAdapt(
          (), impl='mult', target=TF_KL_TARGET, min=TF_KL_MIN, max=TF_KL_MAX,
          vel=TF_KL_VEL, init=scale, name='adapter')
      _, mets = adapter(jnp.full((2, 3), kl, f32), update=True)
      return mets['scale_mean']
    return float(pure(fn, {}, kl))

  def _director(self, scale, kl):
    if kl > (1 + TF_KL_THRES) * TF_KL_TARGET:
      adjusted = scale * (1 + TF_KL_VEL)
    elif kl < (1 / (1 + TF_KL_THRES)) * TF_KL_TARGET:
      adjusted = scale / (1 + TF_KL_VEL)
    else:
      adjusted = scale
    return float(np.clip(adjusted, TF_KL_MIN, TF_KL_MAX))

  @pytest.mark.parametrize('scale,kl', [
      (0.5, 20.0),   # far above target -> grow
      (0.5, 1.0),    # far below target -> shrink
      (0.5, 10.0),   # on target, inside the deadband -> hold
      (0.5, 10.9),   # just inside the upper edge (11.0) -> hold
      (0.5, 11.1),   # just outside -> grow
      (0.5, 9.2),    # just inside the lower edge (9.0909) -> hold
      (0.5, 9.0),    # just outside -> shrink
      (1.0, 20.0),   # already at max -> clipped, stays put
      (1e-5, 1.0),   # already at min -> clipped, stays put
  ])
  def test_update_matches_director(self, scale, kl):
    assert self._ours(scale, kl) == pytest.approx(
        self._director(scale, kl), rel=1e-5)

  def test_multiplier_saturates_at_max_when_kl_stays_above_target(self):
    """The railing seen in every pinpad run: ``goal_kl_max`` is 1.0 and
    ``goal_kl_init`` is also 1.0, so the controller can only ever relax. Once
    KL sits above 11.0 it asks for pressure it is not allowed to apply and
    ``goal_kl_target`` stops being in control -- the effective weight is then
    ``goal_autoencoder_beta`` alone.
    """
    scale = 1.0
    for _ in range(50):
      scale = self._ours(scale, 11.5)
    assert scale == pytest.approx(TF_KL_MAX)

  def test_upper_deadband_edge_is_eleven(self):
    assert (1 + TF_KL_THRES) * TF_KL_TARGET == pytest.approx(11.0)


class TestLossAssembly:
  """The scalar that reaches the optimizer, from real train steps."""

  def test_total_is_rec_plus_beta_times_scaled_kl(self):
    mets = trained_metrics()
    beta = float(agent().config.goal_autoencoder_beta)
    expected = float(mets['goal/rec_mean']) + beta * float(mets['goal/kl_mean'])
    assert float(mets['loss/goal_autoencoder']) == pytest.approx(
        expected, rel=1e-3)

  def test_scaled_kl_is_raw_kl_times_the_multiplier(self):
    mets = trained_metrics()
    expected = (float(mets['goal/kl_raw_mean'])
                * float(mets['goal/kl_adapt_scale_mean']))
    assert float(mets['goal/kl_mean']) == pytest.approx(expected, rel=1e-3)

  def test_outer_loss_scale_is_one(self):
    """TF applies no scale: ``loss = (rec + kl).mean()``."""
    assert float(agent().model.scales['goal_autoencoder']) == 1.0
