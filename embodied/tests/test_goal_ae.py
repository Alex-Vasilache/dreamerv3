"""Unit tests for the assembled VQ / SOM / LipVQ goal autoencoder.

Covers three risks:

  * *drop-in compatibility* -- the agent calls these objects through the same
    duck-typed protocol it used for Director's ``onehot`` / ``mse`` heads, so
    every method that ``agent.py`` and ``hrl/`` actually invoke is exercised
    here against the shapes they pass;
  * *gradient routing per ablation arm* -- which parameter groups are trained
    differs between arms (without a straight-through estimator the encoder is
    reachable only via the commitment term, plus the second reconstruction in
    the SOM arm), and getting this wrong yields a model that trains but learns
    the wrong thing;
  * *numerics* -- the autoencoder actually reduces reconstruction error under
    gradient descent, and the Lipschitz arm's composed bound holds end to end.
"""
import re

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
import pytest

import embodied.jax.nets as nets
import embodied.jax.outs as outs
from dreamerv3.hrl import goal_ae, vq
from dreamerv3.hrl import heads as hrl_heads
from dreamerv3.hrl.heads import _head_inner

f32 = jnp.float32

L, C, DIM, DETER = 4, 5, 3, 16
POLICY_KEYS = r'^(enc|dyn|dec|pol|manager_pol|goal_dec)/'


@pytest.fixture(autouse=True)
def f32_compute():
  old = nets.COMPUTE_DTYPE
  nets.COMPUTE_DTYPE = jnp.float32
  yield
  nets.COMPUTE_DTYPE = old


def build(lip=False, layers=2, units=32, classes=C, **kw):
  dec = goal_ae.GoalVQDecoder(
      DETER, L, classes, DIM, layers=layers, units=units, lip=lip,
      name='goal_dec', **kw)
  enc = goal_ae.GoalVQEncoder(
      dec.codebook, L, DIM, layers=layers, units=units, lip=lip,
      name='goal_enc', **kw)
  return enc, dec


def deters(shape=(2, 3), seed=0, scale=1.0):
  rng = np.random.default_rng(seed)
  return jnp.asarray(rng.normal(0, scale, (*shape, DETER)), f32)


def init_all(enc, dec, x, bdims=2, seed=0):
  def fn(x):
    enc(x, bdims)
    dec(dec.codebook.onehot(jnp.zeros((*x.shape[:bdims], L), jnp.int32)), bdims)
  return nj.init(fn)({}, x, seed=seed)


class TestVQCode:

  def _code(self, seed=0):
    d = jnp.asarray(
        np.random.default_rng(seed).uniform(0, 5, (2, 3, L, C)), f32)
    return goal_ae.VQCode(d), d

  def test_pred_is_a_hard_onehot_at_the_argmin_distance(self):
    code, d = self._code()
    pred = np.asarray(code.pred())
    assert pred.shape == (2, 3, L, C)
    assert set(np.unique(pred)) <= {0.0, 1.0}
    np.testing.assert_array_equal(pred.sum(-1), np.ones((2, 3, L)))
    np.testing.assert_array_equal(pred.argmax(-1), np.asarray(d).argmin(-1))

  def test_sample_is_deterministic_and_equals_pred(self):
    code, d = self._code()
    a = code.sample(jax.random.PRNGKey(0))
    b = code.sample(jax.random.PRNGKey(12345))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, code.pred())

  def test_sample_rejects_extra_sample_shape(self):
    code, d = self._code()
    with pytest.raises(AssertionError):
      code.sample(jax.random.PRNGKey(0), (4,))

  def test_no_straight_through_gradient(self):
    # The defining difference from outs.OneHot: motivation.tex's dual
    # reconstruction exists precisely to avoid the straight-through estimator,
    # so pred() must be gradient-free in the distances.
    d = jnp.asarray(
        np.random.default_rng(1).uniform(0, 5, (2, L, C)), f32)
    g = jax.grad(lambda d: goal_ae.VQCode(d).pred().sum())(d)
    np.testing.assert_array_equal(np.asarray(g), 0.0)

  def test_onehot_head_would_have_had_a_gradient(self):
    # Control for the test above: the Director head this replaces DOES leak a
    # straight-through gradient, so the assertion above is not vacuous. The
    # reduction has to be non-uniform over the class axis -- a plain sum of the
    # one-hot is identically 1 and cancels the estimator's gradient.
    rng = np.random.default_rng(1)
    d = jnp.asarray(rng.uniform(0, 5, (2, L, C)), f32)
    w = jnp.asarray(rng.normal(0, 1, (2, L, C)), f32)
    g = jax.grad(lambda d: (outs.OneHot(-d).pred() * w).sum())(d)
    assert np.abs(np.asarray(g)).max() > 1e-3
    g_vq = jax.grad(lambda d: (goal_ae.VQCode(d).pred() * w).sum())(d)
    np.testing.assert_array_equal(np.asarray(g_vq), 0.0)

  def test_logits_are_negated_distances(self):
    code, d = self._code()
    np.testing.assert_allclose(code.dist.logits, -np.asarray(d), rtol=1e-6)
    code2 = goal_ae.VQCode(d, temp=2.0)
    np.testing.assert_allclose(code2.dist.logits, -np.asarray(d) / 2, rtol=1e-6)

  def test_entropy_is_bounded_by_log_classes(self):
    code, d = self._code()
    ent = np.asarray(code.entropy())
    assert ent.shape == (2, 3, L)
    assert (ent >= -1e-6).all()
    assert (ent <= np.log(C) + 1e-5).all()

  def test_logp_of_its_own_prediction_is_the_max(self):
    code, d = self._code()
    lp = np.asarray(code.logp(code.pred()))
    for c in range(C):
      other = np.asarray(code.logp(jnp.asarray(np.eye(C)[np.full((2, 3, L), c)], f32)))
      assert (lp >= other - 1e-5).all()

  def test_kl_to_itself_is_zero(self):
    code, d = self._code()
    np.testing.assert_allclose(code.kl(code), np.zeros((2, 3, L)), atol=1e-6)

  def test_rejects_nonpositive_temp(self):
    code, d = self._code()
    with pytest.raises(AssertionError):
      goal_ae.VQCode(d, temp=0.0)


class TestEncoderDecoderShapes:

  def test_encoder_output_protocol_matches_the_director_head(self):
    enc, dec = build()
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    out = nj.pure(lambda x: enc(x, 2))(params, x)[1]
    # Wrapped in Agg, exactly like Head.__call__ does for `onehot`.
    assert isinstance(out, outs.Agg)
    inner = _head_inner(out)
    assert inner.dist.logits.shape == (2, 3, L, C)
    assert out.pred().shape == (2, 3, L, C)
    assert out.sample(jax.random.PRNGKey(0)).shape == (2, 3, L, C)
    assert out.entropy().shape == (2, 3)          # Agg summed the L axis

  def test_decoder_output_protocol_matches_the_mse_head(self):
    enc, dec = build()
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    code = jnp.asarray(np.eye(C)[np.zeros((2, 3, L), int)], f32)
    out = nj.pure(lambda c: dec(c, 2))(params, code)[1]
    assert isinstance(out, outs.Agg)
    assert out.pred().shape == (2, 3, DETER)
    assert out.loss(x).shape == (2, 3)            # summed over deter

  def test_latent_shape_and_bdims_1(self):
    enc, dec = build()
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    z2 = nj.pure(lambda x: enc.latent(x, 2))(params, x)[1]
    assert z2.shape == (2, 3, L, DIM)
    flat = x.reshape((6, DETER))
    z1 = nj.pure(lambda x: enc.latent(x, 1))(params, flat)[1]
    assert z1.shape == (6, L, DIM)
    np.testing.assert_allclose(z2.reshape((6, L, DIM)), z1, rtol=1e-5, atol=1e-6)

  def test_decode_from_code_equals_decode_from_looked_up_latent(self):
    enc, dec = build()
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    ids = np.random.default_rng(0).integers(0, C, (2, 3, L))
    code = jnp.asarray(np.eye(C)[ids], f32)
    def both(code):
      a = dec(code, 2).pred()
      b = dec.from_latent(dec.codebook.lookup(code), 2).pred()
      return a, b
    a, b = nj.pure(both)(params, code)[1]
    np.testing.assert_allclose(a, b, rtol=1e-6)

  def test_from_latent_rejects_wrong_latent_shape(self):
    enc, dec = build()
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    with pytest.raises(AssertionError):
      nj.pure(lambda z: dec.from_latent(z, 2))(
          params, jnp.zeros((2, 3, L, DIM + 1), f32))

  def test_encoder_code_agrees_with_the_codebook_quantizer(self):
    enc, dec = build()
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    def fn(x):
      z = enc.latent(x, 2)
      return enc(x, 2).pred(), dec.codebook.quantize(z)['onehot']
    a, b = nj.pure(fn)(params, x)[1]
    np.testing.assert_array_equal(a, b)


class TestParamLayout:

  def test_codebook_lives_under_the_decoder(self):
    enc, dec = build()
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    assert 'goal_dec/codebook/table' in params
    assert params['goal_dec/codebook/table'].shape == (L, C, DIM)
    assert not [k for k in params if k.startswith('goal_enc/codebook')]

  def test_policy_keys_regex_covers_everything_decoding_needs(self):
    # The actor process only receives params matching Agent.policy_keys. If the
    # codebook fell outside it, the actor would decode goals with a stale or
    # uninitialized codebook -- silently, and only in the online/parallel path.
    enc, dec = build()
    code = jnp.asarray(np.eye(C)[np.zeros((2, 3, L), int)], f32)
    needed = nj.init(lambda c: dec(c, 2))({}, code, seed=0)
    assert needed, needed
    unmatched = [k for k in needed if not re.match(POLICY_KEYS, k)]
    assert not unmatched, unmatched

  def test_encoder_params_are_separate_from_the_decoder(self):
    enc, dec = build()
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    assert [k for k in params if k.startswith('goal_enc/')]
    assert [k for k in params if k.startswith('goal_dec/mlp/')]

  def test_lip_creates_bounds_on_every_encoder_and_decoder_layer(self):
    enc, dec = build(lip=True, layers=2)
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    cs = sorted(k for k in params if k.endswith('/c'))
    assert cs == [
        'goal_dec/mlp/linear0/c', 'goal_dec/mlp/linear1/c', 'goal_dec/out/c',
        'goal_enc/mlp/linear0/c', 'goal_enc/mlp/linear1/c', 'goal_enc/out/c']

  def test_no_lip_params_when_disabled(self):
    enc, dec = build(lip=False)
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    assert not [k for k in params if k.endswith('/c')]

  def test_lip_out_false_leaves_the_projection_free(self):
    enc, dec = build(lip=True, lip_out=False, layers=2)
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    assert 'goal_enc/out/c' not in params
    assert 'goal_enc/mlp/linear0/c' in params

  def test_bounds_count_matches_layers(self):
    enc, dec = build(lip=True, layers=2)
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    def fn(x):
      enc(x, 2)
      return len(enc.bounds())
    assert nj.pure(fn)(params, x)[1] == 3      # 2 hidden + output projection


class TestVQGoalLoss:

  def _run(self, enc, dec, params, x, **kw):
    def fn(x):
      return goal_ae.vq_goal_loss(enc, dec, x, 2, **kw)
    return nj.pure(fn)(params, x)[1]

  def test_loss_shape_and_finiteness(self):
    enc, dec = build()
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    loss, mets = self._run(enc, dec, params, x)
    assert loss.shape == (2, 3)
    assert jnp.isfinite(loss).all()
    assert jnp.isfinite(jnp.stack(list(mets.values()))).all()

  def test_som_off_has_no_neighbor_term_and_no_second_reconstruction(self):
    enc, dec = build()
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    loss, mets = self._run(enc, dec, params, x, som=False)
    np.testing.assert_array_equal(mets['vq/som'], 0.0)
    assert 'vq/rec_e' not in mets

  def test_som_on_adds_both(self):
    enc, dec = build()
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    loss, mets = self._run(enc, dec, params, x, som=True)
    assert float(mets['vq/som']) > 0.0
    assert 'vq/rec_e' in mets

  def test_som_arm_loss_equals_the_written_formula(self):
    # Reassemble the loss from its published parts and compare, so a wiring
    # slip (wrong term, wrong weight, missing reconstruction) shows up.
    enc, dec = build()
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    kw = dict(som=True, codebook_scale=1.0, commit_scale=1.0, som_scale=0.9)
    loss, mets = self._run(enc, dec, params, x, **kw)

    def manual(x):
      z_e = enc.latent(x, 2)
      q = dec.codebook.quantize(z_e)
      tgt = jax.lax.stop_gradient(x)
      rec = dec(q['onehot'], 2).loss(tgt) + dec.from_latent(z_e, 2).loss(tgt)
      nb = dec.codebook.neighbors(q['ids'])
      t = vq.vq_losses(z_e, q['z_q'], nb, 'sum')
      return rec + t['codebook'] + t['commit'] + 0.9 * t['som']
    ref = nj.pure(manual)(params, x)[1]
    np.testing.assert_allclose(loss, ref, rtol=1e-5)

  def test_lip_penalty_only_present_with_lip_networks(self):
    x = deters((2, 3))
    enc, dec = build(lip=False)
    params = init_all(enc, dec, x)
    _, mets = self._run(enc, dec, params, x)
    np.testing.assert_array_equal(mets['vq/lip_penalty'], 0.0)
    np.testing.assert_array_equal(mets['vq/lip_bound_max'], 0.0)

    enc, dec = build(lip=True, layers=2, cinit=0.5)
    params = init_all(enc, dec, x)
    _, mets = self._run(enc, dec, params, x)
    # Default impl is logprod: log(0.5**6) over the 6 constrained layers
    # (2 hidden + 1 output projection, encoder and decoder).
    np.testing.assert_allclose(
        mets['vq/lip_penalty'], np.log(0.5 ** 6), rtol=1e-4)
    np.testing.assert_allclose(mets['vq/lip_bound_max'], 0.5, rtol=1e-4)
    _, mets = self._run(enc, dec, params, x, lip_impl='prod')
    np.testing.assert_allclose(mets['vq/lip_penalty'], 0.5 ** 6, rtol=1e-4)

  def test_lip_scale_enters_the_loss(self):
    x = deters((2, 3))
    enc, dec = build(lip=True, layers=2, cinit=2.0)
    params = init_all(enc, dec, x)
    a, _ = self._run(enc, dec, params, x, lip_scale=0.0)
    b, mets = self._run(enc, dec, params, x, lip_scale=1.0)
    np.testing.assert_allclose(b - a, float(mets['vq/lip_penalty']), rtol=1e-4)

  def test_lip_penalty_sum_variant(self):
    x = deters((2, 3))
    enc, dec = build(lip=True, layers=2, cinit=0.5)
    params = init_all(enc, dec, x)
    _, mets = self._run(enc, dec, params, x, lip_impl='sum')
    np.testing.assert_allclose(mets['vq/lip_penalty'], 0.5 * 6, rtol=1e-4)

  def test_quant_err_matches_the_distance_to_the_chosen_entry(self):
    enc, dec = build()
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    _, mets = self._run(enc, dec, params, x)
    def ref(x):
      z = enc.latent(x, 2)
      q = dec.codebook.quantize(z)
      return jnp.linalg.norm(z - q['z_q'], axis=-1).mean()
    np.testing.assert_allclose(
        mets['vq/quant_err'], nj.pure(ref)(params, x)[1], rtol=1e-4)

  def test_usage_metrics_present(self):
    enc, dec = build()
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    _, mets = self._run(enc, dec, params, x)
    for k in ('vq/used_frac', 'vq/perplexity', 'vq/max_prob'):
      assert k in mets
    assert 0.0 <= float(mets['vq/used_frac']) <= 1.0
    assert 1.0 - 1e-5 <= float(mets['vq/perplexity']) <= C + 1e-5


class TestGradientRoutingPerArm:
  """Which parameters each ablation arm actually trains."""

  def _grads(self, som, lip, **kw):
    enc, dec = build(lip=lip, layers=2, **kw)
    x = deters((2, 3), scale=1.0)
    params = init_all(enc, dec, x)
    def loss(params):
      def fn(x):
        return goal_ae.vq_goal_loss(enc, dec, x, 2, som=som)[0].mean()
      return nj.pure(fn)(params, x)[1]
    g = jax.grad(loss)(params)
    return {k: float(np.abs(np.asarray(v)).max()) for k, v in g.items()}

  def test_decoder_and_codebook_train_in_every_arm(self):
    for som in (False, True):
      g = self._grads(som=som, lip=False)
      assert g['goal_dec/codebook/table'] > 1e-8, som
      assert g['goal_dec/mlp/linear0/kernel'] > 1e-8, som
      assert g['goal_dec/out/kernel'] > 1e-8, som

  def test_encoder_trains_in_every_arm(self):
    # Without a straight-through estimator the encoder is reachable only via
    # the commitment term (vq/lipvq arms) plus the second reconstruction (som
    # arm). If either were mis-wired the encoder would silently freeze.
    for som in (False, True):
      g = self._grads(som=som, lip=False)
      assert g['goal_enc/mlp/linear0/kernel'] > 1e-8, som
      assert g['goal_enc/out/kernel'] > 1e-8, som

  def test_commitment_alone_carries_the_encoder_without_som(self):
    enc, dec = build(layers=2)
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    def loss(params, commit_scale):
      def fn(x):
        return goal_ae.vq_goal_loss(
            enc, dec, x, 2, som=False, ste=False,
            commit_scale=commit_scale)[0].mean()
      return nj.pure(fn)(params, x)[1]
    on = jax.grad(loss)(params, 1.0)['goal_enc/out/kernel']
    off = jax.grad(loss)(params, 0.0)['goal_enc/out/kernel']
    assert np.abs(np.asarray(on)).max() > 1e-8
    np.testing.assert_array_equal(np.asarray(off), 0.0)

  def test_som_arm_keeps_training_the_encoder_without_commitment(self):
    # The second reconstruction is an independent path to the encoder.
    enc, dec = build(layers=2)
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    def loss(params):
      def fn(x):
        return goal_ae.vq_goal_loss(
            enc, dec, x, 2, som=True, commit_scale=0.0)[0].mean()
      return nj.pure(fn)(params, x)[1]
    g = jax.grad(loss)(params)['goal_enc/out/kernel']
    assert np.abs(np.asarray(g)).max() > 1e-8

  def test_lipschitz_bounds_train_in_the_lip_arms_only(self):
    g = self._grads(som=False, lip=True, cinit=0.5)
    for k in ('goal_enc/mlp/linear0/c', 'goal_enc/out/c', 'goal_dec/out/c'):
      assert g[k] > 1e-12, k
    g = self._grads(som=False, lip=False)
    assert not [k for k in g if k.endswith('/c')]

  def test_reconstruction_target_is_stop_gradiented(self):
    # dec(...).loss(target) must not backprop into the target; otherwise the
    # world-model deter would be pulled toward its own reconstruction.
    enc, dec = build(layers=2)
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    code = jnp.asarray(np.eye(C)[np.zeros((2, 3, L), int)], f32)
    def fn(x):
      return nj.pure(lambda c, t: dec(c, 2).loss(t))(params, code, x)[1].sum()
    np.testing.assert_array_equal(np.asarray(jax.grad(fn)(x)), 0.0)


class TestLearning:
  """End-to-end numerical checks: the thing actually optimizes."""

  def _train(self, som, lip, steps=600, lr=3e-3, seed=0, ste=None,
             points=64, **kw):
    lkw = {k: kw.pop(k) for k in ('som_scale',) if k in kw}
    enc, dec = build(lip=lip, layers=2, units=64, **kw)
    # A low-dimensional manifold embedded in deter space, so a small code has a
    # real chance of capturing structure.
    rng = np.random.default_rng(seed)
    basis = rng.normal(0, 1, (2, DETER))
    coeff = rng.normal(0, 1, (points, 2))
    x = jnp.asarray(coeff @ basis, f32)[None]                  # (1, N, DETER)
    params = init_all(enc, dec, x)

    def loss(params):
      def fn(x):
        return goal_ae.vq_goal_loss(
            enc, dec, x, 2, som=som, ste=ste, **lkw)[0].mean()
      return nj.pure(fn)(params, x)[1]

    def rec(params):
      def fn(x):
        return goal_ae.vq_goal_loss(
            enc, dec, x, 2, som=som, ste=ste, **lkw)[1]
      return nj.pure(fn)(params, x)[1]

    # Adam, not plain SGD: the real runs use it, and vanilla SGD on a freshly
    # initialized VQ bottleneck is slow enough that a threshold tuned to it
    # would say nothing about whether the objective is wired correctly.
    opt = optax.adam(lr)
    state = opt.init(params)
    grad = jax.jit(jax.grad(loss))
    step = jax.jit(opt.update)
    first = rec(params)
    for _ in range(steps):
      updates, state = step(grad(params), state, params)
      params = optax.apply_updates(params, updates)
    return first, rec(params), params

  @pytest.mark.parametrize('som,lip', [(False, False), (True, False),
                                       (False, True), (True, True)])
  def test_reconstruction_improves(self, som, lip):
    kw = dict(cinit=5.0) if lip else {}
    before, after, params = self._train(som=som, lip=lip, **kw)
    assert float(after['vq/rec_q']) < float(before['vq/rec_q']) * 0.5, (
        float(before['vq/rec_q']), float(after['vq/rec_q']))
    assert np.isfinite(float(after['vq/rec_q']))

  def test_encoder_needs_a_reconstruction_path(self):
    # The LipVQ-VAE released code has neither a straight-through estimator nor
    # a second reconstruction, so nothing carries reconstruction error to the
    # encoder and it trains far worse. This is why ste defaults to True off the
    # SOM arm; the paper itself says it follows VQ-VAE's procedure, which does
    # use the estimator.
    _, ref_after, _ = self._train(som=False, lip=False, ste=False)
    _, ste_after, _ = self._train(som=False, lip=False, ste=True)
    assert float(ste_after['vq/rec_q']) < float(ref_after['vq/rec_q']) / 2

  def test_straight_through_reaches_the_encoder_but_not_the_codebook(self):
    enc, dec = build(layers=2)
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    def only_rec(params, ste):
      def fn(x):
        return goal_ae.vq_goal_loss(
            enc, dec, x, 2, som=False, ste=ste, codebook_scale=0.0,
            commit_scale=0.0)[0].mean()
      return nj.pure(fn)(params, x)[1]
    g = jax.grad(only_rec)(params, True)
    assert np.abs(np.asarray(g['goal_enc/out/kernel'])).max() > 1e-8
    # VQ-VAE: the estimator's stop-gradient keeps reconstruction off the table.
    np.testing.assert_array_equal(
        np.asarray(g['goal_dec/codebook/table']), 0.0)
    g = jax.grad(only_rec)(params, False)
    np.testing.assert_array_equal(np.asarray(g['goal_enc/out/kernel']), 0.0)
    assert np.abs(np.asarray(g['goal_dec/codebook/table'])).max() > 1e-8

  def test_codebook_tracks_the_encoder_output(self):
    # The commitment/codebook pair should leave the quantization error small
    # relative to the latent scale, i.e. the codebook actually covers the
    # latent cloud rather than sitting off to one side.
    _, after, _ = self._train(som=False, lip=False)
    assert float(after['vq/quant_err']) < float(after['vq/z_e_norm'])

  def test_som_arm_reduces_quantization_error(self):
    before, after, _ = self._train(som=True, lip=False)
    assert float(after['vq/quant_err']) < float(before['vq/quant_err'])

  def test_codebook_does_not_fully_collapse(self):
    # VQ's characteristic failure: every input maps to one entry. Not a proof
    # it will not happen at scale, but it would catch a dead codebook here.
    before, after, params = self._train(som=False, lip=False)
    assert float(after['vq/perplexity']) > 1.0

  def test_som_term_organizes_the_codebook_into_a_ring(self):
    # The defining claim of the SOM arm: ring-adjacent entries end up closer
    # together than arbitrary pairs, which is what makes a one-block code edit
    # a small goal edit. Measured against two references: a random codebook
    # scores ~1.0, and C evenly spaced points on a circle score the ideal
    # value below. Uses the project's real C=8 -- at C=5 only 2 of the 4 other
    # entries are ring neighbors and the statistic barely separates.
    classes = 8
    ang = 2 * np.pi * np.arange(classes) / classes
    circle = np.stack([np.cos(ang), np.sin(ang)], 1)
    ideal = (np.linalg.norm(circle - np.roll(circle, 1, 0), axis=-1).mean()
             / np.linalg.norm(circle[:, None] - circle[None, :], axis=-1)[
                 ~np.eye(classes, dtype=bool)].mean())

    def ring_ratio(som_scale):
      _, _, params = self._train(
          som=True, lip=False, classes=classes, points=256, steps=1500,
          som_scale=som_scale)
      t = np.asarray(params['goal_dec/codebook/table'])         # (L, C, DIM)
      nb = np.linalg.norm(t - np.roll(t, 1, axis=1), axis=-1).mean()
      allp = np.linalg.norm(t[:, :, None] - t[:, None, :], axis=-1)
      off = allp[:, ~np.eye(classes, dtype=bool)].mean()
      return float(nb / (off + 1e-8))

    on = ring_ratio(0.9)
    off = ring_ratio(0.0)
    assert off > 0.9, off                       # no topology without the term
    assert on < 0.8, on                          # clearly organized
    assert on < off, (on, off)
    assert on > ideal - 0.2, (on, ideal)         # sanity: not below the ideal

  def test_lipschitz_bounds_are_pulled_down_by_the_penalty(self):
    enc, dec = build(lip=True, layers=2, units=32, cinit=5.0)
    x = deters((1, 32))
    params = init_all(enc, dec, x)
    def loss(params):
      def fn(x):
        return goal_ae.vq_goal_loss(
            enc, dec, x, 2, lip_scale=1.0, lip_impl='sum')[0].mean()
      return nj.pure(fn)(params, x)[1]
    grad = jax.jit(jax.grad(loss))
    start = float(jax.nn.softplus(params['goal_enc/out/c']).max())
    for _ in range(50):
      g = grad(params)
      params = {k: v - 1e-2 * g[k] for k, v in params.items()}
    end = float(jax.nn.softplus(params['goal_enc/out/c']).max())
    assert end < start, (start, end)


class TestEncoderLipschitzProperty:

  def test_encoder_map_is_inf_norm_bounded(self):
    # The architectural claim of the LipVQ arm: deter -> z_e is Lipschitz with
    # the product of the layer bounds. Uses relu (exactly 1-Lipschitz).
    enc, dec = build(lip=True, layers=2, units=32, act='relu', cinit=0.5)
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    def fn(x):
      enc(x, 2)
      return jnp.stack(enc.bounds())
    bounds = np.asarray(nj.pure(fn)(params, x)[1])
    total = float(np.prod(bounds))
    np.testing.assert_allclose(total, 0.5 ** 3, rtol=1e-4)

    rng = np.random.default_rng(9)
    x1 = jnp.asarray(rng.normal(0, 5, (64, DETER)), f32)
    x2 = jnp.asarray(rng.normal(0, 5, (64, DETER)), f32)
    z1 = nj.pure(lambda x: enc.latent(x, 1))(params, x1)[1].reshape((64, -1))
    z2 = nj.pure(lambda x: enc.latent(x, 1))(params, x2)[1].reshape((64, -1))
    lhs = np.abs(np.asarray(z1 - z2)).max(-1)
    rhs = total * np.abs(np.asarray(x1 - x2)).max(-1)
    assert (lhs <= rhs * (1 + 1e-4)).all(), (lhs.max(), rhs.min())

  def _empirical_ratio(self, **kw):
    enc, dec = build(layers=2, units=32, act='relu', **kw)
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    rng = np.random.default_rng(9)
    x1 = jnp.asarray(rng.normal(0, 5, (64, DETER)), f32)
    x2 = jnp.asarray(rng.normal(0, 5, (64, DETER)), f32)
    z1 = nj.pure(lambda x: enc.latent(x, 1))(params, x1)[1].reshape((64, -1))
    z2 = nj.pure(lambda x: enc.latent(x, 1))(params, x2)[1].reshape((64, -1))
    num = np.abs(np.asarray(z1 - z2)).max(-1)
    den = np.abs(np.asarray(x1 - x2)).max(-1)
    return float((num / den).max())

  def test_constraint_actually_reduces_the_realized_lipschitz_ratio(self):
    # Control for the bound test: the same architecture unconstrained has a
    # much larger realized ratio, so the bound above is not vacuously true of
    # any small-initialized network.
    tight = self._empirical_ratio(lip=True, cinit=0.01)
    free = self._empirical_ratio(lip=False)
    assert tight < free / 10.0, (tight, free)
    assert tight <= 0.01 ** 3 * (1 + 1e-4)


class TestLipPenaltyReach:

  def test_penalty_gradient_reaches_only_the_bounds(self):
    # The penalty is a function of the trainable bounds c alone, not of the
    # weights, so however large it grows it cannot drown the reconstruction
    # gradient on the kernels. This is what limits the damage when gamma is
    # mis-scaled for the layer count.
    enc, dec = build(lip=True, layers=2, units=32, cinit=5.0)
    x = deters((2, 3))
    params = init_all(enc, dec, x)
    def penalty(params):
      def fn(x):
        enc(x, 2)
        dec.from_latent(enc.latent(x, 2), 2)
        return goal_ae.lip_penalty(enc.bounds() + dec.bounds(), 'prod')
      return nj.pure(fn)(params, x)[1]
    g = jax.grad(penalty)(params)
    for k, v in g.items():
      a = np.abs(np.asarray(v))
      if k.endswith('/c'):
        assert a.max() > 0.0, k
      else:
        np.testing.assert_array_equal(a, 0.0, err_msg=k)


class TestPenaltyFormPreservesTheGoal:
  """prod vs logprod: same guarantee, same quantity shrunk, different operating point."""

  def _train_and_measure(self, impl, lip_scale, steps=200, lr=3e-3):
    enc, dec = build(lip=True, layers=2, units=32, act='relu', norm='none',
                     strict_bound=True, cinit=3.0)
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(0, 1, (1, 64, DETER)), f32)
    params = init_all(enc, dec, x)

    def prod_bound(params):
      def fn(x):
        enc(x, 2)
        return goal_ae.lip_penalty(enc.bounds(), 'prod')
      return float(nj.pure(fn)(params, x)[1])

    def realized_ratio(params):
      r = np.random.default_rng(1)
      x1 = jnp.asarray(r.normal(0, 5, (128, DETER)), f32)
      x2 = jnp.asarray(r.normal(0, 5, (128, DETER)), f32)
      z1 = nj.pure(lambda a: enc.latent(a, 1))(params, x1)[1].reshape((128, -1))
      z2 = nj.pure(lambda a: enc.latent(a, 1))(params, x2)[1].reshape((128, -1))
      num = np.abs(np.asarray(z1 - z2)).max(-1)
      den = np.abs(np.asarray(x1 - x2)).max(-1)
      return float((num / den).max())

    def loss(params):
      def fn(x):
        return goal_ae.vq_goal_loss(
            enc, dec, x, 2, lip_scale=lip_scale, lip_impl=impl)[0].mean()
      return nj.pure(fn)(params, x)[1]

    opt = optax.adam(lr); state = opt.init(params)
    grad = jax.jit(jax.grad(loss)); step = jax.jit(opt.update)
    before = (prod_bound(params), realized_ratio(params))
    worst = 0.0
    for _ in range(steps):
      upd, state = step(grad(params), state, params)
      params = optax.apply_updates(params, upd)
      worst = max(worst, realized_ratio(params) / prod_bound(params))
    return before, (prod_bound(params), realized_ratio(params)), worst

  @pytest.mark.parametrize('impl,scale', [('prod', 0.05), ('logprod', 0.5)])
  def test_both_forms_shrink_the_same_lipschitz_constant(self, impl, scale):
    # The quantity that bounds the map is prod(b). Whether the penalty is the
    # product or its logarithm, training must drive THAT down -- this is what
    # "same goal" means, and log being strictly increasing is why it holds.
    before, after, _ = self._train_and_measure(impl, scale)
    assert after[0] < before[0], (impl, before[0], after[0])

  @pytest.mark.parametrize('impl,scale', [('prod', 0.05), ('logprod', 0.5),
                                          ('logprod', 0.0)])
  def test_the_bound_holds_throughout_training_for_any_penalty(self, impl, scale):
    # The guarantee comes from the weight normalization, not the penalty: the
    # realized inf-norm ratio never exceeds prod(b) at any point, including
    # with the penalty switched off entirely (scale 0.0).
    _, _, worst_ratio = self._train_and_measure(impl, scale)
    assert worst_ratio <= 1.0 + 1e-4, (impl, scale, worst_ratio)

  def test_a_tighter_bound_yields_a_smoother_map(self):
    # What makes the constraint meaningful is that the bound controls the
    # realized sensitivity. Note it does NOT follow that training reduces that
    # sensitivity: at initialization the encoder sits far below its own bound
    # (measured ratio ~0.04), and fitting the reconstruction moves it UP toward
    # the bound. The constraint only starts to act once the two meet, which is
    # what goal/lip_active_frac reports at run time.
    ratios = []
    for cinit in (0.3, 30.0):
      enc, dec = build(lip=True, layers=2, units=32, act='relu', norm='none',
                       strict_bound=True, cinit=cinit)
      rng = np.random.default_rng(0)
      x = jnp.asarray(rng.normal(0, 1, (1, 64, DETER)), f32)
      params = init_all(enc, dec, x)
      def loss(params):
        def fn(x):
          return goal_ae.vq_goal_loss(enc, dec, x, 2, lip_scale=0.0)[0].mean()
        return nj.pure(fn)(params, x)[1]
      opt = optax.adam(3e-3); state = opt.init(params)
      grad = jax.jit(jax.grad(loss)); step = jax.jit(opt.update)
      for _ in range(200):
        upd, state = step(grad(params), state, params)
        params = optax.apply_updates(params, upd)
      r = np.random.default_rng(1)
      x1 = jnp.asarray(r.normal(0, 5, (128, DETER)), f32)
      x2 = jnp.asarray(r.normal(0, 5, (128, DETER)), f32)
      z1 = nj.pure(lambda a: enc.latent(a, 1))(params, x1)[1].reshape((128, -1))
      z2 = nj.pure(lambda a: enc.latent(a, 1))(params, x2)[1].reshape((128, -1))
      num = np.abs(np.asarray(z1 - z2)).max(-1)
      den = np.abs(np.asarray(x1 - x2)).max(-1)
      def bnd(x):
        enc(x, 2)
        return goal_ae.lip_penalty(enc.bounds(), 'prod')
      final_bound = float(nj.pure(bnd)(params, x)[1])
      ratios.append((float((num / den).max()), final_bound, cinit))
    (tight, tb, _), (loose, lb, _) = ratios
    assert tight < loose, ratios
    # Each stays under its own FINAL bound. Not its initial one: with the
    # penalty off, an ACTIVE constraint feeds gradient back into c, and
    # reconstruction pushes the bounds up (here 0.3^3 = 0.027 at init grew to
    # ~tb). That self-relaxation is why the penalty is not optional.
    assert tight <= tb * (1 + 1e-4), (tight, tb)
    assert loose <= lb * (1 + 1e-4), (loose, lb)
    assert tb > 0.3 ** 3, (tb, 0.3 ** 3)


class TestTrunkNormalizationBreaksTheComposedBound:

  def _ratio(self, norm, scale, cinit=0.5):
    dec = goal_ae.GoalVQDecoder(DETER, L, C, DIM, layers=2, units=32, lip=True,
                                act='relu', norm=norm, cinit=cinit,
                                name='goal_dec')
    enc = goal_ae.GoalVQEncoder(dec.codebook, L, DIM, layers=2, units=32,
                                lip=True, act='relu', norm=norm, cinit=cinit,
                                name='goal_enc')
    x0 = jnp.zeros((1, 2, DETER), f32)
    params = nj.init(lambda x: enc(x, 2))({}, x0, seed=0)
    def bnd(x):
      enc(x, 2)
      return goal_ae.lip_penalty(enc.bounds(), 'prod')
    bound = float(nj.pure(bnd)(params, x0)[1])
    r = np.random.default_rng(0)
    x1 = jnp.asarray(r.normal(0, scale, (1024, DETER)), f32)
    x2 = jnp.asarray(r.normal(0, scale, (1024, DETER)), f32)
    z1 = nj.pure(lambda a: enc.latent(a, 1))(params, x1)[1].reshape((1024, -1))
    z2 = nj.pure(lambda a: enc.latent(a, 1))(params, x2)[1].reshape((1024, -1))
    num = np.abs(np.asarray(z1 - z2)).max(-1)
    den = np.abs(np.asarray(x1 - x2)).max(-1)
    return float((num / den).max()), bound

  def test_without_trunk_norm_the_bound_holds_at_every_input_scale(self):
    # A genuine Lipschitz map: the realized ratio does not depend on how large
    # the inputs are, and never exceeds the product of the layer bounds.
    ratios = []
    for scale in (1e0, 1e-1, 1e-2, 1e-4):
      ratio, bound = self._ratio('none', scale)
      assert ratio <= bound * (1 + 1e-4), (scale, ratio, bound)
      ratios.append(ratio)
    assert max(ratios) / min(ratios) < 1.01, ratios

  def test_rms_trunk_norm_violates_the_composed_bound_at_small_inputs(self):
    # RMS normalization divides by the input's own magnitude, so its gain grows
    # without limit as inputs shrink; it is not 1-Lipschitz and the per-layer
    # bounds therefore do not compose. This is why prod(b) bounds the encoder's
    # LINEAR LAYERS and not the encoder, and why strict_bound requires
    # norm='none'. Documented as a test so the limitation cannot be lost.
    ratio_big, bound = self._ratio('rms', 1e0)
    ratio_small, _ = self._ratio('rms', 1e-3)
    assert ratio_big <= bound, (ratio_big, bound)      # holds at usual scale
    assert ratio_small > bound * 5, (ratio_small, bound)  # fails well below it


class TestManagerRingHead:
  """Ring-aware manager head, exercised directly rather than through the agent."""

  def _head(self, **kw):
    dec = goal_ae.GoalVQDecoder(DETER, L, C, DIM, layers=2, units=32,
                                name='goal_dec')
    head = goal_ae.ManagerRingHead(dec.codebook, L, DIM, layers=2, units=32,
                                   name='manager_pol', **kw)
    x = jnp.asarray(np.random.default_rng(0).normal(0, 1, (2, 3, 16)), f32)
    params = nj.init(lambda t: head(t, 2))({}, x, seed=0)
    return head, dec, params, x

  def test_output_protocol_matches_the_director_manager(self):
    head, dec, params, x = self._head()
    out = nj.pure(lambda t: head(t, 2))(params, x)[1]
    assert isinstance(out, outs.Agg)
    assert out.pred().shape == (2, 3, L, C)
    assert out.sample(jax.random.PRNGKey(0)).shape == (2, 3, L, C)
    assert out.entropy().shape == (2, 3)
    ev = out.sample(jax.random.PRNGKey(0))
    assert out.logp(ev).shape == (2, 3)

  def test_projects_to_the_latent_not_to_free_logits(self):
    head, dec, params, x = self._head()
    assert int(np.asarray(params['manager_pol/out/kernel']).shape[-1]) == L * DIM
    assert 'manager_pol/logit_scale' in params

  def test_advertises_its_entropy_range_like_the_director_head(self):
    # e478's failure: losses.imag_loss_mgr skips any head missing these two
    # attributes, silently, so the entropy adapter never ran and the policy
    # drifted to 16.3 of 16.64 nats over 1.6M steps. The guard is a hasattr on
    # the UNWRAPPED output, so it has to survive the Agg wrapping.
    head, dec, params, x = self._head()
    out = nj.pure(lambda t: head(t, 2))(params, x)[1]
    inner = _head_inner(out)
    assert hasattr(inner, 'minent') and hasattr(inner, 'maxent')
    assert inner.minent == 0.0
    # Totals across the L blocks, matching MLPHead.onehot; imag_loss_mgr
    # divides by L to get the per-block normalizer.
    np.testing.assert_allclose(inner.maxent, L * np.log(C), rtol=1e-6)

  def test_the_advertised_maximum_is_the_entropy_of_a_uniform_policy(self):
    # maxent is only meaningful if a uniform policy actually attains it -- an
    # off-by-L here would make the adapter chase the wrong setpoint without
    # ever erroring.
    head, dec, params, x = self._head()
    out = nj.pure(lambda t: head(t, 2))(params, x)[1]
    inner = _head_inner(out)
    # Uniform logits pushed through the same Agg wrapping the head applies, so
    # the comparison is against the summed-over-L quantity maxent claims to be.
    uniform = outs.Agg(outs.OneHot(jnp.zeros((2, 3, L, C), f32)), 1, jnp.sum)
    np.testing.assert_allclose(
        np.asarray(uniform.entropy()).mean(), inner.maxent, rtol=1e-5)
    # And the per-block entropy the adapter actually normalizes is maxent / L.
    np.testing.assert_allclose(
        np.asarray(_head_inner(uniform).entropy()).mean(), inner.maxent / L,
        rtol=1e-5)

  def test_codebook_is_read_under_stop_gradient(self):
    # The codebook is shaped by the autoencoder objective alone; a REINFORCE
    # gradient reaching it would conflate the two.
    head, dec, params, x = self._head()
    def fn(params):
      return nj.pure(lambda t: head(t, 2).entropy().mean())(params, x)[1]
    g = jax.grad(fn)(params)
    np.testing.assert_array_equal(
        np.asarray(g['goal_dec/codebook/table']), 0.0)
    assert np.abs(np.asarray(g['manager_pol/out/kernel'])).max() > 0.0
    assert np.abs(np.asarray(g['manager_pol/logit_scale'])).max() > 0.0

  def test_scale_can_sharpen_a_near_uniform_policy(self):
    # Squared distances between a block's entries are comparable, so a unit
    # scale gives a near-uniform policy; the trainable scale must peak it.
    ents = []
    for init in (1.0, 50.0):
      head, dec, params, x = self._head(scale_init=init)
      ents.append(float(nj.pure(
          lambda t: head(t, 2).entropy().mean())(params, x)[1]))
    assert ents[1] < ents[0] * 0.6, ents

  def test_neighbouring_entries_receive_similar_probability(self):
    # The point of the head: with an ordered codebook, ring-adjacent entries
    # get similar logits, which is what lets REINFORCE credit generalize.
    head, dec, params, x = self._head()
    ring = np.zeros((L, C, DIM), np.float32)
    ang = 2 * np.pi * np.arange(C) / C
    ring[:, :, 0] = np.cos(ang); ring[:, :, 1] = np.sin(ang)
    params = {**params, 'goal_dec/codebook/table': jnp.asarray(ring, f32)}
    logits = nj.pure(lambda t: _head_inner(head(t, 2)).dist.logits)(params, x)[1]
    p = np.asarray(jax.nn.softmax(logits, -1)).reshape(-1, C)
    best = p.argmax(-1)
    adj = p[np.arange(len(p)), (best + 1) % C]
    far = p[np.arange(len(p)), (best + C // 2) % C]
    assert (adj > far).mean() > 0.95, (adj.mean(), far.mean())


class TestRingSmoothedCredit:
  """Spreading the REINFORCE event over ring neighbours."""

  RC = 8            # the project's C; the module-level C here is 5

  def _event(self, ids):
    return jnp.asarray(np.eye(self.RC)[ids], f32)

  def test_alpha_zero_is_the_identity(self):
    # The arm must differ from goal_som_lipvq by this one number, so alpha=0
    # has to reproduce the standard estimator bit for bit.
    e = self._event(np.random.default_rng(0).integers(0, self.RC, (4, 3, L)))
    np.testing.assert_array_equal(hrl_heads.smooth_skill_event(e, 0.0), e)

  def test_weights_sum_to_one_and_land_on_the_ring_neighbours(self):
    got = np.asarray(hrl_heads.smooth_skill_event(
        self._event(np.array([[3]])), 0.25))[0, 0]
    np.testing.assert_allclose(got.sum(), 1.0, rtol=1e-6)
    np.testing.assert_allclose(got[[2, 3, 4]], [0.25, 0.5, 0.25], rtol=1e-6)
    assert got[[0, 1, 5, 6, 7]].max() == 0.0

  def test_wraps_around_the_ring(self):
    C_ = self.RC
    for k, lo, hi in ((0, C_ - 1, 1), (C_ - 1, C_ - 2, 0)):
      got = np.asarray(hrl_heads.smooth_skill_event(
          self._event(np.array([[k]])), 0.25))[0, 0]
      np.testing.assert_allclose(got[[lo, k, hi]], [0.25, 0.5, 0.25], rtol=1e-6)

  def test_blocks_are_smoothed_independently(self):
    got = np.asarray(hrl_heads.smooth_skill_event(
        self._event(np.array([[0, 4]])), 0.25))[0]
    np.testing.assert_allclose(
        got[0][[self.RC - 1, 0, 1]], [0.25, 0.5, 0.25], rtol=1e-6)
    np.testing.assert_allclose(got[1][[3, 4, 5]], [0.25, 0.5, 0.25], rtol=1e-6)

  def test_smoothing_shifts_credit_from_the_sample_to_its_neighbours(self):
    # The gradient of sum_c w_c log softmax(logits)_c wrt logit_j is w_j - p_j,
    # so the sign depends on the current probabilities; what smoothing changes
    # is the WEIGHT, hence the credit relative to no smoothing.
    rng = np.random.default_rng(1)
    logits = jnp.asarray(rng.normal(0, 1, (1, 1, L, self.RC)), f32)
    ids = np.zeros((1, 1, L), int); ids[..., 0] = 3
    ev = self._event(ids)
    def obj(logits, alpha):
      return (jax.nn.log_softmax(logits, -1)
              * hrl_heads.smooth_skill_event(ev, alpha)).sum()
    g0 = np.asarray(jax.grad(obj)(logits, 0.0))[0, 0, 0]
    ga = np.asarray(jax.grad(obj)(logits, 0.25))[0, 0, 0]
    np.testing.assert_allclose(ga[[2, 4]] - g0[[2, 4]], [0.25, 0.25], rtol=1e-5)
    np.testing.assert_allclose(ga[3] - g0[3], -0.5, rtol=1e-5)
    far = [0, 1, 5, 6, 7]
    np.testing.assert_allclose(ga[far], g0[far], rtol=1e-5)


class TestLineSmoothedCredit:
  """The same kernel on an open path: no wraparound, renormalized endpoints."""

  RC = 8
  A = 0.1           # the alpha goal_som_lipvq_line_smooth actually runs

  def _event(self, ids):
    return jnp.asarray(np.eye(self.RC)[ids], f32)

  def _smooth(self, ids, alpha=None, topology='line'):
    alpha = self.A if alpha is None else alpha
    return np.asarray(hrl_heads.smooth_skill_event(
        self._event(np.atleast_2d(ids)), alpha, topology))[0]

  def test_alpha_zero_is_the_identity_on_a_line_too(self):
    e = self._event(np.random.default_rng(0).integers(0, self.RC, (4, 3, L)))
    np.testing.assert_array_equal(
        hrl_heads.smooth_skill_event(e, 0.0, 'line'), e)

  def test_interior_picks_match_the_ring_kernel(self):
    # Away from the ends a path and a ring are the same neighbourhood, so the
    # renormalization must be a no-op there (row sum is already 1).
    for k in range(1, self.RC - 1):
      line = self._smooth([k])[0]
      ring = self._smooth([k], topology='ring')[0]
      np.testing.assert_allclose(line, ring, rtol=1e-6, atol=1e-7)
      np.testing.assert_allclose(
          line[[k - 1, k, k + 1]], [self.A, 1 - 2 * self.A, self.A], rtol=1e-6)

  def test_no_wraparound_between_the_two_endpoints(self):
    # The whole point of the line: entries 0 and C-1 are the FARTHEST apart, and
    # jnp.roll would hand them each other's credit.
    assert self._smooth([0])[0][self.RC - 1] == 0.0
    assert self._smooth([self.RC - 1])[0][0] == 0.0

  def test_endpoints_are_renormalized_option_a(self):
    # Masked row is (1-2a) on self + a on the one surviving neighbour, summing
    # to 1-a; dividing by that gives ((1-2a)/(1-a), a/(1-a)).
    a = self.A
    want_self, want_nb = (1 - 2 * a) / (1 - a), a / (1 - a)
    lo = self._smooth([0])[0]
    hi = self._smooth([self.RC - 1])[0]
    np.testing.assert_allclose([lo[0], lo[1]], [want_self, want_nb], rtol=1e-6)
    np.testing.assert_allclose(
        [hi[self.RC - 1], hi[self.RC - 2]], [want_self, want_nb], rtol=1e-6)
    # The surviving neighbour gets MORE than alpha -- that is what (a) buys.
    assert want_nb > a

  def test_every_row_is_a_distribution(self):
    for k in range(self.RC):
      row = self._smooth([k])[0]
      np.testing.assert_allclose(row.sum(), 1.0, rtol=1e-6)
      assert (row >= 0).all()

  def test_blocks_are_smoothed_independently_on_a_line(self):
    got = self._smooth([0, 4, self.RC - 1])
    a = self.A
    np.testing.assert_allclose(
        got[0][[0, 1]], [(1 - 2 * a) / (1 - a), a / (1 - a)], rtol=1e-6)
    np.testing.assert_allclose(
        got[1][[3, 4, 5]], [a, 1 - 2 * a, a], rtol=1e-6)
    np.testing.assert_allclose(
        got[2][[self.RC - 1, self.RC - 2]],
        [(1 - 2 * a) / (1 - a), a / (1 - a)], rtol=1e-6)

  def test_entropy_floor_matches_the_configured_headroom(self):
    # goal_som_lipvq_line_smooth exists because alpha=0.25 put this floor at
    # exactly manager_actent_target=0.5, leaving the entropy controller nothing
    # to trade. These are the numbers the config comment claims.
    def norm_ent(row):
      row = row[row > 0]
      return float(-(row * np.log(row)).sum() / np.log(self.RC))
    np.testing.assert_allclose(norm_ent(self._smooth([4])[0]), 0.307, atol=5e-4)
    # 0.168, not the 0.156 an unnormalized (1-a, a) endpoint would give: option
    # (a) hands the dropped weight to the surviving neighbour, not back to the
    # pick, so an endpoint row is slightly less peaked than (1-a, a).
    np.testing.assert_allclose(norm_ent(self._smooth([0])[0]), 0.168, atol=5e-4)
    assert norm_ent(self._smooth([4], alpha=0.25)[0]) == pytest.approx(0.5, 1e-6)

  def test_gradient_reaches_only_the_surviving_neighbour(self):
    rng = np.random.default_rng(2)
    logits = jnp.asarray(rng.normal(0, 1, (1, L, self.RC)), f32)
    ids = np.zeros((1, L), int)            # every block picks endpoint 0
    ev = self._event(ids)
    def obj(logits, alpha):
      return (jax.nn.log_softmax(logits, -1)
              * hrl_heads.smooth_skill_event(ev, alpha, 'line')).sum()
    g0 = np.asarray(jax.grad(obj)(logits, 0.0))[0, 0]
    ga = np.asarray(jax.grad(obj)(logits, self.A))[0, 0]
    a = self.A
    # Class 1 gains exactly the renormalized neighbour weight; class C-1 (the
    # wraparound partner a ring would have credited) gains nothing.
    np.testing.assert_allclose(ga[1] - g0[1], a / (1 - a), rtol=1e-5)
    np.testing.assert_allclose(ga[self.RC - 1] - g0[self.RC - 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(ga[0] - g0[0], (1 - 2 * a) / (1 - a) - 1, rtol=1e-5)
    np.testing.assert_allclose(ga[[2, 3, 4, 5, 6]], g0[[2, 3, 4, 5, 6]], rtol=1e-5)

  def test_ring_kernel_is_unchanged_by_the_refactor(self):
    # Regression guard for the runs in flight (e417/e420/e423/e426 read this
    # source on requeue): topology='ring' must still be the old jnp.roll kernel.
    e = self._event(np.random.default_rng(3).integers(0, self.RC, (2, L)))
    a = 0.25
    want = ((1 - 2 * a) * e + a * jnp.roll(e, 1, -1) + a * jnp.roll(e, -1, -1))
    np.testing.assert_allclose(
        np.asarray(hrl_heads.smooth_skill_event(e, a, 'ring')),
        np.asarray(want), rtol=1e-6)

  def test_rejects_an_unknown_topology(self):
    with pytest.raises(AssertionError):
      hrl_heads.smooth_skill_event(self._event(np.array([[0]])), 0.1, 'grid')
