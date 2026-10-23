"""Tests for the SOM class-step evenness loss (hrl/goal_ae.py evenness_loss)."""
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

import embodied.jax.nets as nets
from dreamerv3.hrl import goal_ae

f32 = jnp.float32
L, C, DIM, DETER = 4, 8, 3, 16


@pytest.fixture(autouse=True)
def f32_compute():
  old = nets.COMPUTE_DTYPE
  nets.COMPUTE_DTYPE = jnp.float32
  yield
  nets.COMPUTE_DTYPE = old


class _Pred:
  def __init__(self, x):
    self.x = x

  def pred(self):
    return self.x


class _Codebook:
  def __init__(self, table):
    self._table = table

  def table(self):
    return self._table


class LinearDec:
  """Goal = offset + flattened latent: steps follow the codebook spacing."""

  def __init__(self, table, offset=5.0):
    self.codebook = _Codebook(table)
    self.offset = offset

  def from_latent(self, z, bdims):
    flat = z.reshape((*z.shape[:bdims], -1))
    return _Pred(self.offset + flat)


def line_table(positions, scales=None):
  """Each block's C entries on a line in its first latent dim."""
  pos = jnp.asarray(positions, f32)
  scales = jnp.ones(L) if scales is None else jnp.asarray(scales, f32)
  t = jnp.zeros((L, C, DIM), f32)
  return t.at[:, :, 0].set(scales[:, None] * pos[None])


def z_q_from(table, n=32, seed=0):
  ids = np.random.default_rng(seed).integers(0, C, (n, L))
  return table[jnp.arange(L)[None], ids]                           # (n, L, D)


class TestStatistic:

  def test_even_line_is_zero(self):
    t = line_table(np.linspace(-1, 1, C))
    w, a, m = goal_ae.evenness_loss(LinearDec(t), z_q_from(t))
    # Not exactly 0: cosine distance is not linear in position along the line.
    assert float(w) < 0.02 and float(a) < 1e-3
    assert abs(float(m['even_end_ratio']) - 1) < 0.15

  def test_bunched_ends_raise_within_only(self):
    pos = np.array([-1.0, -0.95, -0.6, -0.2, 0.2, 0.6, 0.95, 1.0])
    t = line_table(pos)
    w, a, m = goal_ae.evenness_loss(LinearDec(t), z_q_from(t))
    assert float(w) > 0.1
    assert float(a) < 1e-3
    assert float(m['even_end_ratio']) < 0.5

  def test_unequal_blocks_raise_across(self):
    t = line_table(np.linspace(-1, 1, C), scales=[0.3, 0.6, 1.0, 1.5])
    w, a, m = goal_ae.evenness_loss(LinearDec(t), z_q_from(t))
    assert float(a) > 0.1
    assert float(m['even_block_ratio']) > 3

  def test_scale_free(self):
    pos = np.array([-1.0, -0.9, -0.5, -0.1, 0.1, 0.5, 0.9, 1.0])
    t = line_table(pos)
    w1, a1, _ = goal_ae.evenness_loss(LinearDec(t), z_q_from(t))
    w2, a2, _ = goal_ae.evenness_loss(LinearDec(0.5 * t), z_q_from(0.5 * t))
    np.testing.assert_allclose(float(w1), float(w2), rtol=0.1)


def build():
  dec = goal_ae.GoalVQDecoder(DETER, L, C, DIM, layers=2, units=32, name='goal_dec')
  enc = goal_ae.GoalVQEncoder(dec.codebook, L, DIM, layers=2, units=32, name='goal_enc')
  return enc, dec


def deters(n=64, seed=0):
  return jnp.asarray(np.random.default_rng(seed).normal(0, 1, (2, n // 2, DETER)), f32)


def init(enc, dec, x):
  def fn(x):
    enc(x, 2)
    dec(dec.codebook.onehot(jnp.zeros((*x.shape[:2], L), jnp.int32)), 2)
  return nj.init(fn)({}, x, seed=0)


class TestInLoss:

  def test_off_by_default_and_on_adds_terms(self):
    enc, dec = build()
    x = deters()
    params = init(enc, dec, x)
    off = nj.pure(lambda x: goal_ae.vq_goal_loss(enc, dec, x, 2, som=True))(params, x)[1]
    on = nj.pure(lambda x: goal_ae.vq_goal_loss(
        enc, dec, x, 2, som=True, even_within=2.0, even_across=3.0))(params, x)[1]
    assert 'vq/even_within' not in off[1]
    m = on[1]
    expect = 2.0 * m['vq/even_within'] + 3.0 * m['vq/even_across']
    np.testing.assert_allclose(float((on[0] - off[0]).mean()), float(expect), rtol=1e-4)

  def test_gradient_reaches_codebook_and_decoder_not_encoder(self):
    enc, dec = build()
    x = deters()
    params = init(enc, dec, x)

    def loss(p):
      def fn(x):
        z_q = dec.codebook.quantize(enc.latent(x, 2))['z_q']
        w, a, _ = goal_ae.evenness_loss(dec, z_q)
        return w + a
      return nj.pure(fn)(p, x)[1]
    g = jax.grad(loss)(params)
    norm = lambda pre: sum(float(jnp.abs(v).sum()) for k, v in g.items() if k.startswith(pre))
    assert norm('goal_dec/codebook') > 0
    assert norm('goal_dec/mlp') > 0
    assert norm('goal_enc') == 0

  def test_descent_makes_steps_more_even(self):
    enc, dec = build()
    x = deters()
    params = init(enc, dec, x)

    def loss(p):
      def fn(x):
        z_q = dec.codebook.quantize(enc.latent(x, 2))['z_q']
        w, a, m = goal_ae.evenness_loss(dec, z_q)
        return w + a, m
      return nj.pure(fn)(p, x)[1]
    first = float(loss(params)[0])
    step = jax.jit(lambda p: jax.tree.map(
        lambda v, d: v - 0.05 * d, p, jax.grad(lambda q: loss(q)[0])(p)))
    for _ in range(200):
      params = step(params)
    assert float(loss(params)[0]) < 0.5 * first
