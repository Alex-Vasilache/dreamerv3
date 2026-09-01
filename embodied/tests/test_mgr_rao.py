"""Rao's quadratic entropy regularizer on the manager skill head.

Rao supplies the one thing the entropy regularizer structurally cannot see: the
ORDER of the classes. Entropy is invariant to permuting them, so a manager
spread over three adjacent codes and one split between the two ends of the SOM
line have identical entropy and completely different behaviour. These tests pin
the properties that make the term mean that and not something else:

  - the reference values on a C=8 line, so a normalization slip is caught;
  - permutation SENSITIVITY (Rao changes when classes are relabelled) against
    entropy's permutation INVARIANCE -- the whole reason the term exists;
  - the factorization identity that lets whole-code Rao be computed per block
    instead of over 8^8 = 16.7M codes;
  - that the term is exactly inert when ``manager_rao`` is off.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.hrl.heads import (
    class_distance_matrix,
    head_probs,
    head_rao_perdim_time,
    rao_quadratic_entropy,
)

C = 8


def norm_rao(p):
  d = class_distance_matrix(C)
  return float(rao_quadratic_entropy(jnp.asarray(p, jnp.float32), d))


def onehot_head(logits):
  import embodied.jax.outs as outs
  return outs.OneHot(jnp.asarray(logits, jnp.float32))


class TestReferenceValues:
  """Hand-computable points on the C=8 line, normalized so 1.0 = ends split."""

  def test_all_mass_on_one_class_is_zero(self):
    p = np.zeros(C); p[3] = 1.0
    assert norm_rao(p) == pytest.approx(0.0, abs=1e-6)

  def test_half_at_each_end_is_one(self):
    p = np.zeros(C); p[0] = p[-1] = 0.5
    assert norm_rao(p) == pytest.approx(1.0, rel=1e-5)

  def test_uniform_is_well_below_the_maximum(self):
    # 2 * Var(uniform on {0..7}/7) / 0.5 = 3/7 -- a uniform goal code is NOT
    # the most spread-out one in the distance sense; the ends-split is.
    assert norm_rao(np.full(C, 1 / C)) == pytest.approx(3 / 7, rel=1e-5)

  def test_three_adjacent_classes_score_far_below_uniform(self):
    # This is roughly where the trained e534-e573 manager actually sits (0.125
    # measured), at an entropy indistinguishable from the uniform-ish target.
    p = np.zeros(C); p[3:6] = 1 / 3
    assert norm_rao(p) == pytest.approx(0.0544, abs=1e-3)

  def test_the_50_percent_setpoint_is_reachable_at_50_percent_entropy(self):
    # 0.25 / 0.50 / 0.25 on classes 0 / 3 / 7: the shape manager_rao_target=0.5
    # actually asks for, at exactly manager_actent_target=0.5 entropy.
    p = np.zeros(C); p[0] = 0.25; p[3] = 0.5; p[7] = 0.25
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(C))
    assert ent == pytest.approx(0.5, abs=1e-3)
    assert norm_rao(p) == pytest.approx(0.5, abs=0.01)


class TestOrderSensitivity:
  """The property that distinguishes Rao from entropy."""

  def test_entropy_is_permutation_invariant_but_rao_is_not(self):
    adjacent = np.zeros(C); adjacent[3:6] = 1 / 3
    spread = np.zeros(C); spread[[0, 3, 7]] = 1 / 3
    ent = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())
    assert ent(adjacent) == pytest.approx(ent(spread))
    assert norm_rao(spread) > 5 * norm_rao(adjacent)

  def test_moving_mass_further_apart_raises_rao(self):
    vals = []
    for gap in (1, 2, 3, 7):
      p = np.zeros(C); p[0] = p[gap] = 0.5
      vals.append(norm_rao(p))
    assert vals == sorted(vals)
    assert all(b > a for a, b in zip(vals, vals[1:]))


class TestFactorization:
  """Whole-code Rao == sum of per-block Raos, so 8^8 pairs are never formed."""

  def test_whole_code_rao_equals_the_sum_over_blocks(self):
    rng = np.random.default_rng(0)
    L, small_c = 3, 4          # 4^3 = 64 codes: small enough to enumerate
    logits = rng.normal(size=(L, small_c))
    p = np.exp(logits); p /= p.sum(-1, keepdims=True)
    x = np.arange(small_c) / (small_c - 1)

    # Brute force over every whole code pair, with the additive distance.
    codes = np.array(np.meshgrid(*[np.arange(small_c)] * L, indexing='ij'))
    codes = codes.reshape(L, -1).T                       # (small_c**L, L)
    pc = np.prod([p[l, codes[:, l]] for l in range(L)], axis=0)
    d = ((x[codes][:, None, :] - x[codes][None, :, :]) ** 2).sum(-1)
    brute = float(pc @ d @ pc)

    per_block = sum(
        float(np.einsum('a,ab,b->', p[l], (x[:, None] - x[None, :]) ** 2, p[l]))
        for l in range(L))
    assert brute == pytest.approx(per_block, rel=1e-10)


class TestHeadPlumbing:

  def test_rao_perdim_time_drops_the_trailing_step_and_keeps_blocks(self):
    B, T, L = 2, 5, 4
    head = onehot_head(np.zeros((B, T, L, C)))
    out = head_rao_perdim_time(head)
    assert out.shape == (B, T - 1, L)
    # Zero logits = uniform, so every entry is the uniform reference.
    np.testing.assert_allclose(np.asarray(out), 3 / 7, rtol=1e-5)

  def test_returns_none_for_a_head_without_class_logits(self):
    class NoLogits:
      dist = None
    assert head_probs(NoLogits()) is None
    assert head_rao_perdim_time(NoLogits()) is None

  def test_gradient_flows_into_the_logits_and_points_outward(self):
    # The term must be able to move the policy, and must push mass AWAY from
    # the centre (that is what raising Rao means at fixed entropy).
    d = class_distance_matrix(C)

    def rao_of(logits):
      return rao_quadratic_entropy(jax.nn.softmax(logits, -1), d)

    logits = jnp.asarray(np.concatenate([
        np.full(3, -2.0), np.full(2, 2.0), np.full(3, -2.0)]), jnp.float32)
    g = np.asarray(jax.grad(rao_of)(logits))
    assert np.abs(g).sum() > 0
    assert g[0] > 0 and g[-1] > 0        # raise the ends
    assert g[C // 2] < 0                 # lower the centre


class FixedAdapt:
  """AutoAdapt's call contract at a frozen scale (``impl='fixed'`` semantics).

  Same sign convention as the real adapter with ``inverse=True``: the loss is
  ``-scale * reg``, so minimizing it pushes ``reg`` up toward the target.
  """

  def __init__(self, scale=1.0):
    self.scale = float(scale)

  def __call__(self, reg, update=True, target=None, weights=None):
    return -self.scale * reg, {'mean': reg.mean(), 'scale_mean': self.scale}


def _mgr_loss_inputs(logits, **over):
  """`imag_loss_mgr` inputs with a REAL OneHot skill head (so it has logits)."""
  import embodied.jax.outs as outs
  from embodied.tests.test_losses_gradflow import FakeNorm, FakeValue
  Bs, Ts = logits.shape[0], logits.shape[1]
  theta = jnp.zeros((Bs, Ts), jnp.float32)
  events = jax.nn.one_hot(jnp.argmax(logits, -1), logits.shape[-1])
  ins = dict(
      skills={'skill': events},
      mgr_extr_rew=jnp.linspace(
          0, 1, Bs * Ts).reshape(Bs, Ts).astype(jnp.float32),
      mgr_expl_rew=jnp.full((Bs, Ts), 0.1, jnp.float32),
      # con=1 with the default contdisc makes the discounted continuation
      # weight exactly 1 at every step, so the Rao term's contribution to
      # mgr_policy is unweighted and can be compared against a closed form.
      con=jnp.ones((Bs, Ts), jnp.float32),
      manager_policy={'skill': outs.OneHot(logits)},
      mgr_extr_value=FakeValue(theta),
      mgr_extr_slowvalue=FakeValue(theta),
      mgr_expl_value=FakeValue(theta),
      mgr_expl_slowvalue=FakeValue(theta),
      mgr_extr_retnorm=FakeNorm(), mgr_expl_retnorm=FakeNorm(),
      mgr_extr_valnorm=FakeNorm(), mgr_expl_valnorm=FakeNorm(),
      mgr_advnorm=FakeNorm(), update=True)
  ins.update(over)
  return ins


class TestLossWiring:
  """End-to-end through ``imag_loss_mgr``, the function training actually calls."""

  def _logits(self, seed=0, Bs=2, Ts=9, L=3):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(Bs, Ts, L, C)), jnp.float32)

  def test_off_by_default_is_bit_identical(self):
    from dreamerv3.hrl.losses import imag_loss_mgr
    logits = self._logits()
    a, _, _ = imag_loss_mgr(**_mgr_loss_inputs(logits))
    b, _, _ = imag_loss_mgr(**_mgr_loss_inputs(logits, mgr_rao_adapter=None))
    np.testing.assert_array_equal(
        np.asarray(a['mgr_policy']), np.asarray(b['mgr_policy']))

  def test_enabling_it_subtracts_exactly_the_scaled_per_block_rao(self):
    from dreamerv3.hrl.losses import imag_loss_mgr
    logits = self._logits()
    scale = 0.7
    off, _, _ = imag_loss_mgr(**_mgr_loss_inputs(logits))
    on, _, mets = imag_loss_mgr(**_mgr_loss_inputs(
        logits, mgr_rao_adapter=FixedAdapt(scale), mgr_rao_perdim=True))
    rao = np.asarray(head_rao_perdim_time(
        __import__('embodied.jax.outs', fromlist=['x']).OneHot(logits)))
    delta = np.asarray(on['mgr_policy']) - np.asarray(off['mgr_policy'])
    np.testing.assert_allclose(delta, -scale * rao.sum(-1), rtol=1e-5, atol=1e-6)
    assert float(mets['mgr_rao_norm_mean']) == pytest.approx(
        rao.mean(), rel=1e-5)

  def test_the_term_pushes_the_policy_toward_the_ends_of_the_line(self):
    """The whole point: minimizing mgr_policy must raise Rao.

    Differences the on/off gradients, because mgr_policy also carries the
    REINFORCE term, whose gradient w.r.t. the same logits is unrelated and
    larger. The difference is the Rao term's contribution alone.
    """
    from dreamerv3.hrl.losses import imag_loss_mgr

    def loss_of(adapter):
      def fn(logits):
        losses, _, _ = imag_loss_mgr(**_mgr_loss_inputs(
            logits, mgr_rao_adapter=adapter))
        return losses['mgr_policy'].sum()
      return fn

    # A manager camped on the middle three classes -- roughly where the trained
    # SOM-line+LiP arm actually sits (measured normalized Rao 0.125).
    base = np.full((1, 3, 1, C), -3.0, np.float32)
    base[..., 3:6] = 1.0
    base = jnp.asarray(base)
    g_on = np.asarray(jax.grad(loss_of(FixedAdapt(1.0)))(base))
    g_off = np.asarray(jax.grad(loss_of(None))(base))
    g = (g_on - g_off)[0, 0, 0]
    # Descending this gradient raises the end logits and lowers the centre.
    assert g[0] < 0 and g[-1] < 0
    assert g[C // 2] > 0

  def test_scalar_mode_averages_blocks_instead_of_summing(self):
    from dreamerv3.hrl.losses import imag_loss_mgr
    logits = self._logits()
    off, _, _ = imag_loss_mgr(**_mgr_loss_inputs(logits))
    on, _, _ = imag_loss_mgr(**_mgr_loss_inputs(
        logits, mgr_rao_adapter=FixedAdapt(1.0), mgr_rao_perdim=False))
    rao = np.asarray(head_rao_perdim_time(
        __import__('embodied.jax.outs', fromlist=['x']).OneHot(logits)))
    delta = np.asarray(on['mgr_policy']) - np.asarray(off['mgr_policy'])
    np.testing.assert_allclose(delta, -rao.mean(-1), rtol=1e-5, atol=1e-6)
