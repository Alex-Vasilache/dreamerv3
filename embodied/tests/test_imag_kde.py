"""UCB in imagination at full resolution: the ImagKDE buffer counts.

What has to hold:
  - the count equals the explicit sum over buffered picks of the separable
    Gaussian (per-axis widths h_state / h_goal), scaled from the buffered weight
    to the true decayed number of picks;
  - a single pick counts 1 at its own cell and exp(-1/(2h^2)) one class away on
    one axis;
  - start-state context separates the same goal from different starts;
  - inactive steps skip the sum (zeros), decay is one factor per update, the
    ring buffer wraps, masked steps are never inserted, mass counts every pick.
"""
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from dreamerv3.hrl.explore import ImagKDE

L, C = 8, 8


def build(**kw):
  kw.setdefault('decay', 1.0)
  kw.setdefault('size', 64)
  kw.setdefault('chunk', 16)
  kw.setdefault('insert', 4)
  kde = ImagKDE(name='kde', **kw)
  g = jnp.zeros((1, L), jnp.int32)
  s = g if kw.get('use_state', True) else None
  params = nj.init(lambda: (kde.count(g, s), kde.update(g, s, jnp.zeros(1))))({}, seed=0)
  return kde, params


def run(params, fn, *a, seed=0):
  return nj.pure(fn)(params, *a, seed=seed)


def ids(n, seed):
  return jnp.asarray(np.random.default_rng(seed).integers(0, C, (n, L)), jnp.int32)


class TestCount:

  def test_matches_explicit_kernel_sum(self):
    kde, p = build(size=64, insert=64, h_goal=1.3, h_state=2.0)
    g, s = ids(64, 0), ids(64, 1)
    p, _ = run(p, lambda: kde.update(g, s))
    xg, xs = ids(10, 2), ids(10, 3)
    _, n = run(p, lambda: kde.count(xg, xs))
    buf = np.asarray(p['kde/buf/value'])
    d2 = (((np.asarray(xs)[:, None] - buf[None, :, :L]) ** 2).sum(-1) / 2.0 ** 2
          + ((np.asarray(xg)[:, None] - buf[None, :, L:]) ** 2).sum(-1) / 1.3 ** 2)
    want = np.exp(-d2 / 2).sum(-1)          # every buffered weight is 1, mass 64
    np.testing.assert_allclose(np.asarray(n), want, rtol=1e-4)

  def test_one_pick_counts_one_at_itself_and_gaussian_next_door(self):
    kde, p = build(insert=1, h_goal=1.0, use_state=False)
    g = jnp.full((1, L), 3, jnp.int32)
    p, _ = run(p, lambda: kde.update(g))
    near = g.at[0, 5].set(4)
    _, n = run(p, lambda: kde.count(jnp.concatenate([g, near]), None))
    np.testing.assert_allclose(np.asarray(n), [1.0, np.exp(-0.5)], rtol=1e-5)

  def test_start_state_separates_same_goal(self):
    kde, p = build(insert=1, h_state=0.5)
    g, s0 = jnp.full((1, L), 3, jnp.int32), jnp.zeros((1, L), jnp.int32)
    p, _ = run(p, lambda: kde.update(g, s0))
    _, n = run(p, lambda: kde.count(jnp.concatenate([g, g]),
                                   jnp.concatenate([s0, s0 + 7])))
    assert float(n[0]) == pytest.approx(1.0, rel=1e-5)
    assert float(n[1]) < 1e-30

  def test_inactive_skips_and_returns_zero(self):
    kde, p = build(insert=4)
    p, _ = run(p, lambda: kde.update(ids(4, 0), ids(4, 1)))
    _, n = run(p, lambda: kde.count(ids(4, 0), ids(4, 1), active=jnp.array(False)))
    assert np.all(np.asarray(n) == 0)
    row = np.asarray(p['kde/buf/value'])[:1].astype(np.int32)   # a buffered pick
    _, b = run(p, lambda: kde.bonus(jnp.asarray(row[:, L:]), jnp.asarray(row[:, :L]),
                                    active=jnp.array(True)))
    assert float(b[0]) < 1.0

  def test_scaled_to_true_pick_mass(self):
    # 64 picks of the same cell, only 4 buffered: count must read ~64, not 4.
    kde, p = build(insert=4, use_state=False)
    g = jnp.full((64, L), 2, jnp.int32)
    p, _ = run(p, lambda: kde.update(g))
    _, n = run(p, lambda: kde.count(g[:1], None))
    assert float(n[0]) == pytest.approx(64.0, rel=1e-4)


class TestUpdate:

  def test_decay_is_one_factor_per_update(self):
    kde, p = build(decay=0.5, insert=2, use_state=False)
    p, _ = run(p, lambda: kde.update(ids(2, 0)))
    p, m = run(p, lambda: (kde.update(ids(2, 1), weights=jnp.zeros(2)), kde.metrics())[1])
    assert float(m['mass']) == pytest.approx(1.0)
    assert float(m['buffer_weight']) == pytest.approx(1.0)

  def test_masked_steps_never_inserted_and_mass_counts_all(self):
    kde, p = build(insert=8, use_state=False)
    g = jnp.concatenate([jnp.full((3, L), 5, jnp.int32), jnp.zeros((5, L), jnp.int32)])
    w = jnp.asarray([1, 1, 1, 0, 0, 0, 0, 0], jnp.float32)
    p, m = run(p, lambda: (kde.update(g, weights=w), kde.metrics())[1])
    assert float(m['mass']) == pytest.approx(3.0)
    buf = np.asarray(p['kde/buf/value'])[:8]
    assert np.all(buf == 5)

  def test_ring_wraps(self):
    kde, p = build(size=16, chunk=16, insert=6, use_state=False)
    for i in range(3):
      p, _ = run(p, lambda: kde.update(jnp.full((6, L), i + 1, jnp.int32)), seed=i)
    assert int(p['kde/ptr/value']) == 18 % 16
    buf = np.asarray(p['kde/buf/value'])[:, 0]
    assert list(buf[:2]) == [3, 3] and list(buf[12:16]) == [3, 3, 3, 3]


class TestConfig:

  def test_arm_block(self):
    import pathlib
    import yaml
    path = pathlib.Path(__file__).resolve().parents[2] / 'dreamerv3' / 'configs.yaml'
    cfg = yaml.safe_load(path.read_text())
    ag = cfg['ucb_imag_kde']['agent']
    assert ag['mgr_ucb_c'] == 0.0 and ag['mgr_ucb_imag_c'] > 0
    assert ag['mgr_ucb_imag']['impl'] == 'kde' and ag['mgr_ucb_imag']['candidates'] == 32
    assert cfg['defaults']['agent']['mgr_ucb_imag']['impl'] == 'table'


class TestPositions:

  def test_measured_positions_drive_the_kernel(self):
    kde, p = build(insert=1, h_goal=1.0, use_state=False, pos_rate=1.0)
    pos = np.tile(np.array([0, .2, 1, 2, 3, 4, 4.8, 5.0], np.float32), (L, 1))
    p, _ = run(p, lambda: kde.update_positions(jnp.asarray(pos)))
    g = jnp.zeros((1, L), jnp.int32)
    p, _ = run(p, lambda: kde.update(g))
    near_end = g.at[0, 2].set(1)        # 0 -> 1: measured 0.2 apart
    near_mid = g.at[0, 2].set(2)        # 0 -> 2: measured 1.0 apart
    _, n = run(p, lambda: kde.count(jnp.concatenate([near_end, near_mid]), None))
    np.testing.assert_allclose(np.asarray(n), [np.exp(-0.02), np.exp(-0.5)], rtol=1e-4)

  def test_pos_rate_zero_keeps_class_index(self):
    kde, p = build(insert=1, use_state=False, pos_rate=0.0)
    p, _ = run(p, lambda: kde.update_positions(jnp.zeros((L, C))))
    np.testing.assert_array_equal(np.asarray(p['kde/pos/value'])[0], np.arange(C))


class TestClassPositions:

  def test_cumulative_steps_mean_one_and_bunched_ends(self):
    import embodied.jax.nets as nets
    from dreamerv3.hrl import goal_ae
    old = nets.COMPUTE_DTYPE
    nets.COMPUTE_DTYPE = jnp.float32
    try:
      class Codebook:
        def table(self):
          x = np.array([-1.0, -0.95, -0.6, -0.2, 0.2, 0.6, 0.95, 1.0], np.float32)
          t = np.zeros((L, C, 2), np.float32)
          t[:, :, 0] = x
          return jnp.asarray(t)

      class Pred:
        def __init__(self, x):
          self.x = x
        def pred(self):
          return self.x

      class Dec:
        codebook = Codebook()
        def from_latent(self, z, bdims):
          return Pred(5.0 + z.reshape(z.shape[:bdims] + (-1,)))

      _, pos = nj.pure(lambda: goal_ae.class_positions(Dec()))({}, seed=0)
      pos = np.asarray(pos)
      step = np.diff(pos, axis=-1)
      assert np.all(step > 0)
      assert step.mean() == pytest.approx(1.0, rel=1e-4)
      assert step[:, 0].mean() < 0.3 * step[:, 3].mean()
    finally:
      nets.COMPUTE_DTYPE = old
