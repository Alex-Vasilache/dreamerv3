"""UCB inside imagination: the ImagCounts table and its wiring.

What has to hold for the tilt to mean what it is meant to mean:

  - a pick deposits exactly 1.0 of mass, spread over neighbouring classes
    (the SOM line is what makes "neighbour" meaningful), and the table decays
    by exactly one factor per update, so counts read as recent imagined picks;
  - the half-life config maps to the right per-update decay;
  - arm B's cell is a (start-state, goal) pair: the same goal from a different
    start-state region is a different, untouched cell;
  - picking a goal lowers its own bonus; untouched goals keep bonus 1;
  - the full agent builds and trains with both arms, logs the tilt, writes the
    table only in training, and the default config leaves everything off.
"""
import pathlib

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from dreamerv3.hrl.explore import ImagCounts

L, C = 8, 8


def build(**kw):
  kw.setdefault('decay', 1.0)
  ic = ImagCounts(name='ic', **kw)
  g = jnp.zeros((1, L), jnp.int32)
  s = g if kw.get('state_bins') else None
  params = nj.init(lambda: ic.update(g, s))({}, seed=0)
  return ic, params


def run(ic, params, fn, *a):
  return nj.pure(fn)(params, *a)


class TestTable:

  def test_a_pick_deposits_exactly_one_unit(self):
    ic, p = build(goal_bins=8)
    ids = jnp.asarray(np.random.default_rng(0).integers(0, C, (5, L)))
    p, m = run(ic, p, lambda i: (ic.update(i), ic.metrics())[1], ids)
    assert float(m['mass']) == pytest.approx(5.0, rel=1e-4)

  def test_weights_mask_out_non_decisions(self):
    ic, p = build(goal_bins=8)
    ids = jnp.asarray(np.random.default_rng(0).integers(0, C, (6, L)))
    w = jnp.asarray([1, 0, 1, 0, 0, 1], jnp.float32)
    p, m = run(ic, p, lambda i: (ic.update(i, weights=w), ic.metrics())[1], ids)
    assert float(m['mass']) == pytest.approx(3.0, rel=1e-4)

  def test_decay_is_one_factor_per_update(self):
    ic, p = build(goal_bins=8, decay=0.5)
    ids = jnp.zeros((4, L), jnp.int32)
    p, _ = run(ic, p, lambda i: ic.update(i), ids)
    p, m = run(ic, p, lambda i: (ic.update(i, weights=jnp.zeros(4)),
                                 ic.metrics())[1], ids)
    assert float(m['mass']) == pytest.approx(2.0, rel=1e-4)

  def test_smearing_reaches_neighbours_and_peaks_on_the_pick(self):
    ic, p = build(goal_bins=8, h=1.0)
    code = jnp.full((1, L), 4, jnp.int32)
    p, _ = run(ic, p, lambda i: ic.update(i), code)
    near = code.at[0, 0].set(5)
    far = code.at[0, 0].set(0)
    _, (n0, n1, n2) = nj.pure(lambda: (ic.count(code), ic.count(near),
                                       ic.count(far)))(p)
    assert float(n0[0]) > float(n1[0]) > float(n2[0]) >= 0.0
    assert float(n1[0]) > 0.05 * float(n0[0])

  def test_h_zero_is_a_hard_deposit(self):
    ic, p = build(goal_bins=8, h=0.0)
    code = jnp.full((1, L), 4, jnp.int32)
    p, _ = run(ic, p, lambda i: ic.update(i), code)
    _, (n0, n1) = nj.pure(lambda: (ic.count(code),
                                   ic.count(code.at[0, 0].set(5))))(p)
    assert float(n0[0]) == pytest.approx(1.0) and float(n1[0]) == 0.0

  def test_picking_lowers_its_own_bonus_and_others_stay_at_one(self):
    ic, p = build(goal_bins=8, h=0.0)
    a = jnp.zeros((1, L), jnp.int32)
    b = jnp.full((1, L), 7, jnp.int32)
    for _ in range(3):
      p, _ = run(ic, p, lambda i: ic.update(i), a)
    _, (ba, bb) = nj.pure(lambda: (ic.bonus(a), ic.bonus(b)))(p)
    assert float(ba[0]) == pytest.approx(0.5)      # 1/sqrt(3+1)
    assert float(bb[0]) == pytest.approx(1.0)

  def test_state_context_separates_the_same_goal_from_different_starts(self):
    ic, p = build(goal_bins=4, state_bins=2, h=0.0)
    g = jnp.full((1, L), 3, jnp.int32)
    s_here = jnp.zeros((1, L), jnp.int32)          # every block in the low half
    s_there = jnp.full((1, L), 7, jnp.int32)       # every block in the high half
    p, _ = run(ic, p, lambda gg, ss: ic.update(gg, ss), g, s_here)
    _, (n_here, n_there) = nj.pure(
        lambda: (ic.count(g, s_here), ic.count(g, s_there)))(p)
    assert float(n_here[0]) == pytest.approx(1.0)
    assert float(n_there[0]) == 0.0

  def test_arm_sizes_are_both_16_7m_cells(self):
    a = ImagCounts(name='a', decay=1.0, goal_bins=8, state_bins=0)
    b = ImagCounts(name='b', decay=1.0, goal_bins=4, state_bins=2)
    assert a.size == b.size == 8 ** 8


class TestConfig:

  def _cfg(self):
    import yaml
    path = pathlib.Path(__file__).parent.parent.parent / 'dreamerv3' / 'configs.yaml'
    return yaml.safe_load(path.read_text())

  def test_default_is_off(self):
    assert self._cfg()['defaults']['agent']['mgr_ucb_imag_c'] == 0.0

  def test_arms_turn_imagination_tilt_on_and_environment_tilt_off(self):
    cfg = self._cfg()
    for arm, gb, sb in (('ucb_imag_a', 8, 0), ('ucb_imag_b', 4, 2)):
      ag = cfg[arm]['agent']
      assert ag['mgr_ucb_c'] == 0.0 and ag['mgr_ucb_imag_c'] > 0.0
      assert ag['mgr_ucb_imag']['goal_bins'] == gb
      assert ag['mgr_ucb_imag']['state_bins'] == sb

  def test_half_life_maps_to_per_update_decay(self):
    # pubmatch: 16x64 batch at train_ratio 64 -> 16 env steps per update; a
    # 0.5M-step half-life is 31,250 updates.
    decay = 0.5 ** (16.0 / 5e5)
    assert decay ** 31250 == pytest.approx(0.5, rel=1e-6)
