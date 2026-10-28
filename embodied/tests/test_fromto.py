"""From-to novelty (dreamerv3/hrl/fromto.py): counts, exact expected novelty, gradient."""
import itertools

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from dreamerv3.hrl.fromto import FromToCounter, FromToCounts, bin_ids

L, C, SB, GB = 3, 8, 2, 4


def module():
  m = FromToCounts(start_bins=SB, goal_bins=GB, classes=C, blocks=L, name='ft')
  params = nj.init(lambda: m.tab.read())({}, seed=0)
  return m, params


def run(m, params, fn, *a):
  return nj.pure(fn)(params, *a, seed=0)[1]


class TestExpectedNovelty:

  def test_matches_brute_force(self):
    m, p = module()
    rng = np.random.default_rng(0)
    p['ft/tab/value'] = jnp.asarray(rng.integers(0, 20, SB ** L * GB ** L).astype(np.float32))
    start = jnp.asarray(rng.integers(0, C, (5, L)))
    probs = jax.nn.softmax(jnp.asarray(rng.normal(0, 1, (5, L, C)), jnp.float32), -1)
    e, eu = run(m, p, lambda s, q: m.expected_novelty(s, q), start, probs)
    tab = np.asarray(p['ft/tab/value'])
    for n in range(5):
      row = int((bin_ids(np.asarray(start[n]), SB, C) * SB ** np.arange(L - 1, -1, -1)).sum())
      pb = np.asarray(probs[n]).reshape(L, GB, C // GB).sum(-1)
      want = want_u = 0.0
      for cell in itertools.product(range(GB), repeat=L):
        g = int((np.array(cell) * GB ** np.arange(L - 1, -1, -1)).sum())
        b = 1 / np.sqrt(tab[row * GB ** L + g] + 1)
        want += np.prod([pb[l, cell[l]] for l in range(L)]) * b
        want_u += b / GB ** L
      assert float(e[n]) == pytest.approx(want, rel=1e-5)
      assert float(eu[n]) == pytest.approx(want_u, rel=1e-5)

  def test_gradient_raises_untried_combination(self):
    # Toy from the design notes: everything tried from this start except the
    # region where every block is in its top bin.
    m, p = module()
    tab = np.full(SB ** L * GB ** L, 99.0, np.float32)
    start = jnp.zeros((1, L), jnp.int32)                  # start row 0
    tab[(GB ** L) - 1] = 0.0                              # goal cell (3, 3, 3), row 0
    p['ft/tab/value'] = jnp.asarray(tab)
    logits = jnp.zeros((1, L, C))
    g = jax.grad(lambda lg: run(m, p, lambda s, q: m.expected_novelty(s, q)[0].sum(),
                                start, jax.nn.softmax(lg, -1)))(logits)
    g = np.asarray(g)[0]
    assert np.all(g[:, 6:].mean(-1) > 0)                  # top bin (classes 6, 7) pushed up
    assert np.all(g[:, :6].mean(-1) < 0)

  def test_equal_counts_give_no_direction(self):
    m, p = module()
    p['ft/tab/value'] = jnp.full((SB ** L * GB ** L,), 5.0)
    logits = jnp.asarray(np.random.default_rng(1).normal(0, 1, (2, L, C)), jnp.float32)
    start = jnp.zeros((2, L), jnp.int32)
    g = jax.grad(lambda lg: run(m, p, lambda s, q: m.expected_novelty(s, q)[0].sum(),
                                start, jax.nn.softmax(lg, -1)))(logits)
    assert np.abs(np.asarray(g)).max() < 1e-6


class TestCounter:

  def cnt(self, **kw):
    kw.setdefault('h', 0.0)
    return FromToCounter(start_bins=SB, goal_bins=GB, classes=C, blocks=L, **kw)

  def cell(self, s, g):
    srow = int((bin_ids(s, SB, C) * SB ** np.arange(L - 1, -1, -1)).sum())
    gcol = int((bin_ids(g, GB, C) * GB ** np.arange(L - 1, -1, -1)).sum())
    return srow * GB ** L + gcol

  def test_offsets_and_weights(self):
    c = self.cnt(k_min=1, k_max=2, half_life=1e12, pair_weight=0)   # 1/n weights
    codes = [np.full(L, v) for v in (0, 6, 4, 2)]   # distinct start/goal bins
    for i, code in enumerate(codes):
      c.add(code, i == 0, 0)
    tab = np.asarray(c.export())
    assert tab.sum() == pytest.approx(0.5 * 3 + 0.5 * 2)   # 3 pairs at k=1, 2 at k=2
    assert tab[self.cell(codes[0], codes[2])] == pytest.approx(0.5)

  def test_episode_reset_and_workers_are_separate(self):
    c = self.cnt(k_min=1, k_max=1, half_life=1e12)
    c.add(np.zeros(L), True, 0)
    c.add(np.full(L, 7), True, 1)      # other worker: no pair with worker 0
    c.add(np.full(L, 7), True, 0)      # new episode on worker 0: no pair
    assert float(np.asarray(c.export()).sum()) == 0.0
    c.add(np.full(L, 4), False, 0)
    assert float(np.asarray(c.export()).sum()) == pytest.approx(1.0)

  def test_decay_half_life(self):
    c = self.cnt(k_min=1, k_max=1, half_life=4)
    c.add(np.zeros(L), True, 0)
    c.add(np.zeros(L), False, 0)        # one pair at steps=1
    for _ in range(3):
      c.add(np.zeros(L), True, 1)       # advance time, no pairs (episode starts)
    tot = float(np.asarray(c.export()).sum())
    assert tot == pytest.approx(0.5, rel=1e-5)          # added at t=1, read at t=5: halved

  def test_smearing_preserves_mass(self):
    c = self.cnt(k_min=1, k_max=1, half_life=1e12, h=1.0)
    c.add(np.zeros(L), True, 0)
    c.add(np.full(L, 3), False, 0)
    tab = np.asarray(c.export())
    assert tab.sum() == pytest.approx(1.0, rel=1e-5)
    assert (tab > 0).sum() > 1


def test_export_under_transfer_guard():
  # Runs use jax_transfer_guard='disallow'; export must only transfer explicitly.
  c = FromToCounter(start_bins=SB, goal_bins=GB, classes=C, blocks=L, k_min=1, k_max=1)
  c.add(np.zeros(L), True, 0)
  c.add(np.full(L, 7), False, 0)
  with jax.transfer_guard('disallow'):
    out = c.export()
  assert float(jax.device_get(out).sum()) == pytest.approx(1.0, rel=1e-5)
