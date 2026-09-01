"""Count-based novelty over the joint goal code, and its separate critic.

The properties that make this mean what it is supposed to mean:

  - the memory forgets at exactly the rate replay forgets, and a count reads in
    ENV-STEP units, so `n` is "env steps spent near this code in the replay
    window" and not an arbitrary counter;
  - the cell index is the JOINT tuple of blocks. Per-block counting was measured
    to miss real structure (shuffling blocks independently, which leaves every
    marginal identical, spreads the codes over 3-4x more regions), so a test
    pins that two codes agreeing on 7 of 8 blocks are still different cells;
  - novelty falls where the agent has been and stays high where it has not;
  - the third critic is exactly inert when the arm does not enable it.
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from dreamerv3.hrl.explore import CodeCounts

L, C = 8, 8


def build(**kw):
  kw.setdefault('decay', 1.0)
  kw.setdefault('weight', 1.0)
  cc = CodeCounts(name='cc', **kw)
  ids = jnp.zeros((1, 1, L), jnp.int32)
  params = nj.init(lambda i: cc.update(i))({}, ids, seed=0)
  return cc, params


def run(cc, params, fn, *a):
  return nj.pure(fn)(params, *a)


def val(x):
  """First element as a Python float; lookups keep their batch axis."""
  return float(np.asarray(x).reshape(-1)[0])


class TestEnvStepAccounting:
  """A count must read as env steps, which is what ties it to the replay window."""

  def test_one_env_step_of_states_adds_exactly_one_unit(self):
    # train_ratio=64 -> weight 1/64, and a batch of 64 states is 1 env step's
    # worth of insertions, so the table must gain exactly 1.0 of mass.
    cc, params = build(weight=1 / 64.)
    ids = jnp.asarray(np.random.default_rng(0).integers(0, C, (64, L)))
    params, mets = run(cc, params, lambda i: (cc.update(i), cc.metrics())[1], ids)
    assert float(mets['total_mass']) == pytest.approx(1.0, rel=1e-5)

  def test_decay_is_exact_and_independent_of_insertions(self):
    cc, params = build(decay=0.5, weight=1.0)
    ids = jnp.asarray(np.random.default_rng(0).integers(0, C, (10, L)))
    params, m1 = run(cc, params, lambda i: (cc.update(i), cc.metrics())[1], ids)
    assert float(m1['total_mass']) == pytest.approx(10.0, rel=1e-5)
    cc2 = CodeCounts(name='cc', decay=0.5, weight=0.0)
    _, m2 = nj.pure(lambda i: (cc2.update(i), cc2.metrics())[1])(params, ids)
    assert float(m2['total_mass']) == pytest.approx(5.0, rel=1e-5)

  def test_steady_state_mass_equals_the_replay_window(self):
    """decay=exp(-e/R), weight=1/ratio  ->  the table settles at R."""
    replay, ratio, bt = 4000.0, 64.0, 1024.0
    env_per_train = bt / ratio                       # 16 env steps per train step
    decay = float(np.exp(-env_per_train / replay))
    cc, params = build(decay=decay, weight=1 / ratio)
    rng = np.random.default_rng(0)
    for _ in range(3000):                            # well past the time constant
      ids = jnp.asarray(rng.integers(0, C, (int(bt), L)))
      params, mets = run(cc, params, lambda i: (cc.update(i), cc.metrics())[1], ids)
    assert float(mets['total_mass']) == pytest.approx(replay, rel=0.02)


class TestJointNotPerBlock:
  """The cell index must be the joint tuple; this is the whole design point."""

  def test_codes_differing_in_one_block_are_different_cells(self):
    cc, params = build()
    a = np.zeros((1, L), np.int32)
    b = a.copy(); b[0, 3] = C - 1          # agree on 7 of 8 blocks
    params, _ = run(cc, params, lambda i: cc.update(i), jnp.asarray(a))
    _, (na, nb) = run(cc, params, lambda x: (
        cc.lookup(x[0:1])[0], cc.lookup(x[1:2])[0]), jnp.asarray(np.concatenate([a, b])))
    assert val(na) > 0.0, 'the visited code must be counted'
    assert val(nb) == 0.0, 'one differing block must land in another cell'

  def test_a_permuted_code_is_a_different_cell(self):
    cc, params = build()
    a = np.arange(L, dtype=np.int32)[None] % C
    b = a[:, ::-1].copy()
    params, _ = run(cc, params, lambda i: cc.update(i), jnp.asarray(a))
    _, nb = run(cc, params, lambda i: cc.lookup(i)[0], jnp.asarray(b))
    assert val(nb) == 0.0

  def test_the_coarse_grid_merges_what_the_fine_grid_separates(self):
    # Bin arithmetic at C=8: the fine grid (4 bins) pairs classes (0,1) (2,3)
    # (4,5) (6,7); the coarse grid (2 bins) splits 0-3 from 4-7. So class 0 and
    # class 2 are DIFFERENT fine cells but the SAME coarse cell, which is
    # exactly the pair the frontier rule is built on. Class 1 would not work --
    # it shares a fine bin with class 0.
    cc, params = build()
    a = np.zeros((1, L), np.int32)
    b = a.copy(); b[0, 0] = 2
    params, _ = run(cc, params, lambda i: cc.update(i), jnp.asarray(a))
    _, (nf, nc) = run(cc, params, lambda i: cc.lookup(i), jnp.asarray(b))
    assert val(nf) == 0.0, 'different fine cell'
    assert val(nc) > 0.0, 'same coarse cell'


class TestNovelty:

  def test_visited_codes_become_less_novel_and_far_ones_do_not(self):
    cc, params = build(weight=1.0)
    rng = np.random.default_rng(0)
    seen = jnp.asarray(rng.integers(0, 2, (200, L)))       # a corner of the space
    far = jnp.asarray(rng.integers(C - 2, C, (16, L)))     # the opposite corner
    _, before = run(cc, params, lambda i: cc.novelty(i).mean(), seen)
    params, _ = run(cc, params, lambda i: cc.update(i), seen)
    _, after = run(cc, params, lambda i: cc.novelty(i).mean(), seen)
    _, far_nov = run(cc, params, lambda i: cc.novelty(i).mean(), far)
    assert float(after) < float(before)
    assert float(far_nov) > float(after) * 2

  def test_novelty_is_bounded_in_zero_one(self):
    cc, params = build(weight=1.0)
    ids = jnp.asarray(np.random.default_rng(0).integers(0, C, (500, L)))
    params, _ = run(cc, params, lambda i: cc.update(i), ids)
    _, nov = run(cc, params, lambda i: cc.novelty(i), ids)
    nov = np.asarray(nov)
    assert nov.min() > 0.0 and nov.max() <= 1.0

  def test_frontier_gate_damps_regions_with_no_coarse_history(self):
    seen = jnp.asarray(np.zeros((64, L), np.int32))
    plain, p1 = build(weight=1.0, frontier=False)
    gated, p2 = build(weight=1.0, frontier=True)
    # near: different fine cell, same coarse cell (class 2, see the bin
    # arithmetic above). far: a different coarse cell in every block.
    near = np.zeros((1, L), np.int32); near[0, 0] = 2
    far = np.full((1, L), C - 1, np.int32)
    p1, _ = run(plain, p1, lambda i: plain.update(i), seen)
    p2, _ = run(gated, p2, lambda i: gated.update(i), seen)
    _, (pn, pf) = run(plain, p1, lambda x: (
        plain.novelty(x[0:1]), plain.novelty(x[1:2])),
        jnp.asarray(np.concatenate([near, far])))
    _, (gn, gf) = run(gated, p2, lambda x: (
        gated.novelty(x[0:1]), gated.novelty(x[1:2])),
        jnp.asarray(np.concatenate([near, far])))
    # Ungated, both look equally novel: neither fine cell has been visited.
    assert val(pn) == pytest.approx(val(pf), rel=1e-5)
    # Gated, the reachable one keeps its bonus and the unreachable one loses it.
    assert val(gn) > val(gf)
    assert val(gf) < 0.05


class TestMetrics:

  def test_metrics_are_finite_scalars(self):
    cc, params = build(weight=1.0)
    ids = jnp.asarray(np.random.default_rng(0).integers(0, C, (300, L)))
    params, mets = run(cc, params, lambda i: (cc.update(i), cc.metrics())[1], ids)
    for k, v in mets.items():
      v = np.asarray(v)
      assert v.shape == (), (k, v.shape)
      assert np.isfinite(v), (k, v)

  def test_occupancy_threshold_is_not_zero(self):
    """A >0 test would count every cell ever touched and creep up forever."""
    cc, _ = build()
    assert cc.occ_thresh > 0.0

  def test_effective_cells_counts_distinct_cells_when_mass_is_even(self):
    # Participation ratio over the SELECTION table now that fine_eff_cells is no
    # longer emitted: even mass over n cells must read as n.
    cc, params = build(weight=1.0, select_h=0.0)
    ids = jnp.asarray(np.stack([np.full(L, k, np.int32) for k in (0, 2, 4, 6)]))
    params, _ = run(cc, params, lambda i: cc.update_selected(i, 1.0), ids)
    _, mets = run(cc, params, lambda i: cc.metrics(), ids)
    assert float(mets['select_eff_cells']) == pytest.approx(4.0, rel=1e-4)

  def test_coverage_images_have_the_documented_shapes(self):
    cc, params = build(weight=1.0)
    ids = jnp.asarray(np.random.default_rng(0).integers(0, C, (50, L)))
    params, _ = run(cc, params, lambda i: cc.update(i), ids)
    _, imgs = run(cc, params, lambda i: cc.coverage_images(), ids)
    assert np.asarray(imgs['code_coverage_fine']).shape == (256, 256, 1)
    assert np.asarray(imgs['code_coverage_coarse']).shape == (16, 16, 1)
    for v in imgs.values():
      assert np.asarray(v).dtype == np.uint8


class TestDerivedConstants:
  """The decay/weight formulas agent.py uses, checked against the real config."""

  def test_current_big_config_gives_the_documented_numbers(self):
    batch_size, batch_length, train_ratio, replay = 16, 64, 64.0, 1e6
    env_per_train = batch_size * batch_length / train_ratio
    assert env_per_train == 16.0
    assert float(np.exp(-env_per_train / replay)) == pytest.approx(
        0.999984, abs=1e-6)
    assert 1.0 / train_ratio == pytest.approx(1 / 64.)

  def test_a_bigger_replay_buffer_forgets_more_slowly(self):
    env_per_train = 16.0
    slow = float(np.exp(-env_per_train / 5e6))
    fast = float(np.exp(-env_per_train / 1e6))
    assert slow > fast, 'decay must follow replay.size, not be hardcoded'


class TestNovelWeightKnob:
  """`mgr_novel_weight` decouples the count bonus from the recon bonus.

  Motivated by pinpad_five at 2000-step episodes: at equal weights the count
  bonus took ~50% of the manager's exploration signal before any task reward
  existed, and all 4 Director seeds (recon only) found reward by 224k while the
  equal-weight arm managed 1 of 2 by 800k.
  """

  def test_negative_sentinel_reproduces_the_old_equal_weight_behaviour(self):
    from dreamerv3.hrl.losses import resolve_novel_weight
    assert resolve_novel_weight(0.1, -1.0) == 0.1
    assert resolve_novel_weight(0.25, -0.001) == 0.25

  def test_none_also_means_equal_weight(self):
    from dreamerv3.hrl.losses import resolve_novel_weight
    assert resolve_novel_weight(0.1, None) == 0.1

  def test_an_explicit_weight_is_used_verbatim(self):
    from dreamerv3.hrl.losses import resolve_novel_weight
    assert resolve_novel_weight(0.1, 0.03) == 0.03

  def test_zero_is_explicit_and_disables_the_channel(self):
    # 0.0 must NOT fall through to mgr_expl_weight -- it is a real request to
    # keep the critic but stop it from moving the manager, which is the control
    # this arm needs.
    from dreamerv3.hrl.losses import resolve_novel_weight
    assert resolve_novel_weight(0.1, 0.0) == 0.0

  def test_the_shipped_default_is_the_sentinel(self):
    import pathlib, ruamel.yaml as yaml
    path = pathlib.Path(__file__).parent.parent.parent / 'dreamerv3' / 'configs.yaml'
    cfg = yaml.YAML(typ='safe').load(path.read_text())
    assert cfg['defaults']['agent']['mgr_novel_weight'] < 0.0


class TestUCBSelection:
  """Selection resamples the policy's own candidates by novelty.

  The rule this replaced scored candidates by ``log pi + c * bonus`` and took
  the argmax. Because the candidates are already policy draws, that counts the
  policy twice and acts as a best-of-K likelihood filter: measured on real
  manager distributions it moved the expected log-probability of the chosen goal
  from -9.46 to -6.70, discarding roughly a quarter of the entropy the 50%
  target maintains. Sharpening is the opposite of exploring.
  """

  def _pick(self, bonuses, c, seed=0):
    from dreamerv3.hrl.explore import ucb_pick
    import jax
    K = len(bonuses)
    hot = jax.nn.one_hot(jnp.arange(K)[:, None].repeat(L, 1), C, dtype=jnp.float32)
    _, idx = ucb_pick(jax.random.PRNGKey(seed), hot,
                      jnp.asarray(bonuses, jnp.float32), c)
    return int(np.asarray(idx))

  def test_c_zero_is_a_uniform_draw_over_the_candidates(self):
    # This is the property the old rule failed: with no novelty signal the
    # choice must be an unbiased policy sample, not the most likely candidate.
    picks = [self._pick([0.1, 0.9, 0.5, 0.2], 0.0, seed=s) for s in range(400)]
    counts = np.bincount(picks, minlength=4)
    assert counts.min() > 40, f'not uniform: {counts}'

  def test_a_large_c_concentrates_on_the_rarest_candidate(self):
    picks = [self._pick([0.1, 0.9, 0.5, 0.2], 50.0, seed=s) for s in range(200)]
    assert np.bincount(picks, minlength=4).argmax() == 1

  def test_it_never_leaves_the_candidate_set(self):
    for c in (0.0, 1.0, 10.0, 1e3):
      for s in range(20):
        assert self._pick([0.3, 0.7, 0.5], c, seed=s) in (0, 1, 2)

  def test_moderate_c_still_sometimes_takes_a_common_candidate(self):
    # A tilt, not a filter: the policy's other samples must keep real
    # probability. Bonuses here are the spread actually seen between candidates
    # (~0.1); a contrived 0.75 gap would make any c look like a hard filter.
    picks = [self._pick([0.20, 0.30, 0.24], 10.0, seed=s) for s in range(300)]
    counts = np.bincount(picks, minlength=3)
    assert counts[0] > 10, f'common candidate never chosen: {counts}'
    assert counts[1] > counts[0], f'rarest not favoured: {counts}'

  def test_shipped_default_is_off_and_the_arm_turns_it_on(self):
    import pathlib, ruamel.yaml as yaml
    path = pathlib.Path(__file__).parent.parent.parent / 'dreamerv3' / 'configs.yaml'
    cfg = yaml.YAML(typ='safe').load(path.read_text())
    assert cfg['defaults']['agent']['mgr_ucb_c'] == 0.0
    assert cfg['mgr_ucb']['agent']['mgr_novel'] is False
    assert cfg['mgr_ucb']['agent']['mgr_ucb_c'] > 0


class TestSelectionCounts:
  """Selections drive the UCB bonus; visits stay a coverage metric.

  The split exists because the two must behave OPPOSITELY over training: the
  visit window decays so it tracks replay, while the selection table is
  cumulative so the bonus fades like textbook UCB instead of holding full
  strength to the end of the run.
  """

  def test_proposing_a_goal_lowers_its_own_bonus(self):
    # The self-limiting property. A visit-based bonus cannot do this: if the
    # worker never arrives, the visit count never moves.
    cc, params = build()
    ids = jnp.zeros((4, L), jnp.int32)
    _, before = run(cc, params, lambda i: cc.ucb_bonus(i), ids)
    params, _ = run(cc, params, lambda i: cc.update_selected(i, 1.0), ids)
    _, after = run(cc, params, lambda i: cc.ucb_bonus(i), ids)
    assert val(after) < val(before)

  def test_an_unreached_goal_still_gets_counted(self):
    # Propose repeatedly, never visit. The bonus must still decay, or the
    # manager fixates on an unreachable goal forever.
    cc, params = build()
    ids = jnp.zeros((1, L), jnp.int32)
    for _ in range(20):
      params, _ = run(cc, params, lambda i: cc.update_selected(i, 1.0), ids)
    _, (bonus, visits) = run(cc, params, lambda i: (
        cc.ucb_bonus(i), cc.lookup(i)[0]), ids)
    assert val(visits) == 0.0, 'nothing was ever reached'
    assert val(bonus) < 0.3, 'but the bonus decayed anyway'

  def test_the_selection_table_does_not_decay(self):
    # decay applies to visits only; a cumulative selection table is what makes
    # the bonus fade as 1/sqrt(n) over a run.
    cc, params = build(decay=0.5)
    ids = jnp.zeros((1, L), jnp.int32)
    params, _ = run(cc, params, lambda i: cc.update_selected(i, 1.0), ids)
    _, b1 = run(cc, params, lambda i: cc.ucb_bonus(i), ids)
    other = jnp.full((1, L), 3, jnp.int32)
    for _ in range(5):
      params, _ = run(cc, params, lambda i: cc.update_selected(i, 1.0), other)
    _, b2 = run(cc, params, lambda i: cc.ucb_bonus(i), ids)
    assert val(b2) == pytest.approx(val(b1), rel=1e-5), 'must not decay away'

  def test_the_bonus_fades_as_one_over_sqrt_n(self):
    # select_h=0 is the hard-grid path, where a deposit lands wholly in one cell
    # and the law is exact. With smearing the mass spreads, so the same cell
    # receives less than 1.0 per proposal -- covered separately below.
    cc, params = build(select_h=0.0)
    ids = jnp.zeros((1, L), jnp.int32)
    seen = []
    for _ in range(3):
      for _ in range(10):
        params, _ = run(cc, params, lambda i: cc.update_selected(i, 1.0), ids)
      _, b = run(cc, params, lambda i: cc.ucb_bonus(i), ids)
      seen.append(val(b))
    assert seen[0] > seen[1] > seen[2]
    assert seen[0] == pytest.approx(1 / np.sqrt(11), rel=1e-4)
    assert seen[2] == pytest.approx(1 / np.sqrt(31), rel=1e-4)

  def test_visits_and_selections_are_independent_tables(self):
    cc, params = build(weight=1.0)
    ids = jnp.zeros((1, L), jnp.int32)
    params, _ = run(cc, params, lambda i: cc.update(i), ids)          # visit only
    _, (bonus, visits) = run(cc, params, lambda i: (
        cc.ucb_bonus(i), cc.lookup(i)[0]), ids)
    assert val(visits) > 0.0
    assert val(bonus) == pytest.approx(1.0, rel=1e-5), 'a visit is not a selection'

  def test_the_metric_set_is_small_and_finite(self):
    # Deliberately few: fifteen were emitted before and most went unread (the
    # coarse grid saturates so its occupancy and the frontier on it are
    # constants). Keep the accounting check, the smearing check, the fade, and
    # visit coverage.
    cc, params = build(weight=1.0)
    ids = jnp.zeros((2, L), jnp.int32)
    params, _ = run(cc, params, lambda i: cc.update_selected(i, 4.0), ids)
    params, _ = run(cc, params, lambda i: cc.update(i), ids)
    _, mets = run(cc, params, lambda i: cc.metrics(), ids)
    assert set(mets) == {'select_mass', 'select_eff_cells', 'select_bonus_mean',
                         'fine_occupied', 'total_mass'}, sorted(mets)
    for k, v in mets.items():
      assert np.asarray(v).shape == () and np.isfinite(np.asarray(v)), k

  def test_smearing_spreads_mass_to_neighbours_and_conserves_it(self):
    # Smearing is what makes the bonus anneal: with a hard grid most cells stay
    # at exactly zero and score the maximum bonus forever, so the spread between
    # candidates never closes. Total mass must be unchanged -- the accounting
    # that a count equals one decision is what the whole table rests on.
    hard, ph = build(select_h=0.0)
    soft, ps = build(select_h=1.0)
    ids = jnp.full((1, L), 3, jnp.int32)
    ph, _ = run(hard, ph, lambda i: hard.update_selected(i, 1.0), ids)
    ps, _ = run(soft, ps, lambda i: soft.update_selected(i, 1.0), ids)
    _, mh = run(hard, ph, lambda i: hard.metrics(), ids)
    _, ms = run(soft, ps, lambda i: soft.metrics(), ids)
    assert float(mh['select_mass']) == pytest.approx(1.0, rel=1e-4)
    assert float(ms['select_mass']) == pytest.approx(1.0, rel=1e-4), 'mass conserved'
    assert float(ms['select_eff_cells']) > float(mh['select_eff_cells']) * 5, (
        'smearing must spread the deposit over many more cells')

  def test_smeared_mass_peaks_in_the_cell_the_class_quantises_to(self):
    """The kernel must be centred on the cells it deposits into.

    Cell j holds classes [j*C/b, (j+1)*C/b), so its midpoint is
    (j+0.5)*C/b - 0.5. Omitting the -0.5 shifts every centre half a class: at
    C=8, b=4 class 2 quantises to cell 1 but split its mass equally between
    cells 0 and 1, and 3 of 8 classes peaked in the wrong cell.
    """
    cc, params = build(select_h=1.0)
    for cls in range(C):
      ids = jnp.full((1, L), cls, jnp.int32)
      _, dep = run(cc, params, lambda i: cc._deposit(i), ids)
      per_block = np.asarray(dep).reshape(-1)
      # marginal of the first block: sum the joint deposit over the other blocks
      m = per_block.reshape((cc.select,) * L).sum(axis=tuple(range(1, L)))
      assert int(m.argmax()) == (cls * cc.select) // C, (
          f'class {cls} peaks in cell {m.argmax()}, quantises to '
          f'{(cls * cc.select) // C}')

  def test_h_zero_recovers_the_hard_grid(self):
    a, pa = build(select_h=0.0)
    ids = jnp.full((3, L), 2, jnp.int32)
    pa, _ = run(a, pa, lambda i: a.update_selected(i, 1.0), ids)
    _, m = run(a, pa, lambda i: a.metrics(), ids)
    assert float(m['select_eff_cells']) == pytest.approx(1.0, rel=1e-4)
