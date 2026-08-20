"""Epsilon-greedy index jumps (dreamerv3/hrl/explore.py).

The sampler has to hit an exact total distance while respecting a per-block
capacity that depends on the base code, so the properties worth asserting are
arithmetic ones: the realized distance equals the drawn one, no block leaves
its range, the distance distribution matches p(d) = 2d/[dmax(dmax+1)], and the
jump fires at the requested rate.

The last test is the one that matters for correctness of training: the jump
must reach the actor and NOT the two paths the manager learns from.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.hrl import explore

L, C = 8, 8
KEY = jax.random.PRNGKey(0)


def ids_of(onehot):
  return np.asarray(jnp.argmax(onehot, -1))


# --------------------------------------------------------------------------
# the pieces
# --------------------------------------------------------------------------

def test_capacity_is_the_larger_side_and_dmax_sums_it():
  ids = jnp.array([[0, 7, 3, 4, 1, 6, 2, 5]])
  down, up, cap, dmax = explore.capacities(ids, C)
  np.testing.assert_array_equal(np.asarray(down)[0], [0, 7, 3, 4, 1, 6, 2, 5])
  np.testing.assert_array_equal(np.asarray(up)[0], [7, 0, 4, 3, 6, 1, 5, 2])
  np.testing.assert_array_equal(np.asarray(cap)[0], [7, 7, 4, 4, 6, 6, 5, 5])
  assert int(dmax[0]) == 44


def test_dmax_averages_44_over_uniform_codes_not_56():
  """A property of the CODE, not a constant. Only a code with every block at an
  extreme can reach the 56 the axis admits."""
  ids = jax.random.randint(KEY, (20000, L), 0, C)
  _, _, _, dmax = explore.capacities(ids, C)
  dmax = np.asarray(dmax)
  assert 43.5 < dmax.mean() < 44.5, dmax.mean()
  assert dmax.min() >= 32 and dmax.max() <= 56
  assert (dmax == 56).mean() < 1e-3


def test_distance_distribution_matches_the_formula():
  dmax = jnp.full((200000,), 44)
  d = np.asarray(explore.sample_distance(KEY, dmax))
  assert d.min() >= 1 and d.max() <= 44
  want = 2.0 * np.arange(1, 45) / (44 * 45)
  got = np.bincount(d, minlength=45)[1:] / len(d)
  assert np.abs(got - want).max() < 2e-3, np.abs(got - want).max()
  # E[d] = (2/3)(dmax + 1/2)
  assert abs(d.mean() - (2 / 3) * 44.5) < 0.1, d.mean()


def test_spread_realizes_exactly_the_requested_distance():
  ids = jax.random.randint(KEY, (4000, L), 0, C)
  _, _, cap, dmax = explore.capacities(ids, C)
  d = explore.sample_distance(KEY, dmax)
  mag = explore.spread(jax.random.PRNGKey(1), cap, d, C)
  np.testing.assert_array_equal(np.asarray(mag.sum(-1)), np.asarray(d))
  assert (np.asarray(mag) <= np.asarray(cap)).all()


def test_jump_lands_in_range_and_at_the_drawn_distance():
  ids = jax.random.randint(KEY, (4000, L), 0, C)
  out, d = explore.jump(jax.random.PRNGKey(2), ids, C)
  out = np.asarray(out)
  assert out.min() >= 0 and out.max() <= C - 1
  got = np.abs(out - np.asarray(ids)).sum(-1)
  np.testing.assert_array_equal(got, np.asarray(d))


def test_jump_actually_goes_far():
  """The whole point: the realized distance should sit near (2/3)*dmax, far
  above the ~5.7 an ordinary sample from the policy gives."""
  ids = jax.random.randint(KEY, (20000, L), 0, C)
  out, _ = explore.jump(jax.random.PRNGKey(3), ids, C)
  got = np.abs(np.asarray(out) - np.asarray(ids)).sum(-1)
  assert 28 < got.mean() < 32, got.mean()


@pytest.mark.parametrize('eps', [0.0, 0.1, 0.5, 1.0])
def test_apply_jump_fires_at_the_requested_rate(eps):
  ids = jax.random.randint(KEY, (20000, L), 0, C)
  hot = jax.nn.one_hot(ids, C)
  out = explore.apply_jump(jax.random.PRNGKey(4), hot, eps, C)
  changed = (ids_of(out) != np.asarray(ids)).any(-1).mean()
  if eps == 0.0:
    assert changed == 0.0
  else:
    # a jump always moves at least one block, so "changed" == "jumped"
    assert abs(changed - eps) < 0.02, (eps, changed)


def test_apply_jump_moves_the_whole_code_not_single_blocks():
  """A jump is one coherent move, unlike a per-class uniform mixture which
  would perturb blocks independently."""
  ids = jax.random.randint(KEY, (4000, L), 0, C)
  hot = jax.nn.one_hot(ids, C)
  out = ids_of(explore.apply_jump(jax.random.PRNGKey(5), hot, 1.0, C))
  nblocks = (out != np.asarray(ids)).sum(-1)
  assert nblocks.mean() > 5.0, nblocks.mean()


def test_output_is_a_valid_onehot():
  ids = jax.random.randint(KEY, (256, L), 0, C)
  out = np.asarray(explore.apply_jump(
      jax.random.PRNGKey(6), jax.nn.one_hot(ids, C), 0.5, C))
  assert np.allclose(out.sum(-1), 1.0)
  assert np.isin(out, (0.0, 1.0)).all()


def test_is_jittable():
  fn = jax.jit(lambda k, h: explore.apply_jump(k, h, 0.1, C))
  ids = jax.random.randint(KEY, (32, L), 0, C)
  out = fn(jax.random.PRNGKey(7), jax.nn.one_hot(ids, C))
  assert out.shape == (32, L, C)


# --------------------------------------------------------------------------
# the wiring: actor only
# --------------------------------------------------------------------------

def test_only_the_actor_path_explores():
  """The manager is trained by REINFORCE on its own log-probabilities, so a
  jump inside imagination or on the replay sequence would bias the gradient.
  ``_manager_skill_step`` is the env-rollout path and must be the only caller
  that passes explore=True."""
  import inspect
  from dreamerv3.hrl import manager
  src = inspect.getsource(manager)
  callers = [ln.strip() for ln in src.splitlines() if 'explore=True' in ln]
  assert callers, 'nothing requests exploration at all'
  fns = {}
  cur = None
  for ln in src.splitlines():
    s = ln.strip()
    if s.startswith('def '):
      cur = s.split('(')[0][4:]
    if 'explore=True' in s:
      fns.setdefault(cur, 0)
      fns[cur] += 1
  assert set(fns) == {'_manager_skill_step'}, (
      'exploration must be actor-only, found it in: %s' % sorted(fns))
