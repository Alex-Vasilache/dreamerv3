"""Goal-space masking (``goal_mask8``): helpers and agent wiring.

The wiring test exists because a mask head that never receives a gradient
looks exactly like a slow one in a short run: its loss sits at chance either
way. So it asserts the head's parameters actually MOVE under real train steps,
on both goal autoencoders the batch uses (Director AE = `now`, VQ/SOM = `now+SOM`).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3.hrl import goal_mask as gm

from test_goal_ae_agent import make_agent, train_steps


class TestHelpers:

  def test_train_mask_never_empty_and_half_full(self):
    m = np.asarray(gm.sample_train_mask(jax.random.PRNGKey(0), (8000,), 8, 0.5))
    kept = m.sum(-1)
    assert kept.min() >= 1
    assert abs((kept == 8).mean() - 0.5) < 0.02   # partial masks keep 1..7

  def test_keep_round_trips_through_the_goal(self):
    g = jax.random.normal(jax.random.PRNGKey(1), (6, 64))
    keep = gm.hard_keep(jax.random.normal(jax.random.PRNGKey(2), (6, 8)))
    masked = g * gm.expand_groups(keep, 64)
    assert (gm.group_keep(masked, 8) == keep).all()

  def test_hard_keep_keeps_at_least_one(self):
    assert (gm.hard_keep(-jnp.ones((3, 8))).sum(-1) == 1).all()


def _mdec_params(agent):
  return {k: np.asarray(v) for k, v in agent.params.items()
          if k.startswith('goal_mdec/')}


@pytest.mark.parametrize('arm', [(), ('goal_som_line',)], ids=['now', 'som'])
def test_mask_head_trains_and_metrics_are_finite(arm):
  # debug deter is 8 (rssm.blocks 4), so 4 groups of 2 dims.
  a = make_agent(*arm, 'goal_mask8', agent={'goal_mask_groups': 4})
  before = _mdec_params(a)
  assert before, 'goal_mdec has no parameters'
  mets = train_steps(a, steps=3)
  after = _mdec_params(a)
  moved = [k for k in before if not np.allclose(before[k], after[k])]
  assert moved, 'goal_mdec parameters never changed: no gradient reaches the mask head'
  for key in ('goal/mask_loss', 'goal/mask_acc', 'goal/mask_keep_mgr'):
    hits = [k for k in mets if k.endswith(key)]
    assert hits, (key, sorted(mets)[:20])
    assert np.isfinite(np.asarray(mets[hits[0]])).all()
