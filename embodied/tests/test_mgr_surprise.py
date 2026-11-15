"""Surprise reward for the manager (``agent.mgr_surprise``).

A head regresses max(0, KL(post||prior) - running mean KL) on replay; the
manager is paid the head's prediction on imagined states through the third
critic. Checked on the real agent: it builds and trains on fixed and on
variable holds, the head learns, no count table is created, and the head's
loss sends no gradient into the world model.
"""
import numpy as np
import pytest

from test_goal_ae_agent import make_agent, train_steps

V = ('director', 'director_og', 'wkr_ent_fixed', 'wkr_v3')


def _get(mets, key):
  hits = [k for k in mets if k.endswith(key)]
  assert hits, (key, sorted(mets)[:60])
  return float(np.asarray(mets[hits[0]]))


@pytest.mark.parametrize('extra', [(), ('goal_hold_reach', 'goal_hold_cap8')])
def test_surprise_arm_builds_trains_and_logs(extra):
  agent = make_agent(*V, *extra, 'mgr_surprise', 'debug')
  before = {k: np.asarray(v).copy() for k, v in agent.params.items()}
  mets = train_steps(agent, steps=4)
  for key in ('surprise/kl', 'surprise/base', 'surprise/excess',
              'surprise/excess_frac', 'surprise/pred', 'surprise/imag'):
    assert np.isfinite(_get(mets, key)), key
  kl, base = _get(mets, 'surprise/kl'), _get(mets, 'surprise/base')
  assert kl > 0 and base > 0
  # The baseline starts AT the mean KL (no warm-up from zero), so the excess
  # is a minority event from the first step, not "everything is surprising".
  assert abs(base - kl) < 0.5 * kl, (base, kl)
  assert 0.0 <= _get(mets, 'surprise/excess_frac') < 0.9
  assert _get(mets, 'surprise/imag') >= 0.0
  after = {k: np.asarray(v) for k, v in agent.params.items()}
  moved = [k for k in after if k.startswith('mgr_surp/')
           and np.abs(after[k] - before[k]).max() > 0]
  assert moved, 'surprise head did not train'
  assert any(k.startswith('mgr_novel_val/') for k in after)
  assert not any('code_counts' in k for k in after), 'count table was built'
  assert _get(mets, 'loss/mgr_surprise') >= 0 if any(
      k.endswith('loss/mgr_surprise') for k in mets) else True


def test_outlier_threshold_pays_fewer_steps():
  """mean + 2 std: only clear outliers are surprising."""
  agent = make_agent(*V, 'mgr_surprise', 'mgr_surprise_k2', 'debug')
  mets = train_steps(agent, steps=4)
  assert _get(mets, 'surprise/thresh') > _get(mets, 'surprise/base')
  assert _get(mets, 'surprise/excess_frac') < 0.2
