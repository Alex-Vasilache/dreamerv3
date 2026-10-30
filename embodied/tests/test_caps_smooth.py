"""CAPS smoothness terms on the flat actor (agent.imag_loss.smooth / spatial):
the real Agent trains with them and logs finite jump and gap metrics."""
import numpy as np

from embodied.tests.test_goal_ae_agent import make_agent, train_steps


def test_flat_actor_trains_with_temporal_and_spatial_terms():
  agent = make_agent(**{'agent.use_hrl': False, 'agent.imag_loss.smooth': 0.3, 'agent.imag_loss.spatial': 0.5})
  mets = train_steps(agent, steps=3)
  keys = [k for k in mets if k.endswith('smooth_jump') or k.endswith('smooth_gap')]
  assert any(k.endswith('smooth_gap') for k in keys), sorted(mets)[:40]
  assert all(np.isfinite(np.asarray(mets[k])).all() for k in keys)
  assert all(float(np.asarray(mets[k]).mean()) > 0 for k in keys)


def test_spatial_off_logs_no_gap():
  agent = make_agent(**{'agent.use_hrl': False})
  mets = train_steps(agent, steps=3)
  assert not any(k.endswith('smooth_gap') for k in mets)
