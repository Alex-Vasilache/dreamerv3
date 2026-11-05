"""Emotion acting on behaviour and learning: pain, affect, arousal events."""

import json

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'dreamerv3' / 'deploy'))

import policy_runner  # noqa: E402
from embodied.envs import robot  # noqa: E402
from embodied.run.online import ArousalBoost  # noqa: E402

BODY = dict(theta_zero=0.0, lo=-0.12, hi=0.161, zone=0.035)


def test_pain_signal_zones_and_copies_agree():
  for theta in (-0.2, -0.11, -0.09, 0.0, 0.13, 0.15, 0.2):
    for act, chosen in ((None, None), ((0.2, 0.2), (0.8, 0.8))):
      a = policy_runner.pain_signal(theta, BODY, act, chosen)
      b = robot.pain_signal(theta, BODY, act, chosen)
      assert np.allclose(a, b)
  assert policy_runner.pain_signal(0.0, BODY) == (0.0, 0.0, 0.0)
  pain, front, back = policy_runner.pain_signal(0.161, BODY)
  assert front == 1.0 and back == 0.0 and pain == 1.0
  pain, front, back = policy_runner.pain_signal(-0.12 + 0.0175, BODY)
  assert back == 0.5 and front == 0.0
  # A reflex taking 0.6 off the policy's command hurts 0.6 on top.
  pain, _, _ = policy_runner.pain_signal(0.0, BODY, (0.2, 0.2), (0.8, 0.8))
  assert abs(pain - 0.6) < 1e-6


def _runner(state=None, **affect):
  r = policy_runner.PolicyRunner()
  r.body = BODY
  r.affect.update(affect)
  r.emotion_state = state
  return r


def test_modulate_vigour_and_catch():
  calm = dict(arousal=0.9, valence=-0.5, mood=-0.5)
  happy = dict(arousal=0.8, valence=0.5, mood=0.5)
  assert _runner(calm, vigour=1.0).modulate([0.4, -0.4]) == [0.4, -0.4]
  out = _runner(happy, vigour=1.0).modulate([0.4, -0.4])
  assert np.allclose(out, [0.4 * 1.4, -0.4 * 1.4])
  r = _runner(calm, catch=0.5)
  r.pain = policy_runner.pain_signal(0.15, BODY)   # leaning forward
  out = r.modulate([0.0, 0.0])
  assert out[0] > 0 and out[0] == out[1]           # drive forward to catch
  r.pain = policy_runner.pain_signal(-0.11, BODY)  # leaning back
  assert r.modulate([0.0, 0.0])[0] < 0
  assert r.modulate([-1.0, -1.0]) == [-1.0, -1.0]   # clipped
  # What the trainer's limits are measured against is the affected action.
  out = r.modulate([0.0, 0.0])
  assert r._chosen == out


def test_explore_scale():
  assert _runner(None, explore=1.0).explore_scale() == 1.0
  sad = _runner(dict(arousal=1.0, mood=-1.0, valence=-1.0), explore=1.0)
  assert sad.explore_scale() == 2.0
  hurt = _runner(dict(arousal=1.0, mood=0.5, valence=0.0), explore=1.0)
  hurt.pain = (1.0, 1.0, 0.0)
  assert hurt.explore_scale() == 0.5
  assert _runner(dict(arousal=1.0, mood=-1.0, valence=0.0)).explore_scale() == 1.0


def test_arousal_boost(tmp_path):
  boost = ArousalBoost(tmp_path, lr=2.0, updates=4, burst=8, publish=True)
  assert boost.poll() is None   # no file yet
  path = tmp_path / 'emotion_event'
  path.write_text(json.dumps(dict(n=1, t=0, intensity=0.5)))
  boost.polled = 0
  assert boost.poll() is None   # one from before: not new
  path.write_text(json.dumps(dict(n=2, t=0, intensity=0.5)))
  boost.polled = 0
  assert boost.poll() == 0.5
  scales = []
  for _ in range(6):
    scales.append(boost.scale())
    boost.left = max(boost.left - 1, 0)
  assert np.allclose(scales, [2.0, 1.75, 1.5, 1.25, 1.0, 1.0])
  assert ArousalBoost(tmp_path).on is False


def test_arousal_boost_sets_agent_scale(tmp_path):

  import jax.numpy as jnp

  class Agent:
    params = {'opt/lr_scale_model/value': jnp.ones(()),
              'opt/lr_scale_ac/value': jnp.ones(()),
              'pol/w': jnp.zeros(3)}

  boost = ArousalBoost(tmp_path, lr=1.0, updates=2)
  boost.intensity, boost.left = 1.0, 2
  agent = Agent()
  boost.apply(agent)
  assert float(agent.params['opt/lr_scale_model/value']) == 2.0
  assert float(agent.params['opt/lr_scale_ac/value']) == 2.0
  assert agent.params['pol/w'].shape == (3,)
  boost.apply(agent)
  boost.apply(agent)
  assert float(agent.params['opt/lr_scale_ac/value']) == 1.0


def test_sim_env_with_phone_declares_executed_and_cost():
  from dreamerv3 import main as dm
  import elements
  import ruamel.yaml as yaml
  c = yaml.YAML(typ='safe').load(open(ROOT / 'dreamerv3' / 'configs.yaml'))
  cfg = elements.Config(c['defaults'])
  for block in ('robot', 'robot_sim'):
    cfg = cfg.update(c[block])
  cfg = cfg.update({
      'task': 'robot_balance', 'logdir': '/tmp/unused',
      'env.robot.sim_phone': True, 'env.robot.cost_obs': True,
      'env.robot.sim_walls': 0.5, 'env.robot.sim_push': '1:0.1:0.1'})
  env = dm.make_env(cfg, 0)
  assert 'executed/drive' in env.obs_space and 'cost' in env.obs_space
  assert env.model.geom('wall0') is not None
  env.close()
