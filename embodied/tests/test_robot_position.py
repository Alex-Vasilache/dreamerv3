"""The robot's leaky wheel distance ('position', env.robot.pos_tau): the
trainer's observation and cost, and the phone's copy of it."""

import pathlib
import sys
import types

import numpy as np
import pytest

from embodied.envs import robot

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'dreamerv3' / 'deploy'))
from policy_runner import PolicyRunner  # noqa: E402

HZ = 10.0
KW = dict(pos_tau=2.0, pos_penalty=0.5, pos_range=1000.0,
          obs_pos_scale=1e-3, pos_jump=500.0)


def make_env(tmp_path):
  env = robot.SmartphoneRobot(
      'balance', discrete=False, symmetric=False, hz=HZ,
      pause_file=str(tmp_path / 'pause.json'), **KW)
  env._send = lambda left, right, reset: None
  feed = []
  env._recv = lambda: (feed.pop(0), 0.0)
  return env, feed


def sensors(left, right):
  return dict(theta=0.0, angular_velocity=0.0, wheel_speed_l=0.0,
              wheel_speed_r=0.0, wheel_distance_l=left, wheel_distance_r=right,
              battery_voltage=8.0)


# 100 mm a step forward for 10 steps, a counter reset, then standing still.
STREAM = ([sensors(100.0 * i, 100.0 * i) for i in range(11)]
          + [sensors(0.0, 0.0)] + [sensors(0.0, 0.0)] * 20)


def test_position_sums_the_wheels_with_a_leak_and_skips_a_reset(tmp_path):
  env, feed = make_env(tmp_path)
  feed.extend(STREAM)
  drive = dict(drive=np.zeros(2, np.float32))
  pos = [env._advance(drive)['position'] for _ in STREAM]
  leak = np.exp(-1 / (KW['pos_tau'] * HZ))
  expect = sum(100.0 * leak ** k for k in range(10))
  assert pos[10][0] == pytest.approx(expect * KW['obs_pos_scale'], rel=1e-5)
  # The jump back to 0 is a counter reset, not 1000 mm of motion.
  assert pos[11][0] == pytest.approx(pos[10][0] * leak, rel=1e-5)
  # Standing still, it decays back towards where the robot is.
  assert pos[-1][0] < 0.5 * pos[11][0]


def test_displacement_costs_up_to_pos_penalty(tmp_path):
  env, feed = make_env(tmp_path)
  drive = dict(drive=np.zeros(2, np.float32))
  feed.extend([sensors(0.0, 0.0)] * 2)
  still = [float(env._advance(drive)['reward']) for _ in range(2)]
  feed.extend([sensors(400.0 * i, 400.0 * i) for i in range(1, 8)])
  moved = [float(env._advance(drive)['reward']) for _ in range(7)]
  assert still[1] - moved[0] == pytest.approx(
      KW['pos_penalty'] * 400.0 / KW['pos_range'], rel=1e-4)
  assert still[1] - moved[-1] == pytest.approx(KW['pos_penalty'], rel=1e-4)


def test_the_phone_observes_the_same_position(tmp_path):
  env, feed = make_env(tmp_path)
  feed.extend(STREAM)
  runner = PolicyRunner()
  runner.policy = types.SimpleNamespace(meta=dict(
      obs_keys=['orientation', 'position', 'wheels'],
      obs_scale=dict(theta_zero=0.0, obs_theta_scale=0.18, obs_rate_scale=3.0,
                     obs_wheel_scale=3.3e-4, obs_clip=3.0, hz=HZ,
                     **{k: KW[k] for k in ('pos_tau', 'obs_pos_scale',
                                           'pos_jump')})))
  drive = dict(drive=np.zeros(2, np.float32))
  for s in STREAM:
    ours = env._advance(drive)['position']
    np.testing.assert_allclose(runner.observe(s)['position'], ours, rtol=1e-6)


def test_off_by_default():
  env = robot.SmartphoneRobot('balance', discrete=False, symmetric=False)
  assert 'position' not in env.obs_space
