"""Command relabelling rescores rewards exactly as the robot env does."""

import elements
import numpy as np
import pytest

pytest.importorskip('mujoco')

from embodied.envs import robot_mujoco  # noqa: E402
from embodied.envs.robot_relabel import CommandRelabel  # noqa: E402

KW = dict(
    command_reward='bumper_track', bumper_margin=0.0262, bumper_penalty=2.0,
    command_speed=2500.0, command_turn=1000.0, command_sigma=0.4,
    command_p_zero=0.25, command_p_axis=0.5, forward_sign=-1.0,
    turn_source='wheels', action_rate_penalty=0.0)


def robot_config():
  return elements.Config(
      **KW, theta_zero=0.0, bumper_lo=-0.120, bumper_hi=0.161,
      obs_theta_scale=0.18, obs_wheel_scale=3.3e-4, command_step=0.0)


def rate_cost(batch):
  # The command reward's tilt-rate penalty (rate_penalty 0.05, obs scale 3).
  return 0.05 * np.abs(batch['orientation'][..., 1] * 3.0)


def rollout(steps=600, seed=0):
  env = robot_mujoco.RobotMujoco(
      'command', hz=65.0, discrete=False, symmetric=False, sim_noise=0.0,
      **{k: v for k, v in KW.items()})
  rng = np.random.default_rng(seed)
  keys = ('wheels', 'orientation', 'command', 'reward')
  out = {k: [] for k in keys}
  obs = env.step({'drive': np.zeros(2, np.float32), 'reset': True})
  for _ in range(steps):
    act = np.clip(rng.normal(0, 0.6, 2), -1, 1).astype(np.float32)
    obs = env.step({'drive': act, 'reset': False})
    for k in keys:
      out[k].append(np.asarray(obs[k]))
  return {k: np.stack(v)[None] for k, v in out.items()}


def test_rescoring_matches_env():
  batch = rollout()
  relabel = CommandRelabel(robot_config(), relabel=1.0)
  sel = np.ones(1, bool)
  track = relabel._track(batch, batch['command'][sel], sel)
  near = relabel._near(batch['orientation'][sel])
  expected = np.where(near, -2.0, track) - rate_cost(batch)
  assert not near.all()
  np.testing.assert_allclose(batch['reward'], expected, atol=1e-4)


def test_relabelled_reward_is_the_new_commands():
  batch = rollout(seed=1)
  relabel = CommandRelabel(robot_config(), relabel=1.0, seed=3)
  out = relabel(batch)
  assert not np.allclose(out['command'], batch['command'])
  sel = np.ones(1, bool)
  track = relabel._track(out, out['command'], sel)
  near = relabel._near(out['orientation'])
  np.testing.assert_allclose(
      out['reward'], np.where(near, -2.0, track) - rate_cost(out), atol=1e-4)


def test_mirror_keeps_reward():
  batch = rollout(seed=2)
  batch['drive'] = np.random.default_rng(0).uniform(
      -1, 1, batch['wheels'].shape).astype(np.float32)
  relabel = CommandRelabel(robot_config(), mirror=1.0)
  out = relabel(batch)
  np.testing.assert_array_equal(out['wheels'], batch['wheels'][..., ::-1])
  np.testing.assert_array_equal(out['drive'], batch['drive'][..., ::-1])
  np.testing.assert_array_equal(out['command'][..., 1], -batch['command'][..., 1])
  sel = np.ones(1, bool)
  np.testing.assert_allclose(
      relabel._track(out, out['command'], sel),
      relabel._track(batch, batch['command'], sel), atol=1e-6)
