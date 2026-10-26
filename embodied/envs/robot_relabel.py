"""Learner-side data augmentation for the robot's command task.

Both transforms act on training batches drawn from replay ({key: [B, T, ...]}
numpy arrays), before the world model sees them, so every real step teaches
more than the one situation it was recorded in.

Command relabelling (hindsight): the command is exogenous -- nothing the
robot does changes it -- and the command reward ('bumper_track') is a known
function of the observation and the command. So a recorded sequence is just
as valid with any other command in its place, if the reward is rescored for
it. A fraction of each batch gets a fresh command from the phone's sampler,
held over the sequence, and its tracking reward recomputed.

Mirror: the robot is left-right symmetric. Swapping the wheels (observation
and action) and negating the turn command gives another valid sequence with
the same reward.

The context latent stored with the sequence (replay_context) was computed
with the original command and side; the posterior corrects it within a few
steps, since the command and wheels are observed directly.
"""

import math

import numpy as np


class CommandRelabel:

  def __init__(self, robot, relabel=0.0, mirror=0.0, seed=0):
    r = robot
    self.relabel = float(relabel)
    self.mirror = float(mirror)
    if self.relabel and r.command_reward != 'bumper_track':
      raise NotImplementedError(
          f'relabelling rescoring is written for bumper_track, not '
          f'{r.command_reward}')
    if r.turn_source != 'wheels':
      raise NotImplementedError('relabelling needs turn_source wheels')
    self.r = r
    self.rng = np.random.default_rng(seed)

  def __call__(self, batch):
    if 'command' not in batch:
      return batch
    batch = dict(batch)
    B = batch['command'].shape[0]
    if self.mirror:
      m = self.rng.random(B) < self.mirror
      if m.any():
        for key in ('wheels', 'drive'):
          if key in batch:
            x = batch[key].copy()
            x[m] = x[m][..., ::-1]
            batch[key] = x
        cmd = batch['command'].copy()
        cmd[m, :, 1] *= -1
        batch['command'] = cmd
    if self.relabel:
      sel = self.rng.random(B) < self.relabel
      if sel.any():
        new = np.array([self._sample() for _ in range(int(sel.sum()))],
                       np.float32)
        old_track = self._track(batch, batch['command'][sel], sel)
        cmd = batch['command'].copy()
        cmd[sel] = new[:, None, :]
        new_track = self._track(batch, cmd[sel], sel)
        near = self._near(batch['orientation'][sel])
        reward = batch['reward'].copy()
        reward[sel] = np.where(
            near, reward[sel], reward[sel] - old_track + new_track)
        batch['command'] = cmd
        batch['reward'] = reward.astype(batch['reward'].dtype)
    return batch

  def _sample(self):
    """The phone's 'auto' sampler (robot_sim.RobotSim._sample_command)."""
    r, rng = self.r, self.rng
    u = rng.random()
    if u < r.command_p_zero:
      cmd = (0.0, 0.0)
    elif u < r.command_p_zero + r.command_p_axis:
      value = rng.uniform(-1, 1)
      cmd = (value, 0.0) if rng.random() < 0.5 else (0.0, value)
    else:
      rad, phi = math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
      cmd = (rad * math.cos(phi), rad * math.sin(phi))
    if r.command_step:
      cmd = tuple(r.command_step * round(c / r.command_step) for c in cmd)
    return cmd

  def _near(self, orientation):
    r = self.r
    offset = orientation[..., 0] * r.obs_theta_scale
    return ((offset >= r.bumper_hi - r.bumper_margin)
            | (offset <= r.bumper_lo + r.bumper_margin))

  def _track(self, batch, cmd, sel):
    """robot.Robot._command_reward's tracking term, vectorised."""
    r = self.r
    speed = batch['wheels'][sel] / r.obs_wheel_scale
    left, right = speed[..., 0], speed[..., 1]
    forward = r.forward_sign * 0.5 * (left + right) / r.command_speed
    turn = 0.5 * (left - right) / r.command_turn
    cf, ct = cmd[..., 0], cmd[..., 1]
    k = np.maximum(1.0, np.abs(cf) + np.abs(ct) * r.command_turn / r.command_speed)
    err = np.hypot(forward - cf / k, turn - ct / k)
    return np.exp(-((err / r.command_sigma) ** 2))
