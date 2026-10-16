"""The smartphone robot as a simulated two-wheeled inverted pendulum.

Same observations, actions, rewards and episode handling as
`embodied.envs.robot.SmartphoneRobot` -- it subclasses it and replaces only
the link to the phone with a physics model -- so every `env.robot.*` key and
every robot_* config block applies unchanged. Turned on by
`env.robot.sim: True` (config block `sim_robot`); wrap it in
`env.realtime_hz` = `env.robot.sim_hz` to make it cost real time like the
robot does.

The plant, in the phone's units and signs (docs/ROBOT_BRIDGE_FINDINGS.md):

* Tilt theta in rad, positive is a backward lean. Bumper stops at
  bumper_lo/bumper_hi from theta_zero (-6.9 / +9.2 deg measured), inelastic.
* A positive motor output gives a negative wheel speed (forward_sign -1), and
  positive wheel acceleration catches a positive lean (recover_gain -1):
      theta'' = w0^2 sin(theta - theta_eq) - (a / L) cos(theta) - c theta'
  with w0 = 12/s, L = 0.08 m: a point mass ~6.5 cm above the axle. The
  closed-loop replay estimate (tau ~ 50 ms, w0 ~ 20, docs/ONDEVICE_POLICY.md)
  mixes in policy noise; at w0 20 even a tuned linear controller cannot hold
  25 Hz with the one-tick delay, while at 12 it holds 25 Hz loosely (tilt std
  ~2 deg) and 65 Hz tightly (~0.3 deg).
* Each wheel follows its target speed with a first-order lag (tau 50 ms).
  Ground top speed 4900 mm/s at full output. Without friction_comp, outputs
  under 0.06 do nothing and a still wheel needs 0.3 to start (the measured dead
  zone and stiction); with it, the phone's compensation is assumed exact.
* One tick between an observation and the action it prompts taking effect
  (the pipelined protocol), and sensor noise on tilt, rate and wheel speed.
* The body is not reset between episodes, as on the robot: an episode that
  ends on a bumper is followed by one that starts there.

It is a stand-in for tuning learning speed, not a calibrated twin: w0, L and
the motor lag are estimates, and the true balance point theta_eq is drawn per
run within +-1 deg of theta_zero, as it is unknown on the real robot.
"""

import math

import numpy as np

from embodied.envs import robot as robot_env


class RobotSim(robot_env.SmartphoneRobot):

  def __init__(
      self, task, sim_hz=25.0, sim_w0=12.0, sim_length=0.08, sim_damping=0.5,
      sim_motor_tau=0.05, sim_top_speed=4900.0, sim_track=0.12,
      sim_noise=1.0, sim_eq_range=0.0175, sim_substep=0.002, seed=0, **kwargs):
    for key in ('host', 'port', 'reconnect', 'onboard', 'weights_every',
                'pace', 'config', 'policy_dir'):
      kwargs.pop(key, None)
    super().__init__(task, seed=seed, onboard=False, reconnect=False, **kwargs)
    self.hz = float(sim_hz)
    self.w0 = float(sim_w0)
    self.arm = float(sim_length)
    self.damping = float(sim_damping)
    self.motor_tau = float(sim_motor_tau)
    self.top_speed = float(sim_top_speed)
    self.track = float(sim_track)
    self.noise = float(sim_noise)
    self.substeps = max(1, int(math.ceil(1.0 / self.hz / float(sim_substep))))
    self.sim_rng = np.random.RandomState(seed + 7919)
    self.theta_eq = self.theta_zero + self.sim_rng.uniform(
        -sim_eq_range, sim_eq_range)
    # Start leaning on a bumper, as a robot that was just set down does.
    self.theta = self.theta_zero + (
        self.bumper_hi if self.sim_rng.rand() < 0.5 else self.bumper_lo)
    self.omega = 0.0
    self.speed = np.zeros(2)       # mm/s, phone sign
    self.distance = np.zeros(2)    # mm
    self.applied = np.zeros(2)     # motor outputs in effect this tick
    self.previous = np.zeros(2)    # sent last step, takes effect this tick
    self.pending = np.zeros(2)     # sent this step, takes effect next tick
    self.cmd_target = (0.0, 0.0)
    self.cmd_now = np.zeros(2)
    self.cmd_left = 0.0            # seconds until the next random command

  # The phone link, replaced.

  def _connect(self):
    self._link = True

  def _drop(self):
    pass

  def close(self):
    pass

  def _send(self, left, right, reset):
    self._sent_act = (float(left), float(right))
    self.pending = np.array([left, right], np.float64) * self.command_scale
    if reset:
      self.distance[:] = 0.0

  def _recv(self):
    dt = 1.0 / self.hz
    # The phone picks up an action one tick after the observation that
    # prompted it, so this tick runs on the action sent a step ago.
    self.applied, self.previous = self.previous, self.pending
    target = -self.top_speed * np.array(
        [self._motor(u, v) for u, v in zip(self.applied, self.speed)])
    h = dt / self.substeps
    for _ in range(self.substeps):
      accel = (target - self.speed) / self.motor_tau   # mm/s^2
      self.speed += accel * h
      self.distance += self.speed * h
      a = 1e-3 * accel.mean()                           # m/s^2
      alpha = (self.w0 ** 2 * math.sin(self.theta - self.theta_eq)
               - a / self.arm * math.cos(self.theta)
               - self.damping * self.omega)
      self.omega += alpha * h
      self.theta += self.omega * h
      lo = self.theta_zero + self.bumper_lo
      hi = self.theta_zero + self.bumper_hi
      if self.theta <= lo:
        self.theta, self.omega = lo, max(self.omega, 0.0)
      elif self.theta >= hi:
        self.theta, self.omega = hi, min(self.omega, 0.0)
    self._sample_command(dt)
    n = self.noise
    rng = self.sim_rng
    speed = self.speed * (1 + 0.02 * n * rng.randn(2)) + 30 * n * rng.randn(2)
    self._yaw_rate = self.yaw_sign * (
        1e-3 * (self.speed[0] - self.speed[1]) / self.track
        + 0.02 * n * rng.randn())
    self._last = dict(
        wheel_speed_l=float(speed[0]), wheel_speed_r=float(speed[1]),
        wheel_distance_l=float(self.distance[0]),
        wheel_distance_r=float(self.distance[1]),
        theta=float(self.theta + 0.003 * n * rng.randn()),
        angular_velocity=float(self.omega + 0.05 * n * rng.randn()),
        battery_voltage=3.6)
    self._timing = (0.0, 0.0, 0.0, 0.0)
    self._backlog = 0
    self._dropped = 0
    return self._last, dt

  def _motor(self, u, speed):
    """Wheel speed as a fraction of top speed that output u drives toward."""
    u = float(np.clip(u, -1.0, 1.0))
    if self.friction_comp:
      return u
    if abs(speed) < 50.0 and abs(u) < 0.3:
      return 0.0                                        # stiction
    return math.copysign(max(abs(u) - 0.06, 0.0) / 0.94, u)

  def _sample_command(self, dt):
    """The phone's 'auto' command sampler, with its ramp."""
    if self.task != 'command':
      return
    s = self.command_sampler
    rng = self._rng
    self.cmd_left -= dt
    if self.cmd_left <= 0.0:
      self.cmd_left = rng.uniform(s['hold_min'], s['hold_max'])
      u = rng.rand()
      if u < s['p_zero']:
        cmd = (0.0, 0.0)
      elif u < s['p_zero'] + s['p_axis']:
        value = rng.uniform(-1, 1)
        cmd = (value, 0.0) if rng.rand() < 0.5 else (0.0, value)
      else:
        r, phi = math.sqrt(rng.rand()), rng.uniform(0, 2 * math.pi)
        cmd = (r * math.cos(phi), r * math.sin(phi))
      if s['step']:
        cmd = tuple(s['step'] * round(c / s['step']) for c in cmd)
      self.cmd_target = cmd
    target = np.array(self.cmd_target)
    if s['ramp']:
      delta = np.clip(target - self.cmd_now, -s['ramp'] * dt, s['ramp'] * dt)
      self.cmd_now = self.cmd_now + delta
    else:
      self.cmd_now = target
    self._cmd = (float(self.cmd_now[0]), float(self.cmd_now[1]))
    self._cmd_src = 0.0
