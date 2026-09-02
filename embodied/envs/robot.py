import json
import socket
import struct
import time

import elements
import embodied
import numpy as np

PROTOCOL = 1

# Anything that means the link went away rather than that the code is
# wrong: socket errors and timeouts (both OSError), plus a truncated or
# garbled frame from a half-closed connection.
LINKLOST = (OSError, ValueError, struct.error)

# Same wheel PWM table as apps/basicAssembler on the phone, so a discrete
# policy trained here means the same thing if it is later moved on-device.
MOTIONS = (
    ('stop', 0.0, 0.0),
    ('forward', 0.75, 0.75),
    ('backward', -0.75, -0.75),
    ('left', -1.0, 1.0),
    ('right', 1.0, -1.0),
)


class SmartphoneRobot(embodied.Env):
  """DreamerV3 environment backed by the OIST smartphone robot.

  The policy runs here; the phone runs a thin client (see the `dreamerBridge`
  app in the smartphone-robot-android repo) that connects in, applies each
  action to the wheels, waits out the control period, then ships proprio sensor
  readings back. The phone therefore owns the control clock and `step` blocks
  until the next reading arrives.

  The listening socket is opened lazily on the first `step` rather than in the
  constructor, because `main.make_agent` builds one throwaway env just to read
  the spaces and would otherwise leave the port bound.
  """

  def __init__(
      self, task='drive', host='0.0.0.0', port=3000, length=200,
      discrete=True, timeout=20.0, speed_scale=1e-3, fall_angle=0.0,
      spin_penalty=0.1, rate_penalty=0.05, theta_zero=0.0,
      theta_lo=-0.123, theta_hi=0.173, theta_sigma=0.05, drift_penalty=0.1,
      ref_range=0.09, ref_hold=100, seed=0, status_every=0,
      recover_gain=0.0, recover_k=4.0, recover_tol=0.05, recover_max=250,
      recover_min=0.6,
      reconnect=True, logdir=None):
    assert task in ('drive', 'balance', 'track', 'none'), task
    self.task = task
    self.host = host
    self.port = int(port)
    self.length = int(length)
    self.discrete = bool(discrete)
    self.timeout = float(timeout)
    self.speed_scale = float(speed_scale)
    self.fall_angle = float(fall_angle)
    self.spin_penalty = float(spin_penalty)
    self.rate_penalty = float(rate_penalty)
    self.theta_zero = float(theta_zero)
    # The bumper stops, as offsets from upright, and NOT symmetric. The tilt
    # term is normalised per side so the reward reaches 0 at each stop; one
    # symmetric theta_max paid 0.51 at the near stop and 0.31 at the far one,
    # leaving half the reward available for lying against a bumper.
    self.theta_lo = float(theta_lo)
    self.theta_hi = float(theta_hi)
    self.theta_sigma = float(theta_sigma)
    self.drift_penalty = float(drift_penalty)
    self.ref_range = float(ref_range)
    self.ref_hold = int(ref_hold)
    self._rng = np.random.RandomState(seed)
    self.status_every = int(status_every)
    # Sign of the wheel command that pushes the body back toward upright.
    # +1 means drive toward the lean, which is correct for an inverted
    # pendulum when the tilt sign and the PWM sign agree. It is set per rig
    # from two observations -- which way a hand lean reads, and which way
    # positive PWM rolls -- because it cannot be recovered from logs: measured
    # 2026-09-02, action-conditioned tilt response was 0.5 sigma. 0 disables
    # recovery, leaving the robot wherever the last episode ended.
    self.recover_gain = float(recover_gain)
    self.recover_k = float(recover_k)
    self.recover_tol = float(recover_tol)
    self.recover_max = int(recover_max)
    # Floor on the recovery command: a proportional term alone lands under the
    # motor deadband near the setpoint, so recovery stalls short of upright.
    # Measured 2026-09-02: 0.30 PWM produced no wheel motion at all.
    self.recover_min = float(recover_min)
    self._ref = float(theta_zero)
    self.reconnect = bool(reconnect)
    # Per-step timings, because the episode aggregates the logger keeps cannot
    # tell one long stall from many short ones.
    self._timing_file = None
    if logdir:
      path = elements.Path(logdir)
      path.mkdir()
      self._timing_path = str(path / 'timing.csv')
    else:
      self._timing_path = None
    self._listener = None
    self._sock = None
    self._file = None
    self._step = 0
    self._done = True
    self._sent = 0.0
    self._timing = (0.0, 0.0, 0.0, 0.0)
    self._last = dict(
        wheel_speed_l=0.0, wheel_speed_r=0.0, wheel_distance_l=0.0,
        wheel_distance_r=0.0, theta=0.0, angular_velocity=0.0,
        battery_voltage=0.0)

  @property
  def obs_space(self):
    return {
        'wheels': elements.Space(np.float32, (2,)),
        'orientation': elements.Space(np.float32, (3,)),
        'battery': elements.Space(np.float32, (1,)),
        'target': elements.Space(np.float32, (2,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/theta_deg': elements.Space(np.float32),
        'log/distance_l': elements.Space(np.float32),
        'log/distance_r': elements.Space(np.float32),
        'log/theta_ref_deg': elements.Space(np.float32),
        'log/track_err_deg': elements.Space(np.float32),
        'log/latency_ms': elements.Space(np.float32),
        'log/phone_wait_ms': elements.Space(np.float32),
        'log/phone_work_ms': elements.Space(np.float32),
        'log/imu_age_ms': elements.Space(np.float32),
        'log/imu_stale_ms': elements.Space(np.float32),
    }

  @property
  def act_space(self):
    space = {'reset': elements.Space(bool)}
    if self.discrete:
      space['motion'] = elements.Space(np.int32, (), 0, len(MOTIONS))
    else:
      space['wheels'] = elements.Space(np.float32, (2,), -1.0, 1.0)
    return space

  def step(self, action):
    if action['reset'] or self._done:
      return self._reset()
    try:
      return self._advance(action)
    except LINKLOST as e:
      if not self.reconnect:
        raise
      print(f'Lost the robot mid-episode, ending it here: {e}')
      self._drop()
      # Close the episode out rather than leaving replay a dangling sequence;
      # the next step reconnects through _reset.
      self._done = True
      return self._obs(self._last, 0.0, 0.0, is_last=True)

  def _reset(self):
    """Start an episode, waiting as long as it takes for the robot."""
    while True:
      try:
        self._connect()
        self._send(0.0, 0.0, reset=True)
        sensors, latency = self._recv()
        sensors, latency = self._recover(sensors, latency)
        self._step = 0
        self._done = False
        self._ref = float(self.theta_zero)
        return self._obs(sensors, latency, 0.0, is_first=True)
      except LINKLOST as e:
        if not self.reconnect:
          raise
        print(f'Robot did not answer the reset, waiting for it again: {e}')
        self._drop()
        time.sleep(1.0)

  def _recover(self, sensors, latency):
    """Drive the body back under itself so the next episode can start upright.

    Without this an episode that ended against a bumper is followed by one that
    starts there, trips the termination on its first step, and the run collapses
    into a loop of one-step episodes.
    """
    if not self.recover_gain:
      return sensors, latency
    for i in range(self.recover_max):
      error = float(sensors['theta']) - self.theta_zero
      if abs(error) < self.recover_tol:
        break
      command = self.recover_gain * self.recover_k * error
      command = float(np.clip(
          np.sign(command) * max(abs(command), self.recover_min), -1.0, 1.0))
      self._send(command, command, reset=True)
      sensors, latency = self._recv()
    else:
      print('[robot] recovery timed out at %+.1f deg from upright; the body may '
            'be wedged' % np.degrees(float(sensors['theta']) - self.theta_zero))
    self._send(0.0, 0.0, reset=True)
    return self._recv()

  def _advance(self, action):
    if self.task == 'track' and self._step % self.ref_hold == 0:
      self._ref = self.theta_zero + self._rng.uniform(
          -self.ref_range, self.ref_range)
    left, right = self._decode(action)
    self._send(left, right, reset=False)
    sensors, latency = self._recv()
    self._step += 1
    self._record(latency)
    if self.status_every and self._step % self.status_every == 0:
      ref = ('  ref %+5.1f' % np.degrees(self._ref)) if self.task == 'track' else ''
      print('[robot] step %5d  tilt %+5.1f%s  err %+5.1f  w %+5.2f  v %+6.3f'
            % (self._step, np.degrees(float(sensors['theta'])), ref,
               np.degrees(float(sensors['theta']) - (
                   self._ref if self.task == 'track' else self.theta_zero)),
               float(sensors['angular_velocity']),
               0.5 * (float(sensors['wheel_speed_l'])
                      + float(sensors['wheel_speed_r'])) * self.speed_scale),
            flush=True)
    reward, terminal = self._evaluate(sensors)
    self._done = terminal or self._step >= self.length
    return self._obs(
        sensors, latency, reward, is_last=self._done, is_terminal=terminal)

  def close(self):
    for handle in (self._file, self._sock, self._listener):
      try:
        handle and handle.close()
      except OSError:
        pass
    self._file = self._sock = self._listener = None

  def _decode(self, action):
    if self.discrete:
      index = int(action['motion'])
      return MOTIONS[index][1], MOTIONS[index][2]
    left, right = np.clip(np.asarray(action['wheels'], np.float32), -1, 1)
    return float(left), float(right)

  def _evaluate(self, sensors):
    theta = float(sensors['theta'])
    speed_l = float(sensors['wheel_speed_l']) * self.speed_scale
    speed_r = float(sensors['wheel_speed_r']) * self.speed_scale
    # Measured from the setpoint: upright is theta_zero, not zero, so an
    # absolute threshold would trip asymmetrically.
    fallen = (
        bool(self.fall_angle)
        and abs(theta - self.theta_zero) > self.fall_angle)
    if self.task == 'drive':
      forward = 0.5 * (speed_l + speed_r)
      reward = forward - self.spin_penalty * abs(speed_l - speed_r)
    elif self.task == 'track':
      # Attitude control: the inner loop of balancing, which is the part a
      # statically stable rig can still pose. The body self-centres at
      # theta_zero, so holding any other angle takes sustained wheel
      # acceleration -- standing still scores badly as soon as the reference
      # moves away from the equilibrium.
      err = theta - self._ref
      reach = self.theta_hi if err > 0 else abs(self.theta_lo)
      linear = 1.0 - min(1.0, abs(err) / max(reach, 1e-6))
      bonus = np.exp(-((err / self.theta_sigma) ** 2))
      rate = float(sensors['angular_velocity'])
      reward = (
          0.5 * linear + 0.5 * bonus
          - self.rate_penalty * abs(rate)
          - self.drift_penalty * abs(0.5 * (speed_l + speed_r)))
    elif self.task == 'balance':
      # cos(theta) is second-order flat at upright, so on a rig whose tilt only
      # spans a few degrees it delivers almost no gradient. Instead combine a
      # linear term, which has slope everywhere in the reachable range and so
      # still points home from against a bumper, with a narrow bonus that pays
      # for precision near the setpoint.
      #
      # The setpoint is theta_zero, not zero: the IMU reads the phone's mount
      # angle, so true upright sits wherever the body balances, which no
      # calibration here establishes. Estimate it as the midpoint of the tilt
      # range the robot actually reaches and set it per rig.
      offset = theta - self.theta_zero
      reach = self.theta_hi if offset > 0 else abs(self.theta_lo)
      linear = 1.0 - min(1.0, abs(offset) / max(reach, 1e-6))
      bonus = np.exp(-((offset / self.theta_sigma) ** 2))
      rate = float(sensors['angular_velocity'])
      # Without this a constant forward acceleration holds a constant tilt
      # forever, which scores perfectly while driving off the bench.
      drift = abs(0.5 * (speed_l + speed_r))
      reward = (
          0.5 * linear + 0.5 * bonus
          - self.rate_penalty * abs(rate)
          - self.drift_penalty * drift)
    else:
      reward = float(sensors.get('reward', 0.0))
    return float(reward), fallen

  def _obs(self, sensors, latency, reward, is_first=False, is_last=False,
           is_terminal=False):
    theta = float(sensors['theta'])
    return dict(
        wheels=np.array([
            sensors['wheel_speed_l'], sensors['wheel_speed_r']], np.float32),
        orientation=np.array([
            np.sin(theta), np.cos(theta), sensors['angular_velocity']],
            np.float32),
        battery=np.array([sensors['battery_voltage']], np.float32),
        target=np.array([
            self._ref - self.theta_zero, theta - self._ref], np.float32),
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        **{
            'log/theta_deg': np.float32(np.degrees(theta)),
            'log/distance_l': np.float32(sensors.get('wheel_distance_l', 0.0)),
            'log/distance_r': np.float32(sensors.get('wheel_distance_r', 0.0)),
            'log/theta_ref_deg': np.float32(np.degrees(self._ref)),
            'log/track_err_deg': np.float32(np.degrees(theta - self._ref)),
            'log/latency_ms': np.float32(latency * 1e3),
            'log/phone_wait_ms': np.float32(self._timing[0]),
            'log/phone_work_ms': np.float32(self._timing[1]),
            'log/imu_age_ms': np.float32(self._timing[2]),
            'log/imu_stale_ms': np.float32(self._timing[3]),
        },
    )

  # Wire format, both directions: a 4-byte big-endian header length, a UTF-8
  # JSON header, then `blob_len` bytes of binary payload. The blob is unused
  # while the observation is proprio only; it is where camera frames go.

  def _connect(self):
    if self._sock is not None:
      return
    if self._listener is None:
      self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      self._listener.bind((self.host, self.port))
      self._listener.listen(1)
    print(f'Waiting for the robot to connect on {self.host}:{self.port}')
    self._sock, address = self._listener.accept()
    self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    self._sock.settimeout(self.timeout)
    self._file = self._sock.makefile('rb')
    hello, _ = self._read()
    if hello.get('type') != 'hello' or hello.get('protocol') != PROTOCOL:
      raise ValueError(f'Unexpected handshake {hello!r}')
    print(f'Robot connected from {address[0]}:{address[1]}: {hello}')
    self._write(dict(type='hello', protocol=PROTOCOL, discrete=self.discrete))

  def _record(self, latency):
    if not self._timing_path:
      return
    if self._timing_file is None:
      self._timing_file = open(self._timing_path, 'a', buffering=1)
      self._timing_file.write('t,latency_ms,wait_ms,work_ms\n')
    self._timing_file.write('%.6f,%.3f,%.3f,%.3f\n' % (
        time.time(), latency * 1e3, self._timing[0], self._timing[1]))

  def _drop(self):
    """Close the client socket but keep listening, so the robot can return."""
    for handle in (self._file, self._sock):
      try:
        handle and handle.close()
      except OSError:
        pass
    self._file = self._sock = None

  def _send(self, left, right, reset):
    self._sent = time.time()
    self._write(dict(
        type='act', left=float(left), right=float(right), reset=bool(reset)))

  def _recv(self):
    header, _ = self._read()
    if header.get('type') != 'obs':
      raise ValueError(f'Expected an obs message, got {header!r}')
    self._last = header['sensors']
    self._timing = (
        float(header.get('wait_ms', 0.0)), float(header.get('work_ms', 0.0)),
        float(header.get('imu_age_ms', 0.0)),
        float(header.get('imu_stale_ms', 0.0)))
    return self._last, time.time() - self._sent

  def _write(self, header, blob=b''):
    header = dict(header, blob_len=len(blob))
    payload = json.dumps(header).encode('utf-8')
    self._sock.sendall(struct.pack('>I', len(payload)) + payload + blob)

  def _read(self):
    length, = struct.unpack('>I', self._readexactly(4))
    header = json.loads(self._readexactly(length).decode('utf-8'))
    return header, self._readexactly(header.get('blob_len', 0))

  def _readexactly(self, amount):
    if not amount:
      return b''
    data = self._file.read(amount)
    if data is None or len(data) < amount:
      raise ConnectionError(
          f'Robot closed the connection after {len(data or b"")}/{amount} '
          f'bytes; check that the phone app is still running.')
    return data
