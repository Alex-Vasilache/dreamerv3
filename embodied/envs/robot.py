import collections
import io
import json
import os
import pickle
import queue
import socket
import struct
import threading
import time

import elements
import embodied
import numpy as np

PROTOCOL = 3

# Anything that means the link went away rather than that the code is
# wrong: socket errors and timeouts (both OSError), plus a truncated or
# garbled frame from a half-closed connection.
LINKLOST = (OSError, ValueError, struct.error)

# Weight pushes go out in slices this size, so a control frame queued behind a
# push waits for one slice, not for megabytes. 32 KB is ~10 ms on the lab WiFi.
WEIGHT_CHUNK = 32 * 1024


class _Link:
  """One phone connection, read and written by threads of its own.

  The env thread used to do its own socket I/O, so anything that stalled it --
  the ~0.9 s weight export and sendall, the actor applying new weights, a GC --
  left the phone's frames queued in the kernel, and the pipelined `_recv` then
  threw away all but the newest. Measured on e1226 (2026-10-06): one stall of
  40-57 observations every ~30 s, which is what took `fps/policy` from 50 to 32.

  Here the reader drains the socket the moment bytes arrive, into an unbounded
  queue, so a stall on the env thread delays observations but never loses
  them. The writer sends control frames first and weight slices in between, so
  a push never holds the env thread or a reset behind it.
  """

  def __init__(self, sock, timeout):
    self.sock = sock
    self.sock.settimeout(timeout)
    self.frames = queue.Queue()
    self.error = None
    self._closed = False
    self._cond = threading.Condition()
    self._ctrl = collections.deque()
    self._job = None   # [header, blob, offset, chunk, on_done]
    self._reader = threading.Thread(
        target=self._read_loop, name='robot_reader', daemon=True)
    self._writer = threading.Thread(
        target=self._write_loop, name='robot_writer', daemon=True)
    self._reader.start()
    self._writer.start()

  @property
  def pushing(self):
    return self._job is not None

  def get(self, timeout):
    """The oldest frame not yet taken, waiting up to `timeout` for one."""
    try:
      frame = self.frames.get(timeout=timeout)
    except queue.Empty:
      raise TimeoutError(
          f'No frame from the robot in {timeout:.0f}s') from None
    if isinstance(frame, BaseException):
      self.frames.put(frame)  # every later call must fail the same way
      raise frame
    return frame

  def get_nowait(self):
    try:
      frame = self.frames.get_nowait()
    except queue.Empty:
      return None
    if isinstance(frame, BaseException):
      self.frames.put(frame)
      raise frame
    return frame

  def send(self, header, blob=b''):
    """Queue a control frame; it goes out ahead of any weight slice."""
    if self.error is not None:
      raise self.error
    with self._cond:
      self._ctrl.append(_frame(header, blob))
      self._cond.notify()

  def push(self, header, blob, chunked, on_done):
    """Start sending a weight blob in the background. False if one is going."""
    with self._cond:
      if self._job is not None or self.error is not None:
        return False
      chunk = WEIGHT_CHUNK if chunked else max(len(blob), 1)
      self._job = [header, blob, 0, chunk, on_done]
      self._cond.notify()
    return True

  def close(self):
    with self._cond:
      self._closed = True
      self._cond.notify()
    try:
      self.sock.shutdown(socket.SHUT_RDWR)
    except OSError:
      pass
    try:
      self.sock.close()
    except OSError:
      pass

  def _fail(self, error):
    if self.error is None:
      self.error = error
    self.frames.put(error)
    with self._cond:
      self._closed = True
      self._cond.notify()

  def _read_loop(self):
    buffer = bytearray()
    try:
      while not self._closed:
        chunk = self.sock.recv(1 << 16)
        if not chunk:
          raise ConnectionError(
              'Robot closed the connection; check that the phone app is '
              'still running.')
        buffer.extend(chunk)
        while True:
          frame = _parse(buffer)
          if frame is None:
            break
          self.frames.put(frame)
    except BaseException as e:  # noqa: BLE001 -- handed to the env thread
      self._fail(e if isinstance(e, LINKLOST) else ConnectionError(repr(e)))

  def _write_loop(self):
    try:
      while True:
        with self._cond:
          while not self._closed and not self._ctrl and self._job is None:
            self._cond.wait()
          if self._closed:
            return
          ctrl = list(self._ctrl)
          self._ctrl.clear()
          job = self._job
        for data in ctrl:
          self.sock.sendall(data)
        if job is None:
          continue
        header, blob, offset, chunk, on_done = job
        piece = blob[offset:offset + chunk]
        if chunk >= len(blob):
          self.sock.sendall(_frame(header, blob))
        else:
          self.sock.sendall(_frame(dict(
              header, type='wchunk', offset=offset, total=len(blob)), piece))
        job[2] = offset + len(piece)
        if job[2] >= len(blob):
          with self._cond:
            self._job = None
          on_done(header)
    except BaseException as e:  # noqa: BLE001
      self._fail(e if isinstance(e, LINKLOST) else ConnectionError(repr(e)))


def _frame(header, blob=b''):
  """Wire format, both directions: a 4-byte big-endian header length, a UTF-8
  JSON header, then `blob_len` bytes of binary payload."""
  payload = json.dumps(dict(header, blob_len=len(blob))).encode('utf-8')
  return struct.pack('>I', len(payload)) + payload + bytes(blob)


def _parse(buffer):
  """Pull one frame out of `buffer`, leaving a partial one in place."""
  if len(buffer) < 4:
    return None
  length, = struct.unpack('>I', buffer[:4])
  if len(buffer) < 4 + length:
    return None
  header = json.loads(bytes(buffer[4:4 + length]).decode('utf-8'))
  total = 4 + length + header.get('blob_len', 0)
  if len(buffer) < total:
    return None
  blob = bytes(buffer[4 + length:total])
  del buffer[:total]
  return header, blob

# Same wheel PWM table as apps/basicAssembler on the phone, so a discrete
# policy trained here means the same thing if it is later moved on-device.
MOTIONS = (
    ('stop', 0.0, 0.0),
    ('forward', 0.75, 0.75),
    ('backward', -0.75, -0.75),
    ('left', -1.0, 1.0),
    ('right', 1.0, -1.0),
)

# Symmetric drive only: both wheels always get the same value, so the robot can
# translate but never turn. Differential turning is what tears the wheels off
# this rig, and it does nothing for pitch balance anyway -- the pendulum axis is
# unaffected by yaw. Spending the freed action budget on speed granularity
# instead is strictly better for balancing. The slow step is 0.5 rather than
# something smaller because 0.30 PWM was measured to produce no wheel motion at
# all on this hardware.
MOTIONS_SYMMETRIC = (
    ('stop', 0.0, 0.0),
    ('forward_slow', 0.5, 0.5),
    ('forward_fast', 1.0, 1.0),
    ('backward_slow', -0.5, -0.5),
    ('backward_fast', -1.0, -1.0),
)


def _load_policy_pickle(blob):
  """Unpickle a learner policy file written by whichever numpy is on that side.

  numpy renamed `numpy.core` to `numpy._core` in 2.0, and this file crosses
  machines: the learner writes it on Saion, the actor reads it here. Map in
  whichever direction the local install needs rather than assuming one.
  """

  class Unpickler(pickle.Unpickler):

    def find_class(self, module, name):
      for src, dst in (('numpy.core', 'numpy._core'),
                       ('numpy._core', 'numpy.core')):
        if module.startswith(src):
          try:
            return super().find_class(module, name)
          except (ModuleNotFoundError, AttributeError):
            return super().find_class(module.replace(src, dst, 1), name)
      return super().find_class(module, name)

  return Unpickler(io.BytesIO(blob)).load()


class SmartphoneRobot(embodied.Env):
  """DreamerV3 environment backed by the OIST smartphone robot.

  The policy runs here; the phone runs a thin client (see the `dreamerBridge`
  app in the smartphone-robot-android repo) that connects in, applies each
  action to the wheels, waits out the control period, then ships proprio sensor
  readings back. The phone therefore owns the control clock and `step` blocks
  until the next reading arrives.

  Two timing modes, chosen by `pipeline` and announced in the handshake:

  * Lock-step (`pipeline=False`, the protocol-1 behaviour). The phone blocks on
    an action before every tick, so a step costs one control period *plus* the
    round trip and whatever the policy took. Measured on this rig that capped
    the loop at 9.7Hz against a phone ticking at 50Hz: the phone spent a median
    36ms, and a p90 of 260ms, blocked on us.

  * Pipelined (`pipeline=True`, the default). The phone free-runs on its own
    clock, applying the most recent action it has received and reporting every
    tick regardless. The round trip then hides inside the control period and
    the rate is the phone's alone. The cost is a fixed one-tick delay between
    an action and the observation that reflects it -- consistent, so the world
    model can learn it -- and the need to keep up on average: observations we
    are too slow to consume queue up, and `_recv` drops all but the newest,
    counting them as `log/dropped`.

  The listening socket is opened lazily on the first `step` rather than in the
  constructor, because `main.make_agent` builds one throwaway env just to read
  the spaces and would otherwise leave the port bound.
  """

  def __init__(
      self, task='drive', host='0.0.0.0', port=3000, length=200,
      discrete=True, timeout=20.0, speed_scale=1e-3, fall_angle=0.0,
      spin_penalty=0.1, rate_penalty=0.05, theta_zero=0.0,
      theta_lo=-0.122, theta_hi=0.182, theta_sigma=0.05, drift_penalty=0.1,
      drift_clip=1.0, wheel_penalty=0.0, action_rate_penalty=0.0,
      command_scale=1.0, symmetric=True,
      obs_theta_scale=0.18, obs_rate_scale=3.0, obs_wheel_scale=3.3e-4,
      obs_clip=3.0,
      ref_range=0.09, ref_hold=100, seed=0, status_every=0,
      recover_gain=0.0, recover_k=4.0, recover_tol=0.05, recover_max=250,
      recover_min=0.6, command_speed=1500.0, command_turn=1000.0,
      command_sigma=0.3, command_mode='auto', command_hold_min=2.0,
      command_hold_max=5.0, command_p_zero=0.3, command_p_axis=0.4,
      reconnect=True, pipeline=True, onboard=False, config=None,
      policy_dir=None, weights_every=0.0, weights_poll=0.25, logdir=None,
      pace=None):
    assert task in ('drive', 'balance', 'track', 'command', 'none'), task
    self.task = task
    self.host = host
    self.port = int(port)
    # Offer the phone the chance to run the policy itself. It only takes effect
    # if the phone also has weights loaded, so this is safe to leave on: a
    # phone without a policy simply keeps the old behaviour.
    self.onboard = bool(onboard)
    self._onboard = False        # negotiated per connection
    self._onboard_act = None     # the action the phone actually applied
    self._config = config
    self._policy_dir = elements.Path(policy_dir) if policy_dir else None
    # Pushes run on a thread of their own (`_push_loop`), so they can go as
    # often as the learner publishes: `weights_every` is only a floor on the
    # time between two push starts, 0 meaning back to back, and the learner's
    # `online_publish_every` is what actually sets the cadence.
    self._weights_every = float(weights_every)
    self._weights_poll = float(weights_poll)
    self._weights_stamp = None   # the policy the phone holds, as far as we know
    self._push_started = 0.0
    self._pusher = None
    self._closing = threading.Event()
    self._phone_stamp = None     # policy_stamp the phone reported last
    # 'serial' or 'clock', or None to take the phone's default. The phone's
    # RP2040 applies one command per ~83 ms and reads USB only in between, so
    # 'serial' -- one decision per reply -- is what gets every action applied.
    self.pace = pace
    self._seq = None  # the phone's observation counter, echoed in each action
    self.length = int(length)
    self.discrete = bool(discrete)
    self.timeout = float(timeout)
    self.pipeline = bool(pipeline)
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
    # Instantaneous wheel speed is spiky: measured p50 689, p90 3406 but
    # p99 37965. Without a clip the penalty tracks sensor spikes rather
    # than sustained drift -- at scale 1e-4 the p99 alone would cost 0.38
    # of a reward capped at 1.0. Clipping bounds it at drift_penalty.
    self.drift_clip = float(drift_clip)
    # Effort cost on each wheel's own speed. drift only sees the net forward
    # motion, so wheels turning in opposite directions, or oscillating fast
    # around zero mean, cost nothing there. Clipped like drift per wheel.
    self.wheel_penalty = float(wheel_penalty)
    self.symmetric = bool(symmetric)
    # Cost on the change in each wheel's command between steps, mean over the
    # wheels, in the policy's [-1, 1] units. At 50 Hz an exploring policy
    # chatters the wheels every 20 ms and nothing else in the reward minds.
    # A penalty, not a low-pass filter, so no delay is added to the loop.
    self.action_rate_penalty = float(action_rate_penalty)
    self._prev_act = np.zeros(2, np.float32)
    self._sent_act = (0.0, 0.0)
    # The phone multiplies every wheel command by this before driving, so the
    # policy keeps its whole [-1, 1] range while the motors stay off full
    # power, the only regime where a driver was seen to cut out under load.
    self.command_scale = float(command_scale)
    # Observation normalisation. Raw units put wheel speed ~70x above tilt
    # once symlog is applied (std 6.4 against 0.09), so the encoder saw
    # encoder counts and barely saw the angle the reward depends on.
    # Each feature is scaled to roughly unit range instead.
    self.obs_theta_scale = float(obs_theta_scale)
    self.obs_rate_scale = float(obs_rate_scale)
    self.obs_wheel_scale = float(obs_wheel_scale)
    self.obs_clip = float(obs_clip)
    self.motions = MOTIONS_SYMMETRIC if symmetric else MOTIONS
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
    # Task 'command': balance while driving at a commanded forward speed and
    # turn rate, both in [-1, 1]. The phone owns the command -- it samples
    # random ones while training and takes a joystick's from its web page --
    # and reports the one its policy saw with every observation, so the policy
    # input, the recorded observation and the reward here always agree.
    # command_speed is the wheel speed (encoder units, as on the phone screen)
    # that forward 1.0 asks of both wheels; command_turn is what turn 1.0 adds
    # to the left wheel and takes from the right. A negative command_turn
    # flips which way is right.
    self.command_speed = float(command_speed)
    self.command_turn = float(command_turn)
    self.command_sigma = float(command_sigma)
    # How the phone picks commands nobody is steering ('auto': hold a random
    # one for hold_min..hold_max s, zero with p_zero, one axis only with
    # p_axis; 'manual': zero, i.e. balance in place). Sent in the handshake.
    self.command_sampler = dict(
        mode=str(command_mode), hold_min=float(command_hold_min),
        hold_max=float(command_hold_max), p_zero=float(command_p_zero),
        p_axis=float(command_p_axis))
    self._cmd = (0.0, 0.0)
    self._cmd_src = 0.0
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
    self._link = None
    self._chunked = False        # the phone reassembles sliced weight pushes
    self._dropped = 0
    self._backlog = 0
    self._step = 0
    self._done = True
    self._sent = 0.0
    # Mirrored back to the phone with the next action, so its screen can show
    # what the state it just reported was worth. The reward is computed here,
    # not on the phone, so this is the only way the phone can know it.
    self._reward = 0.0
    self._timing = (0.0, 0.0, 0.0, 0.0)
    self._last = dict(
        wheel_speed_l=0.0, wheel_speed_r=0.0, wheel_distance_l=0.0,
        wheel_distance_r=0.0, theta=0.0, angular_velocity=0.0,
        battery_voltage=0.0)

  @property
  def obs_space(self):
    return {
        'wheels': elements.Space(np.float32, (2,)),
        # [tilt, rate], both normalised. cos(theta) is dropped: over the +/-10
        # deg this rig can reach it spans 0.988 to 1.000, a std of 0.002 after
        # symlog, so it is a constant carrying no information.
        'orientation': elements.Space(np.float32, (2,)),
        **({'target': elements.Space(np.float32, (2,))}
           if self.task == 'track' else {}),
        # [forward, turn], each in [-1, 1].
        **({'command': elements.Space(np.float32, (2,))}
           if self.task == 'command' else {}),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/theta_deg': elements.Space(np.float32),
        'log/distance_l': elements.Space(np.float32),
        'log/distance_r': elements.Space(np.float32),
        'log/battery_v': elements.Space(np.float32),
        'log/theta_ref_deg': elements.Space(np.float32),
        'log/track_err_deg': elements.Space(np.float32),
        'log/latency_ms': elements.Space(np.float32),
        'log/phone_wait_ms': elements.Space(np.float32),
        'log/phone_work_ms': elements.Space(np.float32),
        'log/imu_age_ms': elements.Space(np.float32),
        'log/imu_stale_ms': elements.Space(np.float32),
        # Observations discarded as stale in pipelined mode; anything but 0
        # means we are not keeping up with the phone's clock. Always 0 in
        # onboard mode, where every observation is a transition the phone
        # really drove and is kept; there `log/backlog` (frames still queued
        # behind this one) is the measure of keeping up.
        'log/dropped': elements.Space(np.float32),
        'log/backlog': elements.Space(np.float32),
        # Onboard: seconds since the learner published the weights the phone
        # acted on for this step. Sawtooths at the publish cadence when the
        # pushes keep up.
        'log/policy_age_s': elements.Space(np.float32),
        **({k: elements.Space(np.float32) for k in (
            'log/cmd_forward', 'log/cmd_turn', 'log/cmd_joystick',
            'log/forward', 'log/turn')} if self.task == 'command' else {}),
        **({'executed/drive': elements.Space(np.float32, (2,), -1.0, 1.0)}
           if self.onboard and not self.discrete else {}),
    }

  @property
  def act_space(self):
    space = {'reset': elements.Space(bool)}
    if self.discrete:
      space['motion'] = elements.Space(np.int32, (), 0, len(self.motions))
    elif self.symmetric:
      # Named 'drive', not 'wheels': the observation already uses that key and
      # CheckSpaces requires the observation and action keys to be disjoint.
      # One value driving both wheels, not two.
      space['drive'] = elements.Space(np.float32, (1,), -1.0, 1.0)
    else:
      space['drive'] = elements.Space(np.float32, (2,), -1.0, 1.0)
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
        self._reward = 0.0
        self._ref = float(self.theta_zero)
        self._prev_act = np.zeros(2, np.float32)
        self._sent_act = (0.0, 0.0)
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
    self._sent_act = (left, right)
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
    if self.action_rate_penalty:
      act = (np.resize(self._onboard_act, 2)
             if self._onboard and self._onboard_act is not None
             else np.array([left, right], np.float32))
      reward -= self.action_rate_penalty * float(np.mean(np.abs(act - self._prev_act)))
      self._prev_act = act
    self._reward = reward
    self._done = terminal or self._step >= self.length
    return self._obs(
        sensors, latency, reward, is_last=self._done, is_terminal=terminal)

  def close(self):
    self._closing.set()
    self._drop()
    try:
      self._listener and self._listener.close()
    except OSError:
      pass
    self._listener = None

  def _decode(self, action):
    if self.discrete:
      index = int(action['motion'])
      return self.motions[index][1], self.motions[index][2]
    drive = np.clip(np.asarray(action['drive'], np.float32), -1, 1)
    if self.symmetric:
      return float(drive[0]), float(drive[0])
    return float(drive[0]), float(drive[1])

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
          - self.drift_penalty * min(
              abs(0.5 * (speed_l + speed_r)), self.drift_clip)
          - self.wheel_penalty * self._wheel_effort(speed_l, speed_r))
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
      drift = min(abs(0.5 * (speed_l + speed_r)), self.drift_clip)
      reward = (
          0.5 * linear + 0.5 * bonus
          - self.rate_penalty * abs(rate)
          - self.drift_penalty * drift
          - self.wheel_penalty * self._wheel_effort(speed_l, speed_r))
    elif self.task == 'command':
      reward = self._command_reward(sensors)
    else:
      reward = float(sensors.get('reward', 0.0))
    return float(reward), fallen

  def _measured(self, sensors):
    """(forward, turn) the wheels are doing, in command units."""
    left = float(sensors['wheel_speed_l'])
    right = float(sensors['wheel_speed_r'])
    forward = 0.5 * (left + right) / self.command_speed
    turn = 0.5 * (left - right) / self.command_turn
    return forward, turn

  def _command_reward(self, sensors):
    """Half balance, half tracking the commanded forward speed and turn.

    The balance half is task 'balance' unchanged. Tracking pays
    exp(-(error / command_sigma)^2) per axis, the kernel legged-robot velocity
    tracking uses (Rudin et al. 2022), so it saturates rather than letting one
    spiky wheel reading dominate. The drift and wheel penalties of 'balance'
    are kept but measured from the wheel speeds the command asks for, so a
    centred stick (command 0, 0) reproduces the balance task's penalties
    exactly. Still at most 1.0 per step.
    """
    theta = float(sensors['theta'])
    offset = theta - self.theta_zero
    reach = self.theta_hi if offset > 0 else abs(self.theta_lo)
    linear = 1.0 - min(1.0, abs(offset) / max(reach, 1e-6))
    bonus = np.exp(-((offset / self.theta_sigma) ** 2))
    cmd_f, cmd_t = self._cmd
    forward, turn = self._measured(sensors)
    track = 0.5 * (np.exp(-(((forward - cmd_f) / self.command_sigma) ** 2))
                   + np.exp(-(((turn - cmd_t) / self.command_sigma) ** 2)))
    want_l = cmd_f * self.command_speed + cmd_t * self.command_turn
    want_r = cmd_f * self.command_speed - cmd_t * self.command_turn
    err_l = (float(sensors['wheel_speed_l']) - want_l) * self.speed_scale
    err_r = (float(sensors['wheel_speed_r']) - want_r) * self.speed_scale
    drift = min(abs(0.5 * (err_l + err_r)), self.drift_clip)
    return float(
        0.25 * linear + 0.25 * bonus + 0.5 * track
        - self.rate_penalty * abs(float(sensors['angular_velocity']))
        - self.drift_penalty * drift
        - self.wheel_penalty * self._wheel_effort(err_l, err_r))

  def _wheel_effort(self, speed_l, speed_r):
    return 0.5 * (
        min(abs(speed_l), self.drift_clip) + min(abs(speed_r), self.drift_clip))

  def _obs(self, sensors, latency, reward, is_first=False, is_last=False,
           is_terminal=False):
    theta = float(sensors['theta'])
    return dict(
        wheels=np.clip(np.array([
            sensors['wheel_speed_l'], sensors['wheel_speed_r']],
            np.float32) * self.obs_wheel_scale, -self.obs_clip, self.obs_clip),
        orientation=np.clip(np.array([
            (theta - self.theta_zero) / self.obs_theta_scale,
            sensors['angular_velocity'] / self.obs_rate_scale],
            np.float32), -self.obs_clip, self.obs_clip),
        **({'target': np.array(
            [self._ref - self.theta_zero, theta - self._ref], np.float32)}
           if self.task == 'track' else {}),
        **({'command': np.array(self._cmd, np.float32)}
           if self.task == 'command' else {}),
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        # Only in onboard mode: the action the phone already applied for this
        # observation. The driver lifts 'executed/' keys over the actor's own
        # action when it builds the transition.
        # Declared whenever onboard is configured, so it must be filled even
        # while a phone without weights is still bootstrapping on ours: then
        # the executed action is the one we sent.
        **({'executed/drive': self._executed()}
           if self.onboard and not self.discrete else {}),
        **{
            'log/theta_deg': np.float32(np.degrees(theta)),
            'log/distance_l': np.float32(sensors.get('wheel_distance_l', 0.0)),
            'log/distance_r': np.float32(sensors.get('wheel_distance_r', 0.0)),
            'log/battery_v': np.float32(sensors['battery_voltage']),
            'log/theta_ref_deg': np.float32(np.degrees(self._ref)),
            'log/track_err_deg': np.float32(np.degrees(theta - self._ref)),
            'log/latency_ms': np.float32(latency * 1e3),
            'log/phone_wait_ms': np.float32(self._timing[0]),
            'log/phone_work_ms': np.float32(self._timing[1]),
            'log/imu_age_ms': np.float32(self._timing[2]),
            'log/imu_stale_ms': np.float32(self._timing[3]),
            'log/dropped': np.float32(self._dropped),
            'log/backlog': np.float32(self._backlog),
            'log/policy_age_s': np.float32(self._policy_age()),
            **(self._command_logs(sensors) if self.task == 'command' else {}),
        },
    )

  def _command_logs(self, sensors):
    forward, turn = self._measured(sensors)
    return {
        'log/cmd_forward': np.float32(self._cmd[0]),
        'log/cmd_turn': np.float32(self._cmd[1]),
        'log/cmd_joystick': np.float32(self._cmd_src),
        'log/forward': np.float32(forward),
        'log/turn': np.float32(turn),
    }

  # The socket itself is owned by a `_Link`, which reads and writes it on
  # threads of its own; see that class for the wire format and why.

  def _connect(self):
    if self._link is not None:
      return
    if self._listener is None:
      self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      self._listener.bind((self.host, self.port))
      self._listener.listen(1)
    print(f'Waiting for the robot to connect on {self.host}:{self.port}')
    sock, address = self._listener.accept()
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    link = _Link(sock, self.timeout)
    try:
      hello, _ = link.get(self.timeout)
      if hello.get('type') != 'hello' or hello.get('protocol') != PROTOCOL:
        raise ValueError(f'Unexpected handshake {hello!r}')
    except BaseException:
      link.close()
      raise
    print(f'Robot connected from {address[0]}:{address[1]}: {hello}')
    # The phone follows whichever mode we ask for, so a fallback to lock-step
    # needs no reinstall.
    self._onboard = bool(self.onboard and hello.get('onboard', False))
    if self._onboard:
      print(f'[robot] phone is running the policy (weights {hello.get("policy_stamp")});'
            ' recording its actions rather than sending our own')
    # A phone that says nothing about it takes a push as one frame, as before.
    self._chunked = bool(hello.get('wchunk', False))
    self._phone_stamp = hello.get('policy_stamp')
    self._weights_stamp = self._phone_stamp
    link.send(dict(
        type='hello', protocol=PROTOCOL, discrete=self.discrete,
        pipeline=self.pipeline, onboard=self._onboard,
        cmd_scale=self.command_scale,
        **({'command': self.command_sampler}
           if self.task == 'command' else {}),
        # The phone keeps this run's latest weights under `run`, with these
        # settings beside them, so it can replay the policy later on its own.
        run=self._run_name(), settings=self._settings(),
        **({'pace': self.pace} if self.pace else {})))
    self._link = link
    # Note `self.onboard`, not `self._onboard`: a phone that came up without
    # a policy negotiates onboard=False, and pushing to it anyway is exactly
    # how it bootstraps into running its own.
    if self.onboard and self._pusher is None:
      self._pusher = threading.Thread(
          target=self._push_loop, name='robot_pusher', daemon=True)
      self._pusher.start()

  def _run_name(self):
    """The Slurm job name (e1293_robot_25hz) or, outside Slurm, the run dir."""
    name = os.environ.get('SLURM_JOB_NAME')
    if name:
      return name
    if self._policy_dir is not None:
      logdir = self._policy_dir.parent   # <logdir>/online_shared
      # Saion runs live in <run dir>/logdir; local ones are the logdir itself.
      return logdir.parent.name if logdir.name == 'logdir' else logdir.name
    return 'robot'

  def _settings(self):
    """What a player needs to pace and score steps the way training did."""
    return dict(
        task=self.task, length=self.length, theta_zero=self.theta_zero,
        theta_lo=self.theta_lo, theta_hi=self.theta_hi,
        theta_sigma=self.theta_sigma, speed_scale=self.speed_scale,
        drift_penalty=self.drift_penalty, drift_clip=self.drift_clip,
        wheel_penalty=self.wheel_penalty, rate_penalty=self.rate_penalty,
        action_rate_penalty=self.action_rate_penalty,
        command_scale=self.command_scale,
        command_speed=self.command_speed, command_turn=self.command_turn,
        command_sigma=self.command_sigma)

  def _push_loop(self):
    """Send the phone each new policy as soon as the learner publishes it.

    Runs beside the env thread: reading the learner's pickle and building the
    .npz cost ~0.9 s on Saion, and the bytes take about as long again on the
    WiFi, none of which the env thread or the phone's control loop may wait
    for. One push at a time; a policy published while one is in flight is
    skipped in favour of whatever is newest when it lands.
    """
    while not self._closing.wait(self._weights_poll):
      link = self._link
      if link is None or link.pushing or link.error is not None:
        continue
      if time.time() - self._push_started < self._weights_every:
        continue
      try:
        self._push_newest(link)
      except Exception as e:  # noqa: BLE001
        # A failed export must never take the robot down; it just means the
        # phone keeps acting on the weights it already has, which is the whole
        # point of it holding them.
        print(f'[robot] could not push weights: {e}', flush=True)
        time.sleep(5.0)

  def _push_newest(self, link):
    if self._config is None or self._policy_dir is None:
      return
    latest = self._policy_dir / 'policy' / 'latest'
    if not latest.exists():
      return
    stamp = latest.read().strip()
    if not stamp or stamp == self._weights_stamp:
      return
    path = self._policy_dir / 'policy' / f'policy_{stamp}.pkl'
    if not path.exists():
      return
    started = time.time()
    from dreamerv3.deploy import export as exportlib
    data = _load_policy_pickle(path.read_bytes())
    params = data['params'] if isinstance(data, dict) and 'params' in data \
        else data
    blob, _ = exportlib.pack(self._config, params)
    packed = time.time()

    def done(header):
      self._weights_stamp = header['stamp']
      now = time.time()
      print(f'[robot] pushed policy {stamp} to the phone '
            f'({len(blob) / 1e6:.1f} MB, pack {packed - started:.2f}s, '
            f'send {now - packed:.2f}s, age {now - int(stamp) / 1e9:.1f}s)',
            flush=True)

    self._push_started = started
    link.push(dict(type='weights', stamp=stamp), blob, self._chunked, done)

  def _executed(self):
    if self._onboard and self._onboard_act is not None:
      act = self._onboard_act
    else:
      act = self._sent_act
    size = self.act_space['drive'].shape[0]
    return np.resize(np.asarray(act, np.float32), size)

  def _policy_age(self):
    if not self._onboard:
      return 0.0
    try:
      return time.time() - int(self._phone_stamp) / 1e9
    except (TypeError, ValueError):
      return -1.0  # the phone's own boot weights, or no stamp at all

  def _record(self, latency):
    if not self._timing_path:
      return
    if self._timing_file is None:
      self._timing_file = open(self._timing_path, 'a', buffering=1)
      self._timing_file.write('t,latency_ms,wait_ms,work_ms,dropped,backlog\n')
    self._timing_file.write('%.6f,%.3f,%.3f,%.3f,%d,%d\n' % (
        time.time(), latency * 1e3, self._timing[0], self._timing[1],
        self._dropped, self._backlog))

  def _drop(self):
    """Close the client socket but keep listening, so the robot can return."""
    link, self._link = self._link, None
    if link is not None:
      link.close()

  def _send(self, left, right, reset):
    self._sent = time.time()
    if self._link is None:
      raise ConnectionError('No robot connected')
    if self._onboard:
      # The phone has already decided and driven. All that is left to send is
      # the episode boundary and the reward for the step it just reported --
      # neither is latency critical, which is exactly why they can stay here
      # while the action moved to the phone.
      self._link.send(dict(
          type='ctrl', reset=bool(reset), reward=float(self._reward),
          seq=self._seq))
      return
    # `seq` lets a serial-paced phone tell the answer to its newest
    # observation from a late answer to the previous one.
    self._link.send(dict(
        type='act', left=float(left), right=float(right), reset=bool(reset),
        reward=float(self._reward), seq=self._seq))

  def _recv(self):
    if self._link is None:
      raise ConnectionError('No robot connected')
    header, _ = self._link.get(self.timeout)
    self._dropped = 0
    if self.pipeline and not self._onboard:
      # The phone reports every tick whether or not we asked, so anything still
      # queued behind this frame is staler than what is on the wire now. Acting
      # on a stale reading is worse than skipping it, so keep only the newest
      # and count the rest.
      while True:
        newer = self._link.get_nowait()
        if newer is None:
          break
        header = newer[0]
        self._dropped += 1
    # Onboard, nothing is stale: each frame is a step the phone already drove,
    # with the action it drove, so every one is a transition worth keeping. A
    # stall here only delays them.
    self._backlog = self._link.frames.qsize()
    if header.get('type') != 'obs':
      raise ValueError(f'Expected an obs message, got {header!r}')
    self._last = header['sensors']
    self._seq = header.get('seq')
    # The command the phone's policy saw for this observation. A phone that
    # predates commands sends none, which is the same as a centred stick.
    cmd = header.get('cmd') or (0.0, 0.0)
    self._cmd = (float(np.clip(cmd[0], -1, 1)), float(np.clip(cmd[1], -1, 1)))
    self._cmd_src = float(header.get('cmd_src') == 'joystick')
    if self._onboard:
      act = header.get('act')
      if act is None:
        raise ValueError(
            'Onboard mode negotiated but the phone sent no action; refusing '
            'to record a transition whose action we would have to invent')
      self._onboard_act = np.asarray(act, np.float32)
      self._phone_stamp = header.get('policy_stamp', self._phone_stamp)
    self._timing = (
        float(header.get('wait_ms', 0.0)), float(header.get('work_ms', 0.0)),
        float(header.get('imu_age_ms', 0.0)),
        float(header.get('imu_stale_ms', 0.0)))
    return self._last, time.time() - self._sent
