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

# Learner metrics forwarded to the phone for its web dashboard (/dash), as
# (name on the page, metrics.jsonl keys summed). The world-model loss is its
# prediction terms; the decoder terms are named after the observation keys.
STATS = (
    ('wm_loss', ('train/loss/dyn', 'train/loss/rep', 'train/loss/rew',
                 'train/loss/con', 'train/loss/wheels',
                 'train/loss/orientation', 'train/loss/command')),
    ('policy_loss', ('train/loss/policy',)),
    ('value_loss', ('train/loss/value',)),
    ('entropy', ('train/ent/drive',)),
    ('adv_mag', ('train/adv_mag',)),
    ('ret', ('train/ret',)),
    ('fps_train', ('fps/train',)),
    ('updates', ('train/opt/updates',)),
)
STATS_EVERY = 3.0

# The env settings a saved policy carries in its <name>.json for the phone's
# Dreamer Player, beside task, hz and length (`_settings`); each is both an
# attribute here and an env.robot config key, so tools/robot/from_sim.sh
# writes the same file from a sim run's config.
PLAYER_SETTINGS = (
    'theta_zero', 'theta_lo', 'theta_hi', 'theta_sigma', 'speed_scale',
    'drift_penalty', 'drift_clip', 'wheel_penalty', 'yaw_penalty',
    'still_speed', 'rate_penalty', 'action_rate_penalty', 'command_scale',
    'command_bias', 'friction_comp', 'bumper_lo', 'bumper_hi',
    'bumper_margin', 'bumper_penalty', 'command_speed', 'command_turn',
    'command_sigma', 'command_reward', 'command_track_weight',
    'command_balance_weight', 'forward_sign', 'turn_source', 'yaw_rate_max')

def track_position(pos, prev, sensors, tau, hz, jump):
  """One step of the leaky wheel-distance sum; the phone runs a copy
  (deploy/policy_runner.py), so change both."""
  dist = np.array([float(sensors.get('wheel_distance_l', 0.0)),
                   float(sensors.get('wheel_distance_r', 0.0))])
  step = np.zeros(2) if prev is None else dist - np.asarray(prev)
  step[np.abs(step) > jump] = 0.0
  return pos * np.exp(-1.0 / (tau * hz)) + step


def write_event(shared, event):
  """An arousal event for the learner (embodied/run/online.py
  ArousalBoost): <shared>/emotion_event as {n, t, intensity}."""
  path = os.path.join(str(shared), 'emotion_event')
  tmp = path + '.tmp'
  with open(tmp, 'w') as f:
    json.dump(dict(n=event.get('n'), t=event.get('t', time.time()),
                   intensity=float(event.get('intensity', 1.0))), f)
  os.replace(tmp, path)


def pain_signal(theta, body, act=None, chosen=None):
  """Pain from the body, 0 (none) to 1+: (pain, front, back); a copy of
  deploy/policy_runner.py's, so change both. `body` is dict(theta_zero, lo,
  hi, zone): the bumpers' tilts (rad) and how far before them closeness
  starts to hurt; a reflex that overrode the policy (`act` the wheels got,
  `chosen` the policy's) hurts as much as its correction's size."""
  rel = float(theta) - float(body['theta_zero'])
  zone = max(float(body['zone']), 1e-6)
  front = min(1.0, max(0.0, (rel - (float(body['hi']) - zone)) / zone))
  back = min(1.0, max(0.0, ((float(body['lo']) + zone) - rel) / zone))
  pain = max(front, back)
  if act is not None and chosen is not None:
    pain += min(1.0, float(np.mean(np.abs(
        np.asarray(act, np.float32) - np.asarray(chosen, np.float32)))))
  return pain, front, back


def travel_limit(act, pos, lo, hi, state=None):
  """Scale wheel commands that push the robot further the way it has already
  travelled: none below `lo` of leaky travel (`pos`, the mean of the two
  wheels' leaky distances, phone mm over ~pos_tau s), linearly to 0 at `hi`.
  Reaching `hi` latches: pushing on that way stays blocked until the travel
  is back under `lo` (state, a 1-list holding the blocked direction, keeps
  the latch between calls); a proportional fade alone let a policy that kept
  pushing settle into a steady crawl at hi (e30111, 0.08 m/s). Commands that
  brake or reverse pass whole, so short full-speed catches stay full while
  sustained driving stops. hi <= 0: off. A wheel command a turns its wheel
  backwards (travel -a), so it pushes on when -a has the sign of pos. The
  trainer (embodied/envs/robot.py) and the phone (deploy/policy_runner.py)
  each keep a copy, so change both."""
  if hi <= 0:
    return [float(a) for a in act]
  state = state if state is not None else [0.0]
  side = float(np.sign(pos))
  if state[0] and (abs(pos) < lo or side != state[0]):
    state[0] = 0.0
  if abs(pos) >= hi:
    state[0] = side
  if state[0]:
    k = 0.0
  elif abs(pos) <= lo:
    return [float(a) for a in act]
  else:
    k = min(1.0, max(0.0, (hi - abs(pos)) / max(hi - lo, 1e-6)))
  return [float(a) * k if -float(a) * pos > 0 else float(a) for a in act]


class Fatigue:
  """Fades a wheel command the policy holds: each wheel's output is scaled by
  1 - held/steps, so a command kept within `tol` of where it started, in the
  same direction and beyond `dead`, reaches 0 after `steps` control steps.
  Any larger change (or a reversal) restarts that wheel at full strength.
  Short bursts to catch a fall pass nearly whole; steady driving cannot last.
  steps 0: off. The trainer (embodied/envs/robot.py) and the phone
  (deploy/policy_runner.py) each keep a copy, so change both."""

  def __init__(self, steps=0, tol=0.15, dead=0.05):
    self.set(steps, tol, dead)

  def set(self, steps, tol=0.15, dead=0.05):
    self.steps, self.tol, self.dead = int(steps), float(tol), float(dead)
    self.reset()

  def reset(self):
    self.ref = [None, None]
    self.held = [0, 0]

  def __call__(self, act):
    if self.steps <= 0:
      return [float(a) for a in act]
    out = []
    for i, a in enumerate(act[:2]):
      a, ref = float(a), self.ref[i]
      if (ref is not None and abs(a) > self.dead and a * ref > 0
          and abs(a - ref) <= self.tol):
        self.held[i] += 1
      else:
        self.ref[i], self.held[i] = a, 0
      out.append(a * max(0.0, 1.0 - self.held[i] / self.steps))
    return out


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

  def __init__(self, sock, timeout, on_restart=None, on_pause=None, rate=0.0):
    self.sock = sock
    # Weight slices per second as bytes, 0 for as fast as the socket takes
    # them. Over a slow relay the socket takes them far faster than the link
    # drains: megabytes then sit in the relay's buffers ahead of every reply,
    # and the phone sees rewards seconds late. Pacing below the link's rate
    # keeps those buffers empty.
    self._rate = float(rate)
    self._next_slice = 0.0
    # Called on the reader thread when the phone asks for a fresh run, or to
    # pause or resume the robot or the training (with the frame); such a
    # frame is not queued, so the env thread only ever sees observations.
    self._on_restart = on_restart
    self._on_pause = on_pause
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
          if frame[0].get('type') == 'restart' and self._on_restart:
            self._on_restart()
            continue
          if frame[0].get('type') == 'pause' and self._on_pause:
            self._on_pause(frame[0])
            continue
          self.frames.put(frame)
    except BaseException as e:  # noqa: BLE001 -- handed to the env thread
      self._fail(e if isinstance(e, LINKLOST) else ConnectionError(repr(e)))

  def _write_loop(self):
    try:
      while True:
        with self._cond:
          while not self._closed and not self._ctrl:
            if self._job is None:
              self._cond.wait()
            elif (wait := self._next_slice - time.time()) > 0:
              self._cond.wait(wait)
            else:
              break
          if self._closed:
            return
          ctrl = list(self._ctrl)
          self._ctrl.clear()
          job = self._job
        for data in ctrl:
          self.sock.sendall(data)
        if job is None or time.time() < self._next_slice:
          continue
        header, blob, offset, chunk, on_done = job
        piece = blob[offset:offset + chunk]
        if chunk >= len(blob):
          self.sock.sendall(_frame(header, blob))
        else:
          self.sock.sendall(_frame(dict(
              header, type='wchunk', offset=offset, total=len(blob)), piece))
        job[2] = offset + len(piece)
        if self._rate > 0:
          self._next_slice = time.time() + len(piece) / self._rate
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


def _close_socket(sock):
  try:
    sock.shutdown(socket.SHUT_RDWR)
  except OSError:
    pass
  try:
    sock.close()
  except OSError:
    pass


class _Streams:
  """Connections from the phone that carry nothing but weight pushes.

  Each push is cut into one contiguous part per connection and the parts go
  out at once. Over a long-distance link a single TCP connection is capped by
  loss and round-trip time, not by the line (Darmstadt to OIST, 2026-11-03:
  ~200 KB/s each, 4 at once ~640 KB/s), so N connections are ~N times faster.
  They are separate from the control link, so a push never delays a reset or
  a pause and needs no pacing.

  The phone opens them after its hello, to `port`, each greeting with the
  session token our hello gave it; a new control connection starts a new
  session and drops the old streams. Over the relay each stream crosses its
  own ssh connection (train.sh), since channels inside one ssh connection
  would share its single TCP connection and its cap.
  """

  # How long a push waits for the phone to report the new weights installed
  # before it counts as done anyway (a paused or playing phone may not say).
  CONFIRM_S = 60.0

  def __init__(self, host, port, timeout):
    self.port = port
    self._timeout = timeout
    self._lock = threading.Lock()
    self._session = None
    self._socks = []      # (index, socket) of the current session
    self._job = None
    self._listener = None
    try:
      listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      listener.bind((host, port))
      listener.listen(16)
    except OSError as e:
      print(f'[robot] weight streams off, cannot listen on {port}: {e}',
            flush=True)
      return
    self._listener = listener
    threading.Thread(
        target=self._accept_loop, name='robot_wstreams', daemon=True).start()

  @property
  def pushing(self):
    return self._job is not None

  def open_session(self, n):
    """Forget the previous phone connection's streams and return what our
    hello announces, or None when we cannot take any."""
    if self._listener is None:
      return None
    with self._lock:
      old, self._socks = self._socks, []
      self._session = os.urandom(8).hex()
      session = self._session
    for _, sock in old:
      _close_socket(sock)
    return dict(port=self.port, n=int(n), session=session)

  def push(self, stamp, blob, on_done, status):
    """Send `blob` across the streams in the background. False if there are
    none connected or a push is still going: the caller then falls back to
    the control link or waits."""
    with self._lock:
      socks = list(self._socks)
      if not socks or self._job is not None:
        return False
      self._job = threading.Thread(
          target=self._send, args=(stamp, blob, socks, on_done, status),
          name='robot_wpush', daemon=True)
      self._job.start()
    return True

  def close(self):
    listener, self._listener = self._listener, None
    listener and _close_socket(listener)
    with self._lock:
      socks, self._socks = self._socks, []
    for _, sock in socks:
      _close_socket(sock)

  def _send(self, stamp, blob, socks, on_done, status):
    try:
      started = time.time()
      n = len(socks)
      size = -(-len(blob) // n)
      failed = []

      def one(i, sock):
        part = blob[i * size:(i + 1) * size]
        try:
          sock.sendall(_frame(dict(
              type='wpart', stamp=stamp, part=i, parts=n, offset=i * size,
              total=len(blob)), part))
        except OSError as e:
          failed.append((sock, e))

      threads = [threading.Thread(target=one, args=(i, sock), daemon=True)
                 for i, (_, sock) in enumerate(socks)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      if failed:
        bad = {sock for sock, _ in failed}
        with self._lock:
          self._socks = [x for x in self._socks if x[1] not in bad]
        for sock in bad:
          _close_socket(sock)
        print(f'[robot] weight stream lost during a push ({failed[0][1]}); '
              f'{len(socks) - len(bad)} left, will send again', flush=True)
        return
      # sendall only hands the bytes to the kernel, and over the relay
      # megabytes fit in the buffers on the way; so the push lasts until the
      # phone reports running the new weights, and the next one cannot pile
      # up behind it.
      handed = time.time() - started
      deadline = time.time() + self.CONFIRM_S
      while status() is None and time.time() < deadline:
        time.sleep(0.02)
      state = status()
      on_done(dict(stamp=stamp, handed_s=handed, via=f'{n} streams' + (
          '' if state == 'ok' else ', the phone could not load it'
          if state == 'failed' else ', not confirmed by the phone')))
    finally:
      with self._lock:
        self._job = None

  def _accept_loop(self):
    while True:
      listener = self._listener
      if listener is None:
        return
      try:
        sock, address = listener.accept()
      except OSError:
        return
      threading.Thread(target=self._greet, args=(sock, address),
                       daemon=True).start()

  def _greet(self, sock, address):
    try:
      sock.settimeout(self._timeout)
      buffer = bytearray()
      frame = None
      while frame is None:
        chunk = sock.recv(4096)
        if not chunk:
          raise ConnectionError('closed before its hello')
        buffer.extend(chunk)
        frame = _parse(buffer)
      header, _ = frame
      with self._lock:
        ok = (header.get('type') == 'wstream'
              and header.get('session') == self._session)
        if ok:
          self._socks.append((int(header.get('index', 0)), sock))
          self._socks.sort(key=lambda x: x[0])
          count = len(self._socks)
      if not ok:
        raise ValueError(f'not a weight stream of this session: {header}')
      # A send may stall on a dead link; give up on it like on a quiet phone.
      sock.settimeout(self._timeout)
      print(f'[robot] weight stream {header.get("index")} connected from '
            f'{address[0]}:{address[1]} ({count} up)', flush=True)
    except Exception as e:  # noqa: BLE001 -- a bad stream must not end the run
      print(f'[robot] refused a weight stream from {address[0]}: {e}',
            flush=True)
      _close_socket(sock)

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
      self, task='drive', host='0.0.0.0', port=3000, hz=65.0, episode_s=20.0,
      length=0, discrete=True, timeout=20.0, speed_scale=1e-3, fall_angle=0.0,
      spin_penalty=0.1, rate_penalty=0.05, theta_zero=0.0,
      theta_lo=-0.122, theta_hi=0.182, theta_sigma=0.05, drift_penalty=0.1,
      drift_clip=1.0, wheel_penalty=0.0, yaw_penalty=0.0, still_speed=0.0,
      still_ramp_s=0.0,
      action_rate_penalty=0.0, action_rate_quad=0.0,
      command_scale=1.0, symmetric=True,
      obs_theta_scale=0.18, obs_rate_scale=3.0, obs_wheel_scale=3.3e-4,
      obs_clip=3.0,
      ref_range=0.09, ref_hold=100, seed=0, status_every=0,
      recover_gain=0.0, recover_k=4.0, recover_tol=0.05, recover_max=250,
      recover_min=0.6, command_speed=1500.0, command_turn=1000.0,
      command_sigma=0.5, command_sigma_fine=0.0, command_fine_weight=0.5,
      command_reward='product', command_track_weight=2.0,
      command_balance_weight=1.0, forward_sign=-1.0, command_step=0.0,
      command_zeta_steps=40000, command_bias=0.0, command_ramp=0.0,
      friction_comp=False, bumper_lo=-0.120, bumper_hi=0.161,
      bumper_margin=0.0175, bumper_penalty=1.0, tilt_penalty=0.0,
      track_progress=0.0, balance_bumper=0.0,
      turn_source='wheels', yaw_rate_max=1.5,
      yaw_axis='y', yaw_sign=-1.0, command_mode='auto', command_hold_min=2.0,
      command_hold_max=5.0, command_p_zero=0.3, command_p_axis=0.4,
      reconnect=True, pipeline=True, onboard=False, config=None,
      policy_dir=None, weights_every=0.0, weights_dtype='float32',
      weights_kbps=0.0, weights_poll=0.25, weight_streams=0, logdir=None,
      pace=None, pause_file=None, start_paused=False, runaway_s=0.0,
      runaway_drive=0.7, brake_s=1.0, pos_tau=0.0, pos_penalty=0.0,
      pos_range=12000.0, obs_pos_scale=1.67e-4, pos_jump=1000.0,
      pos_fence=0.0, drive_cap0=0.0, drive_cap_s=300.0, home_radius=0.0,
      home_inner=1200.0, home_speed=0.5, home_min=0.25, home_ramp=3000.0,
      home_max_s=30.0, home_track=1634.0, weights_emotion=False,
      fatigue_s=0.0, fatigue_tol=0.15, fatigue_dead=0.05, travel_lo=0.0,
      travel_hi=0.0, affect_vigour=0.0, affect_catch=0.0, affect_explore=0.0,
      pain_zone=0.035, emotion_event=0.0, event_signal='arousal',
      cost_obs=False):
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
    self._base_ok = True
    self._base_health = {}
    self._config = config
    self._policy_dir = elements.Path(policy_dir) if policy_dir else None
    # Set by tools/robot/demo_supervisor.py: where to ask it for a fresh run
    # when the phone's dashboard says so. Unset, the phone is not offered it.
    self._restart_file = os.environ.get('ROBOT_RESTART_FILE') or None
    # The robot and the training can each be paused from the dashboard: the
    # phone holds its wheels, the learner stops updating. Kept in pause_file
    # (online.PAUSE_FILE), which the learner reads and which outlives a
    # reconnect or a resumed run. start_paused: a new run starts with both
    # paused, so an audience can see the untrained robot first; the first
    # time training is then unpaused is minute 0 on the dashboard.
    self._pause_file = pause_file and str(pause_file)
    self._pause_lock = threading.Lock()
    self._pause = self._load_pause(bool(start_paused))
    # Pushes run on a thread of their own (`_push_loop`), so they can go as
    # often as the learner publishes: `weights_every` is only a floor on the
    # time between two push starts, 0 meaning back to back, and the learner's
    # `online_publish_every` is what actually sets the cadence.
    self._weights_every = float(weights_every)
    self._weights_poll = float(weights_poll)
    # float16 halves the push for a slow link (a learner across the world);
    # the phone computes in float32 either way (numpy_policy.load).
    self._weights_dtype = str(weights_dtype)
    self._weights_kbps = float(weights_kbps)  # _Link: pacing, 0 off
    # Weight pushes over this many connections of their own (_Streams), on
    # port + 1, for a phone that offers them; 0 keeps them on the control
    # link. A long-distance link (Saion to Europe: 300 ms, 0.3% loss) caps
    # each TCP connection near 200 KB/s however fast the line, so N of them
    # carry ~N times that; and since none of them carries control frames, a
    # push needs no pacing to keep resets and pauses from queueing behind it.
    self._weight_streams = int(weight_streams)
    self._streams = None
    # Delta pushes (export.encode), for a phone that offers them: `_base` is
    # the last push the phone reported running, as (stamp, arrays), and a new
    # push only carries what changed from it. `_pending` is the push in
    # flight, which becomes the base once the phone reports its stamp.
    self._wdelta = False
    self._base = None
    self._pending = None
    self._wtiming = {}           # the phone's report on the last push
    self._weights_stamp = None   # the policy the phone holds, as far as we know
    self._push_started = 0.0
    self._pusher = None
    self._stats_at = 0.0
    self._stats_offset = 0
    self._stats_updates = 0.0
    self._closing = threading.Event()
    self._phone_stamp = None     # policy_stamp the phone reported last
    # 'serial' or 'clock', or None to take the phone's default. The phone's
    # RP2040 applies one command per ~83 ms and reads USB only in between, so
    # 'serial' -- one decision per reply -- is what gets every action applied.
    self.pace = pace
    self._seq = None  # the phone's observation counter, echoed in each action
    # The control rate the phone runs at (its trainer.json max_hz, which
    # tools/robot/train.sh writes from the same value). Episodes are set in
    # seconds so they last as long at any rate; `length` in steps overrides.
    self.hz = float(hz)
    self.length = int(length) or int(round(self.hz * float(episode_s)))
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
    # Balance task: cost per rad/s of turning, from the gyro (clipped at
    # 2 rad/s). The wheel cost charges a pivot no more than the same speed
    # straight ahead, and on the table policies kept turning in place.
    self.yaw_penalty = float(yaw_penalty)
    # still_speed > 0 (phone mm/s): the balance task's upright score is
    # multiplied by exp(-mean |wheel speed| / still_speed), per wheel, so
    # moving costs a share of the whole reward rather than a small additive
    # term. On the table e2061 and e2078 (wheel_penalty 0.3 and 0.6, clipped
    # at speed_scale 1e-4) balanced while pivoting on one wheel at 1000-2000
    # mm/s, which cost them 0.03-0.12 of a reward near 0.8; at still_speed
    # 1000 the same pivot keeps a third to two thirds of it.
    self.still_speed = float(still_speed)
    # still_ramp_s: phase the stillness factor in over this many seconds of
    # this process's steps (its scale shrinks from 10x still_speed), so
    # balancing is learned first. Measured in the sim: still_speed 1000 from
    # the start took ~10 min to balance on the table against ~5 without it.
    self.still_ramp_s = float(still_ramp_s)
    self.symmetric = bool(symmetric)
    # Cost on the change in each wheel's command between steps, mean over the
    # wheels, in the policy's [-1, 1] units. At 50 Hz an exploring policy
    # chatters the wheels every 20 ms and nothing else in the reward minds.
    # A penalty, not a low-pass filter, so no delay is added to the loop.
    self.action_rate_penalty = float(action_rate_penalty)
    # Squared too (round 6): the actions trained on are sampled (policy
    # minstd 0.1), so |change| is ~0.11 from the noise alone and the L1 term
    # hardly tells small wobble from none; squared, the noise costs ~0.02
    # and a full reversal 4.
    self.action_rate_quad = float(action_rate_quad)
    self._chatter2 = 0.0
    self._prev_act = np.zeros(2, np.float32)
    self._chatter = 0.0
    self._sent_act = (0.0, 0.0)
    # Runaway brake, for live training: once the wheels' mean command has
    # stayed beyond +-runaway_drive in one direction for runaway_s, the
    # wheels are held for brake_s (the phone through a robot pause, which it
    # records as zero action). Early on the table every run passes through a
    # phase of chasing a lean at full drive across the room (sim e19200-
    # e19232: 1-3 min in, 5 of 9 runs; speed penalties only taught lying on
    # a bumper). Trained policies never hold full drive for 1 s, so it stays
    # out of their way and the wheels keep their whole range. 0 = off.
    self.runaway_s = float(runaway_s)
    self.runaway_drive = float(runaway_drive)
    self.brake_s = float(brake_s)
    self._run_steps = 0
    self._run_sign = 0.0
    self._brake_left = 0
    # Where the robot is: each wheel's distance (phone mm) summed with a leak
    # of time constant pos_tau s, so it reads how far the robot has gone from
    # where it was over the last ~pos_tau s and needs no anchor that could
    # jump. An observation ('position', obs_pos_scale x it, ~0.5 m per unit)
    # when pos_tau > 0, and a cost of pos_penalty x the net displacement over
    # pos_range (~1 m; 12178 phone mm per m), clipped at 1. Nothing else
    # tells the policy it has left its spot: speed costs can only make it
    # stop, and on the table early runs stop by lying on a bumper (round 7).
    # The phone keeps the same sum (deploy/policy_runner.py). A step moving a
    # wheel more than pos_jump is a counter reset, not motion.
    self.pos_tau = float(pos_tau)
    self.pos_penalty = float(pos_penalty)
    self.pos_range = float(pos_range)
    self.obs_pos_scale = float(obs_pos_scale)
    self.pos_jump = float(pos_jump)
    self._pos = np.zeros(2, np.float64)
    self._prev_dist = None
    self._pos_step = 0.0
    # pos_fence > 0 (phone mm): the runaway brake also holds the wheels
    # whenever the robot is past this net displacement and still moving
    # away, a virtual wall that leaves speed inside it alone.
    self.pos_fence = float(pos_fence)
    # A speed cap for the first minutes of live training: every wheel
    # command is clipped to +-cap, cap rising linearly from drive_cap0 to 1
    # over drive_cap_s s of this process's steps. Clipped, not scaled, so
    # the small corrections of balancing pass unchanged; the clipped action
    # is what is recorded. 0 = off.
    self.drive_cap0 = float(drive_cap0)
    self.drive_cap_s = float(drive_cap_s)
    self._cap_sent = None
    # Return home: home_radius > 0 (phone mm; 12178 per m) keeps the robot
    # near where the run started. Its position is dead-reckoned from the
    # wheels (x, y, heading; home_track = wheel track in phone mm, fitted to
    # the MuJoCo body's turning: 1634 put its heading within about 2%), so
    # turning on the spot does not count as travel. Past home_radius the
    # episode ends and, before the next one, the robot steers back to the
    # start (forward or backward, whichever faces home; proportional over
    # home_ramp, between home_min and home_speed of full command) until it is
    # within home_inner, or for at most home_max_s. Nothing of the drive home
    # is recorded. The phone drives it from a 'drive' field in ctrl frames.
    self.home_radius = float(home_radius)
    self.home_inner = float(home_inner)
    self.home_speed = float(home_speed)
    self.home_min = float(home_min)
    self.home_ramp = float(home_ramp)
    self.home_max_s = float(home_max_s)
    self.home_track = float(home_track)
    self._xyh = np.zeros(3)
    # Push the RSSM prior and the critic and reward heads with the policy, so
    # the phone can compute its emotion (deploy/emotion.py) for the face.
    self.weights_emotion = bool(weights_emotion)
    self._away_d = np.zeros(2)   # each wheel's travel since the run started
    self._away_prev = None
    self._go_home = False
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
    # 2026-09-02, action-conditioned tilt response was 0.5 sigma. At 65 Hz it
    # can: in e1982's replay positive output (forward, negative wheel speed)
    # is followed by tilt acceleration of the opposite sign (r = -0.5), so
    # positive tilt is a forward lean and the gain for this rig is +1. 0
    # disables recovery, leaving the robot wherever the last episode ended.
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
    # command_sigma_fine > 0: tracking mixes in a second, narrow kernel,
    # (1 - w) * wide + w * fine with w = command_fine_weight. The wide one
    # alone barely tells 0.9 m/s from standing still at sigma 0.6 x 2500
    # mm/s (0.70 vs 1.0): round 6u's best run held the stick at zero on the
    # table by rolling backwards at ~0.9 m/s. The fine kernel pays precision.
    self.command_sigma_fine = float(command_sigma_fine)
    self.command_fine_weight = float(command_fine_weight)
    assert command_reward in (
        'product', 'sum', 'curriculum', 'taper_sum', 'balance',
        'gated', 'bumper', 'bumper_track'), command_reward
    self.command_reward = command_reward
    # 'curriculum' (arXiv 2409.09845, a wheel-legged robot balancing on its
    # wheels): balance + zeta * track_weight * tracking - penalties, with
    # zeta = (tanh(8 s / zeta_steps - 3) + 1) / 2 rising from ~0 to ~1 over
    # zeta_steps recorded steps, so balance is learned first and tracking
    # then takes over. The paper ramps zeta with episode length; nothing ends
    # an episode here, so it ramps with steps (this process's: a resumed run
    # starts the ramp again).
    self.command_track_weight = float(command_track_weight)
    self.command_balance_weight = float(command_balance_weight)
    # Which way forward 1.0 drives the wheels. +1: positive wheel speed is
    # forward. e1302 showed the robot's positive direction is what a person
    # at the joystick calls backward (it leaned and rolled backward on a
    # forward command), hence the default -1. Turning is measured
    # by the gyro, clockwise from above, which is a right turn whichever way
    # the front faces, so it needs no sign.
    self.forward_sign = float(forward_sign)
    self.command_zeta_steps = float(command_zeta_steps)
    # Feed-forward from the command: the phone sends the motors
    # (1 - command_bias) * action + command_bias * the outputs that drive as
    # the command asks (its main.command_motors), so the policy only has to
    # balance around the drive. The command stays in the observation, which
    # is all the policy needs to know the bias. 0: the policy alone.
    self.command_bias = float(command_bias)
    # command_ramp: the phone moves the command toward each new target at
    # this many units per second (0: steps), so the policy has time to lean
    # before it must speed up; it observes and is scored on the ramped one.
    # friction_comp: the phone lifts each action past the motors' dead zone
    # and kicks still wheels loose (its main.compensate; dead zone and
    # stiction measured 2026-10-15 in docs/ROBOT_BRIDGE_FINDINGS.md).
    self.friction_comp = bool(friction_comp)
    # 'bumper' reward: within bumper_margin of either bumper (or past it) is
    # -1, anything else is fine. The stops are the tilt the body rests at on
    # each bumper, measured 2026-10-15 (back -6.7..-7.1 deg, front +9.2..9.4
    # deg), relative to theta_zero; the margin is 1 deg.
    self.bumper_lo = float(bumper_lo)
    self.bumper_hi = float(bumper_hi)
    self.bumper_margin = float(bumper_margin)
    self.bumper_penalty = float(bumper_penalty)  # 'bumper_track' pays minus this
    # 'bumper_track' also pays minus tilt_penalty * (tilt / that side's
    # bumper)^2 off the bumpers, so leaning to the edge of the margin costs
    # something (e6031 drove the real carpet at +4.6 deg, p95 7.5).
    self.tilt_penalty = float(tilt_penalty)
    # The balance task pays minus balance_bumper on or within bumper_margin of
    # either bumper (0, the default: no such term; the upright score alone
    # is ~0 there, as good as a fall the policy cannot see coming).
    self.balance_bumper = float(balance_bumper)
    # fatigue_s > 0: a wheel command held roughly constant fades to 0 over
    # fatigue_s s (Fatigue above); onboard the phone applies it (told in
    # pause frames), here it is applied before the cap.
    self.fatigue_s = float(fatigue_s)
    self._fatigue = Fatigue(round(self.fatigue_s * float(hz)), fatigue_tol, fatigue_dead)
    # travel_hi > 0 (phone mm; needs pos_tau): commands pushing further the
    # way the robot already went fade from travel_lo to 0 at travel_hi of
    # leaky travel (travel_limit above); onboard the phone applies it.
    self.travel_lo, self.travel_hi = float(travel_lo), float(travel_hi)
    assert not self.travel_hi or float(pos_tau) > 0, 'travel_hi needs pos_tau'
    self._travel_latch = [0.0]
    # Emotion acting on the phone's behaviour and on learning
    # (deploy/policy_runner.py modulate/explore_scale, deploy/emotion.py; in
    # the sim through sim_phone.py): gains for vigour with excitement, a
    # catch drive with pain, exploration with distressed arousal; how far
    # before a bumper closeness starts to hurt (rad); how far arousal must
    # rise above its ~10 s average for an event to go to the learner
    # (embodied/run/online.py). 0: off.
    self.affect = dict(vigour=float(affect_vigour), catch=float(affect_catch),
                       explore=float(affect_explore))
    self.pain_zone = float(pain_zone)
    self.emotion_event = float(emotion_event)
    self.event_signal = str(event_signal)
    # cost_obs: report the body's pain as 'cost' beside the reward, for the
    # agent's fear critic (agent.fear); never part of the reward.
    self.cost_obs = bool(cost_obs)
    self._chosen_act = None
    self._cost_pair = (None, None)   # (applied, chosen) of the last step
    self._event_n = None
    # The action a policy run beside this env (the sim's phone) chose and
    # the limits below already shaped, for the step about to be taken.
    self._phone_final = None
    # ... plus track_progress * |c| * clip(forward / c, 0, 1) for a forward
    # command c: the Gaussian pays ~0 until the speed is close (carpet at
    # 600 of 2500 mm/s earns 0.03), so partial progress gets a linear slope.
    self.track_progress = float(track_progress)
    self._total_steps = 0
    # Turn rate from the wheels (half the speed difference / command_turn) or
    # from the phone's raw gyroscope: the body's true yaw rate, which wheel
    # slip cannot fake. The phone stands upright, so yaw is about its long
    # axis (gyro_y, rad/s, counter-clockwise from above positive); yaw_sign
    # -1 makes a right turn positive, matching the joystick. Turn 1.0 asks
    # for yaw_rate_max rad/s.
    assert turn_source in ('wheels', 'gyro'), turn_source
    self.turn_source = turn_source
    self.yaw_rate_max = float(yaw_rate_max)
    self.yaw_axis = str(yaw_axis)
    self.yaw_sign = float(yaw_sign)
    self._yaw_rate = 0.0
    # The axis and sign are set from how the phone sits in the cradle, which
    # nothing here can see; checked against the wheels every 500 steps.
    self._yaw_check = []
    # How the phone picks commands nobody is steering ('auto': hold a random
    # one for hold_min..hold_max s, zero with p_zero, one axis only with
    # p_axis; 'manual': zero, i.e. balance in place). Sent in the handshake.
    # step: random commands are multiples of it, e.g. 0.5 for -1, -0.5, 0,
    # 0.5, 1 on each axis; 0 means continuous.
    self.command_sampler = dict(
        mode=str(command_mode), hold_min=float(command_hold_min),
        hold_max=float(command_hold_max), p_zero=float(command_p_zero),
        p_axis=float(command_p_axis),
        step=float(command_step), ramp=float(command_ramp))
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
        # Each wheel's leaky distance (pos_tau), scaled; see _track_position.
        **({'position': elements.Space(np.float32, (2,))}
           if self.pos_tau else {}),
        'reward': elements.Space(np.float32),
        **({'cost': elements.Space(np.float32)} if self.cost_obs else {}),
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
        'log/brake': elements.Space(np.float32),
        **({k: elements.Space(np.float32) for k in (
            'log/cmd_forward', 'log/cmd_turn', 'log/cmd_joystick',
            'log/forward', 'log/turn', 'log/turn_wheels', 'log/yaw_rate',
            'log/zeta')} if self.task == 'command' else {}),
        **({'executed/drive': elements.Space(np.float32, (2,), -1.0, 1.0)}
           if self._records_executed() else {}),
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
        sensors, latency = self._wait_for_base(sensors, latency)
        sensors, latency = self._wait_while_paused(sensors, latency)
        sensors, latency = self._return_home(sensors, latency)
        sensors, latency = self._recover(sensors, latency)
        self._step = 0
        self._done = False
        self._reward = 0.0
        self._ref = float(self.theta_zero)
        self._fatigue.reset()
        self._prev_act = np.zeros(2, np.float32)
        self._chatter = 0.0
        self._sent_act = (0.0, 0.0)
        self._track_position(sensors)
        self.home_radius and self._away(sensors)
        return self._obs(sensors, latency, 0.0, is_first=True)
      except LINKLOST as e:
        if not self.reconnect:
          raise
        print(f'Robot did not answer the reset, waiting for it again: {e}')
        self._drop()
        time.sleep(1.0)

  def _wait_for_base(self, sensors, latency):
    """Hold the episode start until the phone's base is answering again."""
    if self._base_ok:
      return sensors, latency
    print('[robot] base not answering; waiting for it before the next '
          'episode (%s)' % self._base_health, flush=True)
    start = time.time()
    while not self._base_ok:
      self._send(0.0, 0.0, reset=True)
      sensors, latency = self._recv()
    print('[robot] base back after %.1f s (%s)' % (
        time.time() - start, self._base_health), flush=True)
    return sensors, latency

  def _away(self, sensors):
    """Each wheel's travel since the run started (phone mm), summed from
    step increments: an episode reset may zero the counters (the sim's
    does), and a jump past pos_jump is taken for that."""
    dist = np.array([float(sensors.get('wheel_distance_l', 0.0)),
                     float(sensors.get('wheel_distance_r', 0.0))])
    if self._away_prev is not None:
      step = dist - self._away_prev
      step[np.abs(step) > self.pos_jump] = 0.0
      self._away_d = self._away_d + step
      ds, dh = step.mean(), (step[1] - step[0]) / self.home_track
      x, y, h = self._xyh
      self._xyh = np.array([x + ds * np.cos(h + dh / 2),
                            y + ds * np.sin(h + dh / 2), h + dh])
    self._away_prev = dist
    return self._away_d

  def _home_distance(self):
    return float(np.hypot(self._xyh[0], self._xyh[1]))

  def _return_home(self, sensors, latency):
    if not (self.home_radius and self._go_home):
      return sensors, latency
    self._go_home = False
    print('[robot] %.2f m from the start; driving back' % (
        self._home_distance() / 12178.0), flush=True)
    for _ in range(int(self.home_max_s * self.hz)):
      self._away(sensors)
      dist = self._home_distance()
      if dist < self.home_inner:
        break
      x, y, h = self._xyh
      # Bearing of home from the heading (the direction of positive wheel
      # travel); drive backward when home is behind.
      bearing = np.arctan2(-y, -x) - h
      bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
      sign = 1.0
      if abs(bearing) > np.pi / 2:
        sign = -1.0
        bearing = (bearing - np.pi + np.pi) % (2 * np.pi) - np.pi
      speed = min(1.0, dist / self.home_ramp) * self.home_speed
      speed = max(speed, self.home_min) * max(0.0, np.cos(bearing))
      turn = float(np.clip(bearing, -1.0, 1.0)) * self.home_speed
      # Wheel travel per wheel, then output: forward output turns the wheels
      # backwards (negative speed), so travel u needs output -u.
      left = float(np.clip(-(sign * speed - turn), -1.0, 1.0))
      right = float(np.clip(-(sign * speed + turn), -1.0, 1.0))
      self._send_drive(left, right)
      sensors, latency = self._recv()
    self._send(0.0, 0.0, reset=True)
    return self._recv()

  def _send_drive(self, left, right):
    """Drive the wheels from here between episodes, onboard too."""
    self._sent = time.time()
    if self._onboard:
      self._link.send(dict(
          type='ctrl', reset=False, reward=0.0, seq=self._seq,
          drive=[float(left), float(right)]))
    else:
      self._send(left, right, reset=False)

  def _wait_while_paused(self, sensors, latency):
    """Hold the episode start while the robot is paused from the dashboard.

    The phone keeps reporting while it holds the wheels, and those steps
    used to be recorded: a demo left paused for 4-9 min before starting
    filled replay with the robot lying on a bumper at zero action, and its
    policy learned to lie there (e2831 s01, e9662, e30101 never balanced;
    sessions held under 2.5 min balanced in 2-5 min). Nothing is recorded
    until it runs."""
    if not self._pause['robot']:
      return sensors, latency
    print('[robot] robot paused; recording nothing until it runs', flush=True)
    while self._pause['robot']:
      self._send(0.0, 0.0, reset=True)
      sensors, latency = self._recv()
    print('[robot] robot running; starting an episode', flush=True)
    return sensors, latency

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
    if self._pause['robot']:
      # Paused from the dashboard: close the episode here; the next one
      # starts only once the robot runs again (_wait_while_paused), so the
      # held stretch never reaches replay.
      self._send(0.0, 0.0, reset=False)
      sensors, latency = self._recv()
      self._sent_act = (0.0, 0.0)
      self._done = True
      return self._obs(sensors, latency, 0.0, is_last=True)
    if self.task == 'track' and self._step % self.ref_hold == 0:
      self._ref = self.theta_zero + self._rng.uniform(
          -self.ref_range, self.ref_range)
    if self._phone_final is not None:
      # The sim's phone chose this step's action from the observation it
      # was recorded with, and _limits already shaped it (sim_phone.py).
      left, right = self._phone_final
    else:
      self._chosen_act = self._decode(action)
      left, right = self._limits(*self._chosen_act)
    self._sent_act = (left, right)
    self._cost_pair = ((left, right), self._chosen_act)
    self._send(left, right, reset=False)
    sensors, latency = self._recv()
    if not self._base_ok:
      # The base dropped out: its sensors are frozen and the wheels held, so
      # nothing after this is the world the policy acts in. End the episode
      # here (not terminal: nothing about the state ended it); _reset waits
      # for the base.
      print('[robot] base stopped answering at step %d; ending the episode '
            '(%s)' % (self._step, self._base_health), flush=True)
      self._done = True
      return self._obs(self._last, latency, 0.0, is_last=True)
    self._step += 1
    self._total_steps += 1
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
    if self.task == 'command' and self.turn_source == 'gyro':
      self._check_yaw(sensors)
    # Chatter: the change in each wheel's command since the last step, mean
    # over the wheels, of the action the motors really got.
    act = (np.resize(self._onboard_act, 2)
           if self._onboard and self._onboard_act is not None
           else np.array([left, right], np.float32))
    self._chatter = float(np.mean(np.abs(act - self._prev_act)))
    self._chatter2 = float(np.mean((act - self._prev_act) ** 2))
    self._prev_act = act
    self._runaway(act)
    self._track_position(sensors)
    reward, terminal = self._evaluate(sensors)
    reward -= self.action_rate_penalty * self._chatter
    reward -= self.action_rate_quad * self._chatter2
    if self.pos_penalty:
      reward -= self.pos_penalty * min(
          abs(float(self._pos.mean())) / self.pos_range, 1.0)
    self._reward = reward
    if self.home_radius and (
        self._away(sensors) is not None
        and self._home_distance() > self.home_radius):
      self._go_home = True
    self._done = terminal or self._go_home or self._step >= self.length
    return self._obs(
        sensors, latency, reward, is_last=self._done, is_terminal=terminal)

  def _limits(self, left, right):
    """The trainer's limits on a wheel command: fatigue, travel, the cap,
    and held wheels while paused or braking."""
    if self.fatigue_s:
      left, right = self._fatigue((left, right))
    if self.travel_hi:
      left, right = travel_limit(
          (left, right), float(self._pos.mean()), self.travel_lo, self.travel_hi,
          self._travel_latch)
    if self.drive_cap0:
      cap = self._drive_cap()
      left = float(np.clip(left, -cap, cap))
      right = float(np.clip(right, -cap, cap))
      self._tell_cap(cap)
    if self._pause['robot'] or self._brake_left:
      # The phone holds the wheels only for a policy it runs itself; here the
      # held wheels are this action, and recorded as what they did.
      left, right = 0.0, 0.0
    return float(left), float(right)

  def close(self):
    self._closing.set()
    self._drop()
    self._streams and self._streams.close()
    try:
      self._listener and self._listener.close()
    except OSError:
      pass
    self._listener = None

  def _track_position(self, sensors):
    if not self.pos_tau:
      return
    before = float(self._pos.mean())
    self._pos = track_position(
        self._pos, self._prev_dist, sensors, self.pos_tau, self.hz,
        self.pos_jump)
    self._prev_dist = (float(sensors.get('wheel_distance_l', 0.0)),
                       float(sensors.get('wheel_distance_r', 0.0)))
    self._pos_step = float(self._pos.mean()) - before

  def _runaway(self, act):
    """Count steps of full drive one way; start or end a brake."""
    if not self.runaway_s:
      return
    if self._brake_left:
      self._brake_left -= 1
      if not self._brake_left:
        self._tell_brake(False)
      return
    net = float(np.mean(act))
    sign = float(np.sign(net)) if abs(net) > self.runaway_drive else 0.0
    if not sign:
      self._run_steps = 0
    elif sign == self._run_sign:
      self._run_steps += 1
    else:
      self._run_steps = 1
    self._run_sign = sign
    fenced = (self.pos_fence and abs(float(self._pos.mean())) > self.pos_fence
              and self._pos_step * float(self._pos.mean()) > 0)
    if self._run_steps >= self.runaway_s * self.hz or fenced:
      self._run_steps = 0
      self._brake_left = max(1, int(round(self.brake_s * self.hz)))
      print('[robot] %s, holding the wheels %.1f s' % (
          'past the fence' if fenced else
          'runaway: full drive for %.1f s' % self.runaway_s, self.brake_s),
          flush=True)
      self._tell_brake(True)

  def _drive_cap(self):
    ramp = self._total_steps / max(self.drive_cap_s * self.hz, 1.0)
    return min(1.0, self.drive_cap0 + (1.0 - self.drive_cap0) * ramp)

  def _tell_cap(self, cap):
    # Onboard the phone drives, so it clips (dreamerBridge 'drive_cap' in a
    # pause frame); sent when it has moved by 0.02.
    if not self._onboard or self._link is None:
      return
    # Every 0.02, and the final 1.0 exactly: stopping short of it left the
    # phone clipping at 0.98 for the rest of a run (e30103).
    if self._cap_sent is not None and abs(cap - self._cap_sent) < 0.02 and not (
        cap >= 1.0 and self._cap_sent < 1.0):
      return
    try:
      self._link.send(dict(type='pause', **self._pause_message(),
                           drive_cap=float(cap)))
      self._cap_sent = cap
    except LINKLOST:
      pass

  def _tell_brake(self, on):
    # Onboard the phone drives, so it holds the wheels itself: a robot pause
    # it already understands, ended with the dashboard's own state.
    if not self._onboard or self._link is None:
      return
    message = self._pause_message()
    message['robot'] = bool(on) or message['robot']
    try:
      self._link.send(dict(type='pause', **message))
    except LINKLOST:
      pass

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
      upright = 0.5 * linear + 0.5 * bonus
      if self.still_speed:
        scale = self.still_speed
        if self.still_ramp_s:
          ramp = min(1.0, self._total_steps / (self.still_ramp_s * self.hz))
          scale /= 0.1 + 0.9 * ramp
        upright *= np.exp(-0.5 * (
            abs(float(sensors['wheel_speed_l']))
            + abs(float(sensors['wheel_speed_r']))) / scale)
      reward = (
          upright
          - self.rate_penalty * abs(rate)
          - self.drift_penalty * drift
          - self.wheel_penalty * self._wheel_effort(speed_l, speed_r)
          - self.yaw_penalty * min(abs(self._yaw_rate), 2.0))
      if self.balance_bumper:
        floor = float(sensors.get('theta_floor', theta)) - self.theta_zero
        if (floor >= self.bumper_hi - self.bumper_margin
            or floor <= self.bumper_lo + self.bumper_margin):
          reward -= self.balance_bumper
    elif self.task == 'command':
      reward = self._command_reward(sensors)
    else:
      reward = float(sensors.get('reward', 0.0))
    return float(reward), fallen

  def _measured(self, sensors):
    """(forward, turn) the wheels are doing, in command units."""
    left = float(sensors['wheel_speed_l'])
    right = float(sensors['wheel_speed_r'])
    forward = self.forward_sign * 0.5 * (left + right) / self.command_speed
    if self.turn_source == 'gyro':
      turn = self._yaw_rate / self.yaw_rate_max
    else:
      turn = 0.5 * (left - right) / self.command_turn
    return forward, turn

  def _check_yaw(self, sensors):
    """Print how well the gyro's turn rate agrees with the wheels'.

    Strongly positive: axis and sign are right. Negative: flip yaw_sign. Near
    zero while the wheels do turn: wrong yaw_axis. The slope is the yaw rate
    one unit of wheel turn gives, for choosing yaw_rate_max.
    """
    wheels = 0.5 * (float(sensors['wheel_speed_l'])
                    - float(sensors['wheel_speed_r'])) / self.command_turn
    self._yaw_check.append((wheels, self._yaw_rate))
    if len(self._yaw_check) < 500:
      return
    w, g = np.array(self._yaw_check).T
    self._yaw_check = []
    if w.std() < 1e-6 or g.std() < 1e-6:
      return
    corr = float(np.corrcoef(w, g)[0, 1])
    slope = float(np.polyfit(w, g, 1)[0])
    print(f'[robot] gyro vs wheel turn over 500 steps: corr {corr:+.2f}, '
          f'{slope:+.2f} rad/s per wheel-turn unit, |yaw| p90 '
          f'{np.percentile(np.abs(g), 90):.2f} rad/s', flush=True)

  def _zeta(self):
    return 0.5 * (np.tanh(
        8.0 * self._total_steps / max(self.command_zeta_steps, 1.0) - 3.0) + 1.0)

  def _command_reward(self, sensors):
    """Balance times tracking the commanded forward speed and turn.

    Balance is the balance task's upright score in [0, 1]. Tracking pays
    exp(-(|v - cmd| / command_sigma)^2) on the 2D (forward, turn) error: the
    kernel legged-robot velocity tracking uses (Rudin et al. 2022), so it
    saturates rather than letting one spiky wheel reading dominate.

    'product' (the default) pays balance * tracking. e1299 used 'sum',
    0.5 * balance + 0.5 * tracking with sigma 0.3, and after 15k steps had
    learned to balance and stand still whatever the command (reward 0.62,
    commanded vs measured forward speed correlated -0.11): standing still kept
    the whole balance half and part of the tracking half. Under the product,
    ignoring the command roughly halves the reward, and lying on a bumper
    pays ~0 whatever the wheels do -- which a pure tracking reward would not
    ensure, since nothing ends an episode on this rig. At command (0, 0) it is
    the balance task. sigma 0.5 (was 0.3) makes a partial attempt pay: half
    of a 0.8 command earns 0.53 against 0.08 for standing still.

    Of the balance task's penalties only the tilt-rate one is kept, against
    wobble. Drift duplicated the forward part of tracking, and wheel effort
    was weighted 0 in every command run, so both are gone. At most 1.0.
    """
    theta = float(sensors['theta'])
    offset = theta - self.theta_zero
    reach = self.theta_hi if offset > 0 else abs(self.theta_lo)
    linear = 1.0 - min(1.0, abs(offset) / max(reach, 1e-6))
    bonus = np.exp(-((offset / self.theta_sigma) ** 2))
    cmd_f, cmd_t = self._cmd
    forward, turn = self._measured(sensors)
    # Tracking treats the command as one 2D velocity vector: a Gaussian on
    # the Euclidean error, so neither axis needs its own weight. A zero
    # command is a real target (stand still), not a free pass.
    # The busier wheel gets at most what full forward asks of both: a
    # diagonal is scaled down, keeping its heading, as in the phone's direct
    # drive mode. The taper weight below still reads the stick as given.
    k = max(1.0, abs(cmd_f) + abs(cmd_t) * self.command_turn / self.command_speed)
    err = np.hypot(forward - cmd_f / k, turn - cmd_t / k)
    track = float(np.exp(-((err / self.command_sigma) ** 2)))
    if self.command_sigma_fine:
      fine = float(np.exp(-((err / self.command_sigma_fine) ** 2)))
      track = (1 - self.command_fine_weight) * track + self.command_fine_weight * fine
    balance = 0.5 * linear + 0.5 * bonus
    if self.command_reward == 'curriculum':
      score = (self.command_balance_weight * balance
               + self._zeta() * self.command_track_weight * track)
    elif self.command_reward == 'product':
      score = balance * track
    elif self.command_reward == 'sum':
      score = self.command_balance_weight * balance + self.command_track_weight * track
    elif self.command_reward == 'balance':
      # With command_bias the command drives the wheels itself; the policy is
      # scored on staying upright only.
      score = balance
    elif self.command_reward == 'bumper_track':
      # Tracking alone, -bumper_penalty on or within bumper_margin of a bumper. Balance is
      # implied: a robot cannot hold a lean while standing still, so at a
      # zero command the only way to keep tracking is to stay upright --
      # around its true balance point, not theta_zero, which was set by eye.
      # The bumpers rest on the floor: in the sim on a slope, where the
      # phone's tilt (against gravity) is not the tilt against the floor,
      # judge them by the floor's (theta_floor; the phone does not send it).
      floor = float(sensors.get('theta_floor', theta)) - self.theta_zero
      near = (floor >= self.bumper_hi - self.bumper_margin
              or floor <= self.bumper_lo + self.bumper_margin)
      lean = offset / (self.bumper_hi if offset > 0 else -self.bumper_lo)
      c = cmd_f / k
      progress = abs(c) * min(1.0, max(0.0, forward / c)) if abs(c) > 0.05 else 0.0
      score = -self.bumper_penalty if near else (
          track + self.track_progress * progress - self.tilt_penalty * lean ** 2)
    elif self.command_reward == 'bumper':
      # Touching (or nearly touching) a bumper is -1 whatever else happens.
      # Otherwise, at a zero command half upright and half tracking (which
      # then means standing still); at any other command tracking alone, so
      # leaning to speed up costs nothing. With command_ramp a command only
      # counts as zero once it has ramped all the way there.
      near = (offset >= self.bumper_hi - self.bumper_margin
              or offset <= self.bumper_lo + self.bumper_margin)
      still = abs(cmd_f) < 1e-3 and abs(cmd_t) < 1e-3
      score = -1.0 if near else (
          0.5 * balance + 0.5 * track if still else track)
    elif self.command_reward == 'gated':
      # Tracking counts only while upright: lying on a bumper pays ~0 however
      # the wheels turn, balancing while ignoring the command ~0.5, and
      # balancing while following it 1.
      score = balance * (0.5 + 0.5 * track)
    elif self.command_reward == 'taper_sum':
      # Tracking weight grows linearly with command magnitude, from 0.1 at
      # rest to 0.9 at |cmd| >= 1; the weights sum to 1, so score <= 1.
      movement = min(1.0, float(np.hypot(cmd_f, cmd_t)))
      w_track = 0.1 + 0.8 * movement
      score = (1.0 - w_track) * balance + w_track * track
    else:
      raise ValueError(f'Unknown command reward: {self.command_reward}')
    return float(
        score - self.rate_penalty * abs(float(sensors['angular_velocity'])))

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
        **({'position': np.clip(
            self._pos * self.obs_pos_scale, -self.obs_clip,
            self.obs_clip).astype(np.float32)} if self.pos_tau else {}),
        reward=np.float32(reward),
        **({'cost': np.float32(pain_signal(
            sensors['theta'], dict(theta_zero=self.theta_zero,
                                   lo=self.bumper_lo, hi=self.bumper_hi,
                                   zone=self.pain_zone),
            *self._cost_pair)[0])} if self.cost_obs else {}),
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
           if self._records_executed() else {}),
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
            'log/brake': np.float32(self._brake_left > 0),
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
        'log/turn_wheels': np.float32(0.5 * (
            float(sensors['wheel_speed_l']) - float(sensors['wheel_speed_r']))
            / self.command_turn),
        'log/yaw_rate': np.float32(self._yaw_rate),
        'log/zeta': np.float32(self._zeta()),
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
    link = _Link(sock, self.timeout, on_restart=(
        self._request_restart if self._restart_file else None),
        on_pause=self._request_pause, rate=self._weights_kbps * 1e3)
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
    self._wdelta = bool(hello.get('wdelta', False))
    self._confirm(self._phone_stamp)
    if self._base and str(self._base[0]) != str(self._phone_stamp):
      self._base = None   # the phone holds other weights: next push whole
    # Weights this run did not publish are another run's, typically the one
    # before a dashboard restart, and it would be that run's skill the phone
    # shows until our first push lands. So it holds the wheels until then.
    hold = self._onboard and not self._published(self._phone_stamp)
    wstreams = None
    if self.onboard and self._weight_streams > 0 and hello.get('wstreams'):
      if self._streams is None:
        self._streams = _Streams(self.host, self.port + 1, self.timeout)
      wstreams = self._streams.open_session(self._weight_streams)
    link.send(dict(
        type='hello', protocol=PROTOCOL, discrete=self.discrete,
        pipeline=self.pipeline, onboard=self._onboard,
        cmd_scale=self.command_scale, command_bias=self.command_bias,
        friction_comp=self.friction_comp,
        **({'command': self.command_sampler}
           if self.task == 'command' else {}),
        # The phone keeps this run's latest weights under `run`, with these
        # settings beside them, so it can replay the policy later on its own.
        run=self._run_name(), settings=self._settings(),
        restart=bool(self._restart_file), hold=hold,
        pause=self._pause_message(),
        **({'wstreams': wstreams} if wstreams else {}),
        **({'pace': self.pace} if self.pace else {})))
    self._link = link
    # Note `self.onboard`, not `self._onboard`: a phone that came up without
    # a policy negotiates onboard=False, and pushing to it anyway is exactly
    # how it bootstraps into running its own.
    if self.onboard and self._pusher is None:
      self._pusher = threading.Thread(
          target=self._push_loop, name='robot_pusher', daemon=True)
      self._pusher.start()

  def _published(self, stamp):
    """Whether this run published the weights with this stamp."""
    if self._policy_dir is None or not stamp:
      return True   # nothing to compare against: trust the phone
    return (self._policy_dir / 'policy' / f'policy_{stamp}.pkl').exists()

  def _request_restart(self):
    """The phone's dashboard asked for training from scratch. The supervisor
    sees the file, stops this run and starts the next one in its place."""
    print('[robot] the phone asked for a fresh run', flush=True)
    path = self._restart_file
    with open(path + '.tmp', 'w') as f:
      f.write(f'{time.time():.3f}\n')
    os.replace(path + '.tmp', path)

  def _load_pause(self, start_paused):
    from embodied.run import online
    if self._pause_file and os.path.exists(self._pause_file):
      return online.read_pause(self._pause_file)   # a resumed run keeps its own
    state = dict(robot=False, training=False, train_started=None,
                 from_start=False)
    if start_paused:
      state.update(robot=True, training=True, from_start=True)
      self._save_pause(state)
      print('[robot] starting with the robot and the training paused; '
            'start them from the dashboard', flush=True)
    return state

  def _save_pause(self, state):
    if self._pause_file:
      from embodied.run import online
      online.write_pause(self._pause_file, state)

  def _pause_message(self):
    """What the phone is told: the state, and how long ago training started
    (an age, so the two clocks need not agree)."""
    state = self._pause
    started = state['train_started']
    return dict(robot=state['robot'], training=state['training'],
                train_started_ago=(None if started is None
                                   else max(0.0, time.time() - started)),
                # Started paused and not yet trained: no minute 0 yet.
                waiting=state['from_start'] and started is None,
                **({'fatigue': dict(steps=self._fatigue.steps, tol=self._fatigue.tol,
                                    dead=self._fatigue.dead)}
                   if self.fatigue_s else {}),
                **({'travel': dict(lo=self.travel_lo, hi=self.travel_hi)}
                   if self.travel_hi else {}),
                # Emotion on the phone (PolicyRunner.configure): the affect
                # gains, the body for pain, the arousal event threshold.
                **({'emotion': dict(
                    affect=self.affect, emotion_event=self.emotion_event,
                    event_signal=self.event_signal,
                    body=dict(theta_zero=self.theta_zero, lo=self.bumper_lo,
                              hi=self.bumper_hi, zone=self.pain_zone))}
                   if self.weights_emotion else {}))

  def _request_pause(self, request):
    """The phone's dashboard paused or resumed the robot and/or the
    training. Called on the link's reader thread; answers with the new state."""
    with self._pause_lock:
      state = dict(self._pause)
      for key in ('robot', 'training'):
        if request.get(key) is not None:
          state[key] = bool(request[key])
      # Minute 0 of a run that started paused: the first time it trains.
      if (state['from_start'] and state['train_started'] is None
          and not state['training']):
        state['train_started'] = time.time()
      if state != self._pause:
        self._pause = state
        self._save_pause(state)
        print('[robot] dashboard: robot %s, training %s' % (
            'paused' if state['robot'] else 'running',
            'paused' if state['training'] else 'running'), flush=True)
    link = self._link
    if link is not None:
      try:
        link.send(dict(type='pause', **self._pause_message()))
      except LINKLOST:
        pass   # the next hello carries it

  def _run_name(self):
    """ROBOT_RUN_NAME, else the Slurm job name (e1293_robot_25hz), else the
    run dir."""
    name = os.environ.get('ROBOT_RUN_NAME') or os.environ.get('SLURM_JOB_NAME')
    if name:
      return name
    if self._policy_dir is not None:
      logdir = self._policy_dir.parent   # <logdir>/online_shared
      # Saion runs live in <run dir>/logdir; local ones are the logdir itself.
      return logdir.parent.name if logdir.name == 'logdir' else logdir.name
    return 'robot'

  def _settings(self):
    """What a player needs to pace and score steps the way training did."""
    return dict(task=self.task, hz=self.hz, length=self.length,
                **{k: getattr(self, k) for k in PLAYER_SETTINGS})

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
      if link is None or link.error is not None:
        continue
      if self._onboard and time.time() - self._stats_at > STATS_EVERY:
        self._stats_at = time.time()
        try:
          self._send_stats(link)
        except Exception as e:  # noqa: BLE001 -- display only
          print(f'[robot] could not forward learner metrics: {e}', flush=True)
      if link.pushing or (self._streams and self._streams.pushing):
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

  def _send_stats(self, link):
    """Forward the learner's newest metrics to the phone's dashboard.

    Reads only what metrics.jsonl gained since the last call. Sent to an
    onboard phone only: older apps ignore unknown messages there, but take
    any message for an action in lock-step and pipelined mode.
    """
    if self._policy_dir is None:
      return
    path = self._policy_dir.parent / 'metrics.jsonl'
    if not path.exists():
      return
    with open(str(path), 'rb') as f:
      f.seek(self._stats_offset)
      chunk = f.read()
    end = chunk.rfind(b'\n') + 1
    self._stats_offset += end
    stats = {}
    for line in chunk[:end].splitlines():
      try:
        row = json.loads(line)
      except ValueError:
        continue
      for name, keys in STATS:
        if not any(k in row for k in keys):
          continue
        value = sum(float(row[k]) for k in keys if k in row)
        if name == 'updates':
          self._stats_updates += value
          value = self._stats_updates
        stats[name] = value
      if 'step' in row and any(k.startswith('train/') for k in row):
        stats['train_step'] = float(row['step'])
    if stats:
      link.send(dict(type='stats', stats=stats))

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
    arrays, meta = exportlib.pack_arrays(
        self._config, params, self._weights_dtype, self.weights_emotion)
    base = self._base if self._wdelta else None
    blob = exportlib.encode(
        meta, arrays, stamp=stamp, base=base,
        compress=(self._weights_dtype == 'int8'))
    self._pending = (stamp, arrays)
    packed = time.time()

    def done(header):
      self._weights_stamp = header['stamp']
      now = time.time()
      t = self._wtiming if str(self._wtiming.get('stamp')) == str(stamp) \
          else {}
      phone = (f', phone: parts over {t.get("recv_s")}s, build '
               f'{t.get("build_ms")}ms, install {t.get("install_ms")}ms'
               if t and not t.get('failed') else '')
      handed = header.get('handed_s')
      print(f'[robot] pushed policy {stamp} to the phone '
            f'({len(blob) / 1e6:.2f} MB {"delta" if base else "whole"}, '
            f'pack {packed - started:.2f}s, send {now - packed:.2f}s'
            + (f' (sent off after {handed:.2f}s)' if handed is not None else '')
            + f', age {now - int(stamp) / 1e9:.1f}s, '
            f'{header.get("via", "control link")}{phone})',
            flush=True)

    self._push_started = started
    streams = self._streams
    if streams and streams.push(stamp, blob, done, lambda: self._status(stamp)):
      return
    link.push(dict(type='weights', stamp=stamp), blob, self._chunked, done)

  def _confirm(self, stamp):
    """The phone reports running `stamp`: the push in flight, if it is
    that one, is now what the next delta builds on."""
    pending = self._pending
    if pending and stamp is not None and str(stamp) == str(pending[0]):
      self._base, self._pending = pending, None

  def _status(self, stamp):
    """'ok' once the phone runs `stamp`, 'failed' if it said it could not
    load it, else None (still on its way)."""
    if str(self._phone_stamp) == str(stamp):
      return 'ok'
    t = self._wtiming
    if t.get('failed') and str(t.get('stamp')) == str(stamp):
      return 'failed'
    return None

  def _records_executed(self):
    # Onboard only. The phone reports, with each observation, the action it
    # took AT that observation, which is what a transition pairs with it.
    # Off the phone (the sim) the action this env could report is the one
    # that LED to the observation, a step late: recorded as executed it
    # shifted every action by one step (rounds 7f-10 and E1-E3, 11-04/05).
    # So the sim records the policy's action, also where the brake or the
    # cap changed it.
    return not self.discrete and (bool(self.onboard) or self._phone_runs())

  def _phone_runs(self):
    return False   # the sim's phone (robot_mujoco.py, sim_phone)

  def _executed(self):
    if self._phone_final is not None:
      act = self._phone_final
    elif self._onboard and self._onboard_act is not None:
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
    # False from the moment the phone's base stops answering until it has
    # answered cleanly for a while (dreamerBridge main.BASE_HOLD_S); its
    # sensors are frozen and its wheels held at zero meanwhile. A phone that
    # predates the flag sends none.
    self._base_ok = bool(header.get('base_ok', True))
    self._base_health = {
        k: header[k] for k in (
            'base_reset_cause', 'base_prev_stage', 'base_boots',
            'base_motor_fail', 'base_bus_recoveries', 'base_self_resets')
        if k in header}
    # The command the phone's policy saw for this observation. A phone that
    # predates commands sends none, which is the same as a centred stick.
    cmd = header.get('cmd') or (0.0, 0.0)
    self._cmd = (float(np.clip(cmd[0], -1, 1)), float(np.clip(cmd[1], -1, 1)))
    self._cmd_src = float(header.get('cmd_src') == 'joystick')
    self._yaw_rate = self.yaw_sign * float(
        header.get('gyro_' + self.yaw_axis, 0.0) or 0.0)
    if self._onboard:
      act = header.get('act')
      if act is None:
        raise ValueError(
            'Onboard mode negotiated but the phone sent no action; refusing '
            'to record a transition whose action we would have to invent')
      self._onboard_act = np.asarray(act, np.float32)
      wtiming = header.get('wtiming')
      if wtiming:
        self._wtiming = wtiming
        if wtiming.get('failed'):
          print(f'[robot] the phone could not load push {wtiming.get("stamp")}'
                f' ({wtiming.get("error")}); sending the next one whole',
                flush=True)
          self._base = self._pending = None
      self._phone_stamp = header.get('policy_stamp', self._phone_stamp)
      self._confirm(self._phone_stamp)
      event = header.get('emotion_event')
      if event and event.get('n') != self._event_n and self._policy_dir:
        # The phone's arousal event: the learner changes its weights now.
        self._event_n = event.get('n')
        write_event(self._policy_dir, event)
    self._timing = (
        float(header.get('wait_ms', 0.0)), float(header.get('work_ms', 0.0)),
        float(header.get('imu_age_ms', 0.0)),
        float(header.get('imu_stale_ms', 0.0)))
    return self._last, time.time() - self._sent
