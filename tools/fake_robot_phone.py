"""Stand-in for the phone, for testing embodied/envs/robot.py without hardware.

Speaks the same wire protocol as the `dreamerBridge` Chaquopy app in the
smartphone-robot-android repo, so it doubles as the reference implementation:
a 4-byte big-endian header length, a UTF-8 JSON header, then `blob_len` bytes
of binary payload. The phone owns the control clock -- it applies the action it
is given, waits out the control period, then reports sensors.

The trainer's handshake reply picks the timing mode. Pipelined (the default) is
the one that reaches 50Hz: we tick on our own clock, apply whatever action has
arrived by then, and report every tick without ever blocking on the trainer.
With `pipeline` false we fall back to lock-step, blocking for an action before
each tick.

  .venv/bin/python tools/fake_robot_phone.py --host 127.0.0.1 --port 3000
"""

import argparse
import json
import math
import socket
import struct
import sys
import time

PROTOCOL = 2

# The real robot reports encoder speeds of order 1e3 at full PWM, not 1.
# Match that so rewards here mean the same as rewards on hardware.
COUNTS_PER_PWM = 1000.0


class Connection:
  """Framed JSON over TCP, with a non-blocking read for the pipelined mode."""

  def __init__(self, sock):
    self.sock = sock
    self.buffer = bytearray()

  def write(self, header, blob=b''):
    header = dict(header, blob_len=len(blob))
    payload = json.dumps(header).encode('utf-8')
    self.sock.sendall(struct.pack('>I', len(payload)) + payload + blob)

  def read(self):
    while True:
      frame = self._parse()
      if frame is not None:
        return frame
      self._fill(block=True)

  def read_nowait(self):
    frame = self._parse()
    if frame is not None:
      return frame
    if not self._fill(block=False):
      return None
    return self._parse()

  def _parse(self):
    if len(self.buffer) < 4:
      return None
    length, = struct.unpack('>I', self.buffer[:4])
    if len(self.buffer) < 4 + length:
      return None
    header = json.loads(bytes(self.buffer[4:4 + length]).decode('utf-8'))
    total = 4 + length + header.get('blob_len', 0)
    if len(self.buffer) < total:
      return None
    blob = bytes(self.buffer[4 + length:total])
    del self.buffer[:total]
    return header, blob

  def _fill(self, block):
    self.sock.settimeout(30.0 if block else 0.0)
    try:
      chunk = self.sock.recv(65536)
    except (BlockingIOError, InterruptedError):
      return False
    finally:
      self.sock.settimeout(30.0)
    if not chunk:
      raise ConnectionError('Trainer closed the connection.')
    self.buffer.extend(chunk)
    return True


class Robot:
  """Crude two-wheel cart so the numbers move in a plausible way."""

  def __init__(self, dt):
    self.dt = dt
    self.reset()

  def reset(self):
    self.speed_l = 0.0
    self.speed_r = 0.0
    self.distance_l = 0.0
    self.distance_r = 0.0
    self.theta = 0.0
    self.rate = 0.0

  def apply(self, left, right):
    # First-order lag from PWM to wheel speed, plus a tilt that leans against
    # acceleration and decays back to upright.
    accel_l = (left - self.speed_l) * 0.4
    accel_r = (right - self.speed_r) * 0.4
    self.speed_l += accel_l
    self.speed_r += accel_r
    self.distance_l += self.speed_l * self.dt
    self.distance_r += self.speed_r * self.dt
    self.rate = 0.7 * self.rate - 2.0 * (accel_l + accel_r) - 3.0 * self.theta
    self.theta += self.rate * self.dt

  def sensors(self):
    return dict(
        wheel_speed_l=self.speed_l * COUNTS_PER_PWM,
        wheel_speed_r=self.speed_r * COUNTS_PER_PWM,
        wheel_distance_l=self.distance_l,
        wheel_distance_r=self.distance_r,
        theta=self.theta,
        angular_velocity=self.rate,
        battery_voltage=3.9,
        charger_voltage=0.0,
        coil_voltage=0.0,
    )


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--host', default='127.0.0.1')
  parser.add_argument('--port', type=int, default=3000)
  parser.add_argument('--control_hz', type=float, default=10.0)
  parser.add_argument('--steps', type=int, default=0, help='0 runs forever')
  parser.add_argument('--connect_timeout', type=float, default=300.0)
  args = parser.parse_args(argv)

  dt = 1.0 / args.control_hz
  # The trainer only binds its listening socket once the run loop reaches the
  # first env step, which is well after JAX starts up, so keep retrying.
  deadline = time.time() + args.connect_timeout
  while True:
    try:
      sock = socket.create_connection((args.host, args.port), timeout=30.0)
      break
    except OSError as e:
      if time.time() > deadline:
        raise
      print(f'Waiting for the trainer at {args.host}:{args.port} ({e})')
      time.sleep(1.0)
  sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  conn = Connection(sock)
  conn.write(dict(
      type='hello', protocol=PROTOCOL, control_hz=args.control_hz, robot_id=1))
  hello, _ = conn.read()
  assert hello['type'] == 'hello' and hello['protocol'] == PROTOCOL, hello
  print(f'Connected to trainer at {args.host}:{args.port}: {hello}')

  pipeline = bool(hello.get('pipeline', True))
  print(f'Timing mode: {"pipelined" if pipeline else "lock-step"}')

  robot = Robot(dt)
  step = 0
  next_tick = time.monotonic()
  try:
    while not args.steps or step < args.steps:
      if pipeline:
        # Never block on the trainer: take the newest action that has arrived,
        # keep the last one otherwise, and hold the tick.
        act = None
        while True:
          frame = conn.read_nowait()
          if frame is None:
            break
          act = frame[0]
      else:
        act, _ = conn.read()
      if act is not None:
        assert act['type'] == 'act', act
        if act['reset']:
          robot.reset()
        else:
          robot.apply(act['left'], act['right'])
      next_tick = max(next_tick + dt, time.monotonic()) if pipeline else (
          time.monotonic() + dt)
      remaining = next_tick - time.monotonic()
      remaining > 0 and time.sleep(remaining)
      conn.write(dict(type='obs', step=step, t=time.time(),
                      sensors=robot.sensors()))
      step += 1
      if step % 100 == 0:
        print(f'step {step}  theta={math.degrees(robot.theta):+.1f}deg  '
              f'speed=({robot.speed_l:+.2f}, {robot.speed_r:+.2f})')
  except (ConnectionError, KeyboardInterrupt) as e:
    print(f'Stopped after {step} steps: {type(e).__name__}: {e}')
  finally:
    sock.close()


if __name__ == '__main__':
  sys.exit(main())
