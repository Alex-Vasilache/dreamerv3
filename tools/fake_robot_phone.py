"""Stand-in for the phone, for testing embodied/envs/robot.py without hardware.

Speaks the same wire protocol as the `dreamerBridge` Chaquopy app in the
smartphone-robot-android repo, so it doubles as the reference implementation:
a 4-byte big-endian header length, a UTF-8 JSON header, then `blob_len` bytes
of binary payload. The phone owns the control clock -- it applies the action it
is given, waits out the control period, then reports sensors.

  .venv/bin/python tools/fake_robot_phone.py --host 127.0.0.1 --port 3000
"""

import argparse
import json
import math
import socket
import struct
import sys
import time

PROTOCOL = 1

# The real robot reports encoder speeds of order 1e3 at full PWM, not 1.
# Match that so rewards here mean the same as rewards on hardware.
COUNTS_PER_PWM = 1000.0


class Connection:

  def __init__(self, sock):
    self.sock = sock
    self.file = sock.makefile('rb')

  def write(self, header, blob=b''):
    header = dict(header, blob_len=len(blob))
    payload = json.dumps(header).encode('utf-8')
    self.sock.sendall(struct.pack('>I', len(payload)) + payload + blob)

  def read(self):
    length, = struct.unpack('>I', self._exactly(4))
    header = json.loads(self._exactly(length).decode('utf-8'))
    return header, self._exactly(header.get('blob_len', 0))

  def _exactly(self, amount):
    if not amount:
      return b''
    data = self.file.read(amount)
    if data is None or len(data) < amount:
      raise ConnectionError('Trainer closed the connection.')
    return data


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

  robot = Robot(dt)
  step = 0
  try:
    while not args.steps or step < args.steps:
      act, _ = conn.read()
      assert act['type'] == 'act', act
      if act['reset']:
        robot.reset()
      else:
        robot.apply(act['left'], act['right'])
      time.sleep(dt)
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
