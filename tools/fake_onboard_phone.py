"""Stand-in for the phone in onboard mode: the real PolicyRunner and receiver, no robot.

Runs the same two modules the dreamerBridge app runs
(`dreamerv3/deploy/policy_runner.py`, `dreamerv3/deploy/bridge_link.py`, both
verbatim copies) on a fixed clock, so the trainer's link -- every observation
recorded, weights pushed in the background -- can be measured at a given rate
without a phone. Starts without weights, lets the trainer drive until its first
push lands (the same bootstrap the app does), then reconnects onboard.

Prints, and writes as JSON with --out, what matters for the link: the rate it
held, how late ticks were and what the policy step cost, split by whether a
weight push was arriving at the time, plus how many pushes were installed and
how old the weights were.

  .venv/bin/python tools/fake_onboard_phone.py --port 3000 --hz 50 --seconds 300
"""

import argparse
import json
import pathlib
import socket
import struct
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# policy_runner imports numpy_policy flat, as it does in the app.
sys.path.insert(0, str(ROOT / 'dreamerv3' / 'deploy'))

from dreamerv3.deploy import bridge_link  # noqa: E402
from dreamerv3.deploy.policy_runner import PolicyRunner  # noqa: E402

PROTOCOL = 3


def write(sock, header, blob=b''):
  payload = json.dumps(dict(header, blob_len=len(blob))).encode('utf-8')
  sock.sendall(struct.pack('>I', len(payload)) + payload + blob)


def connect(args, runner, directory):
  deadline = time.time() + args.connect_timeout
  while True:
    try:
      sock = socket.create_connection((args.host, args.port), timeout=30.0)
      break
    except OSError as e:
      if time.time() > deadline:
        raise
      print(f'waiting for the trainer at {args.host}:{args.port} ({e})')
      time.sleep(1.0)
  sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  write(sock, dict(type='hello', protocol=PROTOCOL, control_hz=args.hz,
                   robot_id=1, onboard=runner.ready, policy_stamp=runner.stamp,
                   pace='clock', max_hz=args.hz, wchunk=True))
  receiver = bridge_link.Receiver(
      sock, build=lambda blob, header: PolicyRunner.build_blob(
          blob, directory + '/policy.npz'))
  hello, _ = receiver.get(timeout=60.0)
  assert hello['type'] == 'hello', hello
  onboard = bool(hello.get('onboard')) and runner.ready
  print(f'connected ({"onboard" if onboard else "trainer drives"}): {hello}')
  return sock, receiver, onboard


class Body:
  """A cartpole-ish tilt so the sensors are not constant."""

  def __init__(self, dt):
    self.dt = dt
    self.theta = 0.0
    self.omega = 0.0
    self.left = 0.0
    self.right = 0.0

  def step(self, left, right):
    push = 0.5 * (left + right)
    self.omega += self.dt * (8.0 * self.theta - 6.0 * push)
    self.theta = float(np.clip(self.theta + self.dt * self.omega, -0.12, 0.18))
    self.left, self.right = 3000.0 * left, 3000.0 * right

  def sensors(self):
    return dict(theta=self.theta, angular_velocity=self.omega,
                wheel_speed_l=self.left, wheel_speed_r=self.right,
                wheel_distance_l=0.0, wheel_distance_r=0.0,
                wheel_count_l=0.0, wheel_count_r=0.0, battery_voltage=4.0,
                charger_voltage=0.0, coil_voltage=0.0)


class Commands:
  """The app's 'auto' command sampler, roughly: a random (forward, turn)
  held for 2-5 s, zero 30% of the time. Only a command-task trainer reads it."""

  def __init__(self, seed=0):
    self.rng = np.random.RandomState(seed)
    self.cmd = [0.0, 0.0]
    self.until = 0.0

  def tick(self):
    now = time.monotonic()
    if now >= self.until:
      zero = self.rng.rand() < 0.3
      self.cmd = [0.0, 0.0] if zero else [
          float(x) for x in self.rng.uniform(-1, 1, 2)]
      self.until = now + self.rng.uniform(2.0, 5.0)
    return list(self.cmd)


def pct(values, q):
  return float(np.percentile(values, q)) if len(values) else float('nan')


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--host', default='127.0.0.1')
  parser.add_argument('--port', type=int, default=3000)
  parser.add_argument('--hz', type=float, default=50.0)
  parser.add_argument('--seconds', type=float, default=120.0)
  parser.add_argument('--connect_timeout', type=float, default=600.0)
  parser.add_argument('--out', default='')
  args = parser.parse_args(argv)

  directory = tempfile.mkdtemp(prefix='fake_phone_weights_')
  runner = PolicyRunner()
  body = Body(1.0 / args.hz)
  commands = Commands()
  period = 1.0 / args.hz
  sock, receiver, onboard = connect(args, runner, directory)

  late_ms, policy_ms, during = [], [], []
  installs, ages = [], []
  sent = 0
  seq = 0
  start = None
  next_tick = time.monotonic()
  last_frames = 0
  while True:
    now = time.monotonic()
    if onboard and start is None:
      start = now
    if start is not None and now - start >= args.seconds:
      break
    # Pace on a fixed grid, like the app's max_hz slots.
    next_tick += period
    if next_tick < now - period:
      next_tick = now + period
    remaining = next_tick - time.monotonic()
    if remaining > 0:
      time.sleep(remaining)
    tick = time.monotonic()
    # Whether a weight push was arriving: bytes for one landed this tick.
    pushing = receiver._partial is not None
    if onboard:
      snap = body.sensors()
      cmd = commands.tick()
      _, act = runner.act(snap, cmd)
      body.step(act[0], act[1])
      applied = time.monotonic()
      seq += 1
      write(sock, dict(type='obs', step=seq, seq=seq, t=time.time(), fresh=True,
                       pace='clock', act=act, is_first=runner.was_first,
                       cmd=cmd, cmd_src='auto',
                       wait_ms=0.0, work_ms=(applied - tick) * 1e3,
                       policy_ms=runner.last_ms, policy_stamp=runner.stamp,
                       sensors=snap))
      sent += 1
      if start is not None:
        late_ms.append((tick - next_tick) * 1e3)
        policy_ms.append(runner.last_ms)
        during.append(pushing)
    else:
      seq += 1
      write(sock, dict(type='obs', step=seq, seq=seq, t=time.time(),
                       fresh=True, sensors=body.sensors(),
                       cmd=commands.tick(), cmd_src='auto'))
    # Drain what the receiver finished, as the app's drain_control does.
    while True:
      frame = receiver.get_nowait()
      if frame is None:
        break
      header, payload = frame
      kind = header.get('type')
      if kind == 'weights_ready':
        stamp = header.get('stamp')
        runner.install(payload, stamp)
        try:
          ages.append(time.time() - int(stamp) / 1e9)
        except (TypeError, ValueError):
          pass
        installs.append(header.get('build_ms', 0.0))
        if not onboard:
          print('got a policy; reconnecting onboard')
          receiver.close()
          sock.close()
          sock, receiver, onboard = connect(args, runner, directory)
          next_tick = time.monotonic()
          break
      elif kind == 'act':
        body.step(header['left'], header['right'])
      elif kind == 'ctrl' and header.get('reset'):
        runner.reset()
  receiver.close()
  sock.close()

  late = np.array(late_ms)
  pol = np.array(policy_ms)
  dur = np.array(during, bool)
  stats = dict(
      hz=args.hz, seconds=args.seconds, sent=sent,
      achieved_hz=sent / args.seconds,
      late_p50_ms=pct(late, 50), late_p99_ms=pct(late, 99),
      late_max_ms=float(late.max()) if len(late) else float('nan'),
      late_p99_push_ms=pct(late[dur], 99), late_p99_idle_ms=pct(late[~dur], 99),
      policy_p50_ms=pct(pol, 50), policy_p99_ms=pct(pol, 99),
      policy_p99_push_ms=pct(pol[dur], 99), policy_p99_idle_ms=pct(pol[~dur], 99),
      ticks_during_push=int(dur.sum()),
      weights_installed=len(installs),
      build_ms_p50=pct(installs, 50),
      push_interval_s=(args.seconds / len(installs)) if installs else None,
      weight_age_p50_s=pct(ages, 50), weight_age_max_s=max(ages) if ages else None,
  )
  print(json.dumps(stats, indent=1))
  if args.out:
    pathlib.Path(args.out).write_text(json.dumps(stats, indent=1))


if __name__ == '__main__':
  main()
