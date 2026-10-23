"""Drive the robot through scripted wheel tests, for fitting the MuJoCo model.

The simulator (embodied/envs/robot_mujoco.py) has one motor model and one
contact friction for every floor, while the real robot learns to balance fast
on carpet and has not on the table. This records what the fit needs, per
surface: what the motors do with a given output, and what the body does when
the wheels push against that floor. Each phase is a fixed list of tests; the
recording keeps every tick with the test it belongs to.

  air      the robot on a stand, wheels free (once; the floor plays no part):
           slow ramps (start/stop friction), steps at several outputs both
           ways (spin-up, top speed, coast-down), quick reversals, a random
           +/- signal, one wheel at a time
  bumper   on the surface, resting on a bumper: the same steps with the body
           loaded (they drive it along the floor, alternating direction so it
           stays roughly in place), short kicks that tip it off the bumper,
           spins in place (gyro against wheels: slip and scrub), a random
           signal
  release  on the surface, motors at zero: hold it upright and let go, a few
           times, and push it gently both ways -- the free fall and how the
           wheels roll under it

Point the phone at this computer and run it (the dreamerBridge app, no
trainer attached; tools/robot/stop.sh first):

  .venv/bin/python tools/robot/dev/sysid_probe.py --surface carpet
  .venv/bin/python tools/robot/dev/sysid_probe.py --surface table --phases bumper release
  .venv/bin/python tools/robot/dev/sysid_probe.py --list      # the tests, no robot

Recordings land in ~/logdir/robot_sysid/<date>_<surface>/<phase>.csv. The
robot WILL DRIVE in the bumper phase: about a metre of clear floor.
"""

import argparse
import csv
import json
import math
import os
import select
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_latency_probe import HEADER_KEYS, Link, PROTOCOL, SENSOR_KEYS  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
APP = 'jp.oist.abcvlib.dreamerBridge'
PHONE_FILES = f'/sdcard/Android/data/{APP}/files'


# -- the tests ----------------------------------------------------------------
# A test is (name, seconds, f) with f(t) -> (left, right) motor outputs, t in
# seconds from the test's start. Positive output drives forward.

def hold(l, r=None):
  r = l if r is None else r
  return lambda t: (l, r)


def ramp(peak, seconds):
  """0 -> peak -> 0, linearly, over `seconds`."""
  return lambda t: (peak * (1 - abs(2 * t / seconds - 1)),) * 2


def pulses(values, on, off):
  """Each value held for `on` s, then zero for `off` s."""
  def f(t):
    i, rem = divmod(t, on + off)
    i = int(i)
    if i >= len(values) or rem >= on:
      return (0.0, 0.0)
    v = values[i]
    return v if isinstance(v, tuple) else (v, v)
  return f, len(values) * (on + off)


def square(amp, half):
  return lambda t: (amp,) * 2 if (t // half) % 2 == 0 else (-amp,) * 2


def prbs(amp, period, seed, zero=False):
  rng = np.random.default_rng(seed)
  bits = rng.random(10000) > 0.5
  def f(t):
    b = bits[int(t / period) % len(bits)]
    v = amp if b else (0.0 if zero else -amp)
    return (v, v)
  return f


def tests(phase):
  out = []
  def add(name, f, seconds=None):
    if seconds is None:
      f, seconds = f
    out.append((name, float(seconds), f))
    out.append((name + ':rest', 1.0, hold(0.0)))

  if phase == 'air':
    for peak in (0.5, -0.5):
      add(f'ramp{peak:+.1f}', ramp(peak, 20.0), 20.0)
    for a in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0):
      add(f'step{a:.2f}', pulses([a, -a], 1.5, 1.5))
    add('reverse0.5', square(0.5, 0.6), 4.8)
    add('reverse1.0', square(1.0, 0.6), 4.8)
    add('prbs0.5', prbs(0.5, 0.05, 1), 10.0)
    add('prbs0_0.5', prbs(0.5, 0.05, 2, zero=True), 10.0)
    add('left0.5', pulses([(0.5, 0.0), (-0.5, 0.0)], 1.5, 1.5))
    add('right0.5', pulses([(0.0, 0.5), (0.0, -0.5)], 1.5, 1.5))
  elif phase == 'bumper':
    # Alternating signs keep it near where it started.
    for a in (0.2, 0.3, 0.4, 0.6, 0.8, 1.0):
      add(f'step{a:.1f}', pulses([a, -a], 0.4, 1.6))
    for ms in (60, 120, 200):
      for a in (0.5, 0.8, 1.0):
        add(f'kick{a:.1f}_{ms}ms', pulses([a, -a, a, -a], ms / 1000, 1.5))
    for a in (0.3, 0.5, 0.8):
      add(f'spin{a:.1f}', pulses([(a, -a), (-a, a)], 1.5, 1.0))
    add('prbs0.4', prbs(0.4, 0.06, 3), 10.0)
    add('prbs0_0.6', prbs(0.6, 0.06, 4, zero=True), 10.0)
  elif phase == 'release':
    add('release', hold(0.0), 40.0)
  else:
    raise ValueError(phase)
  return out


PROMPTS = {
    'air': 'Put the robot on a stand with both wheels off the ground.',
    'bumper': 'Set the robot on the {surface}, resting on a bumper, ~1 m clear '
              'on both sides. It will drive.',
    'release': 'Motors stay at zero for 40 s. On the {surface}: hold the robot '
               'upright and let go, catch it before it falls flat, repeat; '
               'then give it a few gentle pushes both ways.',
}


# -- the phone ------------------------------------------------------------------

def adb(*args, check=True):
  return subprocess.run(['adb', *args], check=check, capture_output=True,
                        text=True).stdout


def local_ip():
  for i in ('en0', 'en1', 'en2'):
    out = subprocess.run(['ipconfig', 'getifaddr', i], capture_output=True,
                         text=True).stdout.strip()
    if out:
      return out
  raise SystemExit('no LAN address found; pass --ip')


def point_phone(ip, port, hz):
  """Write trainer.json to point the app here, and start the app if needed."""
  with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, 'trainer.json')
    with open(path, 'w') as f:
      json.dump(dict(ip=ip, port=port, max_hz=hz), f)
    adb('shell', 'mkdir', '-p', PHONE_FILES)
    adb('push', path, PHONE_FILES + '/')
  if not adb('shell', 'pidof', APP, check=False).strip():
    adb('shell', 'monkey', '-p', APP, '-c', 'android.intent.category.LAUNCHER',
        '1')
    subprocess.run([os.path.join(ROOT, 'tools/robot/phone_allow_usb.sh'), '10'])


def enter_pressed():
  return bool(select.select([sys.stdin], [], [], 0)[0]) and (
      sys.stdin.readline() is not None)


def run_phase(link, phase, surface, writer):
  print('\n== ' + phase + ': ' + PROMPTS[phase].format(surface=surface))
  print('   press Enter to start (wheels held at zero until then)', flush=True)
  plan = tests(phase)
  total = sum(s for _, s, _ in plan)
  started = None
  index, t_test = 0, 0.0
  n, shown = 0, None
  while True:
    header = link.read()
    if header.get('type') != 'obs':
      continue
    now = time.monotonic()
    if started is None:
      left = right = 0.0
      name = 'wait'
      if enter_pressed():
        started = t_test = now
        print(f'   running {len(plan)} tests, {total:.0f} s', flush=True)
    if started is not None:
      while index < len(plan) and now - t_test >= plan[index][1]:
        t_test += plan[index][1]
        index += 1
      if index >= len(plan):
        link.write(dict(type='act', left=0.0, right=0.0, reset=False,
                        reward=0.0, seq=header.get('seq')))
        print(f'   done: {n} ticks', flush=True)
        return
      name, _, f = plan[index]
      left, right = (float(np.clip(v, -1, 1)) for v in f(now - t_test))
      if shown != index and not name.endswith(':rest'):
        shown = index
        print(f'   {name}', flush=True)
    link.write(dict(type='act', left=left, right=right, reset=False,
                    reward=0.0, seq=header.get('seq')))
    if started is None:
      continue
    sensors = header.get('sensors', {})
    row = dict(phase=phase, surface=surface, test=name, test_s=now - started,
               cmd_l=left, cmd_r=right)
    for k in HEADER_KEYS:
      row[k] = header.get(k, float('nan'))
    for k in SENSOR_KEYS:
      row[k] = sensors.get(k, float('nan'))
    writer.writerow(row)
    n += 1


def main():
  ap = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--surface', default='none',
                  help='carpet, table, ...: names the recording')
  ap.add_argument('--phases', nargs='+', default=['air', 'bumper', 'release'],
                  choices=['air', 'bumper', 'release'])
  ap.add_argument('--port', type=int, default=3000)
  ap.add_argument('--ip', default=None, help='this computer, as the phone sees it')
  ap.add_argument('--hz', type=float, default=65.0)
  ap.add_argument('--no_point', action='store_true',
                  help='leave trainer.json and the app alone')
  ap.add_argument('--out', default=os.path.expanduser('~/logdir/robot_sysid'))
  ap.add_argument('--list', action='store_true', help='print the tests and exit')
  args = ap.parse_args()

  if args.list:
    for phase in args.phases:
      plan = tests(phase)
      print(f'{phase}: {sum(s for _, s, _ in plan):.0f} s')
      for name, s, _ in plan:
        if not name.endswith(':rest'):
          print(f'  {name:16s} {s:5.1f} s')
    return

  out = os.path.join(args.out, time.strftime('%Y%m%dT%H%M%S') + '_' + args.surface)
  os.makedirs(out, exist_ok=True)
  server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  server.bind(('0.0.0.0', args.port))
  server.listen(1)
  if not args.no_point:
    ip = args.ip or local_ip()
    point_phone(ip, args.port, args.hz)
    print(f'phone pointed at {ip}:{args.port}')
  print('waiting for the phone ...', flush=True)
  sock, addr = server.accept()
  sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  link = Link(sock)
  hello = link.read()
  if hello.get('type') != 'hello' or hello.get('protocol') != PROTOCOL:
    raise SystemExit(f'unexpected hello from {addr}: {hello}')
  print(f'phone connected from {addr[0]} at {hello.get("control_hz")} Hz')
  # As a training run sends it, minus the policy: friction_comp off, the
  # phone's default handling of outputs inside the dead zone. Driven from
  # here rather than by the phone's own policy, the phone paces itself to
  # the serial link: ~39 Hz (2026-10-23) against a training run's 63.
  link.write(dict(type='hello', protocol=PROTOCOL, discrete=False,
                  pipeline=True, onboard=False, run='sysid'))
  fields = ['phase', 'surface', 'test', 'test_s', 'cmd_l', 'cmd_r',
            *HEADER_KEYS, *SENSOR_KEYS]
  try:
    for phase in args.phases:
      path = os.path.join(out, phase + '.csv')
      with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        run_phase(link, phase, args.surface, writer)
      print(f'   wrote {path}')
  except (ConnectionError, KeyboardInterrupt, socket.timeout) as e:
    print(f'stopped: {type(e).__name__}: {e}')
  finally:
    try:
      link.write(dict(type='act', left=0.0, right=0.0, reset=False, reward=0.0))
    except OSError:
      pass
    sock.close()
    server.close()
  print(f'recordings in {out}')


if __name__ == '__main__':
  main()
