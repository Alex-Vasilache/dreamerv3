"""Round-trip the protocol-3 onboard handshake against the real Robot env.

Checks the half that runs on the trainer: that it negotiates onboard mode, that
it stops sending actions of its own, and that the action the phone reports comes
back out of `env.step` on the `executed/drive` channel the driver lifts over the
actor's action.

No robot and no phone -- a socket in a thread plays the phone.

  .venv/bin/python tools/test_onboard_protocol.py
"""

import json
import pathlib
import socket
import struct
import sys
import threading

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from embodied.envs import robot as robotenv  # noqa: E402

PORT = 3907
SENSORS = dict(
    wheel_speed_l=100.0, wheel_speed_r=-50.0, wheel_distance_l=0.0,
    wheel_distance_r=0.0, wheel_count_l=0.0, wheel_count_r=0.0,
    theta=0.05, angular_velocity=0.3, battery_voltage=4.0,
    charger_voltage=0.0, coil_voltage=0.0)

# What the "phone" claims it drove. Deliberately not something the actor would
# produce, so a pass cannot be a coincidence.
PHONE_ACTS = [[0.11, -0.22], [0.33, -0.44], [0.55, -0.66], [-0.77, 0.88]]


def frames(sock):
  buf = bytearray()
  while True:
    while len(buf) >= 4:
      n, = struct.unpack('>I', buf[:4])
      if len(buf) < 4 + n:
        break
      header = json.loads(bytes(buf[4:4 + n]).decode())
      total = 4 + n + header.get('blob_len', 0)
      if len(buf) < total:
        break
      del buf[:total]
      yield header
    chunk = sock.recv(65536)
    if not chunk:
      return
    buf.extend(chunk)


def write(sock, header, blob=b''):
  header = dict(header, blob_len=len(blob))
  payload = json.dumps(header).encode()
  sock.sendall(struct.pack('>I', len(payload)) + payload + blob)


def phone(results):
  sock = socket.create_connection(('127.0.0.1', PORT), timeout=10)
  sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  write(sock, dict(type='hello', protocol=robotenv.PROTOCOL, control_hz=50.0,
                   robot_id=1, onboard=True, policy_stamp='test'))
  stream = frames(sock)
  results['hello'] = next(stream)
  results['ctrl'] = []

  def send(act):
    write(sock, dict(
        type='obs', step=0, t=0.0, fresh=True, act=act, is_first=False,
        wait_ms=0.1, work_ms=3.0, policy_ms=2.5, policy_stamp='test',
        imu_age_ms=7.0, imu_stale_ms=2.0, sensors=SENSORS))

  # The real phone free-runs and reports on its own clock; the trainer replies
  # when it has something to say. So: report once to get things moving, then
  # report again each time a reply lands.
  sent = results['sent'] = []
  index = 0

  def next_act():
    nonlocal index
    act = PHONE_ACTS[index % len(PHONE_ACTS)]
    index += 1
    sent.append(act)
    send(act)

  next_act()
  try:
    for header in stream:
      results['ctrl'].append(header)
      next_act()
  except OSError:
    pass
  finally:
    sock.close()


def main():
  results = {}
  env = robotenv.SmartphoneRobot(
      task='balance', host='127.0.0.1', port=PORT, length=1000,
      discrete=False, symmetric=False, onboard=True, reconnect=False,
      fall_angle=0.0)

  assert 'executed/drive' in env.obs_space, (
      'onboard env must advertise executed/drive so replay knows its shape')

  thread = threading.Thread(target=phone, args=(results,), daemon=True)
  thread.start()

  obs = env.step({'reset': np.array(True), 'drive': np.zeros(2, np.float32)})
  seen = []
  for i in range(6):
    obs = env.step({
        'reset': np.array(False),
        # What the actor would have asked for -- must be discarded.
        'drive': np.array([9.0, 9.0], np.float32)})
    seen.append(np.asarray(obs['executed/drive']).tolist())
  thread.join(timeout=5)

  ok = True
  print('handshake reply :', results.get('hello'))
  if not results.get('hello', {}).get('onboard'):
    print('FAIL: trainer did not accept onboard mode'); ok = False

  kinds = {c.get('type') for c in results.get('ctrl', [])}
  print('downstream types:', kinds)
  if kinds != {'ctrl'}:
    print(f'FAIL: expected only ctrl frames downstream, got {kinds}'); ok = False
  if any('left' in c or 'right' in c for c in results.get('ctrl', [])):
    print('FAIL: trainer still sent an action of its own'); ok = False

  print('executed/drive  :', seen)
  # Not a fixed offset into what the phone sent: the pipelined reader keeps
  # only the newest queued report and counts the rest as dropped, so how far
  # ahead the phone has run when a step lands is a race. What must hold is
  # that every action recorded is one the phone actually drove -- never the
  # [9, 9] the actor asked for -- and that they arrive in order.
  actor_asked = [9.0, 9.0]
  for value in seen:
    if np.allclose(value, actor_asked):
      print('FAIL: the actor\'s own action reached the transition'); ok = False
    if not any(np.allclose(value, a, atol=1e-6) for a in PHONE_ACTS):
      print(f'FAIL: {value} is not something the phone sent'); ok = False
  order = [
      next(i for i, a in enumerate(PHONE_ACTS) if np.allclose(v, a, atol=1e-6))
      for v in seen if any(np.allclose(v, a, atol=1e-6) for a in PHONE_ACTS)]
  steps = {(b - a) % len(PHONE_ACTS) for a, b in zip(order, order[1:])}
  if steps - {1}:
    print(f'FAIL: reports arrived out of order (deltas {steps})'); ok = False

  env.close()
  print()
  print('OK: onboard protocol round-trips and the phone\'s action is what '
        'reaches the transition.' if ok else 'FAILED')
  return 0 if ok else 1


if __name__ == '__main__':
  raise SystemExit(main())
