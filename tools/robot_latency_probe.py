"""Measure the robot's real reaction time, without trusting a single timestamp.

Every latency number this project had before today was self-reported by the
same software stack it was measuring, and stopped at the moment `drive()`
returned -- which only puts a command in the serial writer's one-slot mailbox.
Two of those numbers were provably wrong: `imu_stale_ms` was *negative*, and
`phone_work_ms` was read as the policy cost when the policy is a small part of
it.

This tool takes the trainer's side of the bridge protocol, drives a signal we
choose, and records everything the phone can tell us about each tick. Nothing
here believes a clock it cannot check:

* Delays are recovered by **cross-correlating a signal we injected against a
  signal we measured**, both sampled at the same instants on the phone's own
  clock. A constant offset between two clocks cancels; only the lag survives.
* The one timestamp we cannot avoid trusting -- `SensorEvent.timestamp` -- is
  checked against both candidate clock bases, so "the sensor is 7ms old" is a
  measurement rather than an assumption.
* The fused orientation is compared against the raw gyroscope, which has no
  fusion filter in front of it. A filter's group delay is invisible in every
  timestamp; the only way to see it is to race it against something faster.

Usage, with the dreamerBridge app running and no trainer attached:

    # passive: no wheel commands. Tilt the robot by hand.
    .venv/bin/python tools/robot_latency_probe.py --signal none --seconds 20

    # wheel step response. THE WHEELS WILL TURN -- hold the robot or put it on
    # a stand so they spin free, or it will drive off the desk.
    .venv/bin/python tools/robot_latency_probe.py --signal square --amp 0.5

    # analyse a recording again without the robot
    .venv/bin/python tools/robot_latency_probe.py --analyse run.csv
"""

import argparse
import csv
import json
import math
import socket
import struct
import sys
import time

import numpy as np

PROTOCOL = 3

# Every scalar the phone reports per tick. Sensors are nested one level down in
# the header and get flattened into the same row.
SENSOR_KEYS = (
    'theta', 'angular_velocity', 'wheel_speed_l', 'wheel_speed_r',
    'wheel_distance_l', 'wheel_distance_r', 'wheel_count_l', 'wheel_count_r',
    'battery_voltage',
)
HEADER_KEYS = (
    'step', 'seq', 't', 'fresh', 'pace', 'cycle_ms', 'serial_wait_ms',
    'wait_ms', 'work_ms', 'drive_ms', 'policy_ms', 'warm_ms', 'prepare_ms',
    'prepared', 'ser_ok',
    'imu_age_ms', 'imu_stale_ms', 'imu_age_uptime_ms',
    'gyro_x', 'gyro_y', 'gyro_z', 'gyro_age_ms', 'gyro_count', 'rot_count',
    'req_l', 'req_r', 'sent_l', 'sent_r',
    'ser_queue_ms', 'ser_prep_ms', 'ser_write_ms', 'ser_resp_ms',
    'ser_service_ms', 'ser_first_byte_ms', 'ser_chunks', 'ser_gap_ms',
    'ser_sent', 'ser_dropped', 'ser_timeouts', 'ser_async',
    'fault_l', 'fault_r', 'fault_count', 'batt_safety',
    't_queued', 't_dequeued', 't_reply', 't_drive', 't_obs',
)


class Link:
  """The trainer half of the wire protocol: framed JSON over one TCP socket."""

  def __init__(self, sock):
    self.sock = sock
    self.buffer = bytearray()

  def write(self, header, blob=b''):
    header = dict(header, blob_len=len(blob))
    payload = json.dumps(header).encode('utf-8')
    self.sock.sendall(struct.pack('>I', len(payload)) + payload + blob)

  def read(self, timeout=30.0):
    while True:
      frame = self._parse()
      if frame is not None:
        return frame
      self.sock.settimeout(timeout)
      chunk = self.sock.recv(65536)
      if not chunk:
        raise ConnectionError('phone closed the connection')
      self.buffer.extend(chunk)

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
    del self.buffer[:total]
    return header


def signal_value(kind, t, amp, period):
  """The command we inject at time `t` seconds into the run.

  Both wheels get the same value: this robot drives them together, and a
  differential command would turn instead of tilting.
  """
  if kind == 'none':
    return 0.0
  if kind == 'square':
    return amp if (t % period) < (period / 2) else -amp
  if kind == 'step':
    return 0.0 if t < period else amp
  if kind == 'chirp':
    # 0.2 Hz to 5 Hz over the run: a delay shows up as a phase slope, which is
    # a far better conditioned estimate than a single edge.
    f0, f1, span = 0.2, 5.0, max(period, 1e-6)
    k = (f1 - f0) / span
    phase = 2 * math.pi * (f0 * t + 0.5 * k * min(t, span) ** 2)
    return amp * math.sin(phase)
  if kind == 'prbs0':
    # Between amp and zero rather than +/-amp: exercises the firmware's coast
    # dead zone the way a policy hovering near equilibrium does.
    n = int(t / period)
    rng = np.random.default_rng(n)
    return amp if rng.random() > 0.5 else 0.0
  if kind == 'prbs':
    # Deterministic pseudo-random sign flips at `period`. Flat spectrum, so no
    # frequency is privileged and nothing can resonate.
    n = int(t / period)
    rng = np.random.default_rng(n)
    return amp * (1.0 if rng.random() > 0.5 else -1.0)
  raise ValueError(f'unknown signal {kind!r}')


def record(args):
  server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  server.bind((args.host, args.port))
  server.listen(1)
  print(f'waiting for the phone on {args.host}:{args.port} ...', flush=True)
  sock, addr = server.accept()
  sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  link = Link(sock)

  hello = link.read()
  if hello.get('type') != 'hello' or hello.get('protocol') != PROTOCOL:
    raise SystemExit(f'unexpected hello from {addr}: {hello}')
  print(f'phone connected from {addr[0]}: control_hz={hello.get("control_hz")} '
        f'onboard_offered={hello.get("onboard")}', flush=True)
  # onboard=False on purpose: for this measurement we must own the command
  # signal, so the phone applies what we send instead of its own policy.
  # onboard=False by default: for the actuation measurements we must own the
  # command signal. --onboard lets the phone run its own policy instead, which
  # is the configuration that matters for latency; we then only observe.
  link.write(dict(type='hello', protocol=PROTOCOL, discrete=False,
                  pipeline=True, onboard=bool(args.onboard),
                  **({'pace': args.pace} if args.pace else {})))
  if args.onboard and not hello.get('onboard'):
    raise SystemExit('phone has no policy loaded; cannot run onboard')

  rows = []
  t0 = time.monotonic()
  sent = 0
  next_act = 0.0
  act_period = 1.0 / args.act_hz if args.act_hz > 0 else 0.0
  try:
    while time.monotonic() - t0 < args.seconds:
      header = link.read()
      if header.get('type') != 'obs':
        continue
      now = time.monotonic()
      value = signal_value(args.signal, now - t0, args.amp, args.period)
      # Throttling the action rate is a measurement, not a convenience: when
      # commands arrive slower than the serial link services them, each one is
      # written at a random phase relative to whatever cadence the RP2040 runs
      # on. If its reply latency is a fixed cost it will not move; if the reply
      # is really us waiting for the microcontroller's own loop to come round,
      # the latency collapses towards half a period.
      overrides = dict(
          async_motor=args.async_motor, seq=header.get('seq'),
          **({'slew': args.slew} if args.slew else {}),
          **({'zero': args.zero} if args.zero else {}),
          **({'lead': args.lead} if args.lead is not None else {}),
          **({'spin': True} if args.spin else {}),
          **({'warm': args.warm} if args.warm is not None else {}),
          **({'warm_n': args.warm_n} if args.warm_n is not None else {}),
          **({'prepare': args.prepare} if args.prepare is not None else {}),
          **({'spin_before': args.spin_before} if args.spin_before is not None else {}),
          **({'presample': args.presample} if args.presample is not None else {}),
          **({'trainer_presample': args.trainer_presample} if args.trainer_presample is not None else {}),
          **({'cpus': args.cpus} if args.cpus and len(rows) == 1 else {}),
          **({'prio': args.prio} if args.prio is not None and len(rows) == 1 else {}),
          **({'bench': True} if args.bench and len(rows) == 100 else {}))
      if args.onboard:
        # The phone decided and drove already; mirror what the trainer sends.
        link.write(dict(type='ctrl', reset=False, reward=0.0, **overrides))
        value = float(header.get('act', [float('nan')])[0])
        sent += 1
      elif now - t0 >= next_act:
        next_act = (now - t0) + act_period
        link.write(dict(type='act', left=value, right=value, reset=False,
                        reward=0.0, **overrides))
        sent += 1
      sensors = header.get('sensors', {})
      row = {'recv_t': now - t0, 'cmd': value}
      for k in HEADER_KEYS:
        row[k] = header.get(k, float('nan'))
      for k in SENSOR_KEYS:
        row[k] = sensors.get(k, float('nan'))
      rows.append(row)
      if len(rows) % 100 == 0:
        print(f'  {len(rows)} ticks  theta={math.degrees(sensors.get("theta", 0)):+6.2f}deg'
              f'  service={header.get("ser_service_ms", float("nan")):.1f}ms'
              f'  dropped={header.get("ser_dropped", 0)}', flush=True)
  except (ConnectionError, KeyboardInterrupt, socket.timeout) as e:
    print(f'stopped: {type(e).__name__}: {e}', flush=True)
  finally:
    # Always leave the wheels off, whatever happened above.
    try:
      link.write(dict(type='act', left=0.0, right=0.0, reset=False,
                      reward=0.0))
    except OSError:
      pass
    sock.close()
    server.close()

  if not rows:
    raise SystemExit('no observations recorded')
  fields = list(rows[0].keys())
  with open(args.out, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
  print(f'wrote {args.out}  ({len(rows)} ticks, {sent} actions sent)')
  return args.out


# -- analysis ---------------------------------------------------------------


def lag_samples(a, b, max_lag):
  """Lag of `b` behind `a`, in samples, by cross-correlation.

  Both series are mean-removed and scaled, so a constant offset or a gain
  difference between them cannot move the peak -- only a time shift can. The
  peak is refined by a parabola through its two neighbours, which recovers a
  fraction of a sample and matters here because one sample is a whole control
  period.
  """
  a = np.asarray(a, float)
  b = np.asarray(b, float)
  keep = np.isfinite(a) & np.isfinite(b)
  a, b = a[keep], b[keep]
  if len(a) < 8 * max_lag:
    return float('nan'), 0.0
  a = a - a.mean()
  b = b - b.mean()
  if a.std() < 1e-12 or b.std() < 1e-12:
    return float('nan'), 0.0
  a /= a.std() * len(a)
  b /= b.std()
  lags = np.arange(-max_lag, max_lag + 1)
  corr = np.array([
      np.dot(a[max_lag:len(a) - max_lag], b[max_lag + k:len(b) - max_lag + k])
      for k in lags])
  i = int(np.argmax(corr))
  peak = float(corr[i])
  if 0 < i < len(corr) - 1:
    y0, y1, y2 = corr[i - 1], corr[i] + 1e-30, corr[i + 1]
    denom = (y0 - 2 * y1 + y2)
    offset = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-30 else 0.0
  else:
    offset = 0.0
  # `b` lags `a` when the correlation peaks at a negative shift of b.
  return -(lags[i] + offset), peak


def summarise(name, values, unit='ms'):
  v = np.asarray([x for x in values if np.isfinite(x)], float)
  if not len(v):
    return f'{name:22s} (no data)'
  return (f'{name:22s} mean {v.mean():8.2f}  p50 {np.median(v):8.2f}  '
          f'p95 {np.percentile(v, 95):8.2f}  max {v.max():8.2f} {unit}')


def analyse(path):
  with open(path) as f:
    rows = list(csv.DictReader(f))
  if not rows:
    raise SystemExit(f'{path} is empty')

  def col(name):
    return np.array([float(r.get(name, 'nan') or 'nan') for r in rows])

  n = len(rows)
  t = col('recv_t')
  dt = np.diff(t)
  rate = 1.0 / np.median(dt) if len(dt) else float('nan')
  period_ms = 1e3 / rate

  print(f'\n=== {path}: {n} ticks, {t[-1] - t[0]:.1f}s, '
        f'{rate:.1f} Hz ({period_ms:.1f} ms/tick) ===\n')

  paces = {r.get('pace') for r in rows} - {None, '', 'nan'}
  print(f'-- the loop ({", ".join(sorted(paces)) or "clock"} paced) --')
  print(summarise('tick interval', dt * 1e3))
  print(summarise('drive-to-drive cycle', col('cycle_ms')))
  print(summarise('waiting for the RP2040', col('serial_wait_ms')))
  print(summarise('phone wait_ms', col('wait_ms')))
  print(summarise('phone work_ms', col('work_ms')))
  print(summarise('  of which drive_ms', col('drive_ms')))
  print(summarise('  of which policy_ms', col('policy_ms')))

  print('\n-- the actuation path, past the point drive() used to stop --')
  print(summarise('serial mailbox wait', col('ser_queue_ms')))
  print(summarise('serial prepare', col('ser_prep_ms')))
  print(summarise('serial USB write', col('ser_write_ms')))
  print(summarise('serial RP2040 reply', col('ser_resp_ms')))
  print(summarise('  first reply byte', col('ser_first_byte_ms')))
  fb = col('ser_first_byte_ms')
  if np.isfinite(fb).any():
    print(f'  (the firmware polls USB for 100 us every ~10 ms after replying: '
          f'~64 ms here means the command was waiting when it looked, '
          f'~74 means it missed a poll; p50 {np.nanmedian(fb):.1f})')
  print(summarise('serial round trip', col('ser_service_ms')))
  ser_sent, ser_drop = col('ser_sent'), col('ser_dropped')
  if np.isfinite(ser_sent[-1]) and np.isfinite(ser_sent[0]):
    d_sent = ser_sent[-1] - ser_sent[0]
    d_drop = ser_drop[-1] - ser_drop[0]
    total = d_sent + d_drop
    frac = 100.0 * d_drop / total if total else 0.0
    print(f'commands: {d_sent:.0f} reached the wheels, {d_drop:.0f} were '
          f'overwritten first ({frac:.0f}% never applied)')
    if d_sent > 0:
      print(f'effective wheel command rate: {d_sent / (t[-1] - t[0]):.1f} Hz '
            f'against a {rate:.1f} Hz control loop')

  req, sent_ = col('req_l'), col('sent_l')
  slewed = np.isfinite(req) & np.isfinite(sent_) & (np.abs(req - sent_) > 1e-3)
  if slewed.any():
    print(f'slew limit engaged on {100.0 * slewed.mean():.0f}% of ticks '
          f'(max |requested - sent| = {np.nanmax(np.abs(req - sent_)):.2f})')

  print('\n-- do the sensor timestamps mean what we assume --')
  print(summarise('IMU age (elapsedRT)', col('imu_age_ms')))
  print(summarise('IMU age (nanoTime)', col('imu_age_uptime_ms')))
  a, b = np.nanmedian(col('imu_age_ms')), np.nanmedian(col('imu_age_uptime_ms'))
  if np.isfinite(a) and np.isfinite(b):
    if abs(b - a) > 5.0:
      print(f'  the two bases differ by {b - a:+.0f} ms: sensor timestamps are '
            f'on the {"elapsedRealtime" if abs(a) < abs(b) else "uptime"} clock, '
            f'and only that one may be subtracted from them')
    else:
      print('  the two clocks agree, so the phone has not slept: either base '
            'gives the same age')
  print(summarise('IMU sample staleness', col('imu_stale_ms')))
  stale = col('imu_stale_ms')
  if np.isfinite(stale).any() and np.nanmin(stale) < -0.5:
    print('  NEGATIVE staleness: this number is still measuring the wrong pair '
          'of instants')
  print(summarise('gyro age', col('gyro_age_ms')))
  gyro_n = col('gyro_count')
  rot_n = col('rot_count')
  if np.isfinite(gyro_n[-1]) and gyro_n[-1] > gyro_n[0]:
    span = t[-1] - t[0]
    print(f'sensor rates: rotation vector {(rot_n[-1] - rot_n[0]) / span:.0f} Hz, '
          f'gyroscope {(gyro_n[-1] - gyro_n[0]) / span:.0f} Hz')

  print('\n-- lags, by cross-correlation (nothing here trusts a clock) --')
  max_lag = max(3, int(round(0.4 * rate)))  # look up to 400 ms out

  # 1. The fusion filter. The gyro axis that matches the fused rate best is the
  #    pitch axis; we do not assume which one it is, we find it.
  fused_rate = col('angular_velocity')
  best = None
  for axis in ('gyro_x', 'gyro_y', 'gyro_z'):
    g = col(axis)
    lag, peak = lag_samples(g, fused_rate, max_lag)
    if np.isfinite(lag) and (best is None or abs(peak) > abs(best[2])):
      best = (axis, lag, peak)
  if best:
    axis, lag, peak = best
    print(f'fused theta rate lags raw {axis:7s} by {lag * period_ms:+7.1f} ms '
          f'(r={peak:+.2f})')
    print('  this is the rotation-vector filter\'s group delay: it appears in '
          'no timestamp')

  # 2. The whole actuation path: what we commanded against what the body did.
  cmd = col('cmd')
  if np.nanstd(cmd) > 1e-6:
    for name, series in (('theta', col('theta')),
                         ('fused rate', fused_rate),
                         ('gyro (best axis)', col(best[0]) if best else None),
                         ('wheel speed L', col('wheel_speed_l'))):
      if series is None:
        continue
      lag, peak = lag_samples(cmd, series, max_lag)
      if np.isfinite(lag):
        print(f'{name:20s} lags the command by {lag * period_ms:+7.1f} ms '
              f'(r={peak:+.2f})')
    print('  command -> wheel speed is the actuation path alone; command -> '
          'gyro adds the sensing path on top')
  else:
    print('(no command was injected, so nothing to correlate against; run '
          'with --signal square or prbs for the actuation path)')
  print()


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--analyse', metavar='CSV',
                  help='re-analyse a recording instead of talking to the robot')
  ap.add_argument('--host', default='0.0.0.0')
  ap.add_argument('--port', type=int, default=3000)
  ap.add_argument('--signal', default='none',
                  choices=['none', 'square', 'step', 'chirp', 'prbs', 'prbs0'])
  ap.add_argument('--amp', type=float, default=0.5)
  ap.add_argument('--period', type=float, default=1.0,
                  help='seconds; square/prbs dwell, step delay, chirp span')
  ap.add_argument('--seconds', type=float, default=20.0)
  ap.add_argument('--pace', choices=['serial', 'clock'], default=None,
                  help="ask the phone to pace on the RP2040's reply ('serial') "
                       "or on its CONTROL_HZ clock; default is the phone's own")
  ap.add_argument('--slew', type=float, default=None,
                  help='max change per wheel command on the phone (abcvlib '
                       'default 0.4; 2.0 disables the ramp)')
  ap.add_argument('--cpus', type=int, nargs='+', default=None,
                  help='pin the phone control thread to these CPUs')
  ap.add_argument('--prio', type=int, default=None,
                  help='android thread priority for the control thread '
                       '(-19 urgent audio ... 0 default)')
  ap.add_argument('--warm', type=float, default=None,
                  help='phone WARM_MS: throwaway policy step this long before '
                       'the reply is due (0 disables)')
  ap.add_argument('--prepare', type=int, choices=[0, 1], default=None,
                  help='phone PREPARE_AHEAD: precompute the GRU half early')
  ap.add_argument('--spin_before', type=float, default=None,
                  help='phone SPIN_BEFORE_MS: busy-wait this long before the '
                       'reply is due')
  ap.add_argument('--trainer_presample', type=float, default=None,
                  help='phone TRAINER_PRESAMPLE_MS: ship the obs this long '
                       'before the reply (trainer-driven)')
  ap.add_argument('--presample', type=float, default=None,
                  help='phone PRESAMPLE_MS: decide this long before the reply')
  ap.add_argument('--warm_n', type=int, default=None,
                  help='number of back-to-back warm-up steps')
  ap.add_argument('--spin', action='store_true',
                  help='phone busy-waits for the RP2040 instead of sleeping')
  ap.add_argument('--bench', action='store_true',
                  help='ask the phone to run its policy benchmark mid-run')
  ap.add_argument('--onboard', action='store_true',
                  help='let the phone run its own policy; only observe')
  ap.add_argument('--lead', type=float, default=None,
                  help='phone WAKE_LEAD_MS: delay after the reply before '
                       'sampling (serial pacing)')
  ap.add_argument('--zero', choices=['coast', 'brake', 'min'], default=None,
                  help='how the phone sends commands inside the dead zone')
  ap.add_argument('--async_motor', action='store_true',
                  help='tell the phone to write motor commands without waiting '
                       'for the RP2040 reply')
  ap.add_argument('--act_hz', type=float, default=50.0,
                  help='rate at which actions are sent; below the loop rate '
                       'the phone repeats the last one')
  ap.add_argument('--out', default='latency.csv')
  args = ap.parse_args(argv)

  if args.analyse:
    analyse(args.analyse)
    return 0
  path = record(args)
  analyse(path)
  return 0


if __name__ == '__main__':
  sys.exit(main())
