"""tools/robot/demo_supervisor.py with a stand-in for the learner."""

import pathlib
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUPERVISOR = ROOT / 'tools' / 'robot' / 'demo_supervisor.py'

# Plays dreamerv3/main.py under the supervisor: parks the way online.py does,
# then "trains" until killed, recording that it went live.
FAKE = """
import os, pathlib, sys, time
logdir = pathlib.Path(sys.argv[sys.argv.index('--logdir') + 1])
logdir.mkdir(parents=True, exist_ok=True)
time.sleep(0.3)  # building and compiling
for role in ('learner', 'actor'):
  (logdir / f'parked_{role}').write_text('1')
while not pathlib.Path(os.environ['ONLINE_GO_FILE']).exists():
  time.sleep(0.02)
(logdir / 'live').write_text(os.environ['ROBOT_RUN_NAME'])
time.sleep(1e6)
"""


def wait_for(cond, timeout=20.0):
  end = time.time() + timeout
  while time.time() < end:
    if cond():
      return True
    time.sleep(0.05)
  return False


def test_a_restart_goes_live_on_the_parked_session_and_parks_another(tmp_path):
  fake = tmp_path / 'fake_main.py'
  fake.write_text(FAKE)
  run_dir = tmp_path / 'e1_demo'
  (run_dir / 's04').mkdir(parents=True)   # an earlier job's session
  proc = subprocess.Popen(
      [sys.executable, str(SUPERVISOR), '--run_dir', str(run_dir),
       '--name', 'e1_demo', '--', sys.executable, str(fake)],
      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  try:
    live = lambda s: (run_dir / s / 'logdir' / 'live').exists()
    assert wait_for(lambda: live('s05'))
    assert (run_dir / 's05' / 'logdir' / 'live').read_text() == 'e1_demo_s05'
    # The next one starts once the live one is up, and parks.
    assert wait_for(lambda: (run_dir / 's06' / 'logdir' / 'parked_actor').exists())
    assert not live('s06')
    asked = time.time()
    (run_dir / 's05' / 'restart').write_text('1')
    assert wait_for(lambda: live('s06'))
    assert time.time() - asked < 3.0
    assert wait_for(lambda: (run_dir / 's07' / 'logdir' / 'parked_actor').exists())
    rows = (run_dir / 'sessions.tsv').read_text().splitlines()
    assert rows[1].startswith('s05') and rows[1].endswith('restart from the dashboard')
  finally:
    proc.send_signal(signal.SIGTERM)
    out = proc.communicate(timeout=20)[0].decode()
  assert proc.returncode == 0, out
  # Every session's processes went down with the job.
  ps = subprocess.run(['pgrep', '-f', str(fake)], capture_output=True, text=True)
  assert not ps.stdout.strip(), ps.stdout


# As FAKE, but every session after the first stays in its cold start until
# the test lets it park.
SLOW = FAKE.replace(
    "time.sleep(0.3)  # building and compiling",
    "time.sleep(0.3)  # building and compiling\n"
    "while not logdir.name == 'logdir' or (\n"
    "    logdir.parent.name != 's01' and\n"
    "    not (logdir.parent.parent / 'release').exists()):\n"
    "  time.sleep(0.02)")


def test_a_restart_before_the_next_session_is_ready_waits_on_the_live_one(
    tmp_path):
  fake = tmp_path / 'fake_main.py'
  fake.write_text(SLOW)
  run_dir = tmp_path / 'e2_demo'
  proc = subprocess.Popen(
      [sys.executable, str(SUPERVISOR), '--run_dir', str(run_dir),
       '--name', 'e2_demo', '--', sys.executable, str(fake)],
      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  try:
    live = lambda s: (run_dir / s / 'logdir' / 'live').exists()
    assert wait_for(lambda: live('s01'))
    assert wait_for(lambda: (run_dir / 's02').exists())
    (run_dir / 's01' / 'restart').write_text('1')
    # s02 is still starting: s01 keeps the phone rather than handing it to
    # a session that is not listening yet.
    time.sleep(1.0)
    assert proc.poll() is None
    assert not live('s02')
    assert subprocess.run(['pgrep', '-f', 's01/logdir'],
                          capture_output=True).returncode == 0
    (run_dir / 'release').write_text('1')
    assert wait_for(lambda: live('s02'))
    rows = (run_dir / 'sessions.tsv').read_text().splitlines()
    assert rows[1].startswith('s01') and rows[1].endswith('restart from the dashboard')
  finally:
    proc.send_signal(signal.SIGTERM)
    out = proc.communicate(timeout=20)[0].decode()
  assert proc.returncode == 0, out
  assert 'stays live until s02 is ready' in out, out
