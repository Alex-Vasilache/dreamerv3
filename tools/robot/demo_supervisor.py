"""Training from scratch on demand, inside one job: for live demos.

Runs the learner command it is given as a series of sessions, s01, s02, ...
under --run_dir, each a complete run with its own logdir. One session is live
(the phone is connected to it); the next is already started and parked at
the point where a cold start spends its time (embodied/run/online.py:_park):
agent built and compiled, first weights published. When the phone's
dashboard asks for a fresh run, the live session is killed and the parked one
let go, so training restarts from random weights in a few seconds instead of
the minute or more of a new process, and without giving up the GPU.

  demo_supervisor.py --run_dir DIR --name NAME -- python -u main.py --configs robot ...

The command gets --logdir <session>/logdir appended and three variables:
ONLINE_GO_FILE (online.py waits for it), ROBOT_RESTART_FILE (the robot env
writes it when the phone asks; embodied/envs/robot.py) and ROBOT_RUN_NAME
(<name>_s01, ...; the phone starts a new dashboard history for each).
tools/robot/trainer_job.sh runs this when DEMO=1; tools/robot/demo.sh is the
way in. Sessions are listed in <run_dir>/sessions.tsv.
"""

import argparse
import os
import pathlib
import signal
import subprocess
import sys
import time

POLL = 0.05
KILL_AFTER = 3.0   # seconds of SIGTERM before SIGKILL
# A session that dies this soon after starting, this many times in a row,
# means the command is broken; stop rather than restart it forever.
CRASH_WINDOW = 120.0
CRASH_LIMIT = 3


def log(message):
  print(f'[demo {time.strftime("%H:%M:%S")}] {message}', flush=True)


class Session:

  def __init__(self, run_dir, name, index, command, go):
    self.index = index
    self.dir = run_dir / f's{index:02d}'
    self.dir.mkdir(parents=True, exist_ok=True)
    self.go_file = self.dir / 'go'
    self.restart_file = self.dir / 'restart'
    self.logdir = self.dir / 'logdir'
    self.started = time.time()
    self.went = None
    self.announced = False
    self.waiting = False   # restart asked, the next session not ready yet
    env = dict(
        os.environ,
        ONLINE_GO_FILE=str(self.go_file),
        ROBOT_RESTART_FILE=str(self.restart_file),
        ROBOT_RUN_NAME=f'{name}_s{index:02d}')
    if go:
      self.go()
    self._out = open(self.dir / 'session.log', 'ab')
    # Its own process group, so the learner, the actor and every env process
    # they spawned go down together.
    self.proc = subprocess.Popen(
        command + ['--logdir', str(self.logdir)], env=env,
        stdout=self._out, stderr=subprocess.STDOUT, start_new_session=True)

  @property
  def parked(self):
    """Learner and actor built, compiled and waiting (or already gone past
    that point)."""
    return all((self.logdir / f'parked_{role}').exists()
               for role in ('learner', 'actor'))

  def go(self):
    self.go_file.write_text('1')
    self.went = time.time()

  def alive(self):
    return self.proc.poll() is None

  def stop(self):
    """Kill the whole process group and wait until none of it is left, so
    the port is free for the next session."""
    pgid = self.proc.pid
    deadline = time.time() + KILL_AFTER
    sig = signal.SIGTERM
    while True:
      try:
        os.killpg(pgid, sig)
      except ProcessLookupError:
        break
      except PermissionError:
        # macOS: a group whose members are all zombies refuses signals.
        if self.proc.poll() is not None:
          break
      self.proc.poll()
      time.sleep(POLL)
      if time.time() > deadline:
        sig = signal.SIGKILL
    self.proc.wait()
    self._out.close()


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--run_dir', required=True)
  parser.add_argument('--name', required=True)
  parser.add_argument('command', nargs=argparse.REMAINDER)
  args = parser.parse_args(argv)
  command = args.command[1:] if args.command[:1] == ['--'] else args.command
  if not command:
    parser.error('give the learner command after --')
  run_dir = pathlib.Path(args.run_dir)
  run_dir.mkdir(parents=True, exist_ok=True)
  table = run_dir / 'sessions.tsv'
  if not table.exists():
    table.write_text('session\tstarted\tlive\tended\treason\n')

  # Numbered on from any sessions already here: an old session's logdir
  # would resume its checkpoint, which is the opposite of a fresh run.
  index = max([int(p.name[1:]) for p in run_dir.glob('s[0-9][0-9]*')
               if p.name[1:].isdigit()] or [0])
  live = standby = None

  def start(go):
    nonlocal index
    index += 1
    session = Session(run_dir, args.name, index, command, go)
    log(f's{index:02d} started' + (' (live)' if go else ' (standby)'))
    return session

  def record(session, reason):
    stamp = lambda t: time.strftime('%H:%M:%S', time.localtime(t)) if t else ''
    with open(table, 'a') as f:
      f.write(f's{session.index:02d}\t{stamp(session.started)}\t'
              f'{stamp(session.went)}\t{stamp(time.time())}\t{reason}\n')

  def shutdown(*_):
    log('stopping')
    for session in (live, standby):
      if session is not None:
        session.stop()
        record(session, 'job stopped')
    sys.exit(0)

  signal.signal(signal.SIGTERM, shutdown)
  signal.signal(signal.SIGINT, shutdown)

  live = start(go=True)
  crashes = 0
  asked_at = None
  while True:
    time.sleep(POLL)
    # The next session compiles while this one runs, but only once this one
    # has finished its own start: two cold starts at once slow both.
    if standby is None and live.parked:
      standby = start(go=False)
    if standby is not None and not standby.announced and standby.parked:
      standby.announced = True
      log(f's{standby.index:02d} ready, {time.time() - standby.started:.0f}s '
          'after it started')
    asked = live.restart_file.exists()
    if not asked and live.alive():
      continue
    if asked and asked_at is None:
      asked_at = time.time()
    # A restart asked for before the next session is ready waits for it on
    # the live one: cutting over at once left the phone with nothing to
    # connect to for the rest of a cold start (~1.5 min on Saion; e2495 s04).
    if (asked and live.alive() and standby is not None and standby.alive()
        and not standby.parked):
      if not live.waiting:
        live.waiting = True
        log(f'restart asked; s{live.index:02d} stays live until '
            f's{standby.index:02d} is ready')
      continue
    reason = 'restart from the dashboard' if asked else (
        f'exited with {live.proc.returncode}')
    t0 = asked_at or time.time()
    asked_at = None
    if asked:
      live.stop()
    else:
      live.proc.wait()
      crashes = crashes + 1 if t0 - live.started < CRASH_WINDOW else 0
      if crashes >= CRASH_LIMIT:
        record(live, reason)
        log(f'{crashes} sessions in a row died within {CRASH_WINDOW:.0f}s; '
            f'see {live.dir}/session.log')
        if standby is not None:
          standby.stop()
          record(standby, 'never live')
        sys.exit(1)
    record(live, reason)
    log(f's{live.index:02d} ended: {reason}')
    if standby is not None and standby.alive():
      live, standby = standby, None
      live.go()
      log(f's{live.index:02d} live {time.time() - t0:.1f}s after the request'
          + ('' if live.parked else ' (still starting up)'))
    else:
      if standby is not None:
        record(standby, f'died while parked ({standby.proc.returncode})')
      live, standby = start(go=True), None


if __name__ == '__main__':
  main()
