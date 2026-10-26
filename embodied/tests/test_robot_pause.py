"""Pausing the robot and the training from the phone's dashboard.

The trainer side runs anywhere. The phone side (the app's web server and run
history, pure Python) runs when ROBOT_ANDROID_DIR points at the
smartphone-robot-android checkout (tools/robot/robot.env), and is skipped
otherwise.
"""

import json
import os
import pathlib
import socket
import sys
import time
import urllib.request

import numpy as np
import pytest

from embodied.envs import robot
from embodied.run import online


def make_env(tmp_path, start_paused=False, **kwargs):
  return robot.SmartphoneRobot(
      'balance', pause_file=str(tmp_path / 'online_shared' / 'pause.json'),
      start_paused=start_paused, **kwargs)


def pause_file(tmp_path):
  return tmp_path / 'online_shared' / 'pause.json'


def read_frame(sock):
  buffer = bytearray()
  sock.settimeout(5.0)
  while True:
    frame = robot._parse(buffer)
    if frame is not None:
      return frame[0]
    buffer.extend(sock.recv(1 << 16))


# ---------- trainer side ----------

def test_a_pause_frame_is_handed_over_and_never_queued():
  asked = []
  ours, phone = socket.socketpair()
  link = robot._Link(ours, timeout=5.0, on_pause=asked.append)
  phone.sendall(robot._frame(dict(type='pause', robot=True)))
  phone.sendall(robot._frame(dict(type='obs', seq=1)))
  assert link.get(5.0)[0]['type'] == 'obs'
  assert asked and asked[0]['robot'] is True
  link.close()


def test_nothing_is_paused_without_a_file(tmp_path):
  env = make_env(tmp_path)
  assert not pause_file(tmp_path).exists()
  assert not online.read_pause(pause_file(tmp_path))['training']
  assert env._pause_message() == dict(
      robot=False, training=False, train_started_ago=None, waiting=False)


def test_a_garbled_file_pauses_nothing(tmp_path):
  pause_file(tmp_path).parent.mkdir(parents=True)
  pause_file(tmp_path).write_text('{"training": tr')
  assert online.read_pause(pause_file(tmp_path)) == dict(
      robot=False, training=False, train_started=None, from_start=False)


def test_start_paused_pauses_both_and_the_learner_sees_it(tmp_path):
  env = make_env(tmp_path, start_paused=True)
  state = online.read_pause(pause_file(tmp_path))
  assert state['robot'] and state['training']
  message = env._pause_message()
  assert message['robot'] and message['training'] and message['waiting']
  assert message['train_started_ago'] is None
  watch = online.TrainingPause(pause_file(tmp_path), poll=0.0)
  assert watch() == (True, True)
  assert watch() == (True, False)


def test_the_first_training_start_is_minute_zero(tmp_path):
  env = make_env(tmp_path, start_paused=True)
  env._request_pause(dict(robot=False))
  assert online.read_pause(pause_file(tmp_path))['train_started'] is None
  before = time.time()
  env._request_pause(dict(training=False))
  started = online.read_pause(pause_file(tmp_path))['train_started']
  assert before <= started <= time.time()
  assert not env._pause_message()['waiting']
  # Pausing and resuming later does not move minute 0.
  env._request_pause(dict(training=True))
  env._request_pause(dict(training=False))
  assert online.read_pause(pause_file(tmp_path))['train_started'] == started
  watch = online.TrainingPause(pause_file(tmp_path), poll=0.0)
  assert watch() == (False, False)


def test_a_run_not_started_paused_keeps_its_start_as_minute_zero(tmp_path):
  env = make_env(tmp_path)
  env._request_pause(dict(training=True))
  assert online.read_pause(pause_file(tmp_path))['training']
  env._request_pause(dict(training=False))
  state = online.read_pause(pause_file(tmp_path))
  assert not state['training'] and state['train_started'] is None
  assert not env._pause_message()['waiting']


def test_a_resumed_run_keeps_its_state(tmp_path):
  env = make_env(tmp_path, start_paused=True)
  env._request_pause(dict(robot=False, training=False))
  started = online.read_pause(pause_file(tmp_path))['train_started']
  again = make_env(tmp_path, start_paused=True)   # the same run, resumed
  message = again._pause_message()
  assert not message['robot'] and not message['training']
  assert online.read_pause(pause_file(tmp_path))['train_started'] == started


def test_the_phone_is_answered_with_the_new_state(tmp_path):
  env = make_env(tmp_path, start_paused=True)
  ours, phone = socket.socketpair()
  env._link = robot._Link(ours, timeout=5.0, on_pause=env._request_pause)
  phone.sendall(robot._frame(dict(type='pause', training=False)))
  answer = read_frame(phone)
  assert answer['type'] == 'pause'
  assert answer['robot'] and not answer['training']
  assert 0.0 <= answer['train_started_ago'] < 5.0
  env._link.close()


def test_a_paused_robot_is_sent_zero_actions(tmp_path):
  # A policy on the trainer (not onboard): the held wheels are the action.
  env = make_env(tmp_path, start_paused=True, discrete=False, symmetric=False)
  sent = []

  def send(left, right, reset):
    sent.append((left, right))
    raise ConnectionError('stop here')

  env._send = send
  action = dict(drive=np.array([0.8, -0.5], np.float32))
  with pytest.raises(ConnectionError):
    env._advance(action)
  env._request_pause(dict(robot=False))
  with pytest.raises(ConnectionError):
    env._advance(action)
  assert sent[0] == (0.0, 0.0)
  assert sent[1] == pytest.approx((0.8, -0.5))


def test_the_learner_drops_its_ratio_debt_while_paused():
  # What run_learner does with should_train while paused: called and its
  # answer thrown away, so resuming does not try to catch up.
  import elements
  should_train = elements.when.Ratio(1.0)
  should_train(0)
  should_train(5000)          # paused: 5000 steps of debt, discarded
  assert should_train(5010) == 10


# ---------- phone side ----------

def app_python():
  root = os.environ.get('ROBOT_ANDROID_DIR')
  path = root and pathlib.Path(root) / 'apps/dreamerBridge/src/main/python'
  if not path or not (path / 'webui.py').exists():
    pytest.skip('set ROBOT_ANDROID_DIR to the smartphone-robot-android checkout')
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))
  return path


def free_port():
  with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    return s.getsockname()[1]


def test_the_page_pauses_through_the_phone():
  app_python()
  import webui
  asked = []
  ui = webui.WebUI(None, port=free_port(),
                   on_pause=lambda robot, training: asked.append((robot, training)))
  assert ui.start()

  def post(body):
    request = urllib.request.Request(
        f'http://127.0.0.1:{ui.port}/pause', data=json.dumps(body).encode(),
        method='POST')
    try:
      return json.loads(urllib.request.urlopen(request, timeout=5).read())
    except urllib.error.HTTPError as e:
      return json.loads(e.read())

  assert post(dict(robot=False))['ok']
  assert post(dict(training=False))['ok']
  assert asked == [(False, None), (None, False)]
  ui.handle(dict(t='pause', robot=True))
  assert asked[-1] == (True, None)

  # A trainer that cannot pause: the page is told so.
  ui.on_pause = None
  assert not post(dict(robot=True))['ok']


def test_the_history_keeps_minute_zero_per_run():
  app_python()
  import dashboard
  history = dashboard.RunHistory()
  history.start('e1_demo')
  history.training(None, True)
  snap = history.snapshot()
  assert snap['waiting'] and snap['train_started'] is None
  history.training(60.0, False)
  snap = history.snapshot()
  assert not snap['waiting']
  assert snap['now'] - snap['train_started'] == pytest.approx(60.0, abs=1.0)
  history.start('e1_demo')        # a reconnect keeps it
  assert history.snapshot()['train_started'] == snap['train_started']
  history.start('e2_demo')        # a new run does not
  assert history.snapshot()['train_started'] is None


def test_the_dashboard_has_the_buttons_and_the_shifted_axis():
  path = app_python()
  page = (path / 'webpages' / 'dashboard.html').read_text()
  for needle in ('id="pause-robot"', 'id="pause-train"', "t: 'pause'",
                 'shiftMin()', 'TRAINING STARTS'):
    assert needle in page, needle


def test_the_learner_and_the_actor_can_start_together(tmp_path):
  # Both processes build the env, and so write the starting state, at once.
  import threading
  errors = []

  def start():
    try:
      for _ in range(50):
        online.write_pause(pause_file(tmp_path), dict(robot=True, training=True))
    except Exception as e:  # noqa: BLE001
      errors.append(e)

  threads = [threading.Thread(target=start) for _ in range(4)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  assert not errors, errors
  assert online.read_pause(pause_file(tmp_path))['training']
