"""The robot env's side of a dashboard restart (tools/robot/demo.sh)."""

import socket
import time

from embodied.envs import robot


def link_pair(on_restart=None):
  ours, phone = socket.socketpair()
  return robot._Link(ours, timeout=5.0, on_restart=on_restart), phone


def test_a_restart_frame_is_handed_over_and_never_queued():
  asked = []
  link, phone = link_pair(on_restart=lambda: asked.append(time.time()))
  phone.sendall(robot._frame(dict(type='restart')))
  phone.sendall(robot._frame(dict(type='obs', seq=1)))
  header, _ = link.get(5.0)
  assert header['type'] == 'obs' and len(asked) == 1
  link.close()


def test_without_a_supervisor_a_restart_frame_is_just_a_frame():
  link, phone = link_pair()
  phone.sendall(robot._frame(dict(type='restart')))
  assert link.get(5.0)[0]['type'] == 'restart'
  link.close()


def make_env(tmp_path, monkeypatch, restart=True):
  if restart:
    monkeypatch.setenv('ROBOT_RESTART_FILE', str(tmp_path / 'restart'))
  else:
    monkeypatch.delenv('ROBOT_RESTART_FILE', raising=False)
  monkeypatch.setenv('ROBOT_RUN_NAME', 'e1_demo_s03')
  policy_dir = tmp_path / 'logdir' / 'online_shared'
  (policy_dir / 'policy').mkdir(parents=True)
  return robot.SmartphoneRobot('balance', policy_dir=str(policy_dir))


def test_the_env_asks_the_supervisor_through_its_file(tmp_path, monkeypatch):
  env = make_env(tmp_path, monkeypatch)
  assert not (tmp_path / 'restart').exists()
  env._request_restart()
  assert (tmp_path / 'restart').exists()
  assert env._run_name() == 'e1_demo_s03'


def test_only_weights_this_run_published_count_as_its_own(tmp_path, monkeypatch):
  env = make_env(tmp_path, monkeypatch, restart=False)
  (tmp_path / 'logdir' / 'online_shared' / 'policy' / 'policy_123.pkl').write_bytes(b'')
  assert env._published('123')
  assert not env._published('456')   # the run before: the phone holds
  assert env._published(None)        # a phone with no stamp: nothing to hold
