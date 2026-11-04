"""deploy/emotion.py: the emotion signals, and the phone's runner computing
them from weights exported with emotion=True."""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'dreamerv3' / 'deploy'))

from emotion import Emotion, Running, _kl  # noqa: E402
from policy_runner import PolicyRunner  # noqa: E402
from numpy_policy import NumpyPolicy  # noqa: E402

RUN = pathlib.Path('/work/DoyaU/vasilache/work/e30103_demo_20261104_233808/s01/logdir')


def test_running_z_scores_settle():
  r = Running(50)
  zs = [r(x) for x in np.random.default_rng(0).normal(3.0, 2.0, 2000)]
  assert abs(np.mean(zs[500:])) < 0.3 and 0.6 < np.std(zs[500:]) < 1.6


def test_kl_of_a_distribution_with_itself_is_zero():
  p = np.full((32, 4), 0.25, np.float32)
  assert _kl(p, p) == pytest.approx(0.0, abs=1e-6)


@pytest.fixture(scope='module')
def weights(tmp_path_factory):
  if not RUN.exists():
    pytest.skip('needs a Saion run dir')
  import elements
  import ruamel.yaml as yaml
  sys.path.insert(0, str(ROOT))
  sys.path.insert(0, str(ROOT / 'tools' / 'robot' / 'dev'))
  import export_policy
  from dreamerv3.deploy import export
  config = elements.Config(yaml.YAML(typ='safe').load((RUN / 'config.yaml').read_text()))
  ckpt = sorted(p for p in (RUN / 'ckpt').iterdir() if p.name != 'latest')[-1]
  arrays, meta = export.pack_arrays(
      config, export_policy.load_params(ckpt), 'int8', emotion=True)
  path = tmp_path_factory.mktemp('w') / 'policy.npz'
  path.write_bytes(export.encode(meta, arrays, compress=True))
  return path


def test_the_runner_feels_on_recorded_steps(weights):
  runner = PolicyRunner(str(weights))
  assert runner.emotion is not None
  chunk = sorted((RUN / 'online_shared' / 'experience').glob('*.npz'))[200]
  d = np.load(chunk)
  for t in range(64):
    sensors = dict(theta=float(d['orientation'][t][0]) * 0.18,
                   angular_velocity=float(d['orientation'][t][1]) * 3.0,
                   wheel_speed_l=float(d['wheels'][t][0]) / 3.3e-4,
                   wheel_speed_r=float(d['wheels'][t][1]) / 3.3e-4)
    runner.act(sensors)
    runner.feel(float(d['reward'][t]))
  s = runner.emotion_state
  assert s is not None and all(np.isfinite(v) for v in s.values())
  assert -1 <= s['valence'] <= 1 and 0 <= s['arousal'] <= 1


def test_empowerment_is_finite_and_non_negative(weights):
  p = NumpyPolicy(str(weights))
  emo = Emotion(p, every=1)
  c = p.initial()
  e = emo._empowerment(c['deter'], c['stoch'])
  assert np.isfinite(e) and e >= 0


def test_applied_feeds_the_wheels_action_to_the_next_step():
  runner = PolicyRunner()
  runner.carry = dict(deter=np.zeros(4), stoch=np.zeros((2, 2)), prevact=np.array([0.9, 0.9]))
  runner.applied([0.6, -0.1])
  np.testing.assert_allclose(runner.carry['prevact'], [0.6, -0.1])
