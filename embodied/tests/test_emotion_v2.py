"""Emotion v2 (deploy/emotion.py EmotionV2), impact pain and the catch reflex,
on a toy world model with random weights (no run dir needed)."""

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'dreamerv3' / 'deploy'))

import emotion  # noqa: E402
import policy_runner  # noqa: E402
from embodied.envs import robot  # noqa: E402

D, G, S, C, H, BINS = 6, 2, 2, 3, 4, 15


class ToyPolicy:
  """What EmotionV2 reads of a NumpyPolicy: `w`, `meta`, `last_post`."""

  def __init__(self, seed=0, threat=True, cost=True, scales=(-2.0, 8.0, -0.5, 0.0)):
    rng = np.random.default_rng(seed)
    n = lambda *shape: rng.normal(0, 0.5, shape).astype(np.float32)  # noqa: E731
    w = {}
    for i, fan in enumerate((D, S * C, 2)):
      w[f'dyn/dynin{i}/kernel'], w[f'dyn/dynin{i}/bias'] = n(fan, H), n(H)
      w[f'dyn/dynin{i}norm/scale'] = np.ones(H, np.float32)
    w['dyn/dyngru/kernel'] = n(G, D // G + 3 * H, 3 * D // G)
    w['dyn/dyngru/bias'] = n(3 * D)
    w['dyn/priorlogit/kernel'], w['dyn/priorlogit/bias'] = n(D, S * C), n(S * C)
    heads = ['val', 'rew'] + (['cval'] if threat else []) + (['cst'] if cost else [])
    for name in heads:
      w[f'{name}/head/logits/kernel'] = n(D + S * C, BINS)
      w[f'{name}/head/logits/bias'] = n(BINS)
    lo, hi, clo, chi = scales
    w['retnorm/lo/value'], w['retnorm/hi/value'] = np.float32(lo), np.float32(hi)
    if threat:
      w['cretnorm/lo/value'], w['cretnorm/hi/value'] = np.float32(clo), np.float32(chi)
    self.w = w
    self.meta = dict(stoch=S, classes=C, blocks=G, dynlayers=0, unimix=0.01,
                     obs_scale=dict(hz=65.0),
                     emotion=dict(gamma=1 - 1 / 130, fear_weight=1.0, ret_limit=1.0))
    stoch = np.eye(C, dtype=np.float32)[[0, 2]]
    self.last_post = (n(D), stoch, n(S, C))


def test_fading_constant():
  lam = emotion.fade_lambda(65.0, 0.5)
  assert lam == pytest.approx(math.exp(-1 / 32.5)) and lam == pytest.approx(0.970, abs=1e-3)
  emo = emotion.EmotionV2(ToyPolicy(), hz=65.0)
  emo.S_V, emo.S_T = 1.0, 1.0
  for _ in range(65):   # 1 s of a constant outcome 1
    emo.tick(1.0, 0.0)
  assert emo.faded == pytest.approx(1 - lam ** 65)
  assert emo.faded == pytest.approx(1 - math.exp(-2), abs=1e-6)   # 2 time constants
  # n steps at once equal n single steps.
  a, b = emotion.EmotionV2(ToyPolicy()), emotion.EmotionV2(ToyPolicy())
  a.tick(0.3, 0.2, n=4)
  for _ in range(4):
    b.tick(0.3, 0.2)
  assert a.faded == pytest.approx(b.faded)


def test_return_scales_and_pain_weight_from_the_weights():
  emo = emotion.EmotionV2(ToyPolicy())
  assert emo.S_V == pytest.approx(10.0)       # max(limit 1, hi - lo)
  assert emo.S_T == pytest.approx(1.0)        # 0.5 < limit
  assert emo.w_pain == 1.0 and emo.has_threat and emo.has_cost
  assert emo.outcome(5.0, 0.5) == pytest.approx(5.0 / 10.0 - 0.5 / 1.0)
  assert emo.outlook(20.0, 0.3) == pytest.approx(2.0 - 0.3)
  # No threat critic: T = 0, and pain still counts with weight 1.
  bare = emotion.EmotionV2(ToyPolicy(threat=False, cost=False))
  assert not bare.has_threat and bare.S_T == 1.0 and bare.w_pain == 1.0
  d, s, _ = bare.p.last_post
  assert bare.threat(d, s) == 0.0


def test_pleasure_is_outcome_plus_outlook_change():
  """delta_t + E_t = o_t + gamma O_{t+1} - O_t + (1 - gamma) O_{t+1}
  = o_t + O_{t+1} - O_t, for any gamma: the discount cancels."""
  rng = np.random.default_rng(1)
  for gamma in (0.9, 0.99, 1 - 1 / 130):
    o, O0, O1 = rng.normal(size=3)
    delta = o + gamma * O1 - O0
    assert delta + (1 - gamma) * O1 == pytest.approx(o + O1 - O0)
  # And EmotionV2 adds the outlook's change to the faded outcome.
  emo = emotion.EmotionV2(ToyPolicy())
  outlooks = iter([(10.0, 0.2), (30.0, 0.1)])
  emo.value, emo.threat = (lambda d, s: 0.0), (lambda d, s: 0.0)
  vals = {}

  def value(d, s):
    vals['v'], vals['t'] = next(outlooks)
    return vals['v']
  emo.value, emo.threat = value, (lambda d, s: vals['t'])
  d, s, logit = emo.p.last_post
  emo.step(d, logit, s, 0.0, 0.0)            # first outlook: no change yet
  before = emo.faded
  emo.tick(4.0, 0.25)
  after_tick = emo.faded
  lam = emo.lam
  assert after_tick == pytest.approx(lam * before + (1 - lam) * (0.4 - 0.25))
  sig = emo.step(d, logit, s, 4.0, 0.25)
  change = (3.0 - 0.1) - (1.0 - 0.2)
  assert sig['change'] == pytest.approx(change)
  assert emo.faded == pytest.approx(after_tick + (1 - lam) * change)
  assert -1 <= sig['pleasure'] <= 1 and sig['valence'] == sig['pleasure']


def test_twohot_nll():
  bins = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
  p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
  assert emotion._twohot_nll(p, bins, 0.0) == pytest.approx(-math.log(0.4), abs=1e-6)
  # Halfway between two bins: half of each log probability.
  assert emotion._twohot_nll(p, bins, 0.5) == pytest.approx(
      -0.5 * math.log(0.4) - 0.5 * math.log(0.2), abs=1e-6)
  assert emotion._twohot_nll(p, bins, 9.0) == pytest.approx(-math.log(0.1), abs=1e-6)


def test_empowerment_kl_terms():
  emo = emotion.EmotionV2(ToyPolicy())
  same = np.full((4, S, C), 1 / C)
  v = np.full((4, BINS), 1 / BINS)
  assert emo.empowerment_value(same, v, v) == pytest.approx(0.0, abs=1e-6)
  # Two commands, one latent each side of uniform: state KL / n, values equal.
  p = np.array([[[0.9, 0.1]], [[0.1, 0.9]]])
  mix = p.mean(0)[0]
  kl = sum(0.9 * math.log(0.9 / m) + 0.1 * math.log(0.1 / n) for m, n in [(mix[0], mix[1])])
  v2 = np.full((2, 3), 1 / 3)
  assert emo.empowerment_value(p, v2) == pytest.approx(kl / 1, rel=1e-5)
  p2 = np.repeat(p, 2, axis=1)   # two latents: the state term is per latent
  assert emo.empowerment_value(p2, v2) == pytest.approx(kl, rel=1e-5)
  # Value and threat terms add their own KLs, not divided by n.
  vv = np.array([[0.9, 0.1], [0.1, 0.9]])
  assert emo.empowerment_value(p2, vv, vv) == pytest.approx(3 * kl, rel=1e-5)
  # The rollout: finite, non-negative, one yield per step plus the value.
  d, s, _ = emo.p.last_post
  out = list(emo.empowerment_slices(d, s))
  assert len(out) == emo.horizon + 1 and out[-1] >= 0 and np.isfinite(out[-1])


@pytest.mark.parametrize('mode', ['kl', 'total', 'error', 'unexpected'])
def test_arousal_modes(mode):
  emo = emotion.EmotionV2(ToyPolicy(), arousal=mode)
  d, s, logit = emo.p.last_post
  err, expected = emo.prediction_error(d, s, 0.3, 0.0)
  assert np.isfinite(err) and np.isfinite(expected) and expected > 0
  for i in range(5):
    sig = emo.step(d, logit, s, 0.3 * i, 0.0)
    assert 0 < sig['arousal'] < 1 and -1 <= sig['dominance'] <= 1
  assert sig['unexpected'] == pytest.approx(sig['surprise'] - sig['expected'])
  assert sig['update'] >= -1e-6
  # The belief update is 0 when the posterior is the prior and its sample
  # the prior's likeliest state.
  pr = emotion._probs(emo.prior_logit(d), emo.p.meta['unimix'])
  assert emo.belief_update(d, np.log(pr), emo.mode_stoch(pr)) == pytest.approx(0.0, abs=1e-3)


def test_impact_pain_copies_agree():
  for rate, prev in ((0.0, None), (0.5, 0.0), (1.0, 2.0), (-2.0, 0.0), (3.5, -1.0)):
    assert policy_runner.impact_pain(rate, prev, 1.0, 3.0) == robot.impact_pain(
        rate, prev, 1.0, 3.0)
  assert robot.impact_pain(0.9, 0.0, 1.0, 3.0) == (0.0, 0.9)
  assert robot.impact_pain(2.0, 0.0, 1.0, 3.0)[0] == pytest.approx(0.5)
  assert robot.impact_pain(-4.0, 0.0, 1.0, 3.0)[0] == 1.0


def _runner_v2(**reflex):
  r = policy_runner.PolicyRunner()
  pol = ToyPolicy()
  r.policy = pol
  r.configure(dict(body=dict(theta_zero=0.0, lo=-0.12, hi=0.161, zone=0.035,
                             pain_mode='impact', iota0=1.0, iota1=3.0),
                   reflex=dict(reflex)))
  r._install_emotion(pol)
  return r


def test_runner_feels_with_v2_and_switches_to_v1():
  r = _runner_v2()
  assert isinstance(r.emotion, emotion.EmotionV2)
  r.pain = (0.8, 0.0, 0.0)
  for _ in range(r.FEEL_EVERY):
    r.feel(0.1)
  st = r.emotion_state
  assert st['model'] == 'v2' and st['pain'] == pytest.approx(0.8)
  for k in ('valence', 'arousal', 'dominance', 'threat'):
    assert np.isfinite(st[k])
  # The fast refresh keeps the v2 state and moves pleasure with the outcome.
  r._fast_affect()
  assert r.emotion_state['model'] == 'v2'
  r.configure(dict(model='v1'))
  assert type(r.emotion) is emotion.Emotion


def test_reflex_steps_back_along_the_slope():
  r = _runner_v2(on=True, tmax=1.5, eps=0.1, steps=4)
  # A threat linear in the drive: T(a) = 1 + 2 mean(a).
  r.emotion.threat_after = lambda d, s, acts, steps: 1.0 + 2.0 * np.asarray(acts).mean(-1)
  out = r.reflex([0.5, 0.5])            # T = 2: 0.5 over, slope 2
  assert out == pytest.approx([0.25, 0.25], abs=1e-5)
  assert r.reflex_last == pytest.approx(-0.25, abs=1e-5)
  assert r.reflex([0.1, 0.3]) == [0.1, 0.3]   # T = 1.4 < tmax: untouched
  assert r.reflex_last == 0.0
  out = r.reflex([1.0, 0.8])            # T = 2.8: back by 0.65 each
  assert out == pytest.approx([0.35, 0.15], abs=1e-5)
  # Flat threat: no slope to follow.
  r.emotion.threat_after = lambda d, s, acts, steps: np.full(len(acts), 9.0)
  assert r.reflex([0.5, 0.5]) == [0.5, 0.5]
  # Off (the default): nothing happens.
  off = _runner_v2()
  off.emotion.threat_after = lambda *a: np.full(3, 9.0)
  assert off.reflex([0.5, 0.5]) == [0.5, 0.5]
  # Through modulate(), with the toy world model's own threat.
  real = _runner_v2(on=True, tmax=-1e9)
  out = real.modulate([0.2, 0.2])
  assert all(-1 <= x <= 1 for x in out)


def test_threat_after_rolls_the_toy_model():
  emo = emotion.EmotionV2(ToyPolicy())
  d, s, _ = emo.p.last_post
  t = emo.threat_after(d, s, np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]), 4)
  assert t.shape == (3,) and np.all(np.isfinite(t))


def test_sim_logs_impact_and_bumper_force():
  pytest.importorskip('mujoco')
  from embodied.envs import robot_mujoco
  env = robot_mujoco.RobotMujoco(
      'balance', hz=65.0, discrete=False, symmetric=False, sim_noise=0.0,
      sim_surface='carpet_fit2', cost_obs=True, pain_mode='impact')
  space = env.obs_space
  assert 'log/impact' in space and 'log/bumper_force' in space and 'cost' in space
  # Fall forward onto the front bumper from upright: a hit, then resting.
  env.theta, env.omega = env.theta_zero + 0.5 * env.bumper_hi, 4.0
  forces, pains = [], []
  for _ in range(40):
    env.pending = np.zeros(2)
    sensors, _ = env._recv()
    forces.append(env.physics.bumper_force)
    pains.append(env._pain(sensors))
  assert max(forces) > 0.5 and forces[-1] > 0.1   # the hit, then the weight
  assert max(pains) > 0 and pains[-1] == 0.0       # pain at the hit only
