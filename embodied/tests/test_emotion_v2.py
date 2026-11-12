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


def test_reflex_slope_step_is_capped_by_dmax():
  r = _runner_v2(on=True, tmax=0.0, eps=0.1, dmax=0.2)
  # A nearly flat threat: the uncapped step would go to the clip.
  r.emotion.threat_after = lambda d, s, acts, steps: 1.0 + 0.01 * np.asarray(acts).mean(-1)
  out = r.reflex([0.3, 0.1])
  assert out == pytest.approx([0.1, -0.1], abs=1e-6)
  assert r.reflex_last == pytest.approx(-0.2, abs=1e-6)
  assert r.reflex_threat == pytest.approx(1.002) and r.reflex_thr == 0.0
  uncapped = _runner_v2(on=True, tmax=0.0, eps=0.1)
  uncapped.emotion.threat_after = r.emotion.threat_after
  assert uncapped.reflex([0.3, 0.1]) == [-1.0, -1.0]


def test_reflex_search_takes_the_nearest_safe_command():
  r = _runner_v2(on=True, tmax=0.5, mode='search', offsets='0.25,0.5,1', steps=16)
  seen = {}

  def threat(d, s, acts, steps):
    seen['acts'], seen['steps'] = np.asarray(acts), steps
    return np.abs(np.asarray(acts).mean(-1) - 0.6)   # safe within 0.5 of 0.6

  r.emotion.threat_after = threat
  out = r.reflex([-0.2, -0.2])          # T = 0.8: +0.25 gives 0.55, +0.5 gives 0.3
  assert out == pytest.approx([0.3, 0.3])
  assert r.reflex_last == pytest.approx(0.5)
  assert seen['steps'] == 16 and len(seen['acts']) == 7   # a, +-0.25, +-0.5, +-1
  assert r.reflex([0.5, 0.5]) == [0.5, 0.5]                # T = 0.1: untouched
  assert r.reflex_last == 0.0
  # Nothing safe: the lowest-threat candidate; the clip keeps it in range.
  r.emotion.threat_after = lambda d, s, acts, steps: 2.0 - np.asarray(acts).mean(-1)
  out = r.reflex([0.8, 0.6])
  assert out == [1.0, 1.0]
  # A flat threat: every candidate as bad, so the command stays (the first).
  r.emotion.threat_after = lambda d, s, acts, steps: np.full(len(acts), 9.0)
  assert r.reflex([0.4, 0.4]) == pytest.approx([0.4, 0.4])
  assert r.reflex_last == 0.0


def test_reflex_relative_limit_learns_the_run():
  r = _runner_v2(on=True, mode='search', z=3.0, window=50, tmax=-1e9)
  level = {'t': 0.1}
  rng = np.random.default_rng(0)
  r.emotion.threat_after = lambda d, s, acts, steps: (
      np.full(len(acts), level['t']) + 0.01 * rng.normal())
  # Warm-up: silent for the first `window` steps, whatever tmax says.
  for _ in range(50):
    assert r.reflex([0.2, 0.2]) == [0.2, 0.2]
  assert np.isinf(r.reflex_thr)
  # Then the usual level (~0.1 +- 0.01) passes ...
  for _ in range(100):
    r.reflex([0.2, 0.2])
  assert r.reflex_thr == pytest.approx(0.13, abs=0.02)
  assert r.reflex_last == 0.0
  # ... and a jump to 0.3 fires (here every candidate is as bad: the
  # lowest-threat one, a candidate other than a only if lower).
  level['t'] = 0.3
  r.emotion.threat_after = lambda d, s, acts, steps: 0.3 - 0.2 * (np.asarray(acts).mean(-1) < 0)
  out = r.reflex([0.2, 0.2])
  assert out == pytest.approx([-0.05, -0.05])
  # The limit tracks a slow drift: 1000 steps at 0.3 make 0.3 usual.
  r.emotion.threat_after = lambda d, s, acts, steps: np.full(len(acts), 0.3) + 0.01 * rng.normal()
  for _ in range(1000):
    r.reflex([0.2, 0.2])
  assert r.reflex_thr == pytest.approx(0.33, abs=0.02)
  assert r.reflex([0.2, 0.2]) == [0.2, 0.2]


def test_reflex_relative_limit_ignores_spikes_and_scales_in_log():
  rng = np.random.default_rng(1)
  for zlog in (False, True):
    r = _runner_v2(on=True, mode='search', z=3.0, window=100, zlog=zlog)
    level = {'t': 0.1}
    r.emotion.threat_after = lambda d, s, acts, steps: np.full(
        len(acts), level['t'] * (1 + 0.1 * rng.normal()))
    for _ in range(300):
      r.reflex([0.0, 0.0])
    before = r.reflex_thr
    # Spikes, as around a hit, enter capped: the limit barely moves.
    level['t'] = 10.0
    for _ in range(20):
      r.reflex([0.0, 0.0])
    level['t'] = 0.1
    r.reflex([0.0, 0.0])
    assert r.reflex_thr < 1.5 * before
  # In log space the limit is a multiple of the level: the critic's level
  # falling 100x moves the limit down 100x.
  r = _runner_v2(on=True, mode='search', z=3.0, window=100, zlog=True)
  level = {'t': 0.1}
  r.emotion.threat_after = lambda d, s, acts, steps: np.full(
      len(acts), level['t'] * np.exp(0.1 * rng.normal()))
  for _ in range(400):
    r.reflex([0.0, 0.0])
  hi = r.reflex_thr
  level['t'] = 0.001
  for _ in range(3000):
    r.reflex([0.0, 0.0])
  assert r.reflex_thr == pytest.approx(hi / 100, rel=0.3)
  assert hi == pytest.approx(0.1 * np.exp(0.3), rel=0.2)


def test_reflex_rising_fires_only_while_threat_climbs():
  r = _runner_v2(on=True, tmax=0.5, mode='search', offsets='1', rising=True)
  level = {'t': 1.0}
  r.emotion.threat_after = lambda d, s, acts, steps: level['t'] - 0.6 * np.asarray(acts).mean(-1)
  assert r.reflex([0.0, 0.0]) == [1.0, 1.0]   # first step: nothing to compare, fires
  assert r.reflex([0.0, 0.0]) == [0.0, 0.0]   # same T: not rising, holds
  level['t'] = 1.1
  assert r.reflex([0.0, 0.0]) == [1.0, 1.0]   # rising again: fires


def test_reflex_settings_from_the_env_reach_the_runner():
  env_reflex = dict(on=True, tmax=0.2, eps=0.1, steps=16, mode='search',
                    dmax=0.0, offsets='0.25,0.5', z=2.5, window=650, zlog=True,
                    rising=False, every=4)
  r = _runner_v2(**env_reflex)
  assert r.reflex_cfg == env_reflex
  # The defaults are the first version: slope, 4 steps, absolute limit.
  d = policy_runner.PolicyRunner().reflex_cfg
  assert (d['on'], d['mode'], d['steps'], d['z'], d['dmax'], d['rising'],
          d['every']) == (False, 'slope', 4, 0.0, 0.0, False, 1)
  # And the env sends the phone the same keys from its catch_* settings.
  pytest.importorskip('mujoco')
  from embodied.envs import robot_mujoco
  env = robot_mujoco.RobotMujoco(
      'balance', hz=65.0, discrete=False, symmetric=False, sim_noise=0.0,
      catch_reflex=True, catch_tmax=0.2, catch_steps=16, catch_mode='search',
      catch_offsets='0.25,0.5', catch_z=2.5, catch_zlog=True, catch_every=4,
      sim_phone_log_every=16, sim_phone_threads=1)
  assert env._emotion_settings()['reflex'] == env_reflex
  assert env._phone_args['log_every'] == 16 and env._phone_args['threads'] == 1
  with pytest.raises(AssertionError):
    robot_mujoco.RobotMujoco('balance', hz=65.0, catch_mode='brake')


def test_reflex_every_holds_the_correction_between_evaluations():
  r = _runner_v2(on=True, tmax=0.5, mode='search', offsets='0.5', every=3)
  calls = []

  def threat(d, s, acts, steps):
    calls.append(1)
    return 1.0 - np.asarray(acts).mean(-1)   # over the limit below a = 0.5

  r.emotion.threat_after = threat
  assert r.reflex([0.2, 0.2]) == pytest.approx([0.7, 0.7])   # evaluates: +0.5
  assert r.reflex([0.0, 0.1]) == pytest.approx([0.5, 0.6])   # held, no rollout
  assert r.reflex([0.9, 0.9]) == pytest.approx([1.0, 1.0])   # held, clipped
  assert r.reflex_last == pytest.approx(0.1) and r.reflex_ms == 0.0
  assert len(calls) == 1
  assert r.reflex([0.8, 0.8]) == [0.8, 0.8]                  # evaluates: safe
  assert r.reflex([0.0, 0.0]) == [0.0, 0.0]                  # nothing to hold
  assert len(calls) == 2


def test_reflex_times_itself_on_the_toy_model():
  r = _runner_v2(on=True, tmax=-1e9, mode='search', steps=16)
  out = r.modulate([0.2, 0.2])
  assert all(-1 <= x <= 1 for x in out)
  assert r.reflex_ms > 0 and np.isfinite(r.reflex_threat)


def test_sim_phone_log_every_keeps_the_rare_rows(tmp_path):
  from embodied.envs import sim_phone
  ph = sim_phone.SimPhone.__new__(sim_phone.SimPhone)
  log = tmp_path / 'emotion.csv'
  ph._log = open(log, 'w', buffering=1)
  ph._log.write(','.join(sim_phone.FIELDS) + '\n')
  ph.runner = _runner_v2(on=True, tmax=1e9)
  ph.runner._fast_affect = lambda: None
  ph.runner.emotion_state = dict(arousal=0.0, valence=0.0, mood=0.0, pain=0.0)
  ph.runner.event = None
  ph.runner.policy.explore = 1.0
  ph.steps, ph._event_n, ph.push, ph.force, ph.log_every = 0, None, 0.0, 0.0, 16
  kept = []
  for i in range(64):
    ph.force = 2.0 if i == 5 else 0.0
    ph.push = 0.1 if i == 9 else 0.0
    ph.runner.reflex_last = -0.3 if i == 20 else 0.0
    ph.applied([0.0, 0.0], i)
  ph._log.close()
  import csv
  steps = [int(r['step']) for r in csv.DictReader(open(log))]
  assert steps == [5, 9, 15, 20, 31, 47, 63]   # every 16th (steps 16, 32, ...) plus rare
  row = list(csv.DictReader(open(log)))[3]
  assert float(row['reflex']) == -0.3 and 'reflex_ms' in row


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


def test_efference_copy_variants():
  emo = emotion.EmotionV2(ToyPolicy(), efference=True)
  d, s, logit = emo.p.last_post
  # No override: the efference variants equal the plain ones.
  sig = emo.step(d, logit, s, 0.2, 0.0)
  for k in ('total', 'unexpected', 'kl'):
    assert sig[f'arousal_{k}_eff'] == sig[f'arousal_{k}']
  assert sig['efference'] == 0.0
  # With a different prior state the efference variants move.
  eff = emo.efference_deter(d, s, [1.0, -1.0])
  assert eff.shape == d.shape
  sig = emo.step(d, logit, s, 0.2, 0.0, eff_deter=eff)
  assert sig['efference'] == 1.0
  assert sig['arousal_kl_eff'] != pytest.approx(sig['arousal_kl'])
  assert sig['arousal_total_eff'] != pytest.approx(sig['arousal_total'])


def test_runner_takes_the_efference_copy_only_when_overridden():
  r = _runner_v2()
  r.configure(dict(efference=True))
  assert r.emotion.efference
  r._intended = [0.5, 0.5]
  r.applied([0.5, 0.5])                 # as intended: no copy
  assert r._eff_next is None
  r._intended = [0.5, 0.5]
  r.applied([0.0, 0.0])                 # held by a limit: a copy
  assert r._eff_next is not None and r._eff_next.shape == (D,)


def test_dominance_terms_at_two_horizons():
  emo = emotion.EmotionV2(ToyPolicy(), horizon=16, empower_extra=(32,))
  d, s, _ = emo.p.last_post
  out = list(emo.empowerment_slices(d, s))
  assert len(out) == 33 and all(x is None for x in out[:-1])
  t = emo.empower_terms
  for h in (16, 32):
    for k in ('state', 'value', 'threat'):
      assert t[f'emp{h}_{k}'] >= -1e-9
  assert out[-1] == pytest.approx(t['emp16_state'] + t['emp16_value'] + t['emp16_threat'])
  sig = emo.step(d, emo.p.last_post[2], s, 0.0, 0.0)
  assert 'emp32_value' in sig


def test_pleasure_parts_for_offline_rebuilds():
  emo = emotion.EmotionV2(ToyPolicy())
  d, s, logit = emo.p.last_post
  a = emo.step(d, logit, s, 0.0, 0.0)
  b = emo.step(d, logit, s, 3.0, 0.4)
  assert b['o_reward'] == pytest.approx(3.0 / emo.S_V)
  assert b['o_pain'] == pytest.approx(emo.w_pain * 0.4 / emo.S_T)
  assert b['td'] == pytest.approx(b['o_reward'] - b['o_pain']
                                  + emo.gamma * b['outlook'] - a['outlook'])
  assert b['change'] == pytest.approx(b['outlook'] - a['outlook'])
