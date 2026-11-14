"""The phone's onboard loop inside the sim (env.robot.sim_phone).

On the robot the phone runs the policy itself (deploy/policy_runner.py): it
reads its sensors, acts at once, and sends the trainer the observation with
the action it applied; the trainer pushes it new weights every few seconds.
Everything the phone does with emotion -- the face's signals, pain, the
affect on behaviour, the arousal events that make the learner change its
weights at once -- lives in that loop. This puts the same loop in the sim,
so it can be tested there before it runs on the robot:

* weights: the learner's newest published policy, packed as a push to the
  phone is (export.pack_arrays + encode, int8 by default, with the emotion
  heads), installed once it is `delay` s old (a push reached the phone ~4 s
  after it was published, 11-04);
* each step: PolicyRunner.feel(reward), act(sensors), modulate(act); the
  env then applies the trainer's limits (cap, brake, fatigue, travel) as the
  app does with what the trainer tells it, and hands the result back to
  applied() and to replay as the executed action;
* events: PolicyRunner's (arousal rising `event` above its own ~10 s
  average) are written to <shared>/emotion_event for the learner
  (embodied/run/online.py), as the trainer does with the phone's;
* a CSV of the signals every FEEL_EVERY steps, for the analysis; or with
  `log_every` (env.robot.sim_phone_log_every) > 0 every log_every steps
  plus every step with a push, an event, bumper contact or a reflex
  correction (smaller files, the same counts for those). With
  `heavy_every` (env.robot.sim_phone_heavy_every) > 0 the rows keep every
  column of LIGHT and fill the others only every heavy_every-th row (empty
  cells otherwise; forward-fill when reading): a 60-min run every step was
  ~75 MB with all columns.
"""

import os
import sys
import tempfile
import threading
import time

import numpy as np

_DEPLOY = os.path.join(os.path.dirname(__file__), '..', '..', 'dreamerv3', 'deploy')
if _DEPLOY not in sys.path:
  sys.path.insert(0, _DEPLOY)   # policy_runner imports numpy_policy flat, as on the phone

from policy_runner import PolicyRunner  # noqa: E402

FIELDS = ('t', 'step', 'stamp', 'arousal', 'valence', 'mood', 'dominance',
          'td', 'unexpected', 'empowerment', 'pain', 'hurt', 'fear', 'explore',
          'event', 'push', 'theta', 'act0', 'chosen0',
          # Since 11-12 (emotion v2, impact pain, catch reflex): the threat
          # critic's value, the impact, and the reflex's last correction.
          'threat', 'impact', 'reflex',
          # And every arousal variant (EmotionV2: total, unexpected, kl, each
          # also against the efference copy), the policy's intended action,
          # whether an efference copy differed, and the true bumper force (N).
          'pleasure', 'arousal_total', 'arousal_unexpected', 'arousal_kl',
          'arousal_total_eff', 'arousal_unexpected_eff', 'arousal_kl_eff',
          'efference', 'intended0', 'force',
          # Pleasure's parts (faded = p~; o_reward - o_pain = o; change =
          # O_{t+1} - O_t; outlook = O; td the plain TD error) and the
          # empowerment end-point terms at each recorded rollout length.
          'faded', 'o_reward', 'o_pain', 'change', 'outlook', 'value',
          'emp16_state', 'emp16_value', 'emp16_threat',
          'emp32_state', 'emp32_value', 'emp32_threat',
          # Since round RF (11-12): the reflex's predicted threat T(a) of the
          # intended command, its limit, and its time (ms).
          'reflex_threat', 'reflex_thr', 'reflex_ms',
          # Since round EM (11-14): the Pearce-Hall associability alpha and
          # its arousal, Yu's prior and posterior probability of pain > 0
          # from the cost head, the value and threat critics' two-hot SD (in
          # symlog units), and
          # the empowerment rollouts finished so far and the last one's ms.
          'ph_alpha', 'arousal_ph', 'p_pain_prior', 'q_pain_post',
          'value_std', 'threat_std', 'emp_n', 'emp_ms')

# Written every row with heavy_every > 0; the rest every heavy_every-th row.
LIGHT = frozenset((
    't', 'step', 'arousal', 'dominance', 'pleasure', 'td', 'faded', 'change',
    'pain', 'threat', 'value', 'impact', 'reflex', 'event', 'push', 'theta',
    'act0', 'intended0', 'force', 'reflex_threat', 'ph_alpha', 'p_pain_prior',
    'q_pain_post', 'value_std', 'threat_std', 'emp_n'))


class SimPhone:

  def __init__(self, config, shared, dtype='int8', delay=3.0, poll=1.0,
               affect=None, body=None, event=0.0, event_signal='arousal', log=None,
               seed=0, settings=None, feel_every=4, log_every=0, threads=0,
               heavy_every=0):
    self.config = config
    self.shared = str(shared)
    self.dtype = dtype
    self.delay = float(delay)
    self.poll = float(poll)
    self.runner = PolicyRunner()
    # The phone feels every 4 control steps; the sim can feel (and log)
    # every step (env.robot.sim_phone_feel_every 1) for the analysis.
    self.runner.FEEL_EVERY = max(1, int(feel_every))
    self.log_every = int(log_every)
    self.heavy_every = int(heavy_every)
    self._rows = 0
    # threads > 0: cap BLAS threads in this (the actor's) process. The
    # phone's matrices are small; OpenBLAS's default of one thread per core
    # of the node, in a 3-CPU job beside the learner, made a 16-step reflex
    # rollout 4x slower and the loop fall to ~30 Hz (round RF smokes, 11-12).
    self._threads = None
    if int(threads) > 0:
      try:
        from threadpoolctl import threadpool_limits
        self._threads = threadpool_limits(limits=int(threads), user_api='blas')
      except Exception as e:  # noqa: BLE001
        print(f'[sim phone] could not cap BLAS threads: {e}', flush=True)
    self.force = 0.0      # set by the env: the true bumper contact force
    # `settings`: what the trainer sends the phone (robot.py
    # _emotion_settings); the separate arguments are the older form.
    self.runner.configure(settings or dict(
        affect=affect or {}, body=body, emotion_event=event,
        event_signal=event_signal))
    self._event_n = None
    self._dir = tempfile.mkdtemp(prefix='sim_phone_')
    # (policy, stamp, published_at) built off-thread; each lands on the
    # phone `delay` s after it was published, the newest that has wins.
    self._ready = []
    self._lock = threading.Lock()
    self._seen = None
    self._seed = seed
    self.steps = 0
    self.push = 0.0       # set by the env when it pushes the body
    self._log = open(log, 'a', buffering=1) if log else None
    if self._log and self._log.tell() == 0:
      self._log.write(','.join(FIELDS) + '\n')
    self._closing = threading.Event()
    self._thread = threading.Thread(target=self._loop, daemon=True,
                                    name='sim_phone_weights')
    self._thread.start()

  def close(self):
    self._closing.set()

  # -- weights, as the trainer's pusher builds them ---------------------------

  def _loop(self):
    from dreamerv3.deploy import export as exportlib
    from embodied.envs.robot import _load_policy_pickle
    while not self._closing.wait(self.poll):
      try:
        latest = os.path.join(self.shared, 'policy', 'latest')
        if not os.path.exists(latest):
          continue
        stamp = open(latest).read().strip()
        if not stamp or stamp == self._seen:
          continue
        path = os.path.join(self.shared, 'policy', f'policy_{stamp}.pkl')
        if not os.path.exists(path):
          continue
        with open(path, 'rb') as f:
          data = _load_policy_pickle(f.read())
        params = data['params'] if isinstance(data, dict) and 'params' in data \
            else data
        arrays, meta = exportlib.pack_arrays(self.config, params, self.dtype, True)
        blob = exportlib.encode(meta, arrays, stamp=stamp,
                                compress=(self.dtype == 'int8'))
        policy = PolicyRunner.build_blob(
            blob, os.path.join(self._dir, 'policy.npz'))
        policy.rng = np.random.default_rng((self._seed, int(stamp) % 2**31))
        with self._lock:
          self._ready.append((policy, stamp, int(stamp) / 1e9))
          del self._ready[:-8]
        self._seen = stamp
      except Exception as e:  # noqa: BLE001 -- the run goes on, on old weights
        print(f'[sim phone] could not build weights: {e}', flush=True)

  def _install(self):
    now = time.time()
    with self._lock:
      due = [r for r in self._ready if now - r[2] >= self.delay]
      if not due:
        return
      ready = due[-1]
      self._ready = [r for r in self._ready if r[2] > ready[2]]
    explore = getattr(self.runner.policy, 'explore', 1.0)
    self.runner.install(ready[0], ready[1])
    ready[0].explore = explore

  # -- a step -----------------------------------------------------------------

  def decide(self, sensors, reward, is_first, command=None):
    """The policy's (affected) action for these sensors, or None before the
    first weights arrive (the robot then holds its wheels)."""
    self._install()
    r = self.runner
    if not r.ready:
      return None
    if is_first:
      r.reset()
    else:
      r.feel(reward)
    _, act = r.act(sensors, command)
    return r.modulate(act)

  def applied(self, act, step):
    """The wheels got `act` (after the trainer's limits)."""
    r = self.runner
    if not r.ready:
      return
    r.applied(act)
    self.steps += 1
    st = r.emotion_state
    event = 0.0
    if r.event is not None and r.event['n'] != self._event_n:
      # As the app sends it with its next observation, and the trainer
      # hands it to the learner (robot.write_event).
      self._event_n = r.event['n']
      event = r.event['intensity']
      from embodied.envs.robot import write_event
      write_event(self.shared, r.event)
    if self.log_every > 0:
      due = (self.steps % self.log_every == 0 or bool(event) or bool(self.push)
             or self.force > 0 or r.reflex_last != 0)
    else:
      due = self.steps % r.FEEL_EVERY == 0 or event or self.push
    if self._log and st and due:
      row = {k: st[k] for k in FIELDS if k in st}
      row.update(t=time.time(), step=step, stamp=r.stamp, event=event,
                 push=self.push, theta=r._theta, act0=act[0],
                 chosen0=(r._chosen or [0.0])[0], explore=r.policy.explore,
                 impact=r.impact, reflex=r.reflex_last, force=self.force,
                 intended0=(r._intended or [0.0])[0],
                 reflex_threat=r.reflex_threat, reflex_thr=r.reflex_thr,
                 reflex_ms=r.reflex_ms)
      every, rows = getattr(self, 'heavy_every', 0), getattr(self, '_rows', 0)
      heavy = every <= 0 or rows % every == 0
      self._rows = rows + 1
      self._log.write(','.join(
          '' if not (heavy or k in LIGHT) else
          (f'{row[k]:.3f}' if k == 't' else f'{row[k]:.5g}'
           if isinstance(row.get(k), float) else str(row.get(k, '')))
          for k in FIELDS) + '\n')
