"""Emotion signals from the world model, in numpy: valence and arousal.

Runs beside `numpy_policy.NumpyPolicy` (same weights file, which then also
needs the RSSM prior and the critic and reward heads: export with
`emotion=True`), on the phone for its face and offline on recorded runs
(`tools/robot/dev/emotion_trace.py`). Per step, with the posterior q_t the
policy just formed from the observation and the prior p_t the world model
predicted for it:

  valence     TD error r_t + gamma V(s_t) - V(s_{t-1}) (Schultz et al. 1997):
              better or worse than the critic expected
  surprise    KL(q_t || p_t), nats: how much the observation moved the belief
  novelty     H(p_t): how uncertain the prediction was before it
  unexpected  CE(q_t, p_t) - H(p_t) = H(q_t) + KL - H(p_t): the surprise beyond
              what the prediction's own spread expected (Yu & Dayan 2005's
              unexpected uncertainty; static noise scores ~0)
  reward_err  |r_t - r_hat(s_t)|: reward prediction error
  empowerment one-step-to-n-step channel proxy: roll K fixed action sequences
              n steps through the prior from s_t and take the mean KL of each
              end prediction from their mixture (per latent, summed): how much
              what the robot does changes where it ends up (Klyubin et al. 2005)

and from those, normalised online (running mean and SD, `tau` steps):

  arousal = sigmoid(w_n z(novelty) + w_u z(unexpected) + w_r z(reward_err)
                    - w_e z(empowerment) + w_h z(|TD|) * (1 - control))
  valence = tanh(TD error / its running RMS)        joy / distress, brief
  mood    = tanh(z(V(s_t)), slow)                   hope / fear (Broekens 2015)
  dominance = tanh(z(empowerment))

The last arousal term is helplessness (Maier & Seligman 2016): large errors
with little control, control = sigmoid(z(empowerment)). With a dense reward
the TD error sits near 0 most of the time, so the face shows `face_valence`,
mostly mood with the TD error's flashes on top.

Arousal rises with novelty and unexpected surprise and falls with control;
low control with large errors is the high-arousal corner either way, as
in helplessness or gambling.
"""

import numpy as np

from numpy_policy import F32, linear, rms_norm, silu, sigmoid, softmax  # noqa: E402  (flat layout on the phone)

# Empowerment's action set: the 3x3 grid of constant wheel commands.
ACTIONS = np.array([(l, r) for l in (-1.0, 0.0, 1.0) for r in (-1.0, 0.0, 1.0)],
                   F32)


def _probs(logit, unimix):
  p = softmax(np.asarray(logit, F32), -1)
  if unimix:
    p = (1 - unimix) * p + unimix / p.shape[-1]
  return p


def _entropy(p):
  return float(-(p * np.log(p + 1e-8)).sum())


def _kl(p, q):
  return float((p * (np.log(p + 1e-8) - np.log(q + 1e-8))).sum())


def _twohot_bins(n):
  half = np.linspace(-20, 0, (n - 1) // 2 + 1, dtype=np.float64)
  half = np.sign(half) * (np.exp(np.abs(half)) - 1)   # symexp
  return np.concatenate([half, -half[:-1][::-1]], 0).astype(F32)


class Running:
  """Exponential running mean and SD, for z-scores without a fixed scale."""

  def __init__(self, tau):
    self.a = 1.0 / float(tau)
    self.mean, self.var, self.n = 0.0, 1.0, 0

  def __call__(self, x):
    x = float(x)
    if self.n == 0:
      self.mean, self.var = x, 1e-6
    a = max(self.a, 1.0 / (self.n + 1))
    d = x - self.mean
    self.mean += a * d
    self.var = (1 - a) * (self.var + a * d * d)
    self.n += 1
    return d / np.sqrt(self.var + 1e-8)


class Emotion:

  def __init__(self, policy, gamma=1 - 1 / 130, horizon=8, every=1, tau=650,
               weights=(1.0, 1.0, 1.0, 1.0, 1.0)):
    """`policy` is a loaded NumpyPolicy whose file has the prior and heads.
    `every`: empowerment only every this many steps (it costs K x horizon
    GRU steps), held in between. `tau`: steps of the running normalisation
    (650 = 10 s at 65 Hz). `weights`: (novelty, unexpected, reward error,
    empowerment, helplessness) in the arousal sum."""
    self.p = policy
    missing = [k for k in ('dyn/priorlogit/kernel', 'val/head/logits/kernel',
                           'rew/head/logits/kernel') if k not in policy.w]
    if missing:
      raise ValueError(f'weights lack {missing}: export with emotion=True')
    self.gamma, self.horizon, self.every = float(gamma), int(horizon), int(every)
    self.w = np.asarray(weights, F32)
    self.norm = {k: Running(tau) for k in (
        'novelty', 'unexpected', 'reward_err', 'empowerment', 'td', 'abs_td')}
    self.mood_norm = Running(tau * 6)
    self.pain_w = 1.0
    self.bins = _twohot_bins(policy.w['val/head/logits/bias'].shape[-1])
    self.reset()

  def reset(self):
    self.prev_value = None
    self.steps = 0
    self.empowerment = 0.0

  # -- network pieces the policy file does not use for acting --------------

  def prior_logit(self, deter):
    w, m = self.p.w, self.p.meta
    x = deter
    i = 0
    while f'dyn/prior{i}/kernel' in w:
      x = linear(x, w[f'dyn/prior{i}/kernel'], w[f'dyn/prior{i}/bias'])
      x = silu(rms_norm(x, w[f'dyn/prior{i}norm/scale']))
      i += 1
    logit = linear(x, w['dyn/priorlogit/kernel'], w['dyn/priorlogit/bias'])
    return logit.reshape(*deter.shape[:-1], m['stoch'], m['classes'])

  def head(self, name, deter, stoch):
    w = self.p.w
    x = np.concatenate([deter, stoch.reshape(*deter.shape[:-1], -1)], -1)
    i = 0
    while f'{name}/mlp/linear{i}/kernel' in w:
      x = linear(x, w[f'{name}/mlp/linear{i}/kernel'], w[f'{name}/mlp/linear{i}/bias'])
      x = silu(rms_norm(x, w[f'{name}/mlp/norm{i}/scale']))
      i += 1
    logits = linear(x, w[f'{name}/head/logits/kernel'], w[f'{name}/head/logits/bias'])
    return float((softmax(logits, -1) * self.bins).sum(-1))

  def _core_batch(self, deter, stoch, action):
    """NumpyPolicy._core over a leading batch axis."""
    w, m = self.p.w, self.p.meta
    g, B = m['blocks'], deter.shape[0]
    stoch = stoch.reshape(B, -1)
    action = action / np.maximum(1.0, np.abs(action))
    x0 = silu(rms_norm(linear(deter, w['dyn/dynin0/kernel'], w['dyn/dynin0/bias']), w['dyn/dynin0norm/scale']))
    x1 = silu(rms_norm(linear(stoch, w['dyn/dynin1/kernel'], w['dyn/dynin1/bias']), w['dyn/dynin1norm/scale']))
    x2 = silu(rms_norm(linear(action, w['dyn/dynin2/kernel'], w['dyn/dynin2/bias']), w['dyn/dynin2norm/scale']))
    x = np.concatenate([x0, x1, x2], -1)
    x = np.repeat(x[:, None, :], g, axis=1)
    x = np.concatenate([deter.reshape(B, g, -1), x], -1)          # (B, g, in)
    for i in range(m['dynlayers']):
      x = self._block(x, w[f'dyn/dynhid{i}/kernel'], w[f'dyn/dynhid{i}/bias'])
      x = silu(rms_norm(x.reshape(B, -1), w[f'dyn/dynhid{i}norm/scale'])).reshape(B, g, -1)
    x = self._block(x, w['dyn/dyngru/kernel'], w['dyn/dyngru/bias'])
    reset, cand, update = np.split(x, 3, axis=-1)
    reset, cand, update = (y.reshape(B, -1) for y in (reset, cand, update))
    cand = np.tanh(sigmoid(reset) * cand)
    update = sigmoid(update - 1.0)
    return update * cand + (1 - update) * deter

  @staticmethod
  def _block(x, kernel, bias):
    # x (B, g, in) with kernel (g, in, out): each block its own weights.
    y = np.einsum('bgi,gio->bgo', x, kernel)
    return y + bias.reshape(1, kernel.shape[0], -1)

  def _empowerment(self, deter, stoch):
    for value in self.empowerment_slices(deter, stoch):
      pass
    return value

  def empowerment_slices(self, deter, stoch):
    """_empowerment one rollout step at a time: yields None after each of
    the `horizon` steps, then the value. On the phone the whole rollout holds
    the interpreter for ~100 ms, which starved the 65 Hz control loop to
    ~55 Hz (2026-11-06); a caller that sleeps between slices lets it in."""
    K, unimix = len(ACTIONS), self.p.meta['unimix']
    d = np.repeat(deter[None], K, 0)
    s = np.repeat(stoch[None], K, 0)
    for _ in range(self.horizon):
      d = self._core_batch(d, s, ACTIONS)
      p = _probs(self.prior_logit(d), unimix)            # (K, stoch, classes)
      s = np.eye(p.shape[-1], dtype=F32)[p.argmax(-1)]   # the likeliest latent
      yield None
    mix = p.mean(0, keepdims=True)
    yield float(np.mean([_kl(p[k], mix[0]) for k in range(K)]))

  # -- the step ---------------------------------------------------------------

  def step(self, deter, post_logit, stoch, reward, is_first=False, steps=1,
           pain=0.0):
    """`deter`: the step's deterministic state (precompute's output);
    `post_logit`/`stoch`: the posterior logits and its sample; `reward`: the
    reward that came with this observation. Called every `steps` control
    steps (the phone), `reward` is their discounted sum since the last call.
    Returns a dict of signals. With every <= 0 empowerment is left to the
    caller (`self.empowerment`, e.g. from a thread). `pain` (0-1, from the
    body: closeness to a bumper, a reflex overriding the policy; see
    policy_runner.pain_signal) is a sensation, not a prediction: it raises
    arousal whatever its context, and lowers the face's valence by less the
    more the robot is in control (Salomons et al. 2004)."""
    unimix = self.p.meta['unimix']
    if is_first:
      self.reset()
    q = _probs(post_logit, unimix)
    pr = _probs(self.prior_logit(deter), unimix)
    surprise = _kl(q, pr)
    novelty = _entropy(pr)
    unexpected = _entropy(q) + surprise - novelty
    value = self.head('val', deter, stoch)
    # Fear: the fear critic's prediction of pain to come (agent.fear), when
    # the weights carry one; its rise is anticipatory pain.
    fear = (-self.head('cval', deter, stoch)
            if 'cval/head/logits/kernel' in self.p.w else 0.0)
    r_hat = self.head('rew', deter, stoch)
    reward_err = abs(float(reward) - r_hat)
    td = 0.0 if self.prev_value is None else (
        float(reward) + self.gamma ** steps * value - self.prev_value)
    self.prev_value = value
    if self.every > 0 and self.steps % self.every == 0:
      self.empowerment = self._empowerment(deter, stoch)
    self.steps += 1
    z = {k: self.norm[k](v) for k, v in (
        ('novelty', novelty), ('unexpected', unexpected),
        ('reward_err', reward_err), ('empowerment', self.empowerment))}
    # RMS, not SD: a steady disappointment (lying on a bumper) is a TD error
    # with a mean and little spread, and should read clearly negative, not
    # saturate at -1.
    a = max(self.norm['td'].a, 1.0 / (self.norm['td'].n + 1))
    self._td_ms = (1 - a) * getattr(self, '_td_ms', td * td) + a * td * td
    self.norm['td'].n += 1
    td_scale = np.sqrt(self._td_ms + 1e-8)
    control = float(sigmoid(np.float32(z['empowerment'])))
    helpless = self.norm['abs_td'](abs(td)) * (1 - control)
    drive = (self.w[0] * z['novelty'] + self.w[1] * z['unexpected']
             + self.w[2] * z['reward_err'] - self.w[3] * z['empowerment']
             + self.w[4] * helpless + self.pain_w * 3.0 * pain)
    hurt = float(pain) * (1.0 - 0.5 * control)
    valence = float(np.tanh(td / max(td_scale, 1e-6)))
    mood = float(np.tanh(self.mood_norm(value) / 2.0))
    return dict(
        valence=valence, mood=mood,
        face_valence=float(np.clip(0.7 * mood + 0.3 * valence - hurt, -1, 1)),
        pain=float(pain), hurt=hurt, fear=float(fear),
        helpless=float(helpless),
        arousal=float(sigmoid(np.float32(drive / 2.0))),
        # Without pain, so the phone can refresh pain's share every control
        # step (PolicyRunner._fast_affect) between these 16 Hz updates.
        drive=float(drive - self.pain_w * 3.0 * pain), control=control,
        base_valence=float(0.7 * mood + 0.3 * valence),
        dominance=float(np.tanh(z['empowerment'])),
        td=td, value=value, reward_hat=r_hat, surprise=surprise,
        novelty=novelty, unexpected=unexpected, reward_err=reward_err,
        empowerment=self.empowerment)


# -- v2: pleasure, arousal, dominance (docs/robot/EMOTION.md, the target
# design of 2026-11-12) --------------------------------------------------------

def _twohot_nll(probs, bins, x):
  """-log p(x) of a two-hot categorical over `bins` (ascending): the same
  cross-entropy the reward and critic heads train with."""
  x = float(np.clip(x, bins[0], bins[-1]))
  hi = int(np.clip(np.searchsorted(bins, x, side='right'), 1, len(bins) - 1))
  lo = hi - 1
  span = max(float(bins[hi] - bins[lo]), 1e-12)
  w_hi = (x - float(bins[lo])) / span
  logp = np.log(np.asarray(probs, np.float64) + 1e-8)
  return float(-((1 - w_hi) * logp[..., lo] + w_hi * logp[..., hi]))


def _kl_rows(p, q):
  """KL(p_k || q) for each row k of p (all trailing axes summed)."""
  p = np.asarray(p, np.float64)
  q = np.asarray(q, np.float64)
  d = p * (np.log(p + 1e-8) - np.log(q + 1e-8))
  return d.reshape(d.shape[0], -1).sum(-1)


def fade_lambda(hz, fade_s=0.5):
  """Per-step weight of the fading average: a step's weight falls to 1/e
  after fade_s seconds (0.5 s at 65 Hz: 0.970)."""
  return float(np.exp(-1.0 / (float(fade_s) * float(hz))))


class EmotionV2(Emotion):
  """The three axes, each one quantity of the world model and critics:

    outcome  o_t = r_t / S_V - w c_t / S_T       reward minus pain, now
    outlook  O_t = V_t / S_V - w T_t / S_T       value minus threat
    pleasure p_t = o_t + O_{t+1} - O_t (the TD error plus the expectation
             per step; the discount cancels), faded over ~0.5 s:
             p~_t = lam p~_{t-1} + (1 - lam) p_t, lam = exp(-1/(0.5 s hz));
             pleasure = tanh z(p~)
    arousal  e_t = (1/n)(-log p^(s_t)) - log p^(r_t) - log p^(c_t), the
             prediction error of the prior p^ (state, and the reward and cost
             heads on the prior's state); arousal = sigmoid z(e_t), or with
             arousal='unexpected' sigmoid z(e_t - e^_t), e^_t = (1/n) H(p^_s)
             + H(p^_r) + H(p^_c) its expected size
    dominance  32 random wheel commands in [-1, 1]^2, each held `horizon`
             steps through the prior; C = mean_k[(1/n) KL(s_k || <s>) +
             KL(V_k || <V>) + KL(T_k || <T>)] with V, T the critics' two-hot
             distributions at the end state; dominance = tanh z(C)

  S_V, S_T are the return scales the actor divides the advantages by
  (retnorm, cretnorm: max(limit, p95 - p5)), w the actor's pain weight
  (agent.fear.weight); T is the threat critic, `cval`, which the agent trains
  on minus the cost, so T = -cval. n is the number of latent variables.
  z-scores are running over `tau` calls (~10 s). Without a threat critic
  in the weights T = 0 and w = 1; without a cost head the cost terms drop.

  Per control step `tick(reward, cost)` fades the outcome in; every few
  steps `step(...)` adds the outlook's change and updates arousal and the
  z-scores (PolicyRunner.feel)."""

  version = 'v2'

  def __init__(self, policy, hz=65.0, tau=650, arousal='error', horizon=16,
               commands=32, fade_s=0.5, every=0, seed=0):
    assert arousal in ('error', 'unexpected'), arousal
    self.p = policy
    self.set_policy(policy)
    self.hz = float(hz)
    self.lam = fade_lambda(hz, fade_s)
    self.arousal_mode = arousal
    self.horizon, self.commands, self.every = int(horizon), int(commands), int(every)
    self.rng = np.random.default_rng(seed)
    self.norm = {k: Running(tau) for k in ('pleasure', 'arousal', 'dominance')}
    self.faded = 0.0
    self.reset()

  def set_policy(self, policy):
    """New weights (a push); the running statistics stay."""
    w = policy.w
    missing = [k for k in ('dyn/priorlogit/kernel', 'val/head/logits/kernel',
                           'rew/head/logits/kernel') if k not in w]
    if missing:
      raise ValueError(f'weights lack {missing}: export with emotion=True')
    self.p = policy
    meta = policy.meta.get('emotion', {})
    self.gamma = float(meta.get('gamma', 1 - 1 / 130))
    self.has_threat = 'cval/head/logits/kernel' in w
    self.has_cost = 'cst/head/logits/kernel' in w
    self.w_pain = float(meta.get('fear_weight', 1.0)) if self.has_threat else 1.0
    if not self.w_pain:
      self.w_pain = 1.0
    limit = float(meta.get('ret_limit', 1.0))

    def scale(name):
      # Saved as '<name>/lo/value' (ninjax variables); '<name>/lo' also read.
      lo = w.get(f'{name}/lo/value', w.get(f'{name}/lo'))
      hi = w.get(f'{name}/hi/value', w.get(f'{name}/hi'))
      if lo is None or hi is None:
        return 1.0
      return max(limit, float(np.asarray(hi, np.float64)) - float(np.asarray(lo, np.float64)))

    self.S_V, self.S_T = scale('retnorm'), scale('cretnorm')
    self.bins = _twohot_bins(w['val/head/logits/bias'].shape[-1])
    self.rbins = _twohot_bins(w['rew/head/logits/bias'].shape[-1])
    self.cbins = (_twohot_bins(w['cst/head/logits/bias'].shape[-1])
                  if self.has_cost else None)

  def reset(self):
    """A new episode: the outlook starts afresh (pleasure's fading and the
    z-scores carry on)."""
    self.prev_outlook = None
    self.steps = 0
    self.empowerment = getattr(self, 'empowerment', 0.0)

  # -- pieces ------------------------------------------------------------------

  def head_probs(self, name, deter, stoch):
    w = self.p.w
    x = np.concatenate([deter, stoch.reshape(*deter.shape[:-1], -1)], -1)
    i = 0
    while f'{name}/mlp/linear{i}/kernel' in w:
      x = linear(x, w[f'{name}/mlp/linear{i}/kernel'], w[f'{name}/mlp/linear{i}/bias'])
      x = silu(rms_norm(x, w[f'{name}/mlp/norm{i}/scale']))
      i += 1
    logits = linear(x, w[f'{name}/head/logits/kernel'], w[f'{name}/head/logits/bias'])
    return softmax(logits, -1)

  def _mean(self, probs, bins):
    return (np.asarray(probs, np.float64) * bins).sum(-1)

  def value(self, deter, stoch):
    return float(self._mean(self.head_probs('val', deter, stoch), self.bins))

  def threat(self, deter, stoch):
    """T: expected future pain (minus the cval critic, which is trained on
    minus the cost). 0 without a threat critic. Batched over a leading axis."""
    if not self.has_threat:
      return np.zeros(deter.shape[:-1]) if deter.ndim > 1 else 0.0
    t = -self._mean(self.head_probs('cval', deter, stoch), self.bins)
    return t if deter.ndim > 1 else float(t)

  def outcome(self, reward, cost):
    return float(reward) / self.S_V - self.w_pain * float(cost) / self.S_T

  def outlook(self, value, threat):
    return float(value) / self.S_V - self.w_pain * float(threat) / self.S_T

  def tick(self, reward, cost, n=1):
    """One control step's outcome into the fading average (n steps of it)."""
    lam = self.lam ** int(n)
    self.faded = lam * self.faded + (1 - lam) * self.outcome(reward, cost)
    return self.faded

  def pleasure_now(self):
    """tanh z(p~) from the current running statistics, without updating
    them: for the per-step face between step() calls."""
    nm = self.norm['pleasure']
    if nm.n == 0:
      return 0.0
    return float(np.tanh((self.faded - nm.mean) / np.sqrt(nm.var + 1e-8)))

  def mode_stoch(self, probs):
    return np.eye(probs.shape[-1], dtype=F32)[probs.argmax(-1)]

  def rollout(self, deter, stoch, actions, steps):
    """Hold each row of `actions` for `steps` steps through the prior from
    (deter, stoch), taking the likeliest latent each step. Yields after each
    step; the last value is (deter, stoch, prior probs)."""
    K = len(actions)
    d = np.repeat(deter[None], K, 0)
    s = np.repeat(stoch[None], K, 0)
    unimix = self.p.meta['unimix']
    p = None
    for _ in range(int(steps)):
      d = self._core_batch(d, s, actions)
      p = _probs(self.prior_logit(d), unimix)
      s = self.mode_stoch(p)
      yield None
    yield d, s, p

  def threat_after(self, deter, stoch, actions, steps):
    """The threat critic's T after holding each action `steps` steps."""
    for out in self.rollout(deter, stoch, np.asarray(actions, F32), steps):
      pass
    d, s, _ = out
    return np.asarray(self.threat(d, s), np.float64).reshape(-1)

  def empowerment_slices(self, deter, stoch):
    """C_t, one rollout step per yield (None), then the value."""
    actions = self.rng.uniform(-1, 1, (self.commands, 2)).astype(F32)
    for out in self.rollout(deter, stoch, actions, self.horizon):
      if out is None:
        yield None
    d, s, p = out
    yield self.empowerment_value(p, self.head_probs('val', d, s),
                                 self.head_probs('cval', d, s)
                                 if self.has_threat else None)

  def empowerment_value(self, p, v, t=None):
    """mean_k[(1/n) KL(p_k || <p>) + KL(v_k || <v>) + KL(t_k || <t>)]."""
    n = p.shape[-2]
    c = _kl_rows(p, p.mean(0)) / n + _kl_rows(v, v.mean(0))
    if t is not None:
      c = c + _kl_rows(t, t.mean(0))
    return float(c.mean())

  def _empowerment(self, deter, stoch):
    for value in self.empowerment_slices(deter, stoch):
      pass
    return value

  def prediction_error(self, deter, stoch, reward, cost):
    """(e, e^): the prior's error on this step's state, reward and cost,
    and its expected size."""
    pr = _probs(self.prior_logit(deter), self.p.meta['unimix'])
    n = pr.shape[-2]
    s = np.asarray(stoch, np.float64).reshape(pr.shape)
    err = -float((s * np.log(pr + 1e-8)).sum()) / n
    exp = _entropy(pr) / n
    sp = self.mode_stoch(pr)
    pr_r = self.head_probs('rew', deter, sp)
    err += _twohot_nll(pr_r, self.rbins, reward)
    exp += _entropy(pr_r)
    if self.has_cost:
      pr_c = self.head_probs('cst', deter, sp)
      err += _twohot_nll(pr_c, self.cbins, cost)
      exp += _entropy(pr_c)
    return err, exp

  # -- the step ------------------------------------------------------------------

  def step(self, deter, post_logit, stoch, reward, cost=0.0, is_first=False,
           steps=1):
    """Every few control steps, after tick() has faded in each step's
    outcome: `reward`, `cost` are this step's (for arousal). Adds the
    outlook's change since the last call to the faded pleasure."""
    if is_first:
      self.reset()
    value = self.value(deter, stoch)
    threat = self.threat(deter, stoch)
    outlook = self.outlook(value, threat)
    change = 0.0 if self.prev_outlook is None else outlook - self.prev_outlook
    self.prev_outlook = outlook
    # The outlook's change lands in one step of the fading average.
    self.faded += (1 - self.lam) * change
    err, expected = self.prediction_error(deter, stoch, reward, cost)
    if self.every > 0 and self.steps % self.every == 0:
      self.empowerment = self._empowerment(deter, stoch)
    self.steps += 1
    z_p = self.norm['pleasure'](self.faded)
    z_a = self.norm['arousal'](err - expected if self.arousal_mode == 'unexpected'
                               else err)
    z_d = self.norm['dominance'](self.empowerment)
    pleasure = float(np.tanh(z_p))
    return dict(
        pleasure=pleasure, valence=pleasure, face_valence=pleasure,
        mood=pleasure, arousal=float(sigmoid(np.float32(z_a))),
        dominance=float(np.tanh(z_d)), faded=float(self.faded),
        outlook=outlook, change=change, value=value, threat=float(threat),
        fear=float(threat), pain=float(cost), surprise=err, expected=expected,
        unexpected=err - expected, empowerment=float(self.empowerment),
        td=change, version='v2')
