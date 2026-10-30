"""Transition-count exploration (TCE): push the manager toward rarely made start->goal transitions.

Counts (``FromToCounter``, host side, at environment interaction): every real
step t adds the pairs (code of the state at t-k -> code of the state at t) for
k in [k_min, k_max] (default 1..8, one manager goal's reach), weight 1/n each,
so a real step adds 1 in total. Codes are the goal encoder's mode for the
current state, coarsened to ``start_bins`` / ``goal_bins`` regions per block;
the cell layout is ``(start_bins,)*L + (goal_bins,)*L`` row-major, as in
``explore.ImagCounts``. Counts decay with a half-life in env steps (stored
inflated by 2^(t/half_life) and deflated at export, so there is no per-step
pass over the table). Every ``export_every`` env steps the table is decayed,
smeared over neighbouring bins (Gaussian, width ``h`` in classes) and copied
into the agent's ``FromToCounts`` variable.

Use (``FromToCounts.expected_novelty``, in the manager loss): at a decision
state with start region s, the bonus of goal region z is 1/sqrt(n(s, z) + 1),
and the policy's expected novelty E_pi[b] = sum_z pi(z|s) b(s, z) is computed
exactly by contracting the start region's row (goal_bins^L cells) with the
per-block policy probabilities summed into bins. The manager's entropy term
becomes  weight * [entropy + c * E_pi[b]]  with the entropy controller's own
weight: the same amount of randomness, aimed at goal regions not yet reached
from here. With equal counts the extra term has zero gradient direction.
"""
import collections

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
i32 = jnp.int32


def bin_ids(ids, bins, classes):
  return np.minimum(np.asarray(ids, np.int64) * bins // classes, bins - 1)


def smear_kernel(bins, classes, h):
  """(bins, bins), column-normalized Gaussian over bin centres (class units)."""
  if h <= 0:
    return np.eye(bins, dtype=np.float32)
  centers = (np.arange(bins) + 0.5) * (classes / bins) - 0.5
  d = centers[:, None] - centers[None, :]
  k = np.exp(-(d ** 2) / (2.0 * h ** 2))
  return (k / k.sum(0, keepdims=True)).astype(np.float32)


class FromToCounts(nj.Module):
  """Device copy of the from-to table and the exact expected novelty."""

  def __init__(self, start_bins=2, goal_bins=4, classes=8, blocks=8, start_rows=0):
    self.start_bins, self.goal_bins = int(start_bins), int(goal_bins)
    self.classes, self.blocks = int(classes), int(blocks)
    # start_rows > 0: the start side is a hashed world-model state, passed in
    # as a row id directly (see Agent._start_hash), not a goal code to bin.
    self.hashed = int(start_rows) > 0
    self.rows = int(start_rows) if self.hashed else self.start_bins ** self.blocks
    self.cols = self.goal_bins ** self.blocks
    self.tab = nj.Variable(jnp.zeros, (self.rows * self.cols,), f32, name='tab')

  def start_row(self, start_ids):
    if self.hashed:
      return start_ids.astype(i32).reshape(-1)
    q = jnp.minimum(start_ids.astype(i32) * self.start_bins // self.classes,
                    self.start_bins - 1)
    strides = self.start_bins ** jnp.arange(self.blocks - 1, -1, -1)
    return (q * strides).sum(-1)

  def expected_novelty(self, start_ids, probs):
    """``start_ids (N, L)`` ints (or ``(N,)`` row ids when hashed), ``probs (N, L, C)`` -> E_pi[b] (N,) and the
    uniform-policy E[b] (N,) for reference."""
    tab = jax.lax.stop_gradient(self.tab.read())
    row = self.start_row(start_ids)                                  # (N,)
    idx = row[:, None] * self.cols + jnp.arange(self.cols)[None]
    bonus = 1.0 / jnp.sqrt(tab[idx] + 1.0)                           # (N, cols)
    N, L, C = probs.shape
    # Class c -> bin min(c * bins // C, bins - 1), as in the counter; bins need
    # not divide C (3 bins over 8 classes: 3 / 3 / 2).
    assign = jax.nn.one_hot(jnp.minimum(jnp.arange(C) * self.goal_bins // C,
                                        self.goal_bins - 1), self.goal_bins, dtype=f32)
    pb = jnp.einsum('nlc,cb->nlb', probs.astype(f32), assign)

    def contract(x, p):
      # Row-major: the first remaining goal axis is block l.
      for l in range(L):
        x = x.reshape(N, self.goal_bins, -1)
        x = (x * p[:, l, :, None]).sum(1)
      return x.reshape(N)

    # Uniform over CLASSES (an untrained policy), binned like the policy.
    uni = jnp.broadcast_to(assign.mean(0), pb.shape)
    self.last_row_mass = tab[idx].sum(-1)                            # evidence per state
    return contract(bonus, pb), contract(bonus, uni)


class FromToCounter:
  """Host-side counter fed from the environment driver (see module docstring)."""

  def __init__(self, start_bins=2, goal_bins=4, classes=8, blocks=8,
               k_min=1, k_max=8, half_life=5e5, h=0.5, export_every=10000,
               pair_weight=1.0, prop_weight=1.0, half_life_frac=0.0, half_life_min=2.5e4,
               start_rows=0):
    self.sb, self.gb, self.C, self.L = start_bins, goal_bins, classes, blocks
    # start_rows > 0: ``add`` gets the start cell as a hashed row id.
    self.start_rows = int(start_rows)
    self.ks = list(range(int(k_min), int(k_max) + 1))
    # Each experienced (from, to) pair is a full look at that combination: if a
    # reward were there it would have shown, so every pair counts 1 (a real
    # step then adds up to k_max - k_min + 1). ``pair_weight=0`` gives 1/n.
    self.w = float(pair_weight) if pair_weight > 0 else 1.0 / len(self.ks)
    # Manager proposals also count (start code at a real decision -> the goal
    # it proposed): a goal proposed but never reached gains counts each time
    # and fades, instead of staying maximally novel forever. 0 disables.
    self.prop_w = float(prop_weight)
    self.half_life = float(half_life)
    # Growing half-life (half_life_frac > 0): h(t) = max(frac * t, half_life_min)
    # replaces the fixed one. Memory then scales with training time: a count
    # made at step s weighs (s/t)^(ln2/frac) at step t, total mass grows ~t and
    # the bonus differences fade by themselves.
    self.hl_frac = float(half_life_frac)
    self.hl_min = float(half_life_min)
    self.h = float(h)
    self.export_every = int(export_every)
    self.cols = self.gb ** self.L
    rows = self.start_rows if self.start_rows > 0 else self.sb ** self.L
    self.tab = np.zeros(rows * self.cols, np.float32)
    self.hist = collections.defaultdict(lambda: collections.deque(maxlen=max(self.ks)))
    self.steps = 0
    self._sstr = self.sb ** np.arange(self.L - 1, -1, -1)
    self._gstr = self.gb ** np.arange(self.L - 1, -1, -1)
    self._smear = jax.jit(self._smear_fn)

  def add(self, code, is_first, worker, proposal=None, start=None):
    """One real step of one env: ``code`` (L,) class ids of the current state;
    ``proposal`` (L,) the goal the manager proposed at this step, if it made a
    decision here; ``start`` the current state's hashed row (hashed mode)."""
    hist = self.hist[worker]
    if is_first:
      hist.clear()
    code = np.asarray(code, np.int64)
    g = int((bin_ids(code, self.gb, self.C) * self._gstr).sum())
    s_now = (int(start) if self.start_rows > 0 else
             int((bin_ids(code, self.sb, self.C) * self._sstr).sum()))
    inflate = self._inflate()
    if proposal is not None and self.prop_w > 0:
      gp = int((bin_ids(np.asarray(proposal, np.int64), self.gb, self.C) * self._gstr).sum())
      self.tab[s_now * self.cols + gp] += self.prop_w * inflate
    for k in self.ks:
      if len(hist) >= k:
        self.tab[hist[-k] * self.cols + g] += self.w * inflate
    hist.append(s_now)
    self.steps += 1

  def _inflate(self):
    """exp(integral of ln2 / h(u) du) up to now: 2^(t/half_life) for a fixed
    half-life; half_life <= 0 turns decay off (counts only grow, so bonuses
    shrink as 1/sqrt(n) and the push anneals by itself). With a growing
    half-life h(t) = max(frac * t, h_min), constant h_min until t0 = h_min / frac,
    then log2 = (1 + ln(t / t0)) / frac."""
    if self.hl_frac > 0:
      t0 = self.hl_min / self.hl_frac
      if self.steps <= t0:
        return 2.0 ** (self.steps / self.hl_min)
      return 2.0 ** ((1.0 + np.log(self.steps / t0)) / self.hl_frac)
    return 1.0 if self.half_life <= 0 else 2.0 ** (self.steps / self.half_life)

  def due(self):
    return self.steps > 0 and self.steps % self.export_every == 0

  def _smear_fn(self, tab):
    if self.start_rows > 0:   # hashed starts have no neighbours: goal axes only
      shape = (self.start_rows,) + (self.gb,) * self.L
      axes = list(enumerate(shape))[1:]
    else:
      shape = (self.sb,) * self.L + (self.gb,) * self.L
      axes = list(enumerate(shape))
    x = tab.reshape(shape)
    for axis, bins in axes:
      k = jnp.asarray(smear_kernel(bins, self.C, self.h))
      x = jnp.moveaxis(jnp.tensordot(k, x, axes=([1], [axis])), 0, axis)
    return x.reshape(-1)

  def export(self):
    """Decayed and smeared table (device array) in true pick units."""
    deflate = 1.0 / self._inflate()
    # Explicit device_put: implicit host->device copies are disallowed in runs.
    return self._smear(jax.device_put(self.tab * np.float32(deflate)))
