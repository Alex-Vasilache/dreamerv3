"""From-to novelty for the manager: directed entropy toward untried goal regions.

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

  def __init__(self, start_bins=2, goal_bins=4, classes=8, blocks=8):
    self.start_bins, self.goal_bins = int(start_bins), int(goal_bins)
    self.classes, self.blocks = int(classes), int(blocks)
    self.rows = self.start_bins ** self.blocks
    self.cols = self.goal_bins ** self.blocks
    self.tab = nj.Variable(jnp.zeros, (self.rows * self.cols,), f32, name='tab')

  def start_row(self, start_ids):
    q = jnp.minimum(start_ids.astype(i32) * self.start_bins // self.classes,
                    self.start_bins - 1)
    strides = self.start_bins ** jnp.arange(self.blocks - 1, -1, -1)
    return (q * strides).sum(-1)

  def expected_novelty(self, start_ids, probs):
    """``start_ids (N, L)`` ints, ``probs (N, L, C)`` -> E_pi[b] (N,) and the
    uniform-policy E[b] (N,) for reference."""
    tab = jax.lax.stop_gradient(self.tab.read())
    row = self.start_row(start_ids)                                  # (N,)
    idx = row[:, None] * self.cols + jnp.arange(self.cols)[None]
    bonus = 1.0 / jnp.sqrt(tab[idx] + 1.0)                           # (N, cols)
    N, L, C = probs.shape
    pb = probs.astype(f32).reshape(N, L, self.goal_bins, C // self.goal_bins).sum(-1)

    def contract(x, p):
      # Row-major: the first remaining goal axis is block l.
      for l in range(L):
        x = x.reshape(N, self.goal_bins, -1)
        x = (x * p[:, l, :, None]).sum(1)
      return x.reshape(N)

    uni = jnp.full_like(pb, 1.0 / self.goal_bins)
    return contract(bonus, pb), contract(bonus, uni)


class FromToCounter:
  """Host-side counter fed from the environment driver (see module docstring)."""

  def __init__(self, start_bins=2, goal_bins=4, classes=8, blocks=8,
               k_min=1, k_max=8, half_life=5e5, h=0.5, export_every=10000):
    self.sb, self.gb, self.C, self.L = start_bins, goal_bins, classes, blocks
    self.ks = list(range(int(k_min), int(k_max) + 1))
    self.w = 1.0 / len(self.ks)
    self.half_life = float(half_life)
    self.h = float(h)
    self.export_every = int(export_every)
    self.cols = self.gb ** self.L
    self.tab = np.zeros((self.sb ** self.L) * self.cols, np.float32)
    self.hist = collections.defaultdict(lambda: collections.deque(maxlen=max(self.ks)))
    self.steps = 0
    self._sstr = self.sb ** np.arange(self.L - 1, -1, -1)
    self._gstr = self.gb ** np.arange(self.L - 1, -1, -1)
    self._smear = jax.jit(self._smear_fn)

  def add(self, code, is_first, worker):
    """One real step of one env: ``code`` (L,) class ids of the current state."""
    hist = self.hist[worker]
    if is_first:
      hist.clear()
    code = np.asarray(code, np.int64)
    g = int((bin_ids(code, self.gb, self.C) * self._gstr).sum())
    inflate = 2.0 ** (self.steps / self.half_life)
    for k in self.ks:
      if len(hist) >= k:
        s = int((bin_ids(hist[-k], self.sb, self.C) * self._sstr).sum())
        self.tab[s * self.cols + g] += self.w * inflate
    hist.append(code)
    self.steps += 1

  def due(self):
    return self.steps > 0 and self.steps % self.export_every == 0

  def _smear_fn(self, tab):
    shape = (self.sb,) * self.L + (self.gb,) * self.L
    x = tab.reshape(shape)
    for axis, bins in enumerate(shape):
      k = jnp.asarray(smear_kernel(bins, self.C, self.h))
      x = jnp.moveaxis(jnp.tensordot(k, x, axes=([1], [axis])), 0, axis)
    return x.reshape(-1)

  def export(self):
    """Decayed and smeared table (device array) in true pick units."""
    deflate = 2.0 ** (-self.steps / self.half_life)
    return self._smear(jnp.asarray(self.tab * np.float32(deflate)))
