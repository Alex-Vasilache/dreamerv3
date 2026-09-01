"""Epsilon-greedy index jumps for the manager, on an ordered codebook.

At the entropy the regularizer holds, the manager's Gaussian samples land close
to its mode: two draws in the same state differ by ~5.7 of the 56 index steps
the code admits, and decode to goals ~0.84 similar. That concentration is what
makes credit transfer to neighbouring classes, so it cannot simply be widened
-- raising the entropy target buys reach only by flattening the distribution
until the ordering stops being used (at target 0.8 the mode/neighbour ratio
falls from 2.9 to 1.27), and a uniform mixture is absorbed by the entropy
controller shrinking sigma to compensate.

This buys reach on a separate axis instead. With probability ``eps`` the
sampled code is replaced by a deliberately distant one:

  1. take the mode ``c`` of the manager's current distribution;
  2. a block at class ``c_l`` can shift by at most ``max(c_l, C-1-c_l)``, so the
     largest reachable index distance is ``dmax(c) = sum_l max(c_l, C-1-c_l)``.
     Note this is a property of the CODE: over uniform codes it averages 44,
     not 56, and only a code with every block at an extreme reaches 56;
  3. draw a target distance with ``p(d) = 2d / [dmax (dmax + 1)]`` for
     ``d = 1..dmax``, which normalizes exactly and favours large jumps
     (``E[d] = (2/3)(dmax + 1/2)``, about 30 in practice);
  4. realize exactly that distance by spreading ``d`` unit steps over the
     blocks within each block's own capacity, then choosing a direction per
     block among those the bounds allow.

Only worth doing because the index is ordered. On Director's codebook a large
index gap does not imply a distant goal -- its measured curve is flat and even
rises at the far end -- so jumping along that axis would buy nothing.

APPLIED IN THE ENVIRONMENT ROLLOUT ONLY. The manager is trained by REINFORCE on
the log-probability of its own samples, so drawing from a different
distribution during training biases the gradient. Importance weighting is the
textbook fix but is empty here: the policy assigns a jumped code a probability
of order 1e-61, which underflows float32, so such a sample would contribute
nothing anyway. Restricting the jump to the actor changes which states reach
the replay buffer -- the coverage we want -- and leaves imagination on-policy.
"""
import jax
import ninjax as nj
import numpy as np
import jax.numpy as jnp

i32 = jnp.int32
f32 = jnp.float32


def capacities(ids, classes):
  """Per-block room to move, and the total reachable index distance.

  Returns ``(down, up, cap, dmax)`` where ``down``/``up`` are how far the block
  can move toward class 0 / class C-1, ``cap = max(down, up)`` is the largest
  magnitude it admits, and ``dmax`` sums that over the blocks.
  """
  ids = i32(ids)
  down = ids
  up = (classes - 1) - ids
  cap = jnp.maximum(down, up)
  return down, up, cap, cap.sum(-1)


def sample_distance(key, dmax):
  """Draw ``d`` with ``p(d) = 2d / [dmax (dmax + 1)]``, ``d = 1..dmax``.

  Inverse-CDF: ``P(D <= d) = d(d+1) / [dmax(dmax+1)]``, so inverting
  ``d(d+1) = u * dmax(dmax+1)`` gives ``d = ceil((-1 + sqrt(1 + 4uM)) / 2)``.
  """
  m = f32(dmax) * (f32(dmax) + 1.0)
  u = jax.random.uniform(key, jnp.shape(dmax), f32)
  d = jnp.ceil((-1.0 + jnp.sqrt(1.0 + 4.0 * u * m)) / 2.0)
  return jnp.clip(i32(d), 1, jnp.maximum(i32(dmax), 1))


def spread(key, cap, d, classes):
  """Split ``d`` into per-block magnitudes with ``|delta_l| <= cap_l``.

  Each block contributes ``cap_l`` unit slots; choosing ``d`` of the
  ``sum_l cap_l`` slots without replacement enforces both ``sum |delta_l| = d``
  and the per-block bound at once, and every valid assignment is reachable.
  Implemented by ranking uniform keys, so it stays a fixed-shape operation.
  """
  steps = jnp.arange(classes - 1)
  valid = steps < cap[..., None]                      # (..., L, C-1)
  noise = jax.random.uniform(key, valid.shape, f32)
  noise = jnp.where(valid, noise, 2.0)                # invalid slots sort last
  flat = noise.reshape(*noise.shape[:-2], -1)
  rank = jnp.argsort(jnp.argsort(flat, -1), -1).reshape(valid.shape)
  return (rank < d[..., None, None]).astype(i32).sum(-1) * valid.any(-1)


def jump(key, ids, classes):
  """A code at a random large index distance from ``ids``."""
  down, up, cap, dmax = capacities(ids, classes)
  k_d, k_s, k_dir = jax.random.split(key, 3)
  d = sample_distance(k_d, dmax)
  mag = spread(k_s, cap, d, classes)
  # Direction: take whichever side has room. cap is the max of the two, so at
  # least one always does; when both do, pick at random.
  can_up = mag <= up
  can_down = mag <= down
  coin = jax.random.uniform(k_dir, mag.shape, f32) < 0.5
  go_up = jnp.where(can_up & can_down, coin, can_up)
  delta = jnp.where(go_up, mag, -mag)
  return jnp.clip(ids + delta, 0, classes - 1), d


def apply_jump(key, onehot, eps, classes):
  """Replace the sampled code by a jump from its mode, with probability ``eps``.

  ``onehot`` is ``(..., L, C)``. The mode is taken per block; the decision to
  jump is made once per code, not once per block, so a jump moves the whole
  code coherently -- which is the point, and is what a per-class mixture like
  ``unimix`` cannot do.
  """
  if not eps:
    return onehot
  k_take, k_jump = jax.random.split(key, 2)
  ids = jnp.argmax(onehot, -1)
  jumped, _ = jump(k_jump, ids, classes)
  hot = jax.nn.one_hot(jumped, classes, dtype=onehot.dtype)
  take = jax.random.uniform(k_take, ids.shape[:-1], f32) < eps
  return jnp.where(take[..., None, None], hot, onehot)


def ucb_pick(key, onehots, bonuses, c):
  """Choose among policy samples with probability proportional to exp(c * bonus).

  The candidates are ALREADY draws from the manager's policy, so they already
  carry its preference. Scoring them by ``log pi + c * bonus`` and taking the
  argmax counts the policy twice and turns the rule into a best-of-K likelihood
  filter -- measured on real manager distributions that shifts the expected
  log-probability of the chosen goal from -9.46 to -6.70, discarding about a
  quarter of the entropy the 50% target exists to maintain, and a directly
  sampled goal is the best of its own 8 only 12% of the time. Sharpening is the
  opposite of what an exploration mechanism should do.

  Weighting the policy's own samples by ``exp(c * bonus)`` instead is
  self-normalised importance sampling toward ``pi(z) * exp(c * bonus(z))``:
  at ``c = 0`` it reproduces an exact policy sample (measured E[log pi] -9.45
  against an unbiased -9.46), and as ``c`` grows it moves toward rarely proposed
  goals, lowering the log-probability rather than raising it.

  Implemented with the Gumbel-max trick so it is one argmax, no cumulative sums.
  ``onehots`` is ``(K, ..., L, C)`` and ``bonuses`` is ``(K, ...)``.
  """
  logits = c * bonuses
  g = jax.random.gumbel(key, logits.shape, dtype=logits.dtype)
  best = jnp.argmax(logits + g, 0)
  idx = best[None, ..., None, None]
  return jnp.take_along_axis(onehots, idx, 0)[0], best


# --- Count-based novelty over the joint goal code ---------------------------
#
# The manager's exploration bonus in Director is the goal autoencoder's own
# reconstruction error. That couples the signal to the autoencoder's quality --
# a better autoencoder explores less by construction -- and it fades over
# training exactly when the manager is most able to use it. This is an
# alternative signal: reward the manager for reaching goal codes it has rarely
# reached lately.
#
# Counting does not need the encoder to DESCRIBE an unfamiliar state, only to
# LABEL it consistently, which a poorly-reconstructing encoder still does. The
# problem is instead that C^L = 8^8 = 16.7M codes against 500k decisions per run
# leaves every exact count at 0 or 1. Two things fix that:
#
#   Smearing. Each visit marks a neighbourhood, not a cell, so one visit speaks
#   for its surroundings. Implemented as a coarse grid: quantizing each block to
#   ``bins`` levels puts 8^8 codes into bins^8 cells.
#
#   Joint, not per-block. A code is a point in 8 dimensions and the blocks are
#   correlated across visits -- shuffling each block independently (which leaves
#   every per-block marginal identical) spreads the measured codes over 3-4x
#   more regions. So the cell index is the JOINT tuple, never per-block counts.
#
# Two resolutions are kept, because they answer different questions. The FINE
# grid asks "have I been in this exact spot"; the COARSE grid asks "have I been
# anywhere around here at all", and its saturation is what makes it a
# reachability test rather than a second novelty measure. A frontier cell is
# empty on the fine grid but occupied on the coarse one: new, yet adjacent to
# somewhere the agent has been. Plain novelty instead points at the FARTHEST
# region, which is typically unreachable.
#
# Grid sizes are measured, not guessed (experiments/manager_rao): writing alpha
# for the exponent in occupancy ~ n^alpha, alpha is 0.39/0.61/0.74/0.83/0.87 at
# 2/3/4/5/6 bins per block and the 90th/10th percentile count ratio falls
# 46.9 -> 2.6 over the same range. alpha -> 1 puts every visit in its own cell
# and pins the reward at its maximum (no signal); a saturating grid stops
# discriminating (also no signal). 4 bins keeps a 5.3x count spread while
# reducing the space 256-fold; 2 bins saturates, which is what the coarse test
# wants.
class CodeCounts(nj.Module):
  """Decaying visit counts over the joint goal code, at two resolutions.

  Both tables decay so that the memory forgets at exactly the rate replay
  forgets. With ``weight = 1/train_ratio`` per inserted state and
  ``decay = exp(-env_steps_per_train / replay_size)`` per training step, an env
  step contributes exactly 1.0 in total however often replay happens to resample
  it, and the table settles at a total mass of ``replay_size``. A count then
  reads directly as "env steps spent near this code in the last replay window",
  which is the same slice of experience the world model trains on.
  """

  fine: int = 4
  coarse: int = 2
  # Grid for the SELECTION table, independent of the visit grid above. Coarser
  # on purpose: with 4 bins there are 65,536 cells against ~500k decisions in a
  # 4M run, so the grid can never fill, a never-selected candidate at bonus 1.0
  # always exists, the candidate spread never shrinks and the tilt SHARPENS over
  # training instead of fading (measured: override rate 67%->75% between 100k and
  # 300k). 3 bins gives 6,561 cells, ~76 selections each, so counts saturate, the
  # spread compresses and the bonus genuinely anneals.
  select: int = 4
  # Kernel width for the selection counts, in CLASS units (a block has C=8
  # classes). 0 recovers the hard grid exactly. A visit deposits mass that
  # decays with index distance instead of landing in one cell, which is only
  # meaningful because the SOM line makes index distance track goal distance.
  select_h: float = 1.0
  classes: int = 8
  blocks: int = 8
  # A cell counts as occupied once at least this much mass sits in it. Mass is
  # in ENV-STEP units (see the weight/decay note above), so 1.0 means "at least
  # one env step was spent near this code inside the replay window". It must not
  # be 0: decayed cells approach zero without reaching it, so a >0 test would
  # count every cell ever touched and creep upward forever.
  occ_thresh: float = 1.0

  def __init__(self, decay, weight, frontier=False, eps=1.0):
    self.decay = float(decay)
    self.weight = float(weight)
    self.frontier = bool(frontier)
    self.eps = float(eps)
    self.n_fine = int(self.fine ** self.blocks)
    self.n_coarse = int(self.coarse ** self.blocks)
    self.n_select = int(self.select ** self.blocks)
    self.fine_tab = nj.Variable(
        jnp.zeros, (self.n_fine,), f32, name='fine_tab')
    self.coarse_tab = nj.Variable(
        jnp.zeros, (self.n_coarse,), f32, name='coarse_tab')
    # Goals the manager PROPOSES, as opposed to states it reaches. Deliberately
    # CUMULATIVE (no decay) while the two above are a sliding window, because
    # the two serve opposite purposes:
    #
    #   visits  (decaying)   "where have we been lately" -- a coverage measure,
    #                        which must track the replay window to stay
    #                        comparable with what the world model trains on;
    #   selects (cumulative) the UCB bonus, which must FADE. Textbook UCB anneals
    #                        precisely because its counts never decay: everything
    #                        gets tried, every count grows, every bonus shrinks
    #                        as 1/sqrt(n). On a sliding window the counts settle
    #                        at a steady level instead, so the tilt would keep
    #                        its full strength to the end of training and push
    #                        the policy around with what is by then noise.
    #
    # Counting proposals rather than arrivals is also what makes the bonus
    # self-limiting: a goal the worker can never reach still gets counted, so the
    # manager stops fixating on it. A visit-based bonus would leave an
    # unreachable goal maximally attractive forever.
    self.select_tab = nj.Variable(
        jnp.zeros, (self.n_select,), f32, name='select_tab')

  def _key(self, ids, bins):
    """Flat index of the JOINT cell -- never a per-block reduction."""
    q = jnp.minimum((ids.astype(i32) * bins) // self.classes, bins - 1)
    place = (bins ** jnp.arange(self.blocks)).astype(i32)
    return (q * place).sum(-1)

  def lookup(self, ids):
    """Counts at ``ids`` on both grids. ``ids`` is ``(..., L)`` int."""
    shape = ids.shape[:-1]
    flat = ids.reshape((-1, self.blocks))
    n_f = self.fine_tab.read()[self._key(flat, self.fine)]
    n_c = self.coarse_tab.read()[self._key(flat, self.coarse)]
    return n_f.reshape(shape), n_c.reshape(shape)

  def novelty(self, ids):
    """Exploration reward in (0, 1]: ``1/sqrt(n+eps)``, optionally frontier-gated."""
    n_f, n_c = self.lookup(ids)
    nov = 1.0 / jnp.sqrt(n_f + self.eps)
    if self.frontier:
      # Reachability gate: scale down regions with no coarse-grid history, so
      # the reward points just outside the explored set instead of at the
      # farthest, typically unreachable, corner of the code space.
      nov = nov * (1.0 - jnp.exp(-n_c / jnp.maximum(self.eps, 1e-6)))
    return nov

  def _deposit(self, ids):
    """Smeared mass for each proposed code, as a flat ``select**blocks`` vector.

    A Gaussian over the joint code factorises into one factor per block, so a
    smeared deposit is the outer product of L short vectors and the whole batch
    is one contraction -- cost is the table's size, not a neighbour search.
    Each block's profile is normalised, so one decision deposits exactly 1.0 in
    total and the mass-equals-decisions accounting is unchanged.
    """
    flat = f32(ids.reshape((-1, self.blocks)))
    # Cell j holds classes [j*C/b, (j+1)*C/b), so its midpoint in class units is
    # (j + 0.5)*C/b - 0.5. Dropping the -0.5 shifts every centre half a class and
    # misassigns mass: at C=8, b=4 it put class 2 (which quantises to cell 1)
    # equally into cells 0 and 1, and 3 of 8 classes peaked in the wrong cell.
    centers = ((jnp.arange(self.select, dtype=f32) + 0.5)
               * (self.classes / self.select) - 0.5)
    d = flat[:, :, None] - centers[None, None, :]        # (N, L, select)
    k = jnp.exp(-(d ** 2) / (2.0 * self.select_h ** 2))
    k = k / jnp.maximum(k.sum(-1, keepdims=True), 1e-12)
    out = k[:, 0, :]
    for l in range(1, self.blocks):
      out = (out[:, :, None] * k[:, l, None, :]).reshape(out.shape[0], -1)
    return out

  def update_selected(self, ids, weight):
    """Add the goals the manager proposed. One entry per DECISION, never decayed.

    ``weight`` should be ``1/train_ratio`` so each real decision contributes 1.0
    however often replay resamples it, matching the accounting used for visits.
    """
    if self.select_h <= 0.0:
      flat = ids.reshape((-1, self.blocks))
      w = jnp.full((flat.shape[0],), float(weight), f32)
      cur = self.select_tab.read()
      self.select_tab.write(cur.at[self._key(flat, self.select)].add(w))
      return
    add = self._deposit(ids).sum(0) * float(weight)
    self.select_tab.write(self.select_tab.read() + add)

  def ucb_bonus(self, ids):
    """``1/sqrt(selections+1)`` -- the optimism term, on PROPOSALS not arrivals."""
    shape = ids.shape[:-1]
    flat = ids.reshape((-1, self.blocks))
    n = self.select_tab.read()[self._key(flat, self.select)]
    return (1.0 / jnp.sqrt(n + 1.0)).reshape(shape)

  def update(self, ids):
    """Decay both tables, then add this batch. Real (replay) states only."""
    flat = ids.reshape((-1, self.blocks))
    w = jnp.full((flat.shape[0],), self.weight, f32)
    for tab, bins in ((self.fine_tab, self.fine), (self.coarse_tab, self.coarse)):
      cur = tab.read() * self.decay
      tab.write(cur.at[self._key(flat, bins)].add(w))
    # Materialize the selection table even when UCB is off. ``metrics()`` reads
    # it unconditionally, and a ninjax Variable cannot be created inside a
    # non-creating pure call -- so without this the REWARD-only arm would raise
    # the first time it logged. Copying 65k floats is free next to the model.
    self.select_tab.write(self.select_tab.read())

  def metrics(self):
    """Cheap scalars: two reductions over 65k and 256 floats, no per-step cost."""
    f, c = self.fine_tab.read(), self.coarse_tab.read()
    occ_f = (f > self.occ_thresh).astype(f32)
    occ_c = (c > self.occ_thresh).astype(f32)
    # Participation ratio: how many cells the mass is EFFECTIVELY spread over,
    # which unlike raw occupancy is not dominated by cells holding a trace.
    pr = lambda t: jnp.square(t.sum()) / jnp.maximum(jnp.square(t).sum(), 1e-8)
    # Frontier: empty on the fine grid but inside an occupied coarse cell. Every
    # coarse cell contains (fine/coarse)^L fine cells, so this needs no scatter.
    per = float((self.fine // self.coarse) ** self.blocks)
    reach = occ_c.sum() * per
    s = self.select_tab.read()
    # Deliberately few. Earlier versions emitted fifteen of these and most were
    # never read: the coarse grid saturates by ~1M so its occupancy and the
    # frontier built on it are constants, and occupied-fraction duplicates
    # occupied. What is kept is one accounting check, one that says whether
    # smearing is live, and one that tracks the fade.
    pr = lambda t_: jnp.square(t_.sum()) / jnp.maximum(jnp.square(t_).sum(), 1e-8)
    occ_s = (s > self.occ_thresh).astype(f32)
    return {
        # must track (env steps)/K -- this check caught two real accounting bugs
        'select_mass': s.sum(),
        # ~1 means a hard deposit, thousands means smearing is working
        'select_eff_cells': pr(s),
        # the annealing: should fall as proposals accumulate
        'select_bonus_mean': (occ_s * (1.0 / jnp.sqrt(s + 1.0))).sum() /
                             jnp.maximum(occ_s.sum(), 1.0),
        # visit-side coverage, the one number worth keeping from that table
        'fine_occupied': (self.fine_tab.read() > self.occ_thresh).astype(f32).sum(),
        'total_mass': self.fine_tab.read().sum(),
    }

  def coverage_images(self):
    """Report-only pictures of the two tables (log scale, uint8).

    The flat key is lexicographic over blocks, so reshaping the fine table to
    (256, 256) splits it into the low four blocks along one axis and the high
    four along the other -- a space-filling layout of the whole code space.
    Built only in ``report``, never in the train step.
    """
    out = {}
    for name, tab, side in (
        ('coarse', self.coarse_tab.read(), int(np.sqrt(self.n_coarse))),
        ('fine', self.fine_tab.read(), int(np.sqrt(self.n_fine)))):
      v = jnp.log1p(tab).reshape((side, side))
      v = v / jnp.maximum(v.max(), 1e-8)
      out[f'code_coverage_{name}'] = (v * 255).astype(jnp.uint8)[..., None]
    return out
