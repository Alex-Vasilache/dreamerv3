"""Blockwise vector-quantization bottleneck with a circular 1D SOM topology.

Replaces Director's per-block categorical + straight-through one-hot with a
per-block codebook lookup, following ``motivation.tex`` Modification A. There
are ``L`` independent codebooks, one per goal-code block, each holding ``C``
embeddings of dimension ``D`` arranged on a *circular* 1D grid:

    z_q^(l) = argmin_{e in E^(l)} || z_e^(l) - e ||^2

The manager's action space is unchanged -- it still emits one ``(L, C)``
one-hot per decision -- so everything downstream of the code (masking, reuse,
struct diagnostics, REINFORCE) keeps working. Only the *decoder input* changes,
from the flattened one-hot to the flattened gathered embeddings.

Topology. ``motivation.tex`` specifies neighbors ``N(k) = {(k-1) mod C,
(k+1) mod C}`` on a ring, which sidesteps the boundary handling in the official
SOM-VAE code: that implementation is a non-periodic 2D grid whose edge cells
substitute a ZERO vector for the missing neighbor (``tf.where(k1_not_top, ...,
tf.zeros(...))`` in ``som_vae/somvae_model.py``), so border codes are actively
pulled toward the origin. A ring has no border. The official code also includes
the winner itself in the neighbor set; here that is off by default
(``include_self``) because the winner is already pulled by the codebook term.

Gradient routing. The two quadratic terms are kept split with explicit
stop-gradients (standard VQ-VAE / LipVQ-VAE):

    codebook  = || sg[z_e] - z_q ||^2     (moves the codebook toward the encoder)
    commit    = || z_e - sg[z_q] ||^2     (moves the encoder toward the codebook)

With equal weights this is *exactly* the gradient of the official SOM-VAE's
un-split ``loss_commit = mean((z_e - z_q)^2)``, which is what
``motivation.tex``'s single ``alpha`` term denotes -- see
``test_goal_vq.py::TestVQLosses::test_split_equals_unsplit_gradients``.
"""
import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient


def _agg(x, dims, impl):
  """Reduce the trailing ``dims`` axes by sum (``motivation.tex``) or mean."""
  axes = tuple(range(-dims, 0))
  if impl == 'sum':
    return x.sum(axes)
  if impl == 'mean':
    return x.mean(axes)
  raise NotImplementedError(impl)


def vq_losses(z_e, z_q, neighbors=None, agg='sum', nb_mask=None,
              commit_joint=False):
  """The VQ / SOM quadratic terms, reduced to the leading batch dims.

  Args:
    z_e: ``(..., L, D)`` continuous encoder output.
    z_q: ``(..., L, D)`` quantized output (gathered embeddings).
    neighbors: ``(..., L, N, D)`` SOM neighbors of the winning entries, or
      ``None`` to skip the SOM term.
    nb_mask: ``(..., L, N)`` 1 where the neighbor exists. Required by the open
      ``line`` topology, whose end entries have a neighbor on one side only.
    agg: ``sum`` over ``(L, D)`` per ``motivation.tex``'s ``sum_l ||.||^2``, or
      ``mean`` as in both reference implementations.
    commit_joint: emit the original SOM-VAE's SINGLE commitment term
      ``alpha ||z_e - z_q||^2``, with no stop-gradient on either side, instead
      of the equal-weighted ``codebook + commit`` split. The two are gradient-
      identical -- the split sends ``2(z_e - z_q)`` to ``z_e`` from ``commit``
      and ``2(z_q - z_e)`` to the codebook from ``codebook``, which is exactly
      what the joint term's two partial derivatives are -- so this changes the
      logged value (the split double-counts it) but not the optimization. It
      exists so an arm can be literally the equation in the source paper.
      Reported under ``commit``, with ``codebook`` held at zero.

  Returns a dict of arrays shaped like the leading batch dims of ``z_e``.
  """
  assert z_e.shape == z_q.shape, (z_e.shape, z_q.shape)
  if commit_joint:
    out = {
        'codebook': jnp.zeros(z_e.shape[:-2], f32),
        'commit': _agg(jnp.square(z_e - z_q), 2, agg),
    }
  else:
    out = {
        'codebook': _agg(jnp.square(sg(z_e) - z_q), 2, agg),
        'commit': _agg(jnp.square(z_e - sg(z_q)), 2, agg),
    }
  if neighbors is None:
    out['som'] = jnp.zeros(z_e.shape[:-2], f32)
  else:
    assert neighbors.shape[:-2] == z_e.shape[:-1], (
        neighbors.shape, z_e.shape)
    assert neighbors.shape[-1] == z_e.shape[-1], (neighbors.shape, z_e.shape)
    err = jnp.square(neighbors - sg(z_e)[..., None, :])
    if nb_mask is not None:
      assert nb_mask.shape == neighbors.shape[:-1], (
          nb_mask.shape, neighbors.shape)
      err = err * nb_mask[..., None]
    out['som'] = _agg(err, 3, agg)
  return out


class BlockCodebook(nj.Module):
  """``L`` independent circular 1D codebooks of ``C`` entries in ``R^D``.

  Fields:
    stddev: init scale. ``0.05`` matches the official SOM-VAE
      (``truncated_normal_initializer(stddev=0.05)``).
    include_self: include the winning entry in ``neighbors`` (official SOM-VAE
      does; ``motivation.tex``'s ``N(k)`` does not).
  """

  stddev: float = 0.05
  include_self: bool = False
  # Neighborhood radius. 1 reproduces SOM-VAE's immediate neighborhood; larger
  # values pull more entries toward each encoding (measured: strictly worse).
  radius: int = 1
  # 'ring' wraps, so every entry has 2*radius neighbors and there is no
  # boundary. 'line' is an open path: the end entries have neighbors on one
  # side only, and the missing ones are MASKED OUT of the loss rather than
  # replaced by a zero vector, which is what the official SOM-VAE code does for
  # its 2D grid's edge cells and which pulls border entries toward the origin.
  # A line spans a wider range of code distances than a ring: for C=8 an evenly
  # spaced path has an adjacent-to-arbitrary distance ratio of 0.333 against a
  # ring's 0.533, because a ring's wrap keeps its farthest pairs closer.
  topology: str = 'ring'      # ring | line

  def __init__(self, blocks, classes, dim):
    assert blocks >= 1 and classes >= 1 and dim >= 1, (blocks, classes, dim)
    self.blocks = int(blocks)
    self.classes = int(classes)
    self.dim = int(dim)

  def table(self):
    """The codebook, ``(L, C, D)``."""
    return self.value('table', self._init)

  def _init(self):
    shape = (self.blocks, self.classes, self.dim)
    return self.stddev * jax.random.truncated_normal(
        nj.seed(), -2.0, 2.0, shape, f32)

  def distances(self, z_e):
    """Squared L2 distance to every entry: ``(..., L, D) -> (..., L, C)``."""
    self._check(z_e)
    diff = z_e[..., :, None, :].astype(f32) - self.table()
    return jnp.square(diff).sum(-1)

  def encode(self, z_e):
    """Nearest-entry indices, ``(..., L, D) -> (..., L)`` int32."""
    return jnp.argmin(self.distances(z_e), -1).astype(i32)

  def onehot(self, ids):
    """``(..., L) -> (..., L, C)`` float one-hot, the manager's code format."""
    return jax.nn.one_hot(ids, self.classes, dtype=f32)

  def lookup(self, onehot):
    """Gather embeddings from a one-hot code: ``(..., L, C) -> (..., L, D)``.

    Differentiable in both arguments, so a soft (non-one-hot) code produces the
    corresponding convex combination of entries.
    """
    assert onehot.shape[-2:] == (self.blocks, self.classes), onehot.shape
    return jnp.einsum('...lc,lcd->...ld', onehot.astype(f32), self.table())

  def quantize(self, z_e):
    """Full forward pass of the bottleneck.

    Returns ``ids`` ``(..., L)``, ``onehot`` ``(..., L, C)``, ``z_q``
    ``(..., L, D)`` and the squared distances ``(..., L, C)``. ``z_q`` carries
    gradient to the codebook but NOT to ``z_e``: ``argmin`` is not
    differentiable, so the encoder is trained through the commitment term (and,
    for the SOM arm, through the second reconstruction from ``z_e``) rather
    than through a straight-through estimator.
    """
    dist = self.distances(z_e)
    ids = jnp.argmin(dist, -1).astype(i32)
    onehot = self.onehot(ids)
    return {'ids': ids, 'onehot': onehot, 'z_q': self.lookup(onehot),
            'dist': dist}

  def neighbors(self, ids):
    """Circular neighbors of the winners: ``(..., L) -> (..., L, N, D)``.

    ``N`` is ``2 * radius`` (``k +- 1 .. k +- radius`` mod ``C``), plus one with
    ``include_self``, where
    the winner comes first to match the official SOM-VAE's stacking order.
    Note ``C <= 2`` degenerates (both neighbors coincide, and for ``C == 1``
    they coincide with the winner); the project uses ``C == 8``.
    """
    C = self.classes
    assert 1 <= self.radius <= C // 2, (self.radius, C)
    assert self.topology in ('ring', 'line'), self.topology
    picks = [ids] if self.include_self else []
    for r in range(1, self.radius + 1):
      if self.topology == 'ring':
        picks += [(ids - r) % C, (ids + r) % C]
      else:
        picks += [jnp.clip(ids - r, 0, C - 1), jnp.clip(ids + r, 0, C - 1)]
    return jnp.stack([self.lookup(self.onehot(p)) for p in picks], -2)

  def neighbor_mask(self, ids):
    """``(..., L) -> (..., L, N)`` 1 where that neighbor exists.

    All ones on a ring. On a line the positions past either end do not exist;
    ``neighbors`` clamps their index, so this mask is what keeps the resulting
    duplicate out of the loss.
    """
    C = self.classes
    cols = [jnp.ones(ids.shape, f32)] if self.include_self else []
    for r in range(1, self.radius + 1):
      if self.topology == 'ring':
        cols += [jnp.ones(ids.shape, f32), jnp.ones(ids.shape, f32)]
      else:
        cols += [(ids - r >= 0).astype(f32), (ids + r <= C - 1).astype(f32)]
    return jnp.stack(cols, -1)

  def soft_probs(self, z_e, temp=1.0):
    """``softmax(-d^2 / temp)`` over entries: the differentiable stand-in for
    the categorical probabilities the Director encoder used to emit, so the
    geometry diagnostics (``goal/struct_corr``) keep a soft code to measure."""
    assert temp > 0.0, temp
    return jax.nn.softmax(-self.distances(z_e) / temp, -1)

  def metrics(self, ids):
    """Codebook-usage diagnostics; VQ's main failure mode is dead entries.

    Gradient-free by construction (``ids`` come from ``argmin``), but the
    reductions are still floored so no infinite derivative can enter the graph.
    """
    counts = self.onehot(ids).reshape((-1, self.blocks, self.classes)).sum(0)
    probs = counts / jnp.maximum(counts.sum(-1, keepdims=True), 1.0)
    # log is floored: a class with zero count gives probs == 0 exactly, and
    # both log(0) and its derivative are infinite.
    entropy = -(probs * jnp.log(jnp.maximum(probs, 1e-12))).sum(-1)
    return {
        'used_frac': (counts > 0).astype(f32).mean(),
        'perplexity': jnp.exp(entropy).mean(),
        'max_prob': probs.max(-1).mean(),
    }

  def _check(self, z_e):
    assert z_e.shape[-2:] == (self.blocks, self.dim), (
        z_e.shape, (self.blocks, self.dim))
