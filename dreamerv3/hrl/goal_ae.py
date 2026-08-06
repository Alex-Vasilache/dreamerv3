"""Lipschitz-constrained SOM-VAE goal autoencoder (``motivation.tex``).

Drop-in replacements for Director's ``goal_enc`` / ``goal_dec`` pair that swap
the categorical + straight-through bottleneck for a blockwise vector-quantizer
(``hrl.vq``), optionally with a circular SOM neighborhood loss and optionally
with Lipschitz-constrained linear layers (``hrl.lipschitz``). The four ablation
arms of ``motivation.tex`` are the two booleans:

    ============  ===  ===  ============================================
    arm           som  lip  loss terms beyond reconstruction from z_q
    ============  ===  ===  ============================================
    baseline       -    -   (Director head; not this module at all)
    vq             F    F   codebook + commit
    som            T    F   codebook + commit + neighbors + recon from z_e
    lipvq          F    T   codebook + commit + L_Lipschitz
    som+lipvq      T    T   all of the above
    ============  ===  ===  ============================================

``vq`` is not one of the four arms in the write-up but falls out for free and
is worth running: it separates "quantization instead of categorical sampling"
from the SOM and Lipschitz components, so a difference between the baseline and
the som/lipvq arms can be attributed.

Interface compatibility. The manager still emits an ``(L, C)`` one-hot per
decision, so masking, reuse, the struct diagnostics and REINFORCE are all
untouched -- only the decoder's *input* changes, from the flattened one-hot to
the flattened gathered embeddings. ``GoalVQEncoder.__call__`` returns a
``VQCode``, which implements the same ``pred/sample/entropy/logp/kl`` protocol
as the ``outs.OneHot`` the Director head returned (including ``.dist.logits``,
used by the geometry diagnostics), and ``GoalVQDecoder.__call__`` returns the
same ``outs.Agg(outs.MSE(...), 1, sum)`` the ``mse`` head returned.

Codebook ownership. The codebook lives under ``goal_dec/codebook/`` because it
is needed at *policy* time (decoding the manager's code) and that keeps
``Agent.policy_keys``' existing ``^(...|goal_dec)/`` regex correct without
change. The encoder holds a reference to the same module instance.
"""
import jax
import jax.numpy as jnp
import ninjax as nj

import embodied.jax.nets as nets
import embodied.jax.outs as outs

from .lipschitz import LipLinear, LipMLP, lip_penalty
from .vq import BlockCodebook, vq_losses

f32 = jnp.float32
sg = jax.lax.stop_gradient


class VQCode(outs.Output):
  """``outs.OneHot``-compatible view of a vector-quantized encoding.

  Differences from ``outs.OneHot``, both deliberate:

    * ``sample`` is ``pred``. Quantization is a deterministic nearest-neighbor
      lookup, not a draw from a posterior; Director's stochastic categorical
      sample has no analogue here.
    * ``pred`` returns a plain hard one-hot with NO straight-through term.
      ``motivation.tex`` specifies dual reconstruction precisely so the encoder
      is trained without the straight-through estimator, and a residual
      ``probs - sg(probs)`` would silently reintroduce it.

  ``logits`` are the negated squared distances (scaled by ``temp``), so
  ``softmax(logits)`` is the natural soft code and ``argmax(logits)`` is
  exactly the quantizer's ``argmin`` over distances.
  """

  def __init__(self, dist_sq, temp=1.0):
    assert temp > 0.0, temp
    self.dist_sq = dist_sq
    self.classes = dist_sq.shape[-1]
    self.dist = outs.Categorical(-dist_sq.astype(f32) / temp)

  def pred(self):
    return jax.nn.one_hot(self.dist.pred(), self.classes, dtype=f32)

  def sample(self, seed, shape=()):
    assert shape == (), (
        'VQ encoding is deterministic; batched sampling is not meaningful.')
    return self.pred()

  def logp(self, event):
    return (jax.nn.log_softmax(self.dist.logits, -1) * event).sum(-1)

  def entropy(self):
    return self.dist.entropy()

  def kl(self, other):
    return self.dist.kl(other.dist)


def _wrap_code(code):
  """Match the Director head's ``Agg`` wrapping so ``_head_inner`` still works."""
  return outs.Agg(code, 1, jnp.sum)


def _wrap_recon(pred):
  """Match the ``mse`` head: sum the squared error over the deter axis."""
  return outs.Agg(outs.MSE(pred), 1, jnp.sum)


class GoalVQEncoder(nj.Module):
  """``deter -> z_e (L, D) -> nearest-entry code``.

  ``latent`` returns the continuous pre-quantization output (needed for the
  commitment/SOM terms and the second reconstruction); ``__call__`` returns the
  discrete ``VQCode``, matching the old ``goal_enc`` head's signature.
  """

  layers: int = 3
  units: int = 1024
  act: str = 'silu'
  norm: str = 'none'
  bias: bool = True
  winit: str = 'trunc_normal_in'
  binit: str = 'zeros'
  outscale: float = 0.1
  lip: bool = False
  lip_out: bool = True
  per_row: bool = True
  cinit: float = -1.0
  clamp: bool = True
  temp: float = 1.0

  def __init__(self, codebook, blocks, dim):
    self.codebook = codebook
    self.blocks = int(blocks)
    self.dim = int(dim)
    self.mlp = LipMLP(
        self.layers, self.units, act=self.act, norm=self.norm, bias=self.bias,
        winit=self.winit, binit=self.binit, lip=self.lip, per_row=self.per_row,
        cinit=self.cinit, clamp=self.clamp, name='mlp')
    # The output layer is constrained too when ``lip`` is on: LipVQ-VAE's Fig. 3
    # and its released code disagree about whether the last projection is
    # regularized, and leaving it free would make the composed bound vacuous --
    # an unconstrained final layer can rescale the latent arbitrarily.
    if self.lip and self.lip_out:
      self.out = LipLinear(
          (self.blocks, self.dim), bias=self.bias, winit=self.winit,
          binit=self.binit, outscale=self.outscale, per_row=self.per_row,
          cinit=self.cinit, clamp=self.clamp, name='out')
    else:
      self.out = nets.Linear(
          (self.blocks, self.dim), bias=self.bias, winit=self.winit,
          binit=self.binit, outscale=self.outscale, name='out')

  def latent(self, x, bdims):
    """``(..., deter) -> (..., L, D)`` continuous encoder output ``z_e``."""
    bshape = x.shape[:bdims]
    x = x.reshape((*bshape, -1))
    return self.out(self.mlp(x)).astype(f32)

  def __call__(self, x, bdims):
    return _wrap_code(VQCode(self.codebook.distances(
        self.latent(x, bdims)), self.temp))

  def bounds(self):
    """Per-layer Lipschitz bounds from the most recent forward pass."""
    if not self.lip:
      return []
    out = self.mlp.bounds()
    if self.lip_out:
      out = out + [self.out.bound()]
    return out


class GoalVQDecoder(nj.Module):
  """``code -> embeddings -> deter``. Owns the codebook (see module docstring)."""

  layers: int = 3
  units: int = 1024
  act: str = 'silu'
  norm: str = 'none'
  bias: bool = True
  winit: str = 'trunc_normal_in'
  binit: str = 'zeros'
  outscale: float = 0.1
  lip: bool = False
  lip_out: bool = True
  per_row: bool = True
  cinit: float = -1.0
  clamp: bool = True
  stddev: float = 0.05
  include_self: bool = False

  def __init__(self, shape, blocks, classes, dim):
    self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
    self.blocks = int(blocks)
    self.classes = int(classes)
    self.dim = int(dim)
    self.codebook = BlockCodebook(
        blocks, classes, dim, stddev=self.stddev,
        include_self=self.include_self, name='codebook')
    self.mlp = LipMLP(
        self.layers, self.units, act=self.act, norm=self.norm, bias=self.bias,
        winit=self.winit, binit=self.binit, lip=self.lip, per_row=self.per_row,
        cinit=self.cinit, clamp=self.clamp, name='mlp')
    if self.lip and self.lip_out:
      self.out = LipLinear(
          self.shape, bias=self.bias, winit=self.winit, binit=self.binit,
          outscale=self.outscale, per_row=self.per_row, cinit=self.cinit,
          clamp=self.clamp, name='out')
    else:
      self.out = nets.Linear(
          self.shape, bias=self.bias, winit=self.winit, binit=self.binit,
          outscale=self.outscale, name='out')

  def from_latent(self, z, bdims):
    """Decode from continuous embeddings ``(..., L, D)`` -- the ``s_hat_e`` path."""
    assert z.shape[-2:] == (self.blocks, self.dim), z.shape
    bshape = z.shape[:bdims]
    x = z.reshape((*bshape, -1))
    return _wrap_recon(self.out(self.mlp(x)).astype(f32))

  def __call__(self, code, bdims):
    """Decode from a one-hot code ``(..., L, C)`` -- the ``s_hat_q`` path."""
    return self.from_latent(self.codebook.lookup(code), bdims)

  def bounds(self):
    if not self.lip:
      return []
    out = self.mlp.bounds()
    if self.lip_out:
      out = out + [self.out.bound()]
    return out


def vq_goal_loss(
    enc, dec, deter, bdims, som=False, ste=None, agg='sum',
    codebook_scale=1.0, commit_scale=1.0, som_scale=0.9, lip_scale=1e-6,
    lip_impl='prod', rec_scale=1.0):
  """Assemble the full goal-autoencoder objective for the VQ arms.

  Returns ``(loss, metrics)`` where ``loss`` has the leading batch dims of
  ``deter``. Mirrors ``motivation.tex``'s

      || s - s_q ||^2 + || s - s_e ||^2
      + alpha sum_l || z_e - z_q ||^2
      + beta  sum_l sum_{e in N(z_q)} || e - sg[z_e] ||^2
      + gamma L_Lipschitz

  with the single ``alpha`` term expressed as the equal-weighted
  ``codebook + commit`` split (identical gradients; see ``hrl.vq``). The second
  reconstruction and the neighborhood term are the SOM arm only; the Lipschitz
  penalty is nonzero only when the networks were built with ``lip=True``.

  ``ste`` selects how the reconstruction error reaches the encoder, and
  defaults to ``not som``:

    * SOM arm (``ste=False``): via the second reconstruction from ``z_e``, as
      ``motivation.tex`` and the official SOM-VAE specify. No straight-through
      estimator anywhere.
    * VQ / LipVQ arms (``ste=True``): via the straight-through estimator
      ``z_e + sg[z_q - z_e]``, i.e. standard VQ-VAE (van den Oord et al.),
      which the LipVQ-VAE paper says it follows ("We follow the training
      procedure in [25]"). Its *released* code omits both the estimator and a
      second reconstruction, which leaves the encoder trained by the
      commitment term alone -- measurably barely trained; see
      ``test_goal_ae.py::TestLearning::test_encoder_needs_a_reconstruction_path``.
      Pass ``ste=False, som=False`` to reproduce that reference exactly.

  Under the straight-through path the codebook receives no reconstruction
  gradient (the ``sg`` blocks it) and is trained by the codebook term, exactly
  as in VQ-VAE.
  """
  ste = (not som) if ste is None else bool(ste)
  target = sg(deter.astype(f32))
  z_e = enc.latent(deter, bdims)
  quant = dec.codebook.quantize(z_e)

  z_rec = z_e + sg(quant['z_q'] - z_e) if ste else quant['z_q']
  rec_q = dec.from_latent(z_rec, bdims).loss(target)
  rec = rec_q
  if som:
    rec_e = dec.from_latent(z_e, bdims).loss(target)
    rec = rec + rec_e

  neighbors = dec.codebook.neighbors(quant['ids']) if som else None
  terms = vq_losses(z_e, quant['z_q'], neighbors, agg)
  loss = (
      rec_scale * rec
      + codebook_scale * terms['codebook']
      + commit_scale * terms['commit']
      + som_scale * terms['som'])

  bounds = enc.bounds() + dec.bounds()
  penalty = lip_penalty(bounds, lip_impl)
  if bounds:
    loss = loss + lip_scale * penalty

  metrics = {
      'vq/rec_q': rec_q.mean(),
      'vq/codebook': terms['codebook'].mean(),
      'vq/commit': terms['commit'].mean(),
      'vq/som': terms['som'].mean(),
      'vq/z_e_norm': jnp.linalg.norm(z_e, axis=-1).mean(),
      'vq/z_q_norm': jnp.linalg.norm(quant['z_q'], axis=-1).mean(),
      'vq/quant_err': jnp.sqrt(quant['dist'].min(-1)).mean(),
      'vq/lip_penalty': penalty,
      'vq/lip_bound_max': (
          jnp.stack(bounds).max() if bounds else jnp.zeros((), f32)),
  }
  if som:
    metrics['vq/rec_e'] = rec_e.mean()
  metrics.update({f'vq/{k}': v for k, v in dec.codebook.metrics(
      quant['ids']).items()})
  return loss, metrics
