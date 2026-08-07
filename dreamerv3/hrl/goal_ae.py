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
import numpy as np

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
  strict_bound: bool = False
  dtype: str = 'default'
  temp: float = 1.0

  def __init__(self, codebook, blocks, dim):
    self.codebook = codebook
    self.blocks = int(blocks)
    self.dim = int(dim)
    self.mlp = LipMLP(
        self.layers, self.units, act=self.act, norm=self.norm, bias=self.bias,
        winit=self.winit, binit=self.binit, lip=self.lip, per_row=self.per_row,
        cinit=self.cinit, clamp=self.clamp, strict_bound=self.strict_bound,
        dtype=self.dtype, name='mlp')
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

  def code_from_latent(self, z_e):
    """Wrap a precomputed ``z_e`` as the discrete code, so a caller that needs
    both (the training loss does) pays for only one encoder pass."""
    return _wrap_code(VQCode(self.codebook.distances(z_e), self.temp))

  def __call__(self, x, bdims):
    return self.code_from_latent(self.latent(x, bdims))

  def bounds(self):
    """Per-layer Lipschitz bounds from the most recent forward pass."""
    if not self.lip:
      return []
    out = self.mlp.bounds()
    if self.lip_out:
      out = out + [self.out.bound()]
    return out

  def scales(self):
    """Realized per-unit rescale factors; 1.0 means the constraint is inert."""
    if not self.lip:
      return []
    out = self.mlp.scales()
    if self.lip_out:
      out = out + [self.out.scale()]
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
  strict_bound: bool = False
  dtype: str = 'default'
  stddev: float = 0.05
  include_self: bool = False
  radius: int = 1
  topology: str = 'ring'

  def __init__(self, shape, blocks, classes, dim):
    self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
    self.blocks = int(blocks)
    self.classes = int(classes)
    self.dim = int(dim)
    self.codebook = BlockCodebook(
        blocks, classes, dim, stddev=self.stddev,
        include_self=self.include_self, radius=self.radius,
        topology=self.topology, name='codebook')
    self.mlp = LipMLP(
        self.layers, self.units, act=self.act, norm=self.norm, bias=self.bias,
        winit=self.winit, binit=self.binit, lip=self.lip, per_row=self.per_row,
        cinit=self.cinit, clamp=self.clamp, strict_bound=self.strict_bound,
        dtype=self.dtype, name='mlp')
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

  def scales(self):
    """Realized per-unit rescale factors; 1.0 means the constraint is inert."""
    if not self.lip:
      return []
    out = self.mlp.scales()
    if self.lip_out:
      out = out + [self.out.scale()]
    return out


def vq_goal_loss(
    enc, dec, deter, bdims, som=False, ste=None, agg='sum',
    codebook_scale=1.0, commit_scale=1.0, som_scale=0.9, lip_scale=0.03,
    lip_impl='logprod', rec_scale=1.0, z_e=None, commit_joint=False,
    commit_adapt=None):
  """Assemble the full goal-autoencoder objective for the VQ arms.

  Returns ``(loss, metrics)`` where ``loss`` has the leading batch dims of
  ``deter``. Mirrors ``motivation.tex``'s

      || s - s_q ||^2 + || s - s_e ||^2
      + alpha sum_l || z_e - z_q ||^2
      + beta  sum_l sum_{e in N(z_q)} || e - sg[z_e] ||^2
      + gamma L_Lipschitz

  with the single ``alpha`` term expressed as the equal-weighted
  ``codebook + commit`` split (identical gradients; see ``hrl.vq``), or as the
  literal single term when ``commit_joint=True``. The second
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
  # ``z_e`` may be supplied by a caller that already ran the encoder (the agent
  # needs the code for its entropy / geometry diagnostics); recomputing it here
  # would double the encoder cost every train step.
  z_e = enc.latent(deter, bdims) if z_e is None else z_e
  quant = dec.codebook.quantize(z_e)

  z_rec = z_e + sg(quant['z_q'] - z_e) if ste else quant['z_q']
  rec_q = dec.from_latent(z_rec, bdims).loss(target)
  rec = rec_q
  if som:
    rec_e = dec.from_latent(z_e, bdims).loss(target)
    rec = rec + rec_e

  neighbors = dec.codebook.neighbors(quant['ids']) if som else None
  nb_mask = dec.codebook.neighbor_mask(quant['ids']) if som else None
  terms = vq_losses(z_e, quant['z_q'], neighbors, agg, nb_mask=nb_mask,
                    commit_joint=commit_joint)

  # Optional controller on the commitment weight. ``commit_adapt`` is called
  # with the batch's codebook perplexity and returns the weight to use in place
  # of ``commit_scale``.
  #
  # Perplexity rather than ``used_frac`` because ``used_frac`` is pinned at
  # 0.99 to 1.00 in every healthy run and only moves once the codebook is
  # already dead, while perplexity varies smoothly across the healthy range
  # (6.3 to 7.6 of a possible 8) and falls to ~1.3 on collapse. Perplexity
  # cannot reach 8, which would need all entries used equally, so the target
  # must be set inside the reachable range.
  #
  # The controller raises the weight while perplexity is ABOVE target, so it
  # settles on the LARGEST commitment weight that still keeps the codebook
  # alive. Driving it the other way would be ill-posed: a weight of zero
  # trivially maximizes perplexity by untying the encoder from the codebook.
  usage = dec.codebook.metrics(quant['ids'])
  if commit_adapt is not None:
    commit_scale = sg(commit_adapt(sg(usage['perplexity'])))

  loss = (
      rec_scale * rec
      + codebook_scale * terms['codebook']
      + commit_scale * terms['commit']
      + som_scale * terms['som'])

  bounds = enc.bounds() + dec.bounds()
  scales = enc.scales() + dec.scales()
  penalty = lip_penalty(bounds, lip_impl)
  if bounds:
    loss = loss + lip_scale * penalty

  # Diagnostics only. Every one of these is stop-gradiented, and the sqrt /
  # norm reductions are floored away from zero: d(sqrt(x))/dx is infinite at
  # x = 0, and a collapsed codebook makes an exact-zero quantization distance
  # entirely reachable (perplexity was measured at ~1.5 early in training). A
  # single inf/NaN anywhere in the traced graph is enough to poison the shared
  # backward pass for every optimizer group, so metrics must not be able to
  # introduce one.
  tiny = 1e-12
  metrics = {
      'vq/rec_q': rec_q.mean(),
      'vq/codebook': terms['codebook'].mean(),
      'vq/commit': terms['commit'].mean(),
      'vq/som': terms['som'].mean(),
      'vq/z_e_norm': jnp.sqrt(jnp.square(z_e).sum(-1) + tiny).mean(),
      'vq/z_q_norm': jnp.sqrt(jnp.square(quant['z_q']).sum(-1) + tiny).mean(),
      'vq/quant_err': jnp.sqrt(
          jnp.maximum(quant['dist'].min(-1), 0.0) + tiny).mean(),
      'vq/lip_penalty': penalty,
      'vq/lip_bound_max': (
          jnp.stack(bounds).max() if bounds else jnp.zeros((), f32)),
      # Fraction of output units whose weights are actually being rescaled.
      # 0.0 means the constraint is inert and the arm is equivalent to plain
      # VQ; this is what says whether the Lipschitz arm is doing anything.
      'vq/lip_active_frac': (
          jnp.concatenate([(s < 1.0).astype(f32).reshape((-1,))
                           for s in scales]).mean()
          if scales else jnp.zeros((), f32)),
  }
  if som:
    metrics['vq/rec_e'] = rec_e.mean()
  metrics.update({f'vq/{k}': v for k, v in usage.items()})
  if commit_adapt is not None:
    metrics['vq/commit_scale'] = commit_scale
  metrics = {k: sg(v) for k, v in metrics.items()}
  return loss, metrics


class ManagerRingHead(nj.Module):
  """Manager policy whose logits come from distances to the goal codebook.

  The Director manager emits ``L`` categoricals over ``C`` arbitrary labels, so
  a REINFORCE update that favours entry ``k`` raises only ``k``. Once the SOM
  term has ordered the codebook, entry ``k+1`` decodes to a nearby goal with a
  likely similar return, and that generalization is unavailable to a policy
  whose classes carry no metric. Here the trunk emits a continuous
  ``v^(l) in R^d`` per block and the logits are

      logit_c = -s * || v^(l) - e^(l)_c ||^2 ,

  so entries adjacent on the ring receive similar probability by construction,
  and moving ``v`` toward one entry raises its neighbours as a side effect.
  The output is the same ``outs.OneHot`` the Director head produced, so
  sampling, log-probabilities, entropy and the straight-through code are
  unchanged downstream.

  Two details matter. The codebook is read under a stop-gradient: it is shaped
  by the autoencoder objective alone, and letting the manager's REINFORCE
  gradient move it would confound the two. And the distances are divided by
  their per-block standard deviation before ``s`` is applied, which makes ``s``
  the standard deviation of the logits and therefore a temperature in units the
  codebook's own scale cannot shift. ``scale_init`` is set so the head starts at
  or BELOW the manager's entropy target. It has to: ``s`` receives gradient only
  from REINFORCE and barely moves in practice (5.000 to 5.008 over 100k steps in
  e483), and the entropy adapter is one-directional -- it raises entropy toward
  the target and shrinks to its floor when entropy is above it. Starting below
  the target is therefore the only mis-specification the adapter can correct. At
  ``scale_init = 8`` the normalized entropy is 0.16 on a random codebook, 0.47 on
  an evenly spaced line (the arrangement the SOM term drives toward) and 0.27 to
  0.44 across e483's trained blocks, all at or under the 0.5 target.
  """

  layers: int = 3
  units: int = 1024
  act: str = 'silu'
  norm: str = 'rms'
  bias: bool = True
  winit: str = 'trunc_normal_in'
  binit: str = 'zeros'
  outscale: float = 0.1
  unimix: float = 0.0
  scale_init: float = 8.0   # std-normalized; starts at/below the entropy target
  per_block_scale: bool = True

  def __init__(self, codebook, blocks, dim):
    self.codebook = codebook
    self.blocks = int(blocks)
    self.dim = int(dim)
    self.mlp = nets.MLP(
        self.layers, self.units, act=self.act, norm=self.norm, bias=self.bias,
        winit=self.winit, binit=self.binit, name='mlp')
    self.out = nets.Linear(
        (self.blocks, self.dim), bias=self.bias, winit=self.winit,
        binit=self.binit, outscale=self.outscale, name='out')

  def __call__(self, x, bdims):
    bshape = jax.tree.leaves(x)[0].shape[:bdims]
    v = self.out(self.mlp(x.reshape((*bshape, -1)))).astype(f32)
    table = sg(self.codebook.table())                      # (L, C, D)
    dist = jnp.square(v[..., :, None, :] - table).sum(-1)  # (..., L, C)
    # Normalize by the per-block mean distance before scaling. Absolute squared
    # distances depend on the codebook's and latent's scale, which drift during
    # training: measured at initialization they are ~0.05, so an unscaled logit
    # spread of ~0.05 gives a policy at 1.000 normalized entropy -- uniform, and
    # unable to commit to a goal. Dividing by the mean makes the logits
    # dimensionless, so `scale` has the same meaning at any codebook scale and
    # at any point in training. The mean is stop-gradiented: it sets the
    # temperature, it is not something the policy should optimize.
    # Normalize by the per-block STANDARD DEVIATION of the distances, not their
    # mean. Softmax is shift-invariant, so what sets the entropy is the SPREAD
    # of the logits, and dividing by the std makes that spread equal to `scale`
    # by construction. Dividing by the mean does not: the spread then also
    # depends on where the trunk's point v sits relative to the codebook, since
    # a v near the centroid is about equally far from every entry. Measured on
    # e483's trained codebook, mean-normalization gives 0.72 to 0.92 normalized
    # entropy at a fixed scale as v moves from 0.5 to 2 codebook radii out,
    # against 0.68 to 0.70 for std-normalization, and it puts the 0.5 entropy
    # target at scale 21.4 rather than 4.5. e483 ran with scale 5 under
    # mean-normalization and sat at 0.763 with the scale parameter frozen
    # (5.000 to 5.008 over 100k steps), because the entropy adapter is
    # one-directional -- it raises entropy toward the target and cannot lower
    # it -- so nothing corrected the mis-specification.
    #
    # The floor at 1e-3 of the mean keeps a collapsed codebook finite: with all
    # entries equal the distances are equal, and equal logits are a uniform
    # policy rather than a division by zero.
    d_mean = dist.mean(-1, keepdims=True)
    d_std = dist.std(-1, keepdims=True)
    dist = dist / (sg(jnp.maximum(d_std, 1e-3 * d_mean)) + 1e-8)
    code = outs.OneHot(-self._scale() * dist, self.unimix)
    # The manager's entropy adapter skips, SILENTLY, any head that does not
    # advertise its entropy range:
    #
    #     if not hasattr(inner, 'minent') or not hasattr(inner, 'maxent'):
    #       continue                              # losses.imag_loss_mgr
    #
    # ``MLPHead.onehot`` sets them (embodied/jax/heads.py); this head did not,
    # so it was dropped from the regularizer entirely. e478 ran 1.6M steps with
    # ``mgr_ent_loss`` identically 0.0 and the skill entropy drifting to 16.3 of
    # a maximum 16.64 nats -- a uniform manager, and no error anywhere. Both are
    # TOTALS over the L blocks, matching ``MLPHead.onehot``'s ``outer *
    # log(classes)``; ``imag_loss_mgr`` divides by L to normalize per block.
    code.minent = 0.0
    code.maxent = float(self.blocks * np.log(self.codebook.classes))
    return _wrap_code_onehot(code)

  def _scale(self):
    shape = (self.blocks, 1) if self.per_block_scale else ()
    raw = self.value('logit_scale', self._make_scale, shape)
    return jax.nn.softplus(raw.astype(f32))

  def _make_scale(self, shape):
    from .lipschitz import softplus_inv
    return jnp.broadcast_to(
        softplus_inv(jnp.asarray(self.scale_init, f32)), shape).astype(f32)


def _wrap_code_onehot(code):
  """Same ``Agg`` wrapping the Director ``onehot`` head applied."""
  return outs.Agg(code, 1, jnp.sum)
