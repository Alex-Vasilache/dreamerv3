"""Lipschitz-constrained linear layers (LipVQ-VAE, Vuong et al. 2025).

Implements the row-wise weight normalization of LipVQ-VAE Eq. (4), which is
itself the ``Lipschitz MLP`` of Liu et al. (SIGGRAPH 2022) that the paper cites:
for a layer with weight matrix ``W`` and a trainable bound ``c``,

    W_i <- W_i * min(1, softplus(c_i) / sum_j |W_ij|)

so the layer's inf-norm operator norm ``max_i sum_j |W_ij|`` is bounded by
``softplus(c_i)``. With 1-Lipschitz activations, the network's inf-norm
Lipschitz constant is bounded by the product of the per-layer bounds, which is
the quantity the ``L_Lipschitz`` penalty shrinks.

Two details where the LipVQ-VAE paper and its released code disagree with the
project's ``motivation.tex`` write-up; both are exposed as options here:

  * ``motivation.tex`` writes the normalization as an unconditional rescale
    ``W_i / sum_j |W_ij| * softplus(c_i)``. The paper's text ("If the absolute
    row sum is already smaller than softplus(c_l), no rescaling is applied")
    and the released code (``torch.minimum(1.0, softplus(c)/absrowsum)`` in
    ``robomimic/models/vq_vae/backbone_lfqvae_v5.py``) both clamp at 1. We
    follow the paper/code; ``clamp=False`` restores the unconditional form.
  * The penalty is a PRODUCT over layers in the paper (and in Liu et al.,
    where it equals the network's Lipschitz bound) but a SUM in
    ``motivation.tex``. ``lip_penalty`` supports both.

Indexing note: ``embodied.jax.nets.Linear`` stores its kernel as ``(in, out)``
and computes ``x @ kernel``, i.e. the LipVQ "row" ``W_i`` (torch layout
``(out, in)``) is a *column* of our kernel. The absolute row sum is therefore a
reduction over axis 0, not axis 1.
"""
import math
from typing import Callable

import jax
import jax.numpy as jnp
import ninjax as nj

import embodied.jax.nets as nets

f32 = jnp.float32


def softplus_inv(y):
  """Inverse of ``softplus``: ``log(exp(y) - 1)``, numerically stable.

  For ``y > 20`` softplus is the identity to well within f32 precision, and
  ``expm1(y)`` would overflow, so the identity branch is used there.
  """
  y = jnp.asarray(y, f32)
  safe = jnp.minimum(y, 20.0)
  return jnp.where(y > 20.0, y, jnp.log(jnp.expm1(safe) + 1e-30))


def abs_row_sum(kernel):
  """``sum_j |W_ij|`` per output unit, for an ``(in, out)`` kernel -> ``(out,)``."""
  return jnp.abs(kernel.astype(f32)).sum(0)


def lip_scale(kernel, bound, clamp=True, eps=1e-12):
  """Per-output-unit rescale factor enforcing ``sum_j |W_ij| <= bound``.

  Args:
    kernel: ``(in, out)`` weight matrix.
    bound: ``softplus(c)``; scalar or ``(out,)``.
    clamp: keep the layer unchanged where it already satisfies the bound
      (paper/reference behavior). ``False`` rescales unconditionally, matching
      the ``motivation.tex`` form of the equation.
  """
  ratio = jnp.asarray(bound, f32) / (abs_row_sum(kernel) + eps)
  ratio = jnp.broadcast_to(ratio, (kernel.shape[-1],))
  return jnp.minimum(1.0, ratio) if clamp else ratio


def lip_normalize(kernel, c, clamp=True, eps=1e-12):
  """Row-wise normalized kernel (LipVQ-VAE Eq. 4). ``c`` is the raw bound."""
  bound = jax.nn.softplus(jnp.asarray(c, f32))
  return kernel.astype(f32) * lip_scale(kernel, bound, clamp, eps)[None, :]


def lip_penalty(bounds, impl='logprod'):
  """``L_Lipschitz`` from the per-layer bounds.

  ``prod`` is the form written in the LipVQ-VAE paper and in Liu et al., and
  equals the network's inf-norm Lipschitz bound; ``sum`` is the
  ``motivation.tex`` form. ``logprod`` (default) is ``log`` of the product, and
  is what this project uses.

  The reason is that ``prod`` does not survive a change of depth or width. With
  the 8 constrained layers here, each bound initialized to a 1024-unit layer's
  absolute row sums (~27.8), the product is ~6e10, so LipVQ-VAE's gamma=1e-6 --
  chosen for a single constrained layer, where the product is ~10 -- makes the
  penalty 99.96% of the goal-autoencoder loss. ``logprod`` is monotone in the
  same quantity, so it shrinks exactly what ``prod`` shrinks, but its gradient
  with respect to a layer's bound is ``1/bound`` rather than the product of the
  other layers' bounds, which keeps a single gamma usable across architectures.

  An empty layer list yields ``0`` for every impl (an off switch, not a product
  identity of 1, so that ``gamma * penalty`` vanishes when Lipschitz is off).
  """
  bounds = [jnp.asarray(b, f32) for b in bounds]
  if not bounds:
    return jnp.zeros((), f32)
  if impl == 'prod':
    out = bounds[0]
    for b in bounds[1:]:
      out = out * b
    return out
  if impl == 'sum':
    out = bounds[0]
    for b in bounds[1:]:
      out = out + b
    return out
  if impl == 'logprod':
    # log(prod(bounds)) computed as sum(log(bounds)); the direct product
    # overflows f32 well before it becomes large enough to matter.
    out = jnp.log(jnp.maximum(bounds[0], 1e-12))
    for b in bounds[1:]:
      out = out + jnp.log(jnp.maximum(b, 1e-12))
    return out
  raise NotImplementedError(impl)


class LipLinear(nj.Module):
  """``nets.Linear`` with the LipVQ-VAE row-wise weight normalization.

  Fields:
    per_row: one trainable bound per output unit (the paper's "introduced for
      every row i", and the released code). ``False`` uses a single scalar
      bound per layer (Liu et al., and the ``c_l`` subscript of Eq. 4).
    cinit: initial value of the *bound* ``softplus(c)``. Negative (default)
      initializes it to the layer's own initial absolute row sums, so the
      normalization is exactly inert at initialization and the penalty has to
      pull the bound down -- Liu et al.'s recipe. A fixed small value would
      instead scale a 1024-unit layer down by ~20x at init and destroy the
      forward signal.
    clamp: see ``lip_scale``.
  """

  bias: bool = True
  winit: str | Callable = nets.Initializer('trunc_normal')
  binit: str | Callable = nets.Initializer('zeros')
  outscale: float = 1.0
  per_row: bool = True
  cinit: float = -1.0
  clamp: bool = True

  def __init__(self, units):
    # Mirrors ``nets.Linear``: an int or a tuple output shape, the latter
    # flattened into the kernel and reshaped back afterwards. The Lipschitz
    # bound is per flattened output unit either way, which is the right
    # granularity -- the reshape does not mix units.
    self.units = (units,) if isinstance(units, int) else tuple(units)
    self.size = math.prod(self.units)
    self._insize = None

  def __call__(self, x):
    # Accept whatever dtype the caller supplies rather than pinning to the
    # global COMPUTE_DTYPE: the goal-AE trunks run in f32 under
    # ``LipMLP.dtype='float32'``, which is what makes ``norm='none'`` -- and so
    # the composed Lipschitz bound -- trainable.
    nets.ensure_dtypes(x, fwd=x.dtype, bwd=x.dtype)
    self._insize = x.shape[-1]
    kernel = self._kernel()
    y = x @ lip_normalize(kernel, self._c(), self.clamp).astype(x.dtype)
    if self.bias:
      y += self.value('bias', nets.init(self.binit), self.size).astype(x.dtype)
    return y.reshape((*y.shape[:-1], *self.units))

  def bound(self):
    """The layer's inf-norm Lipschitz bound, ``max_i softplus(c_i)``."""
    return jax.nn.softplus(self._c().astype(f32)).max()

  def scale(self):
    """Realized rescale factor per output unit (1.0 == constraint inactive)."""
    bound = jax.nn.softplus(self._c().astype(f32))
    return lip_scale(self._kernel(), bound, self.clamp)

  def _kernel(self):
    assert self._insize is not None, 'LipLinear must be called before use.'
    return self.value('kernel', self._scaled_winit, (self._insize, self.size))

  def _c(self):
    return self.value('c', self._make_c)

  def _make_c(self):
    shape = (self.size,) if self.per_row else ()
    if self.cinit >= 0.0:
      return jnp.full(shape, softplus_inv(self.cinit), f32)
    rows = abs_row_sum(self._kernel())
    target = rows if self.per_row else rows.max()
    return jnp.broadcast_to(softplus_inv(target), shape).astype(f32)

  def _scaled_winit(self, *args, **kwargs):
    return nets.init(self.winit)(*args, **kwargs) * self.outscale


class LipMLP(nj.Module):
  """Plain MLP trunk whose linear layers are optionally Lipschitz-constrained.

  Mirrors ``nets.MLP`` (``layers`` x [Linear -> Norm -> act]) but lets the
  Lipschitz constraint be switched on per instance.

  Normalization and the strength of the guarantee. A rescaling norm layer
  (``rms``/``layer``) multiplies activations by a data-dependent factor, so
  with one present the per-layer bounds no longer compose into a certified
  global Lipschitz constant: the weight normalization still bounds each linear
  map's inf-norm operator norm, and the penalty still shrinks those bounds, but
  ``prod(bounds)`` is then a property of the linear layers only, not of the
  network. ``strict_bound=True`` requires ``norm='none'`` and makes
  ``prod(bounds)`` a true bound on the whole trunk (LipVQ-VAE Fig. 3 shows only
  ``Linear -> Lipschitz Reg -> activation``).

  It is off by default because a 3x1024 unnormalized trunk is not trainable at
  this project's scale: with ``norm='none'`` the goal autoencoder produces a
  NaN within 2-16 train steps on dmc_hopper_hop at debug scale, for every seed
  and every ablation arm, while the same configuration with ``norm='rms'`` runs
  clean (the Director trunk it replaces also uses ``rms``). Keeping ``rms``
  additionally leaves each arm a single-factor change from the Director
  baseline. Note the LipVQ-VAE released code does not remove normalization
  either; it applies its ``LipschitzMLP`` as one layer among unconstrained
  ones.
  """

  act: str = 'silu'
  norm: str = 'none'
  bias: bool = True
  winit: str | Callable = nets.Initializer('trunc_normal')
  binit: str | Callable = nets.Initializer('zeros')
  lip: bool = False
  per_row: bool = True
  cinit: float = -1.0
  clamp: bool = True
  strict_bound: bool = False
  # 'default' follows nets.COMPUTE_DTYPE (bfloat16 in training); 'float32'
  # runs this trunk in f32 regardless. Needed for norm='none': the
  # unnormalized trunk produces non-finite values within a few steps in
  # bfloat16 and trains without incident in float32 (measured).
  dtype: str = 'default'

  def __init__(self, layers, units):
    self.layers = int(layers)
    self.units = int(units)
    if self.dtype != 'default':
      # Only the LipLinear path tolerates a non-global dtype; nets.Linear
      # asserts nets.COMPUTE_DTYPE. That is the only case that needs it: the
      # f32 trunk exists to make norm='none' trainable, and norm='none' only
      # buys anything when the Lipschitz constraint is on.
      assert self.lip, (
          f'dtype={self.dtype!r} requires lip=True; the unconstrained path '
          'uses nets.Linear, which pins the global compute dtype.')
    if self.lip and self.strict_bound:
      assert self.norm == 'none', (
          f'strict_bound requires norm=none, got {self.norm!r}: a rescaling '
          'normalization layer voids the composed weight-normalization bound.')
    self._bounds = []

  def __call__(self, x):
    shape = x.shape[:-1]
    x = x.astype(f32 if self.dtype == 'float32' else nets.COMPUTE_DTYPE)
    x = x.reshape([-1, x.shape[-1]])
    bounds = []
    for i in range(self.layers):
      if self.lip:
        layer = self.sub(
            f'linear{i}', LipLinear, int(self.units), bias=self.bias,
            winit=self.winit, binit=self.binit, per_row=self.per_row,
            cinit=self.cinit, clamp=self.clamp)
        x = layer(x)
        bounds.append(layer.bound())
      else:
        x = self.sub(
            f'linear{i}', nets.Linear, self.units, bias=self.bias,
            winit=self.winit, binit=self.binit)(x)
      # ``Norm('none')`` is a no-op that still asserts the global compute
      # dtype, so skip the submodule entirely rather than route an f32 trunk
      # through it. Parameter names are unaffected: 'none' creates none.
      if self.norm != 'none':
        x = self.sub(f'norm{i}', nets.Norm, self.norm)(x)
      x = nets.act(self.act)(x)
    self._bounds = bounds
    x = x.reshape((*shape, x.shape[-1]))
    return x

  def scales(self):
    """Per-layer realized rescale factors from the most recent ``__call__``."""
    return [self.sub(f'linear{i}', LipLinear, int(self.units)).scale()
            for i in range(self.layers)] if self.lip else []

  def bounds(self):
    """Per-layer inf-norm bounds of the linear maps, most recent ``__call__``.

    Composes into a bound on the whole trunk only under ``strict_bound``; see
    the class docstring.
    """
    return list(self._bounds)
