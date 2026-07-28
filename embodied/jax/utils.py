import functools

import jax
import jax.numpy as jnp
import ninjax as nj

from . import internal

sg = jax.lax.stop_gradient
f32 = jnp.float32
i32 = jnp.int32

COMPUTE_DTYPE = jnp.bfloat16


class Normalize(nj.Module):

  rate: float = 0.01
  limit: float = 1e-8
  perclo: float = 5.0
  perchi: float = 95.0
  debias: bool = True

  def __init__(self, impl):
    self.impl = impl
    if self.debias and self.impl != 'none':
      self.corr = nj.Variable(jnp.zeros, (), f32, name='corr')
    if self.impl == 'none':
      pass
    elif self.impl == 'meanstd':
      self.mean = nj.Variable(jnp.zeros, (), f32, name='mean')
      self.sqrs = nj.Variable(jnp.zeros, (), f32, name='sqrs')
    elif self.impl == 'perc':
      self.lo = nj.Variable(jnp.zeros, (), f32, name='lo')
      self.hi = nj.Variable(jnp.zeros, (), f32, name='hi')
    else:
      raise NotImplementedError(self.impl)

  def __call__(self, x, update, weights=None):
    if update:
      self.update(x, weights)
    return self.stats()

  def update(self, x, weights=None):
    """Accumulate running statistics of ``x``.

    ``weights`` (broadcastable to ``x``, nonnegative) restricts the statistics
    to the entries that are real data. Packed variable-length timelines (e.g.
    the manager's block-pooled decision axis) forward-fill their unused tail,
    so without a weight the running mean/spread is dominated by repeated copies
    of the last real entry.
    """
    x = sg(f32(x))
    if weights is not None:
      weights = sg(f32(jnp.broadcast_to(weights, x.shape)))
    if self.impl == 'none':
      pass
    elif self.impl == 'meanstd':
      self._update(self.mean, self._mean(x, weights))
      self._update(self.sqrs, self._mean(jnp.square(x), weights))
    elif self.impl == 'perc':
      self._update(self.lo, self._perc(x, self.perclo, weights))
      self._update(self.hi, self._perc(x, self.perchi, weights))
    else:
      raise NotImplementedError(self.impl)
    if self.debias and self.impl != 'none':
      self._update(self.corr, 1.0)

  def stats(self):
    corr = 1.0
    if self.debias and self.impl != 'none':
      corr /= jnp.maximum(self.rate, self.corr.read())
    if self.impl == 'none':
      return 0.0, 1.0
    elif self.impl == 'meanstd':
      mean = self.mean.read() * corr
      std = jnp.sqrt(jax.nn.relu(self.sqrs.read() * corr - mean ** 2))
      std = jnp.maximum(self.limit, std)
      return mean, std
    elif self.impl == 'perc':
      lo, hi = self.lo.read() * corr, self.hi.read() * corr
      return sg(lo), sg(jnp.maximum(self.limit, hi - lo))
    else:
      raise NotImplementedError(self.impl)

  def _mean(self, x, weights=None):
    if weights is None:
      x = x.mean()
      axes = internal.get_data_axes()
      if axes:
        x = jax.lax.pmean(x, axes)
      return x
    # Weighted mean across devices: average the numerator and denominator
    # separately so the ratio stays the global weighted mean.
    num, den = (x * weights).sum(), weights.sum()
    axes = internal.get_data_axes()
    if axes:
      num = jax.lax.pmean(num, axes)
      den = jax.lax.pmean(den, axes)
    return num / jnp.maximum(den, 1e-8)

  def _perc(self, x, q, weights=None):
    if weights is not None:
      # Drop zero-weight entries from the order statistic entirely (percentiles
      # have no meaningful fractional-weight form here; the weights this takes
      # are 0/1 validity masks).
      x = jnp.where(weights > 0, x, jnp.nan)
    axes = internal.get_data_axes()
    if axes:
      x = jax.lax.all_gather(x, axes)
    x = jnp.nanpercentile(x, q) if weights is not None else jnp.percentile(x, q)
    return x

  def _update(self, var, x):
    var.write((1 - self.rate) * var.read() + self.rate * sg(x))


class AutoAdapt(nj.Module):
  """Adaptive Lagrange multiplier (Director ``tfutils.AutoAdapt``).

  Holds a scalar/per-dim scale that grows when the regulated quantity (``reg``)
  is below ``target`` and shrinks when above (or vice versa if ``inverse``).
  Use ``inverse=True`` for entropy-style regularizers (we want loss to push up
  on entropy when it falls below target). ``inverse=False`` for KL-style
  regularizers (we want stronger push when KL is above target).

  ``one_sided=True`` disables the shrink branch: the scale still grows to
  correct a violation but never relaxes back down once the target is met, for
  regularizers where overshooting the target is harmless or beneficial (e.g.
  a loss ceiling that may drift back above target later in training under a
  fixed scale -- relaxing early would leave less pressure to correct that).
  """

  vel: float = 0.1
  thres: float = 0.1

  def __init__(
      self, shape, impl, target, min, max, inverse=False, one_sided=False,
      init=1.0):
    self.shape = tuple(shape)
    self.impl = impl
    self.target = float(target)
    self.min = float(min)
    self.max = float(max)
    self.inverse = bool(inverse)
    self.one_sided = bool(one_sided)
    if impl in ('mult', 'prop'):
      init_val = float(init)
      self.scale_var = nj.Variable(
          lambda s: jnp.full(s, init_val, f32), self.shape, name='scale')
    elif impl == 'fixed':
      self.fixed_scale = float(init)
    else:
      raise NotImplementedError(impl)

  def __call__(self, reg, update=True, target=None, weights=None):
    reg = f32(reg)
    if weights is not None:
      weights = sg(f32(jnp.broadcast_to(weights, reg.shape)))
    if update:
      self.update(reg, target, weights)
    scale = self.scale()
    # Broadcast scale over leading reduction dims of reg.
    while scale.ndim < reg.ndim:
      scale = scale[None]
    loss = scale * (-reg if self.inverse else reg)
    if weights is None:
      reg_mean, reg_std = reg.mean(), reg.std()
    else:
      den = jnp.maximum(weights.sum(), 1e-8)
      reg_mean = (reg * weights).sum() / den
      reg_std = jnp.sqrt(jnp.maximum(
          (jnp.square(reg) * weights).sum() / den - jnp.square(reg_mean), 0.0))
    metrics = {
        'mean': reg_mean,
        'std': reg_std,
        'scale_mean': self.scale().mean(),
        'scale_std': self.scale().std(),
    }
    if target is not None:
      metrics['target'] = f32(target).mean()
    return loss, metrics

  def scale(self):
    if self.impl == 'fixed':
      return jnp.full(self.shape, self.fixed_scale, f32)
    return sg(self.scale_var.read())

  def update(self, reg, target=None, weights=None):
    """Step the multiplier toward ``target``.

    ``weights`` (broadcastable to ``reg``) restricts the tracked average to the
    entries that are real data, for packed/padded axes -- see
    ``Normalize.update``.
    """
    if self.impl == 'fixed':
      return
    tgt = self.target if target is None else f32(target)
    # Reduce all leading dims that are not part of self.shape.
    reduce_ndim = reg.ndim - len(self.shape)
    if reduce_ndim > 0:
      axes_red = tuple(range(reduce_ndim))
      if weights is None:
        avg = reg.mean(axes_red)
      else:
        w = jnp.broadcast_to(f32(weights), reg.shape)
        avg = ((reg * w).sum(axes_red) /
               jnp.maximum(w.sum(axes_red), 1e-8))
    else:
      avg = reg
    # Aggregate across data axes for multi-device runs.
    axes = internal.get_data_axes()
    if axes:
      avg = jax.lax.pmean(avg, axes)
    below = avg < (1.0 / (1.0 + self.thres)) * tgt
    above = avg > (1.0 + self.thres) * tgt
    if self.inverse:
      below, above = above, below
    if self.one_sided:
      below = jnp.zeros_like(below)  # never relax once the target is met
    s = self.scale_var.read()
    if self.impl == 'mult':
      adjusted = jnp.where(
          above, s * (1.0 + self.vel),
          jnp.where(below, s / (1.0 + self.vel), s))
    elif self.impl == 'prop':
      direction = avg - tgt
      if self.inverse:
        direction = -direction
      if self.one_sided:
        direction = jnp.maximum(direction, 0.0)
      adjusted = s + self.vel * direction
    else:
      raise NotImplementedError(self.impl)
    adjusted = jnp.clip(adjusted, self.min, self.max)
    self.scale_var.write(adjusted)


class Ratchet(nj.Module):
  """Deterministic value that moves linearly from ``init`` toward ``final`` by
  at most ``vel`` per call, then holds at ``final``. Open-loop (no feedback
  from any regulated quantity) -- unlike ``AutoAdapt``, which reacts to a
  measured signal, this only tracks a step count. Used to anneal an
  ``AutoAdapt`` target itself over training without discontinuous jumps (e.g.
  mask sparsity target 1.0 -> 0.3 as goal-space exploration narrows).
  """

  vel: float = 0.01

  def __init__(self, shape, init, final):
    self.shape = tuple(shape)
    self.init = float(init)
    self.final = float(final)
    self.value_var = nj.Variable(
        lambda s: jnp.full(s, float(init), f32), self.shape, name='value')

  def __call__(self, update=True):
    if update:
      self.step()
    return sg(self.value_var.read())

  def step(self):
    v = self.value_var.read()
    if self.final >= self.init:
      adjusted = jnp.minimum(v + self.vel, self.final)
    else:
      adjusted = jnp.maximum(v - self.vel, self.final)
    self.value_var.write(adjusted)


class RmsTracker(nj.Module):

  """EMA of batch mean(loss**2); read() returns sqrt for per-term RMS scaling."""

  rate: float = 0.01
  limit: float = 1e-8

  def __init__(self):
    self.sqrs = nj.Variable(jnp.ones, (), f32, name='sqrs')

  def __call__(self, x, update):
    x = sg(f32(x))
    m2 = jnp.mean(jnp.square(x))
    axes = internal.get_data_axes()
    if axes:
      m2 = jax.lax.pmean(m2, axes)
    if update:
      self.sqrs.write((1 - self.rate) * self.sqrs.read() + self.rate * m2)
    return jnp.sqrt(jnp.maximum(self.sqrs.read(), self.limit))


class SlowModel:

  def __init__(self, model, *, source, rate=1.0, every=1):
    assert rate == 1 or rate < 0.5, rate
    self.source = source
    self.model = model
    self.rate = rate
    self.every = every
    name = self.model.path + '_count'
    self.count = nj.Variable(jnp.zeros, (), i32, name=name)

  def __getattr__(self, name):
    self._initonce()
    return getattr(self.model, name)

  def __call__(self, *args, **kwargs):
    self._initonce()
    return self.model(*args, **kwargs)

  def update(self):
    self._initonce()
    mix = jnp.where(self.count.read() % self.every == 0, self.rate, 0)
    fn = lambda src, dst: mix * src + (1 - mix) * dst
    values = jax.tree.map(fn, self.source.values, self.model.values)
    [self.model.write(k, v) for k, v in values.items()]
    self.count.write(self.count.read() + 1)

  def _initonce(self, *args, method=None, **kwargs):
    assert self.source.values, 'no parameters to track'
    if not self.model.values:
      p = self.model.path + '/'
      nj.context().update({p + k: v for k, v in self.source.values.items()})
    assert self.model.values.keys() == self.source.values.keys(), (
        self.model.values.keys(), self.source.values.keys())


class LayerScan:

  def __init__(self, module, count, names=('__call__',)):
    self.module = module
    self.count = count
    self.names = names

  def __call__(self, *args, **kwargs):
    # Magic methods need to be forwarded explicitly.
    return self.__getattr__('__call__')(*args, **kwargs)

  def __getattr__(self, name):
    value = getattr(self.module, name)
    if name in self.names:
      assert callable(value)
      value = nj.pure(value, nested=True)
      value = functools.partial(
          layer_scan, value, self.module.path, self.count)
    return value


def layer_scan(fn, scope, count, inp, *args, **kwargs):
  isinner = lambda k: k.startswith(scope + '/')

  args_ = jax.tree.map(lambda x: x[0], args)  # Copy structure
  kwargs_ = jax.tree.map(lambda x: x, kwargs)  # Copy structure
  state_ = {k: v[0] if isinner(k) else v for k, v in nj.context().items()}
  state, _, accessed, modified, created = fn(
      state_, inp, *args_, ignore=True, track=True,
      seed=nj.seed(None, True), **kwargs_)

  # print('-' * 79)
  # print('accessed:', accessed)
  # print('modified:', modified)
  # print('created:', created)

  inner = lambda xs: {k: v for k, v in xs.items() if isinner(k)}
  outer = lambda xs: {k: v for k, v in xs.items() if not isinner(k)}

  unchanging = {
      k: v for k, v in nj.context().items()
      if k in accessed and k not in modified and k not in created}
  unchanging_inner = inner(unchanging)
  unchanging_outer = outer(unchanging)

  creations = {k: v for k, v in state.items() if k in created}
  creations_inner = inner(creations)
  creations_outer = outer(creations)
  nj.context().update(creations_outer)
  del creations_inner  # Will be created inside the scan.

  # Inner values do not exist yet, so we only keep them in the creations. This
  # is fine, because inner values cannot change across scan iterations anyways.
  # Outer values can change over iterations, so we need to thread them even
  # during creation.
  changing_inner = inner({
      # k: v for k, v in state.items()
      k: v for k, v in nj.context().items()
      if k in modified and k not in created})
  changing_outer = outer({
      k: v for k, v in state.items()
      if k in modified})

  # f = lambda x: {k: v.shape for k, v in x.items()}
  # print('-' * 79)
  # print('unchanging_inner', f(unchanging_inner))
  # print('unchanging_outer', f(unchanging_outer))
  # print('creations_inner', f(inner(creations)))
  # print('creations_outer', f(creations_outer))
  # print('changing_inner', f(changing_inner))
  # print('changing_outer', f(changing_outer))

  def body(carry, x):
    inp, changing_outer = carry
    arg, seed, unchanging_inner, changing_inner = x
    state = {
        **unchanging_inner, **unchanging_outer,
        **changing_inner, **changing_outer}
    state, out = fn(state, inp, *arg, **kwargs, seed=seed)
    out, *other = out if isinstance(out, tuple) else (out,)
    changing = {k: v for k, v in state.items() if k in modified}
    changing_inner = inner(changing)
    changing_outer = outer(changing)
    creations = {k: v for k, v in state.items() if k in created}
    creations_inner = inner(creations)
    carry = (out, changing_outer)
    y = (other, creations_inner, changing_inner)
    return carry, y

  seeds = nj.seed(count, True)
  carry, ys = jax.lax.scan(
      f=body,
      init=(inp, changing_outer),
      xs=(args, seeds, unchanging_inner, changing_inner),
      length=count)
  out, changing_outer = carry
  other, creations_inner, changing_inner = ys

  if nj.context().modify:
    nj.context().update(creations_inner)
    nj.context().update(changing_inner)
    nj.context().update(changing_outer)

  return (out, *other) if len(other) else out
