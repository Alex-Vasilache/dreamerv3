"""Decode a hand-edited goal code into the image the world model predicts.

The demo needs one function: given an ``(L, C)`` code, produce the 64x64 frame
the agent would show as its goal. That is the same three steps the agent itself
runs in ``Agent.report`` --

    code --goal_dec--> deter --dyn._prior--> feat --dec--> image

-- so this module does not reimplement any of them. It builds the real
``dreamerv3.agent.Agent`` module tree, runs ``nj.init`` over exactly that path
(which creates parameters only for the three modules the path touches, ~55M of
the checkpoint's 232M), overwrites them from a milestone checkpoint, and jits
the result. Whatever the training run would have rendered is what comes out.

What the agent thinks of the state
----------------------------------
``values`` reads the same ``feat`` with the agent's own heads instead of the
image decoder, so a code can be judged as well as looked at. The three critics
are the three the HRL agent trains, kept apart because they answer different
questions:

* ``mgr_extr_val(feat)`` -- the manager's discounted **task** return from here.
* ``mgr_expl_val(feat)`` -- the manager's **exploration** return, whose reward is
  the goal autoencoder's own squared reconstruction error, so it scores how
  unfamiliar the state is rather than how good it is.
* ``wkr_goal_val(feat, g)`` -- the worker's return under the cosine goal reward.
  Unlike the other two this one needs a goal as well as a state, so it is the
  only critic that can be asked about a *pair* of codes: ``g`` defaults to the
  state itself (the worker standing on its goal, its best case) and the demo
  also asks it across a saved/current pair.

``rew`` and ``con`` come along because they are what the manager's critic is
built out of. Values are un-normalized with each head's ``valnorm`` stats, which
is a no-op on the pinned runs (``valnorm.impl: none``) and correct anyway if a
later pin trains with a real one.

All of it inherits the same caveat as the frame: these heads were trained on
features of *real* states, and a hand-edited code is only as real as the goal
decoder makes it.

Generalized codes
-----------------
Both arms take a real-valued class index per block, so a block can sit between
two classes or outside the eight the model was trained on. What that index
means differs between the arms, because only one of them has a geometry to
extrapolate in.

**Ours** turns the index into an embedding (``class_embeddings``). Inside
``[0, C-1]`` it walks the piecewise-linear path through the block's ``C``
codebook entries, so every integer lands exactly on a trained entry. Outside,
it leaves the end entry along the block's *fitted line* -- the first principal
direction of all ``C`` entries, oriented toward increasing class -- taking one
*average* segment length per class:

    z(t) = e_0        + t          * s_l * d_l          t < 0
         = (1-u) e_i  + u e_{i+1}  0 <= t <= C-1,  i = floor(t), u = t - i
         = e_{C-1}    + (t-(C-1))  * s_l * d_l          t > C-1

with ``d_l`` the fitted direction and ``s_l`` the mean of the block's ``C-1``
segment lengths. Anchoring on the real end entries keeps ``z`` continuous at 0
and ``C-1``. The earlier version of this simply repeated the end segment, which
made extrapolation inherit whichever end gap happened to be shortest -- and
measured across all four environments, the two end gaps are always the shortest
in the block (~0.4 against an interior mean of ~0.6). Using the fitted
direction and the average length makes a step outside cost the same as a
typical step inside, and follows where the line as a whole points rather than
where its last pair happened to sit.

**Director** has no embedding, so there is nothing to fit a line to: its ``C``
one-hots are mutually equidistant by construction and their principal
directions are degenerate. It keeps the weight form (``class_weights``),

    i = clip(floor(t), 0, C-2);  u = t - i;  w[i] = 1-u;  w[i+1] = u

one-hot at every integer in range, linear between, and the straight-line
extension of the one-hot family outside -- ``t = -1`` gives ``w = [2, -1, ...]``.
Those 64 numbers go straight into the decoder MLP as input activations. The
extrapolation is defined but uncalibrated, because the input space carries no
learned metric. That asymmetry is a finding to look at, not a bug to hide.
"""
import os
import pickle
import sys

import numpy as np

# ``source_dreamerv3_env.sh`` exports DREAMERV3_CONV_IMPL=reference, the einsum
# fallback that works around a cuDNN bug on the V100 nodes. It is read once at
# ``embodied.jax.nets`` import time and it is 5-10x slower; nothing here runs on
# a V100, so drop it before that import happens. ``test_conv_impls_agree`` in
# check_render.py asserts the two paths render the same frame.
os.environ.pop('DREAMERV3_CONV_IMPL', None)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
  sys.path.insert(0, _REPO)

import elements  # noqa: E402
import ninjax as nj  # noqa: E402

import runs as runsmod  # noqa: E402  (same directory)

SPACE_CACHE = os.path.join(_HERE, '.space_cache.pkl')

# How many points the goal-distance walk decodes per refinement round. Fixed so
# the search costs one jit compile rather than one per requested distance.
SEARCH_POINTS = 48


def class_weights(idx, classes):
  """Real-valued class indices ``(..., L)`` -> class weights ``(..., L, C)``.

  The Director form: one-hot on integers in range, linear between, linear
  extension of the one-hot family outside. See the module docstring.
  """
  idx = np.asarray(idx, np.float32)
  lo = np.clip(np.floor(idx), 0, classes - 2).astype(np.int64)
  frac = idx - lo
  out = np.zeros(idx.shape + (classes,), np.float32)
  ax = np.indices(idx.shape)
  out[(*ax, lo)] = 1.0 - frac
  out[(*ax, lo + 1)] = frac
  return out


def fit_lines(table):
  """``(L, C, D)`` codebook -> per-block unit direction ``(L, D)`` and mean
  segment length ``(L,)``.

  The direction is the first principal component of the block's entries, signed
  so that it points from low class index to high. The length is the mean of the
  ``C-1`` distances between consecutive entries -- a typical step along the
  block, rather than either end's.
  """
  table = np.asarray(table, np.float32)
  blocks, classes = table.shape[:2]
  rank = np.arange(classes) - (classes - 1) / 2.0
  dirs = np.empty(table.shape[::2], np.float32)   # (L, D)
  for i, entries in enumerate(table):
    _, _, vt = np.linalg.svd(entries - entries.mean(0), full_matrices=False)
    axis = vt[0]
    # Orient toward increasing class: the projections should grow with index.
    if float(((entries @ axis) * rank).sum()) < 0:
      axis = -axis
    dirs[i] = axis
  steps = np.linalg.norm(np.diff(table, axis=1), axis=-1).mean(1)
  return dirs, steps.astype(np.float32)


def class_embeddings(idx, table, dirs, steps):
  """Real-valued class indices ``(..., L)`` -> embeddings ``(..., L, D)``.

  The ours form: the codebook path inside the trained range, the fitted line at
  one average segment per class outside it. See the module docstring.
  """
  idx = np.asarray(idx, np.float32)
  blocks, classes = table.shape[:2]
  assert idx.shape[-1] == blocks, (idx.shape, blocks)
  cols = np.arange(blocks)

  lo = np.clip(np.floor(idx), 0, classes - 2).astype(np.int64)
  frac = (idx - lo)[..., None]
  z = (1.0 - frac) * table[cols, lo] + frac * table[cols, lo + 1]

  stride = steps[:, None] * dirs                                  # (L, D)
  below = table[cols, 0] + idx[..., None] * stride
  above = table[cols, classes - 1] + (idx - (classes - 1))[..., None] * stride
  z = np.where((idx < 0)[..., None], below, z)
  z = np.where((idx > classes - 1)[..., None], above, z)
  return z.astype(np.float32)


def _env_spaces(task):
  """Observation/action spaces for a task, cached so MuJoCo runs once ever."""
  cache = {}
  if os.path.exists(SPACE_CACHE):
    with open(SPACE_CACHE, 'rb') as f:
      cache = pickle.load(f)
  if task not in cache:
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    from embodied.envs import dmc
    suite, rest = task.split('_', 1)
    assert suite == 'dmc', task
    env = dmc.DMC(rest, repeat=2, size=(64, 64))
    notlog = lambda k: not k.startswith('log/')
    obs = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act = {k: v for k, v in env.act_space.items() if k != 'reset'}
    env.close()
    cache[task] = (obs, act)
    tmp = SPACE_CACHE + '.tmp'
    with open(tmp, 'wb') as f:
      pickle.dump(cache, f)
    os.replace(tmp, SPACE_CACHE)
  return cache[task]


class GoalRenderer:
  """One checkpoint, loaded once, rendering goal codes on demand."""

  def __init__(self, task, arm, step=runsmod.STEP, platform='cpu',
               compute_dtype=None):
    os.environ.setdefault('JAX_PLATFORMS', platform)
    import jax
    import jax.numpy as jnp
    from embodied.jax import internal
    from dreamerv3.agent import Agent as AgentCls

    self.task, self.arm = task, arm
    self.info = runsmod.info(task, arm)

    config = elements.Config.load(runsmod.config_path(task, arm))
    setup = {k: v for k, v in config.jax.items()
             if k not in _option_keys()}
    # The run trained on a GPU and preallocated it; we are a small CPU service
    # feeding host arrays into a jitted function, so those three differ.
    setup.update(platform=platform, prealloc=False, transfer_guard=False)
    if compute_dtype:
      setup.update(compute_dtype=compute_dtype)
    internal.setup(**setup)
    self.compute_dtype = str(setup['compute_dtype'])
    obs_space, act_space = _env_spaces(task)
    agent_cfg = elements.Config(
        **config.agent, logdir=config.logdir, seed=config.seed,
        jax=config.jax, batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica, replicas=config.replicas)

    # Bypass embodied.jax.Agent.__new__, which additionally builds the
    # sharded/JIT-compiled train+policy wrapper we have no use for here.
    model = object.__new__(AgentCls)
    model.__init__(obs_space, act_space, agent_cfg)
    self.model = model
    self.blocks, self.classes = [int(x) for x in model.skill_shape]
    self.imgkeys = list(model.dec.imgkeys)
    assert self.imgkeys, 'checkpoint has no image decoder to render'
    self.imgkey = self.imgkeys[0]

    # The VQ decoder takes embeddings, so the demo can put a block anywhere in
    # the block's embedding space; Director's takes the flattened class
    # weights. Everything downstream of the decoded deter is shared.
    self.is_vq = hasattr(model.goal_dec, 'from_latent')
    self.dim = int(model.goal_dec.dim) if self.is_vq else None

    def decode(inp):
      return (model.goal_dec.from_latent(inp, 1) if self.is_vq
              else model.goal_dec(inp, 1)).pred()

    def render(inp):
      goal = decode(inp)
      feat = model._feat_from_goal(goal)
      reset = jnp.zeros(inp.shape[:1], bool)
      _, _, recons = model.dec(
          model.dec.initial(inp.shape[0]), feat, reset, training=False)
      img = recons[self.imgkey].pred().astype(jnp.float32)
      img = jnp.clip(img * 255.0, 0, 255).astype(jnp.uint8)
      return img, goal.astype(jnp.float32)

    def goals(inp):
      return decode(inp).astype(jnp.float32)

    # ``target`` is the goal the worker critic is told to aim at. It is a second
    # code rather than the first so the caller can ask the worker about a pair.
    def values(inp, target):
      goal = decode(inp)
      feat = model._feat_from_goal(goal)
      tensor = model.feat2tensor(feat)
      out = {}
      for name, head, norm in (
          ('mgr_extr', model.mgr_extr_val, model.mgr_extr_valnorm),
          ('mgr_expl', model.mgr_expl_val, model.mgr_expl_valnorm),
      ):
        offset, scale = norm.stats()
        out[name] = head(tensor, 1).pred() * scale + offset
      offset, scale = model.wkr_goal_valnorm.stats()
      out['wkr_goal'] = model.wkr_goal_val(
          model._feat_goal2tensor(feat, decode(target)), 1
      ).pred() * scale + offset
      out['reward'] = model.rew(tensor, 1).pred()
      out['cont'] = model.con(tensor, 1).prob(1)
      return {k: v.astype(jnp.float32) for k, v in out.items()}

    if self.is_vq:
      probe = np.zeros((1, self.blocks, self.dim), np.float32)
    else:
      probe = np.zeros((1, self.blocks, self.classes), np.float32)
      probe[:, :, 0] = 1.0
    params = nj.init(render)({}, jnp.asarray(probe), seed=0)
    # The critics are a second path off the same ``feat``, so they need their own
    # init pass; the heads it adds are ~10M more of the checkpoint. Only an HRL
    # checkpoint has them -- a flat one would keep rendering without values.
    self.has_values = hasattr(model, 'mgr_extr_val')
    if self.has_values:
      params = {**params, **nj.init(values)(
          {}, jnp.asarray(probe), jnp.asarray(probe), seed=0)}
    self.param_keys = sorted(params.keys())
    # ``from_latent`` never touches the codebook, so nothing above creates it;
    # fetch it alongside the render parameters instead of reading the
    # checkpoint a second time.
    extra = ('goal_dec/codebook/table',) if self.is_vq else ()
    loaded, extras = _load_params(
        runsmod.ckpt_path(task, arm, step), params, extra)
    self.codebook_table = extras.get('goal_dec/codebook/table')
    self.line_dirs, self.line_steps = (
        fit_lines(self.codebook_table) if self.is_vq else (None, None))
    # Keep the ~220MB of weights resident on the device: passing host arrays
    # would re-transfer all of them on every single render call, which cost
    # more than the forward pass itself.
    self.params = jax.device_put(loaded)
    # ``nj.pure`` also returns the (unchanged) state. Dropping it inside the
    # jit lets XLA eliminate it; returning it made every call materialize all
    # 220MB of weights again and dominated the render time.
    pure = nj.pure(render)
    self._fn = jax.jit(lambda state, w: pure(state, w)[1])
    puregoals = nj.pure(goals)
    self._gfn = jax.jit(lambda state, w: puregoals(state, w)[1])
    if self.has_values:
      purevalues = nj.pure(values)
      self._vfn = jax.jit(lambda state, w, t: purevalues(state, w, t)[1])
    self._jnp = jnp
    # Compile now so the first user interaction is not the one that pays. jit
    # recompiles per batch shape, so warm the ones the page asks for by name: a
    # single frame, the four codes a saved-state comparison sends, and the grid
    # the goal-distance walk searches over.
    for size in (1, 4, SEARCH_POINTS):
      probe = np.zeros((size, self.blocks), np.float32)
      self.goal_vectors(probe)
      if self.has_values and size <= 4:
        self.values(probe)
    self.render(np.zeros((1, self.blocks), np.float32))

  def decoder_input(self, idx):
    """Class indices ``(B, L)`` -> whatever this arm's goal decoder takes."""
    if self.is_vq:
      return class_embeddings(
          idx, self.codebook_table, self.line_dirs, self.line_steps)
    return class_weights(idx, self.classes)

  def render(self, idx):
    """Class indices ``(B, L)`` -> ``(B, 64, 64, 3)`` uint8 and ``(B, D)`` deter."""
    idx = np.atleast_2d(np.asarray(idx, np.float32))
    inp = self.decoder_input(idx)
    img, goal = self._fn(self.params, self._jnp.asarray(inp))
    return np.asarray(img), np.asarray(goal)

  def goal_vectors(self, idx):
    """Class indices ``(B, L)`` -> the ``deter`` goals they decode to ``(B, D)``."""
    idx = np.atleast_2d(np.asarray(idx, np.float32))
    inp = self._jnp.asarray(self.decoder_input(idx))
    return np.asarray(self._gfn(self.params, inp))

  def values(self, idx, goal_idx=None):
    """Class indices ``(B, L)`` -> ``{head: (B,)}`` for every critic and head.

    ``goal_idx`` is the code the worker critic aims at, defaulting to ``idx``
    itself -- the worker already standing on its goal. Pass a different code to
    ask what the worker makes of going from one state to another.
    """
    assert self.has_values, 'checkpoint has no HRL critics'
    idx = np.atleast_2d(np.asarray(idx, np.float32))
    tgt = idx if goal_idx is None else np.atleast_2d(
        np.asarray(goal_idx, np.float32))
    assert tgt.shape == idx.shape, (idx.shape, tgt.shape)
    out = self._vfn(self.params,
                    self._jnp.asarray(self.decoder_input(idx)),
                    self._jnp.asarray(self.decoder_input(tgt)))
    return {k: np.asarray(v, np.float32) for k, v in out.items()}

  def code_at_goal_distance(self, code, target, span, rng,
                            points=SEARCH_POINTS, rounds=3):
    """A code whose goal sits ``target`` away in goal space, not in class steps.

    Class distance is exact by counting and goal distance is not, because the
    map from one to the other is the decoder's. What makes it solvable anyway is
    that the demo's class indices are real-valued: the goal moves *continuously*
    along a ray in code space, so a requested distance can be hit by walking the
    ray rather than searched for over codes. A random direction is drawn, scaled
    so its largest block step is one class, and cut off where the first block
    would leave ``span`` (the columns the page shows). Then the first crossing of
    ``target`` is bracketed on a grid and the bracket refined, which does not
    assume the distance grows monotonically with the step -- it does not have to,
    and near the edge of the trained range it does not.

    Returns the code, the distance actually reached, and whether the ray ran out
    of room before getting there.
    """
    code = np.asarray(code, np.float32).reshape(-1)
    base = self.goal_vectors(code)[0]
    target = float(target)
    if target <= 0:
      return dict(code=code.tolist(), dist=0.0, capped=False)

    delta = rng.uniform(-1.0, 1.0, code.shape).astype(np.float32)
    if not np.any(delta):
      delta = np.ones_like(delta)
    delta /= np.abs(delta).max()
    lo, hi = span
    # How far each block can be dragged before it leaves the grid; the ray stops
    # at the first one to hit the edge.
    room = np.where(delta > 0, (hi - code) / delta, (lo - code) / delta)
    stop = float(max(0.0, room.min()))
    at = lambda ts: code[None] + np.asarray(ts, np.float32)[:, None] * delta

    # The distance is reported from a fresh single decode of the code that comes
    # out, not from the batched grid it was found on: a batch of 48 and a batch
    # of 1 reduce the decoder's matmuls in a different order, and the demo should
    # print the number the user gets when they render that code by itself.
    def result(code, capped):
      # Rounded before it is measured, not after: the page keys its caches on
      # three decimals, so the code it ends up holding is the one the reported
      # distance has to belong to.
      code = np.asarray(code, np.float32).reshape(-1).round(3)
      dist = float(np.linalg.norm(self.goal_vectors(code)[0] - base))
      return dict(code=code.tolist(), dist=dist, capped=capped)

    left, right = 0.0, stop
    for _ in range(rounds):
      ts = np.linspace(left, right, points)
      dists = np.linalg.norm(self.goal_vectors(at(ts)) - base, axis=-1)
      reach = np.nonzero(dists >= target)[0]
      if not len(reach):
        return result(at(ts)[int(np.argmax(dists))], True)
      i = int(reach[0])
      left, right = float(ts[max(i - 1, 0)]), float(ts[i])
    return result(at([right])[0], False)

  def goal_scale(self, seed=0, samples=64):
    """Typical goal-space distances, so a requested one has a scale to sit on.

    ``step`` is the median over moving a single block of a random code by one
    class, ``span`` the median over pairs of unrelated random codes.
    """
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, self.classes, (samples, self.blocks)).astype(
        np.float32)
    goals = self.goal_vectors(codes)
    pairs = np.linalg.norm(goals[:, None] - goals[None, :], axis=-1)
    span = float(np.median(pairs[np.triu_indices(samples, 1)]))
    moved = codes.copy()
    blocks = rng.integers(0, self.blocks, samples)
    rows = np.arange(samples)
    moved[rows, blocks] += np.where(moved[rows, blocks] < self.classes - 1, 1, -1)
    step = float(np.median(np.linalg.norm(
        self.goal_vectors(moved) - goals, axis=-1)))
    return dict(step=step, span=span)

  def codebook(self):
    """``(L, C, D)`` codebook table, or ``None`` for the Director arm."""
    if self.codebook_table is None:
      return None
    return np.asarray(self.codebook_table)


# ``embodied.jax.agent.Options`` fields are consumed by the wrapper we skip;
# everything else in ``config.jax`` is what ``internal.setup`` wants. Read the
# names off the dataclass so a new option cannot silently be passed to setup().
def _option_keys():
  from embodied.jax.agent import Options
  return tuple(Options.__dataclass_fields__)


def _load_params(path, template, extra=()):
  """Checkpoint values for the keys ``nj.init`` created, plus any ``extra``."""
  with open(path, 'rb') as f:
    data = pickle.load(f)
  saved = data['params']
  missing = [k for k in (*template, *extra) if k not in saved]
  assert not missing, f'checkpoint {path} is missing {missing[:5]}'
  out = {}
  for key, ref in template.items():
    val = saved[key]
    assert tuple(val.shape) == tuple(ref.shape), (key, val.shape, ref.shape)
    out[key] = np.asarray(val, ref.dtype)
  return out, {k: np.asarray(saved[k], np.float32) for k in extra}
