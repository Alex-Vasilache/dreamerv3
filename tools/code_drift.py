"""Does a goal code keep pointing at the same PLACE across training?

For each milestone checkpoint of a run:
  1. encode a fixed set of real pinpad trajectories (known player positions)
     with that checkpoint's world model -> deter per state;
  2. collect two code sets: the goal encoder's code for each probe state, and
     the codes the manager actually emits there (mode + one sample);
  3. decode every code with that checkpoint's goal decoder and find the nearest
     probe state by cosine_max (the worker's own reward) -> a grid position.

Each checkpoint is read in its OWN feature space, so a pure change of
coordinates (world model re-expressing the same states) does not register.
What registers is a change of MEANING: code z pointing at a different place.

Pairs of checkpoints are compared on the same codes (codes taken from the
EARLIER checkpoint): displacement of the pointed-at position in grid cells,
and the fraction still pointing at the same pad / non-pad region.

Usage (GPU job):
  python -u tools/code_drift.py --run <run_dir> [--run ...] --out drift.json
"""
import argparse, json, os, pathlib, pickle, sys, time

folder = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(folder))
sys.path.insert(1, str(folder / 'dreamerv3'))

import numpy as np
import elements
import ninjax as nj

B, T = 32, 256          # probe episodes x steps


def probe_data(task, seed=0):
  from embodied.envs.pinpad import PinPad
  env = PinPad(task.split('_', 1)[1], length=10**6)
  rng = np.random.RandomState(seed)
  img = np.zeros((B, T, 64, 64, 3), np.uint8)
  act = np.zeros((B, T), np.int32)
  first = np.zeros((B, T), bool)
  pos = np.zeros((B, T, 2), np.int32)
  seqlen = np.zeros((B, T), np.int32)
  for b in range(B):
    obs = env.step({'action': 0, 'reset': True})
    a, hold = 0, 0
    for t in range(T):
      if t == 0:
        cur = obs
      else:
        if hold <= 0:            # persistent random walk: covers the grid
          a, hold = rng.randint(1, 5), rng.randint(1, 7)
        hold -= 1
        cur = env.step({'action': a, 'reset': False})
        act[b, t] = a           # action that LED to this observation
      img[b, t] = cur['image']
      first[b, t] = (t == 0)
      pos[b, t] = env.player
      seqlen[b, t] = len(env.sequence)
  layout = env.layout
  return dict(image=img, action=act, is_first=first, pos=pos,
              seqlen=seqlen, layout=layout, pads=sorted(env.pads))


class Model:
  def __init__(self, run_dir, platform):
    import jax, jax.numpy as jnp
    from embodied.jax import internal
    from embodied.jax.agent import Options
    from dreamerv3.agent import Agent as AgentCls
    from dreamerv3 import main as m
    self.jax, self.jnp = jax, jnp
    config = elements.Config.load(f'{run_dir}/logdir/config.yaml')
    self.task = config.task
    setup = {k: v for k, v in config.jax.items()
             if k not in Options.__dataclass_fields__}
    setup.update(platform=platform, prealloc=False, transfer_guard=False)
    internal.setup(**setup)
    env = m.make_env(config, 0)
    obs_space = {k: v for k, v in env.obs_space.items()
                 if not k.startswith('log/')}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    env.close()
    self.obs_space = obs_space
    cfg = elements.Config(
        **config.agent, logdir=config.logdir, seed=config.seed,
        jax=config.jax, batch_size=config.batch_size,
        batch_length=config.batch_length, replay_context=config.replay_context,
        report_length=config.report_length, replica=config.replica,
        replicas=config.replicas)
    model = object.__new__(AgentCls)
    model.__init__(obs_space, act_space, cfg)
    self.model = model
    self.L, self.C = [int(x) for x in model.skill_shape]

    def encode(obs, prevact, reset):
      bs = reset.shape[0]
      _, _, tokens = model.enc(model.enc.initial(bs), obs, reset, False)
      _, _, feat = model.dyn.observe(
          model.dyn.initial(bs), tokens, {'action': prevact}, reset, False)
      return (model.feat2deter(feat).astype(jnp.float32),
              model.feat2tensor(feat))

    def state_codes(deter):
      enc = model.goal_enc(deter, 1)
      enc = enc['skill'] if isinstance(enc, dict) else enc
      return enc.pred().astype(jnp.float32)            # (N, L, C) one-hot

    def mgr_codes(tensor, key):
      from dreamerv3.hrl.tensors import mgr_as_dict
      out = mgr_as_dict(model.manager_pol(tensor, 1))['skill']
      return (out.pred().astype(jnp.float32),
              out.sample(key).astype(jnp.float32))

    def decode(code):
      return model.goal_dec(code, 1).pred().astype(jnp.float32)

    self.fns = dict(encode=encode, state_codes=state_codes,
                    mgr_codes=mgr_codes, decode=decode)
    self.compiled = {}
    self.cache = {}
    self.params = None

  def init_and_load(self, ckpt, probe):
    jax, jnp = self.jax, self.jnp
    obs, prevact, reset = self._obs(probe, 0, 2)
    if self.params is None:
      params = nj.init(self.fns['encode'])({}, obs, prevact, reset, seed=0)
      deter, tensor = nj.pure(self.fns['encode'])(params, obs, prevact, reset, seed=jax.random.PRNGKey(0))[1]
      d0, t0 = deter.reshape(-1, deter.shape[-1]), tensor.reshape(-1, tensor.shape[-1])
      params.update(nj.init(self.fns['state_codes'])({}, d0, seed=0))
      code = nj.pure(self.fns['state_codes'])(params, d0, seed=jax.random.PRNGKey(0))[1]
      params.update(nj.init(self.fns['mgr_codes'])(
          {}, t0, jax.random.PRNGKey(0), seed=0))
      params.update(nj.init(self.fns['decode'])({}, code, seed=0))
      self.template = params
    if ckpt not in self.cache:
      with open(ckpt, 'rb') as f:
        saved = pickle.load(f)['params']
      loaded = {}
      for k, ref in self.template.items():
        assert k in saved, f'{ckpt} missing {k}'
        v = saved[k]
        assert tuple(v.shape) == tuple(ref.shape), (k, v.shape, ref.shape)
        loaded[k] = np.asarray(v, ref.dtype)
      del saved
      self.cache[ckpt] = jax.device_put(loaded)
    self.params = self.cache[ckpt]
    for name, fn in self.fns.items():
      if name not in self.compiled:
        pure = nj.pure(fn)
        self.compiled[name] = jax.jit(lambda p, key, *a, pure=pure: pure(p, *a, seed=key)[1])

  def _obs(self, probe, lo, hi):
    jnp = self.jnp
    img = probe['image'][lo:hi]
    bs, t = img.shape[:2]
    obs = {}
    for k, sp in self.obs_space.items():
      if k == 'image':
        obs[k] = jnp.asarray(img)
      elif k == 'is_first':
        obs[k] = jnp.asarray(probe['is_first'][lo:hi])
      else:
        obs[k] = jnp.zeros((bs, t) + tuple(sp.shape), sp.dtype)
    prevact = jnp.asarray(probe['action'][lo:hi])
    reset = jnp.asarray(probe['is_first'][lo:hi])
    return obs, prevact, reset

  def run(self, name, *args):
    return self.compiled[name](self.params, self.jax.random.PRNGKey(0), *args)


def cosmax_argmax(goals, states, jnp):
  """Nearest probe state to each goal under cosine_max; returns (idx, value)."""
  gn = jnp.linalg.norm(goals, axis=-1, keepdims=True)
  sn = jnp.linalg.norm(states, axis=-1, keepdims=True)
  idx, val = [], []
  for i in range(0, goals.shape[0], 1024):
    g, n = goals[i:i + 1024], gn[i:i + 1024]
    dot = g @ states.T
    norm = jnp.maximum(n, sn.T) ** 2
    s = dot / (norm + 1e-12)
    idx.append(np.asarray(jnp.argmax(s, -1)))
    val.append(np.asarray(jnp.max(s, -1)))
  return np.concatenate(idx), np.concatenate(val)


def region(layout, pos):
  ch = layout[pos[..., 0], pos[..., 1]]
  return ch


def analyse_run(run_dir, platform, steps=None):
  t0 = time.time()
  mdl = Model(run_dir, platform)
  probe = probe_data(mdl.task)
  jnp = mdl.jnp
  mdir = pathlib.Path(run_dir) / 'logdir' / 'ckpt_milestones'
  avail = sorted(int(p.name) for p in mdir.iterdir() if (p / 'done').exists())
  steps = [s for s in avail if steps is None or s in steps]
  pos_flat = probe['pos'].reshape(-1, 2)
  N = pos_flat.shape[0]
  per = {}
  for s in steps:
    mdl.init_and_load(str(mdir / f'{s:012d}' / 'agent.pkl'), probe)
    deters, tensors = [], []
    for lo in range(0, B, 8):
      obs, prevact, reset = mdl._obs(probe, lo, lo + 8)
      d, tz = mdl.run('encode', obs, prevact, reset)
      deters.append(np.asarray(d)); tensors.append(np.asarray(tz))
    deter = np.concatenate(deters).reshape(N, -1)
    tensor = np.concatenate(tensors).reshape(N, -1)
    scode = np.asarray(mdl.run('state_codes', jnp.asarray(deter)))
    mmode, msamp = mdl.run('mgr_codes', jnp.asarray(tensor),
                           mdl.jax.random.PRNGKey(s % 2**31))
    per[s] = dict(deter=deter, scode=scode, mmode=np.asarray(mmode),
                  msamp=np.asarray(msamp))
    print(f'  [{os.path.basename(run_dir)[:5]}] loaded {s}  ({time.time()-t0:.0f}s)',
          flush=True)

  def point(s, codes):
    """Position each code decodes to, in checkpoint s's own feature space."""
    mdl.init_and_load(str(mdir / f'{s:012d}' / 'agent.pkl'), probe)
    goals = np.asarray(mdl.run('decode', jnp.asarray(codes)))
    idx, val = cosmax_argmax(jnp.asarray(goals), jnp.asarray(per[s]['deter']), jnp)
    return pos_flat[idx], val

  layout = probe['layout']
  out = dict(run=os.path.basename(run_dir), task=mdl.task, steps=steps,
             self_consistency={}, pairs={})
  # how faithfully each checkpoint's code points back at its source state
  for s in steps:
    p, v = point(s, per[s]['scode'])
    d = np.abs(p - pos_flat).sum(-1)
    out['self_consistency'][s] = dict(
        median_cells=float(np.median(d)), within2=float((d <= 2).mean()),
        same_region=float((region(layout, p) == region(layout, pos_flat)).mean()),
        cosmax=float(v.mean()))
  pairs = [(a, b) for a, b in zip(steps, steps[1:])]
  if len(steps) > 2:
    pairs.append((steps[0], steps[-1]))
  rng = np.random.RandomState(1)
  for a, b in pairs:
    res = {}
    for kind in ('scode', 'mmode', 'msamp'):
      codes = per[a][kind]
      pa, va = point(a, codes)
      pb, vb = point(b, codes)
      d = np.abs(pa - pb).sum(-1)
      perm = rng.permutation(len(codes))
      dch = np.abs(pa - pb[perm]).sum(-1)       # chance: unrelated codes
      ra, rb = region(layout, pa), region(layout, pb)
      res[kind] = dict(
          n_unique=int(len(np.unique(codes.argmax(-1), axis=0))),
          median_cells=float(np.median(d)), mean_cells=float(d.mean()),
          within2=float((d <= 2).mean()),
          same_region=float((ra == rb).mean()),
          chance_median_cells=float(np.median(dch)),
          chance_same_region=float((ra == rb[perm]).mean()),
          cosmax_a=float(va.mean()), cosmax_b=float(vb.mean()))
    out['pairs'][f'{a}->{b}'] = res
    print(f'  [{out["run"][:5]}] {a}->{b}: state-codes median {res["scode"]["median_cells"]:.1f} cells '
          f'(chance {res["scode"]["chance_median_cells"]:.1f}), mgr-codes '
          f'{res["mmode"]["median_cells"]:.1f} (chance {res["mmode"]["chance_median_cells"]:.1f})',
          flush=True)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run', action='append', required=True)
  ap.add_argument('--steps', default='')
  ap.add_argument('--platform', default='cuda')
  ap.add_argument('--out', required=True)
  args = ap.parse_args()
  steps = [int(float(s)) for s in args.steps.split(',')] if args.steps else None
  results = []
  for r in args.run:
    print('==', r, flush=True)
    try:
      results.append(analyse_run(r, args.platform, steps))
    except Exception as e:
      import traceback; traceback.print_exc()
      results.append(dict(run=os.path.basename(r), error=repr(e)))
    json.dump(results, open(args.out, 'w'), indent=1)
  print('wrote', args.out)


if __name__ == '__main__':
  main()
