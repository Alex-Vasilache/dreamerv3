"""Goal-code <-> goal-space geometry, measured five ways from one checkpoint.

`diag_goal_struct_corr.py` answers "does code similarity track goal similarity"
under `cosine_max`. That is the right question for Director's worker, whose
reward IS `cosine_max(goal, deter)`, but it is the wrong question for a
Lipschitz claim, and it says nothing about the SHAPE of the map. This adds:

  1. CORRELATION, under two geometries.
     `cosine_max` (what the worker is rewarded on, and what motivation.tex
     Fig. goal_struct_corr plots) and Euclidean DISTANCE (what the Lipschitz
     condition is actually written in: ||dec(z1)-dec(z2)|| <= c ||z1-z2||).
     A code can track one and not the other -- cosine_max is scale-free, so it
     is blind to a decoder that preserves directions while stretching lengths,
     which is exactly the failure a Lipschitz bound exists to prevent. Both are
     reported, with Pearson r and Spearman rho (the relation is expected to be
     monotone but not linear, and rho does not assume linearity).

  2. THE EMPIRICAL LIPSCHITZ CONSTANT, which needs no correlation at all.
     For every pair, the ratio ||dec(z_i) - dec(z_j)|| / ||z_i - z_j|| for the
     decoder and ||z_e(s_i) - z_e(s_j)|| / ||s_i - s_j|| for the encoder. Their
     maximum IS the constant the penalty claims to shrink, so this is the
     direct test of whether the `_prod` arms' penalty did anything, rather than
     an inference from `goal/lip_bound_max` (a bound, not the realized value).

  3. THE CODE->GOAL LANDSCAPE.
     Take a state's code, walk ONE block through all C classes, decode each,
     and measure how far the decoded goal moved. Plotted against index distance
     |c - c_ref| this is flat for Director (entry indices are arbitrary labels)
     and should ramp monotonically for a SOM on a line, where adjacent entries
     are trained to be neighbours. This is the property the architecture is for,
     stated as directly as it can be stated.

  4. THE SAME IN TWO DIMENSIONS.
     Sweep two blocks jointly over the full C x C lattice, decode all of them,
     and project the decoded goals to 2D. A structure-preserving code draws a
     recognizable lattice; an arbitrary one draws a tangle. This is the figure
     that shows what the numbers mean.

  5. THE GOAL->CODE LANDSCAPE.
     Walk a straight line in goal space between two real states in equal steps
     and watch the code: a Lipschitz encoder gives a smooth ramp in latent
     distance and blocks that flip a few at a time, an unconstrained one gives
     a step function. Plus a perturbation sweep -- add noise of growing size to
     a state and measure the code's response -- whose slope near zero is the
     local Lipschitz constant of the encoder.

Works on both goal-AE families: `goal_ae_impl: vq` (the e510-e549 arms) and the
Director categorical head (e502-e509), so the baseline sits on the same axes.
States are collected from a live rollout exactly as diag_goal_struct_corr.py
does. Nothing here touches the training code path -- the model is only ever
called forward, through `nj.pure` with the restored parameters.

  python -u diag_goal_geometry.py --run_dir /work/.../e510_..._j4676965 \
      --out results/e510.npz
  # a specific milestone rather than the latest checkpoint:
  python -u diag_goal_geometry.py --run_dir ... \
      --ckpt_path .../logdir/ckpt_milestones/000002000000
"""
import argparse
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent /
                      'goal_code_struct_corr'))

from functools import partial as bind

import elements
import embodied
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import dreamerv3.main as m
from diag_goal_struct_corr import load_config, pairwise_cosmax, pearson_offdiag
from dreamerv3.hrl.heads import _head_inner


# ---------------------------------------------------------------- statistics

def pairwise_dist(x):
  """Euclidean distance matrix, the geometry the Lipschitz condition uses."""
  x = np.asarray(x, np.float64)
  sq = (x * x).sum(-1)
  d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
  return np.sqrt(np.maximum(d2, 0.0))


def offdiag(a):
  n = a.shape[0]
  return a[~np.eye(n, dtype=bool)]


def spearman_offdiag(a, b):
  """Pearson on ranks. No scipy in this env, and rank correlation is the
  honest summary here: the code-vs-goal relation is expected to be monotone,
  not linear, and Pearson understates a monotone curve."""
  af, bf = offdiag(a), offdiag(b)
  ar = np.argsort(np.argsort(af)).astype(np.float64)
  br = np.argsort(np.argsort(bf)).astype(np.float64)
  ac, bc = ar - ar.mean(), br - br.mean()
  denom = np.sqrt((ac * ac).sum()) * np.sqrt((bc * bc).sum()) + 1e-12
  return float((ac * bc).sum() / denom)


def ratio_stats(num, den, eps=1e-9):
  """Distribution of ||f(x)-f(y)|| / ||x-y|| over pairs.

  The max is the empirical Lipschitz constant. It is also the least robust
  statistic there is -- one near-duplicate pair with a tiny denominator sets
  it -- so the percentiles are what should be read, and pairs closer than a
  floor are dropped rather than allowed to dominate."""
  num, den = offdiag(num), offdiag(den)
  keep = den > max(eps, np.percentile(den, 1.0))
  r = num[keep] / den[keep]
  return dict(max=float(r.max()), p999=float(np.percentile(r, 99.9)),
              p99=float(np.percentile(r, 99.0)), p50=float(np.median(r)),
              mean=float(r.mean()), n=int(r.size))


def index_distance(ids, topology, classes):
  """Per-pair mean distance between entry indices, on the codebook's own
  topology. A ring wraps; a line does not; Director has no topology at all and
  the quantity is meaningless there -- which is the point of measuring it."""
  diff = np.abs(ids[:, None, :].astype(np.int64) - ids[None, :, :])
  if topology == 'ring':
    diff = np.minimum(diff, classes - diff)
  return diff.mean(-1)


def pca2(x):
  """Two leading principal components, for the lattice figure."""
  x = np.asarray(x, np.float64)
  x = x - x.mean(0, keepdims=True)
  _, s, vt = np.linalg.svd(x, full_matrices=False)
  return x @ vt[:2].T, (s[:2] ** 2) / max((s ** 2).sum(), 1e-12)


# ------------------------------------------------------------------- model

def build(run_dir, ckpt_path=None):
  config = load_config(run_dir)
  agent = m.make_agent(config)
  cp = elements.Checkpoint(directory=elements.Path(config.logdir) / 'ckpt')
  cp.agent = agent
  if ckpt_path:
    cp.load(path=elements.Path(ckpt_path), keys=['agent'])
  else:
    cp.load(keys=['agent'])
  # embodied sets jax_transfer_guard='disallow'; everything below pulls arrays
  # back to the host for numpy.
  jax.config.update('jax_transfer_guard', 'allow')
  return config, agent


def make_fns(config, agent):
  """Forward-only views of the goal autoencoder, for both implementations.

  encode: deter -> (z_e, z_q, ids, onehot, soft)
    z_e is the continuous pre-discretization latent. The quantized arms have
    one by construction; for the Director head the analogous quantity is the
    categorical logits, which is what is returned there, flagged by `impl` in
    the output so no one reads the two as the same object.
  decode: onehot code -> goal vector (the deter-space target the worker chases)
  """
  model = agent.model
  impl = str(getattr(config.agent, 'goal_ae_impl', 'director'))

  def encode(d):
    if impl == 'vq':
      z_e = model.goal_enc.latent(d, 1)
      q = model.goal_dec.codebook.quantize(z_e)
      code = model.goal_enc.code_from_latent(z_e)
      soft = jax.nn.softmax(_head_inner(code).dist.logits, -1)
      return z_e, q['z_q'], q['ids'], q['onehot'], soft
    enc = model.goal_enc(d, 1)
    head = enc['skill'] if isinstance(enc, dict) else enc
    onehot = head.pred()
    logits = _head_inner(head).dist.logits
    soft = jax.nn.softmax(logits, -1)
    return logits, onehot, jnp.argmax(onehot, -1), onehot, soft

  def decode(code):
    return model.goal_dec(code, 1).pred()

  def call(fn, *args):
    return [np.asarray(x) for x in nj.pure(fn)(
        agent.params, *[jnp.asarray(a) for a in args])[1]]

  return impl, (lambda d: call(encode, d)), (lambda c: call(
      lambda x: (decode(x),), c)[0])


def collect_states(config, agent, n_envs, stride, want):
  """A pool of deter states from a live rollout, thinned within each worker."""
  steps = n_envs * stride * (-(-want // n_envs))
  driver = embodied.Driver(
      [bind(m.make_env, config, i) for i in range(n_envs)],
      parallel=(n_envs > 1))
  driver.reset(agent.init_policy)
  by_env = [[] for _ in range(n_envs)]
  driver.on_step(lambda tran, w: 'log/struct_diag_deter' in tran and
                 by_env[w].append(
                     np.asarray(tran['log/struct_diag_deter']).copy()))
  driver(agent.policy, steps=steps)
  driver.close()
  pool = [np.stack(d, 0)[::stride] for d in by_env if d]
  assert pool, 'no states collected -- is policy_struct_diag reaching policy()?'
  return np.concatenate(pool, 0)


# ------------------------------------------------------------ measurements

def correlations(deters, z_e, z_q, ids, soft, topology, classes):
  """Section 1 and 2: correlation under both geometries, plus the constants."""
  sim_goal = pairwise_cosmax(deters)
  dist_goal = pairwise_dist(deters)

  n = len(ids)
  sim_code = {
      'soft': pairwise_cosmax(soft.reshape(n, -1)),
      'hard': 1.0 - (ids[:, None, :] != ids[None, :, :]).mean(-1),
      'embed': pairwise_cosmax(z_q.reshape(n, -1)),
  }
  idx = index_distance(ids, topology, classes)
  half = classes / 2.0 if topology == 'ring' else float(classes - 1)
  sim_code['index'] = 1.0 - idx / half

  dist_code = {
      'embed': pairwise_dist(z_q.reshape(n, -1)),
      'latent': pairwise_dist(z_e.reshape(n, -1)),
      'index': idx,
  }

  out = {}
  for key, sc in sim_code.items():
    out[f'cosmax/{key}/pearson'] = pearson_offdiag(sim_goal, sc)
    out[f'cosmax/{key}/spearman'] = spearman_offdiag(sim_goal, sc)
  for key, dc in dist_code.items():
    out[f'dist/{key}/pearson'] = pearson_offdiag(dist_goal, dc)
    out[f'dist/{key}/spearman'] = spearman_offdiag(dist_goal, dc)
  return out, sim_goal, dist_goal, sim_code, dist_code


def lipschitz(deters, z_e, z_q, decoded):
  """The realized constants, encoder and decoder, over the sampled pairs."""
  n = len(deters)
  d_state = pairwise_dist(deters)
  d_latent = pairwise_dist(z_e.reshape(n, -1))
  d_embed = pairwise_dist(z_q.reshape(n, -1))
  d_goal = pairwise_dist(decoded)
  return {
      'enc': ratio_stats(d_latent, d_state),   # state -> continuous latent
      'dec': ratio_stats(d_goal, d_embed),     # codebook embedding -> goal
  }


def code_sweep(deters, onehot, decode, blocks, classes, n_ref, rng):
  """Section 3: move one block to every class and watch the decoded goal.

  Returns (n_ref, L, C) displacements, aligned so that column c holds the
  distance from the reference goal to the goal obtained by forcing that block
  to class c, and (n_ref, L) reference class indices to measure |c - c_ref|
  against.
  """
  pick = rng.choice(len(deters), size=min(n_ref, len(deters)), replace=False)
  base = onehot[pick]                                     # (M, L, C)
  ref_goal = decode(base)                                 # (M, D)
  ref_ids = base.argmax(-1)                               # (M, L)
  disp = np.zeros((len(pick), blocks, classes), np.float64)
  for b in range(blocks):
    for c in range(classes):
      mod = base.copy()
      mod[:, b, :] = 0.0
      mod[:, b, c] = 1.0
      disp[:, b, c] = np.linalg.norm(decode(mod) - ref_goal, axis=-1)
  return disp, ref_ids, np.linalg.norm(ref_goal, axis=-1)


def grid_sweep(deters, onehot, decode, classes, blocks_pair, rng):
  """Section 4: the full C x C lattice for two blocks, projected to 2D."""
  i = int(rng.integers(len(deters)))
  base = onehot[i:i + 1]
  b0, b1 = blocks_pair
  codes = np.repeat(base, classes * classes, axis=0)
  for a in range(classes):
    for b in range(classes):
      k = a * classes + b
      codes[k, b0, :] = 0.0; codes[k, b0, a] = 1.0
      codes[k, b1, :] = 0.0; codes[k, b1, b] = 1.0
  goals = decode(codes)
  xy, var = pca2(goals)
  return xy.reshape(classes, classes, 2), var


def interp_sweep(deters, encode, n_pairs, n_steps, rng):
  """Section 5a: a straight walk in goal space, watched in code space."""
  idx = rng.choice(len(deters), size=(n_pairs, 2), replace=False)
  ts = np.linspace(0.0, 1.0, n_steps)
  lat = np.zeros((n_pairs, n_steps), np.float64)
  flips = np.zeros((n_pairs, n_steps), np.float64)
  for p, (a, b) in enumerate(idx):
    path = (1.0 - ts)[:, None] * deters[a] + ts[:, None] * deters[b]
    z_e, _, ids, _, _ = encode(path.astype(deters.dtype))
    lat[p] = np.linalg.norm(
        (z_e - z_e[0]).reshape(n_steps, -1), axis=-1)
    flips[p] = (ids != ids[0]).mean(-1)
  return ts, lat, flips


def perturb_sweep(deters, encode, scales, n_ref, rng):
  """Section 5b: response of the code to a perturbation of known size.

  The slope of latent displacement against perturbation size, as the size goes
  to zero, is the encoder's local Lipschitz constant.
  """
  pick = rng.choice(len(deters), size=min(n_ref, len(deters)), replace=False)
  base = deters[pick]
  z0, _, ids0, _, _ = encode(base)
  scale = np.linalg.norm(base, axis=-1).mean()
  lat = np.zeros(len(scales))
  flips = np.zeros(len(scales))
  for k, eps in enumerate(scales):
    noise = rng.normal(size=base.shape).astype(base.dtype)
    noise /= (np.linalg.norm(noise, axis=-1, keepdims=True) + 1e-12)
    pert = base + (eps * scale) * noise
    z1, _, ids1, _, _ = encode(pert.astype(base.dtype))
    lat[k] = np.linalg.norm(
        (z1 - z0).reshape(len(base), -1), axis=-1).mean()
    flips[k] = (ids1 != ids0).mean()
  return np.asarray(scales) * scale, lat, flips


# ------------------------------------------------------------------- driver

def run(run_dir, out_path, ckpt_path=None, n_envs=8, stride=8, n_states=1024,
        n_batches=4, n_ref=64, n_pairs=16, n_steps=33, seed=0):
  config, agent = build(run_dir, ckpt_path)
  impl, encode, decode = make_fns(config, agent)
  blocks, classes = [int(x) for x in config.agent.skill_shape]
  topology = str(getattr(config.agent.goal_vq, 'topology', 'ring')) \
      if impl == 'vq' else 'none'
  name = pathlib.Path(str(run_dir).rstrip('/')).name
  print(f'=== {name} | impl={impl} topology={topology} L={blocks} C={classes}')

  deters = collect_states(config, agent, n_envs, stride, n_states * n_batches)
  print(f'collected {len(deters)} states, D={deters.shape[-1]}')

  rng = np.random.default_rng(seed)
  perm = rng.permutation(len(deters))
  res = {}

  per_batch = []
  lip_batch = []
  keep = None
  for b in range(n_batches):
    idx = perm[b * n_states:(b + 1) * n_states]
    if len(idx) < n_states:
      break
    d = deters[idx]
    z_e, z_q, ids, onehot, soft = encode(d)
    corr, sim_goal, dist_goal, sim_code, dist_code = correlations(
        d, z_e, z_q, ids, soft, topology, classes)
    per_batch.append(corr)
    lip_batch.append(lipschitz(d, z_e, z_q, decode(onehot)))
    if keep is None:  # one batch's raw matrices, for the scatter figures
      sub = rng.permutation(n_states)[:512]
      keep = dict(
          sim_goal=sim_goal[np.ix_(sub, sub)],
          dist_goal=dist_goal[np.ix_(sub, sub)],
          **{f'sim_{k}': v[np.ix_(sub, sub)] for k, v in sim_code.items()},
          **{f'dist_{k}': v[np.ix_(sub, sub)] for k, v in dist_code.items()})

  for key in per_batch[0]:
    vals = np.array([p[key] for p in per_batch])
    res[f'corr/{key}/mean'] = vals.mean()
    res[f'corr/{key}/std'] = vals.std()
    res[f'corr/{key}/per_batch'] = vals
  for side in ('enc', 'dec'):
    for stat in ('max', 'p999', 'p99', 'p50', 'mean'):
      vals = np.array([l[side][stat] for l in lip_batch])
      res[f'lip/{side}/{stat}/mean'] = vals.mean()
      res[f'lip/{side}/{stat}/std'] = vals.std()

  print('\n  correlation (mean +/- std over batches)')
  for key in sorted(k for k in per_batch[0]):
    print(f'    {key:28s} {res[f"corr/{key}/mean"]:+.3f} '
          f'+/- {res[f"corr/{key}/std"]:.3f}')
  print('\n  empirical Lipschitz ratio')
  for side in ('enc', 'dec'):
    print(f'    {side}: p50={res[f"lip/{side}/p50/mean"]:.4g} '
          f'p99={res[f"lip/{side}/p99/mean"]:.4g} '
          f'max={res[f"lip/{side}/max/mean"]:.4g}')

  idx = perm[:n_states]
  d0 = deters[idx]
  _, _, _, onehot0, _ = encode(d0)
  disp, ref_ids, ref_norm = code_sweep(
      d0, onehot0, decode, blocks, classes, n_ref, rng)
  res['sweep/disp'] = disp
  res['sweep/ref_ids'] = ref_ids
  res['sweep/ref_norm'] = ref_norm

  # Displacement as a function of index distance, the headline curve: flat
  # means the index is a label, a ramp means it is a coordinate.
  dist_axis = np.arange(classes)
  curve = np.full((classes,), np.nan)
  for k in range(classes):
    m_ = np.zeros_like(disp, bool)
    for b in range(blocks):
      dd = np.abs(np.arange(classes)[None, :] - ref_ids[:, b:b + 1])
      if topology == 'ring':
        dd = np.minimum(dd, classes - dd)
      m_[:, b, :] = dd == k
    if m_.any():
      curve[k] = disp[m_].mean()
  res['sweep/index_distance'] = dist_axis
  res['sweep/curve'] = curve
  print('\n  decoded-goal displacement vs index distance')
  print('    ' + '  '.join(f'{k}:{v:.3g}' for k, v in
                           zip(dist_axis, curve) if v == v))

  xy, var = grid_sweep(d0, onehot0, decode, classes, (0, 1), rng)
  res['grid/xy'] = xy
  res['grid/var'] = var

  ts, lat, flips = interp_sweep(d0, encode, n_pairs, n_steps, rng)
  res['interp/t'] = ts
  res['interp/latent'] = lat
  res['interp/flips'] = flips

  scales = np.geomspace(1e-3, 1.0, 13)
  eps, plat, pflips = perturb_sweep(d0, encode, scales, n_ref, rng)
  res['perturb/eps'] = eps
  res['perturb/latent'] = plat
  res['perturb/flips'] = pflips
  slope = plat[1] / max(eps[1], 1e-12)
  res['perturb/slope_small'] = slope
  print(f'\n  encoder local slope at eps={eps[1]:.3g}: {slope:.4g}')

  meta = dict(name=name, impl=impl, topology=topology, blocks=blocks,
              classes=classes, task=str(config.task), run_dir=str(run_dir),
              ckpt=str(ckpt_path or 'latest'))
  out = pathlib.Path(out_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(str(out), meta=np.array(str(meta)), **keep, **res)
  print(f'\nwrote {out_path}')
  return res


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run_dir', required=True)
  p.add_argument('--out', required=True)
  p.add_argument('--ckpt_path', default=None)
  p.add_argument('--n_envs', type=int, default=8)
  p.add_argument('--stride', type=int, default=8)
  p.add_argument('--n_states', type=int, default=1024)
  p.add_argument('--n_batches', type=int, default=4)
  p.add_argument('--n_ref', type=int, default=64)
  p.add_argument('--seed', type=int, default=0)
  a = p.parse_args()
  run(a.run_dir, a.out, ckpt_path=a.ckpt_path, n_envs=a.n_envs,
      stride=a.stride, n_states=a.n_states, n_batches=a.n_batches,
      n_ref=a.n_ref, seed=a.seed)


if __name__ == '__main__':
  main()
