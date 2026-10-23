#!/usr/bin/env python3
"""Offline weight sweep for the SOM evenness loss (hrl/goal_ae.py evenness_loss).

The evenness weight trades evenly spaced class steps against reconstruction
inside the goal autoencoder, which can be trained on its own. So instead of
full agent runs:

  collect  roll out a trained OG+SOM run's own policy in its task and store
           the world model's deter states (the goal autoencoder's input):
             python even_sweep.py collect --run_dir <run> --out <states.npz>
  fit      train a FRESH SOM-line goal autoencoder (same architecture and
           goal_vq settings as that run) on those states, once per weight
           pair, and record reconstruction, codebook use and the class-step
           geometry every --eval_every steps:
             python even_sweep.py fit --states <states.npz> --run_dir <run> \
                 --out <result.npz> --weights 0,0 1,1 10,10

Adam at a fixed lr, the same for every weight, so the comparison is between
weights rather than an attempt to reproduce the online schedule. The state
pool is fixed, which the online run's is not; the full batch covers that.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np


def collect(run_dir, out, want=32768, n_envs=8, stride=4):
  from diag_goal_geometry import build, collect_states
  config, agent = build(run_dir)
  deters = collect_states(config, agent, n_envs, stride, want)
  np.savez_compressed(out, deter=deters.astype(np.float16), task=str(config.task))
  print('collected', deters.shape, 'from', config.task)


def geometry(goal):
  """goal (n, L, C, G) -> step CV, end ratio, block ratio, order (Spearman)."""
  g = goal.astype(np.float64)
  nrm = np.linalg.norm(g, axis=-1)
  dot = np.einsum('nlcg,nleg->nlce', g, g)
  m = np.maximum(nrm[..., :, None], nrm[..., None, :]) + 1e-12
  M = (1 - dot / (m * m)).mean(0)                                   # (L, C, C)
  L, C, _ = M.shape
  steps = np.array([[M[l, c, c + 1] for c in range(C - 1)] for l in range(L)])
  block = steps.mean(1)
  gap = np.abs(np.arange(C)[:, None] - np.arange(C)[None])
  off = ~np.eye(C, dtype=bool)
  rho = []
  for l in range(L):
    for c in range(C):
      x, y = gap[c][off[c]], M[l, c][off[c]]
      rx = np.array([x[x < v].size + (x == v).sum() / 2 for v in x])
      ry = np.argsort(np.argsort(y))
      rho.append(np.corrcoef(rx, ry)[0, 1])
  return dict(step_cv=steps.std() / steps.mean(),
              within_cv=np.mean(steps.std(1) / block),
              end_ratio=steps[:, [0, C - 2]].mean() / steps.mean(),
              block_ratio=block.max() / block.min(),
              order=np.nanmean(rho), step_mean=steps.mean(), M=M)


def fit(states, run_dir, out, weights, steps=50000, batch=1024, lr=1e-4,
        eval_every=5000, seed=0):
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  import optax
  import elements
  import embodied.jax.nets as nets
  from dreamerv3.hrl import goal_ae
  from diag_goal_struct_corr import load_config

  nets.COMPUTE_DTYPE = jnp.float32
  config = load_config(run_dir)
  a = config.agent
  cfg = a.goal_vq
  L, C = [int(x) for x in a.skill_shape]
  data = np.load(states)['deter'].astype(np.float32)
  rng = np.random.default_rng(seed)
  perm = rng.permutation(len(data))
  test, train = data[perm[:2048]], data[perm[2048:]]
  G = data.shape[-1]
  print(f'{config.task}: {len(train)} train / {len(test)} test states, deter {G}, L={L} C={C}')

  results = {}
  for w_in, w_ac in weights:
    dec = goal_ae.GoalVQDecoder(
        (G,), L, C, int(cfg.dim), stddev=float(cfg.stddev),
        include_self=bool(cfg.include_self), topology=str(cfg.topology),
        **a.goal_vq_dec, name='goal_dec')
    enc = goal_ae.GoalVQEncoder(dec.codebook, L, int(cfg.dim), temp=float(cfg.temp),
                                **a.goal_vq_enc, name='goal_enc')
    kw = dict(som=bool(cfg.som), ste=True, agg=str(cfg.agg),
              codebook_scale=float(cfg.codebook_scale), commit_scale=float(cfg.commit_scale),
              som_scale=float(cfg.som_scale), even_within=w_in, even_across=w_ac)

    def loss_fn(x):
      loss, mets = goal_ae.vq_goal_loss(enc, dec, x[None], 2, **kw)
      mets.pop('vq/_rec_bt')
      return loss.mean(), mets

    def sweep(x):
      z_q = dec.codebook.quantize(enc.latent(x[None], 2))['z_q'][0][:64]
      table = dec.codebook.table()
      eye = jnp.eye(L)[:, None, :, None]
      z = z_q[:, None, None] * (1 - eye)[None] + (eye * table[:, :, None, :])[None]
      return dec.from_latent(z, 3).pred()

    x0 = jnp.asarray(train[:batch])
    params = nj.init(lambda x: (loss_fn(x), sweep(x)))({}, x0, seed=seed)
    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
    ostate = opt.init(params)

    @jax.jit
    def step(params, ostate, x):
      (loss, mets), grads = jax.value_and_grad(
          lambda p: nj.pure(loss_fn)(p, x)[1], has_aux=True)(params)
      upd, ostate = opt.update(grads, ostate, params)
      return optax.apply_updates(params, upd), ostate, mets

    evalf = jax.jit(lambda p, x: nj.pure(loss_fn)(p, x)[1][1])
    sweepf = jax.jit(lambda p, x: nj.pure(sweep)(p, x)[1])
    rows = []
    for i in range(steps + 1):
      if i % eval_every == 0:
        m = {k: float(v) for k, v in evalf(params, jnp.asarray(test)).items()}
        geo = geometry(np.asarray(sweepf(params, jnp.asarray(test[:64]))))
        rows.append(dict(step=i, rec=m['vq/rec_q'], perplexity=m.get('vq/perplexity', np.nan),
                         **{k: v for k, v in geo.items() if k != 'M'}))
        r = rows[-1]
        print(f'  w=({w_in},{w_ac}) step {i:6d} rec {r["rec"]:8.3f} perp {r["perplexity"]:.2f} '
              f'step_cv {r["step_cv"]:.3f} within_cv {r["within_cv"]:.3f} end {r["end_ratio"]:.2f} '
              f'blocks {r["block_ratio"]:.2f} order {r["order"]:.3f}', flush=True)
        last_M = geo['M']
      if i == steps:
        break
      x = jnp.asarray(train[rng.integers(0, len(train), batch)])
      params, ostate, _ = step(params, ostate, x)
    results[f'{w_in},{w_ac}'] = dict(
        rows={k: np.array([r[k] for r in rows]) for k in rows[0]}, M=last_M)
  np.savez_compressed(out, task=str(config.task), run=str(run_dir),
                      **{f'{k}|{m}': v for k, d in results.items()
                         for m, v in [*d['rows'].items(), ('M', d['M'])]})
  print('wrote', out)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('mode', choices=['collect', 'fit'])
  p.add_argument('--run_dir', required=True)
  p.add_argument('--out', required=True)
  p.add_argument('--states')
  p.add_argument('--weights', nargs='+', default=['0,0', '0.1,0.1', '1,1', '10,10', '1,0'])
  p.add_argument('--steps', type=int, default=50000)
  p.add_argument('--lr', type=float, default=1e-4)
  a = p.parse_args()
  if a.mode == 'collect':
    collect(a.run_dir, a.out)
  else:
    w = [tuple(float(v) for v in s.split(',')) for s in a.weights]
    fit(a.states, a.run_dir, a.out, w, steps=a.steps, lr=a.lr)


if __name__ == '__main__':
  main()
