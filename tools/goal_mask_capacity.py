"""Offline learnability check for goal-space masking (goal_mask8).

Trains the Director-arm goal autoencoder (3x1024 SiLU+RMS MLPs, 8x8 one-hot
code with straight-through sampling, MSE-sum reconstruction, KL to uniform
held near 10 nats like ``goal_kl``) plus the 8-way keep head on real ``deter``
states from a run's replay, with and without masking. Answers one question
before spending GPU time: can a 24-bit code carry the keep pattern AND the kept
values? Reports mask accuracy and kept-dim reconstruction against the
unmasked baseline.

  python tools/goal_mask_capacity.py <run_dir> [steps]
"""
import glob
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, __file__.rsplit('/tools/', 1)[0])
from dreamerv3.hrl import goal_mask  # noqa: E402

L, C, G = 8, 8, 8


def load(run_dir, n=40000):
  fs = sorted(glob.glob(run_dir + '/logdir/replay/*.npz'))[-400:]
  xs = [np.load(f)['dyn/deter'][::4] for f in fs]
  x = np.concatenate(xs).astype(np.float32)
  return x[np.random.RandomState(0).permutation(len(x))[:n]]


def init_mlp(key, sizes):
  ps = []
  for a, b in zip(sizes[:-1], sizes[1:]):
    key, k = jax.random.split(key)
    ps.append({'w': jax.random.truncated_normal(k, -2, 2, (a, b)) / np.sqrt(a),
               'b': jnp.zeros(b), 'g': jnp.ones(b)})
  return ps


def mlp(ps, x, out_scale=0.1):
  for i, p in enumerate(ps):
    x = x @ p['w'] + p['b']
    if i < len(ps) - 1:
      x = x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + 1e-6) * p['g']
      x = jax.nn.silu(x)
  return x


def make(key, D):
  k1, k2, k3 = jax.random.split(key, 3)
  return {'enc': init_mlp(k1, [D, 1024, 1024, 1024, L * C]),
          'dec': init_mlp(k2, [L * C, 1024, 1024, 1024, D]),
          'mdec': init_mlp(k3, [L * C, 256, 256, G]),
          'logkl': jnp.zeros(())}


def loss_fn(p, x, key, masked):
  k1, k2 = jax.random.split(key)
  B, D = x.shape
  keep = (goal_mask.sample_train_mask(k1, (B,), G, 0.5) if masked
          else jnp.ones((B, G)))
  dk = goal_mask.expand_groups(keep, D)
  logits = mlp(p['enc'], x * dk).reshape(B, L, C) * 1.0
  probs = jax.nn.softmax(logits, -1)
  idx = jax.random.categorical(k2, logits)
  hard = jax.nn.one_hot(idx, C)
  code = (hard + probs - jax.lax.stop_gradient(probs)).reshape(B, L * C)
  pred = mlp(p['dec'], code)
  rec = (jnp.square(pred - x) * dk).sum(-1)
  kl = (probs * (jnp.log(probs + 1e-8) - jnp.log(1.0 / C))).sum((-1, -2))
  scale = jnp.exp(p['logkl'])
  mlogit = mlp(p['mdec'], code)
  mloss = -(keep * jax.nn.log_sigmoid(mlogit)
            + (1 - keep) * jax.nn.log_sigmoid(-mlogit)).sum(-1)
  total = rec + jax.lax.stop_gradient(scale) * kl + (mloss if masked else 0.0)
  # Director encdec_kl AutoAdapt: grow the KL weight above target 10, shrink below.
  adapt = -p['logkl'] * jax.lax.stop_gradient(jnp.clip(kl.mean() - 10.0, -1, 1)) * 0.01
  acc = (goal_mask.hard_keep(mlogit) == keep).mean()
  full = (goal_mask.hard_keep(mlogit) == keep).all(-1).mean()
  # Kept-dim error per kept dim, comparable across masked/unmasked.
  rec_dim = (jnp.square(pred - x) * dk).sum() / dk.sum()
  return total.mean() + adapt, dict(rec_dim=rec_dim, kl=kl.mean(), mloss=mloss.mean(),
                                    acc=acc, full=full)


def run(x, masked, steps, B=256):
  p = make(jax.random.PRNGKey(0), x.shape[1])
  opt = optax.chain(optax.clip_by_global_norm(1000.0), optax.adam(3e-4))
  st = opt.init(p)

  @jax.jit
  def step(p, st, xb, key):
    (l, m), g = jax.value_and_grad(loss_fn, has_aux=True)(p, xb, key, masked)
    up, st = opt.update(g, st, p)
    return optax.apply_updates(p, up), st, m

  rng = np.random.RandomState(1)
  key = jax.random.PRNGKey(2)
  t0 = time.time()
  for i in range(steps + 1):
    key, k = jax.random.split(key)
    p, st, m = step(p, st, jnp.asarray(x[rng.randint(0, len(x), B)]), k)
    if i % (steps // 8) == 0:
      m = {k: float(v) for k, v in m.items()}
      print(f'  masked={masked} step {i:5d} rec/dim {m["rec_dim"]:.4f} kl {m["kl"]:5.1f} '
            f'mloss {m["mloss"]:.2f} acc {m["acc"]:.3f} full {m["full"]:.3f} '
            f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
  x = load(sys.argv[1])
  steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
  print('states', x.shape, 'var/dim', float(x.var(0).mean()))
  run(x, True, steps)
  run(x, False, steps)
