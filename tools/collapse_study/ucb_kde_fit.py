"""Offline fit of the full-resolution imagination UCB (hrl/explore.py ImagKDE).

Widths h_goal / h_state and the tilt strength c are chosen on finished V+SOM
runs before any training run, from what the tilt would see:

  states   world-model states from the run's own policy (collect_states, as in
           the goal-geometry tools), stoch taken as the prior's mode;
  buffer   picks the manager would make: start-state code (goal-encoder mode)
           + one policy sample per state, BUF rows, scaled to the true decayed
           pick mass (MASS, the 'mass' the table arms logged at steady state);
  probes   held-out states, K policy samples each = the tilt's candidates.

Per (h_goal, h_state): candidate counts, bonus 1/sqrt(n+1), and the spread of
the bonus among one state's candidates (what the tilt can act on). Per c: the
Gumbel tilt's expected log-prob shift and novelty gain, computed exactly from
the softmax over each state's candidates.

  collect: python tools/collapse_study/ucb_kde_fit.py collect <run_dir> <ckpt|final> <out.npz>
  metric:  python tools/collapse_study/ucb_kde_fit.py metric <run_dir> <ckpt|final> <..._metric.npz>
  fit:     python tools/collapse_study/ucb_kde_fit.py fit <out_dir>
"""
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K, BUF, PROBE = 32, 131072, 2048
MASS = 9.2e7          # steady-state decayed pick mass (2048 picks/update, 0.5M half-life)
H_GOAL = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
H_STATE = [0.5, 1.0, 2.0, 4.0, 8.0]
CS = [1.0, 3.0, 10.0]


def collect(run_dir, ckpt, out):
  sys.path.insert(0, os.path.join(REPO, 'experiments', 'goal_geometry'))
  from diag_goal_geometry import build, collect_states
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  from dreamerv3.hrl.manager import mgr_as_dict
  ck = None if ckpt == 'final' else os.path.join(run_dir, 'logdir', 'ckpt_milestones', ckpt)
  config, agent = build(run_dir, ck)
  n_states = BUF // 4 + PROBE
  cache = out.replace('.npz', '_deter.npy')
  if os.path.exists(cache):
    deter = np.load(cache)
  else:
    deter = collect_states(config, agent, 16, 4, n_states)[:n_states]
    np.save(cache, deter)
  model = agent.model
  import embodied.jax.nets as nets
  cdt = nets.COMPUTE_DTYPE

  def fn(d, nsamp):
    d = d.astype(cdt)
    logit = model.dyn._prior(d)
    stoch = jax.nn.one_hot(jnp.argmax(logit, -1), logit.shape[-1], dtype=cdt)
    feat = dict(deter=d, stoch=stoch)
    dist = mgr_as_dict(model.manager_pol(model.feat2tensor(feat), 1))['skill']
    samples = [dist.sample(nj.seed()) for _ in range(nsamp)]
    goal = jnp.stack([jnp.argmax(s, -1) for s in samples], 1)          # (N, k, L)
    logp = jnp.stack([dist.logp(s) for s in samples], 1)               # (N, k)
    enc = model.goal_enc(d, 1)
    code = enc['skill'] if isinstance(enc, dict) else enc
    start = jnp.argmax(code.pred(), -1)                                # (N, L)
    return goal, logp, start

  res = {}
  for name, sl, k in (('buf', slice(0, BUF // 4), 4), ('probe', slice(BUF // 4, n_states), K)):
    outs = []
    for i in range(sl.start, sl.stop, 1024):
      d = jnp.asarray(deter[i:min(i + 1024, sl.stop)])
      outs.append([np.asarray(x) for x in nj.pure(lambda d: fn(d, k))(
          agent.params, d, seed=jax.random.PRNGKey(i))[1]])
    res[name] = [np.concatenate(z, 0) for z in zip(*outs)]
  g, _, s = res['buf']
  bufrows = np.concatenate([np.repeat(s[:, None], 4, 1), g], -1).reshape(-1, 2 * s.shape[-1])
  pg, plogp, ps = res['probe']
  np.savez_compressed(out, buf=bufrows.astype(np.int8), probe_goal=pg.astype(np.int8),
                      probe_logp=plogp.astype(np.float32), probe_start=ps.astype(np.int8),
                      task=str(config.task), run=os.path.basename(run_dir), ckpt=ckpt)
  print('wrote', out, bufrows.shape, pg.shape)


def metric(run_dir, ckpt, out, n_start=32, n_base=8, steps=8):
  """Which class distance tracks where the worker actually goes?

  Per block l and class pair (c, c'): base codes (uniform random) with block l
  set to c or c', decoded to goals; from real start states (cached deter of
  ``collect``) the worker is imagined ``steps`` steps toward each goal. Stores
    M[l, c, c']  decoded-goal distance (1 - cosine_max), mean over bases;
    O[l, c, c']  outcome distance: 1 - cosine_max between the deter the
                 worker reaches, mean over bases x starts.
  Class-index distance is |c - c'|; measured positions are cumulative M steps.
  """
  sys.path.insert(0, os.path.join(REPO, 'experiments', 'goal_geometry'))
  from diag_goal_geometry import build
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  import embodied.jax.nets as nets
  sys.path.insert(0, os.path.join(REPO, 'dreamerv3'))
  ck = None if ckpt == 'final' else os.path.join(run_dir, 'logdir', 'ckpt_milestones', ckpt)
  config, agent = build(run_dir, ck)
  model = agent.model
  cdt = nets.COMPUTE_DTYPE
  deter = np.load(out.replace('_metric.npz', '_deter.npy'))
  rng = np.random.default_rng(0)
  starts = deter[rng.choice(len(deter), n_start, replace=False)]
  L, C = [int(x) for x in config.agent.skill_shape]

  def cosd(a, b):
    a, b = a.astype(jnp.float32), b.astype(jnp.float32)
    na, nb = jnp.linalg.norm(a, axis=-1), jnp.linalg.norm(b, axis=-1)
    m = jnp.maximum(na, nb) + 1e-8
    return 1 - (a * b).sum(-1) / (m * m)

  def fn(d0, base_ids, l):
    # Goals: (n_base, C) codes, block l swept.
    ids = jnp.repeat(base_ids[:, None], C, 1).at[:, :, l].set(jnp.arange(C)[None])
    code = jax.nn.one_hot(ids, C, dtype=jnp.float32).reshape(-1, L, C)
    goal = model.goal_dec(code, 1).pred()                       # (n_base*C, G)
    G = goal.reshape(n_base, C, -1)
    Md = cosd(G[:, :, None], G[:, None]).mean(0)                # (C, C)
    # Rollouts: every (start, goal) pair.
    d = jnp.repeat(d0.astype(cdt), n_base * C, 0)                # (S*nb*C, D)
    g = jnp.tile(goal, (d0.shape[0], 1))
    logit = model.dyn._prior(d)
    stoch = jax.nn.one_hot(jnp.argmax(logit, -1), logit.shape[-1], dtype=cdt)
    pol = model._worker_policy_fixed_goal(g)
    carry, _, _ = model.dyn.imagine(dict(deter=d, stoch=stoch), pol, steps, False)
    end = carry['deter'].reshape(d0.shape[0], n_base, C, -1)
    Od = cosd(end[:, :, :, None], end[:, :, None]).mean((0, 1))  # (C, C)
    return Md, Od

  M = np.zeros((L, C, C)); O = np.zeros((L, C, C))
  base = jnp.asarray(rng.integers(0, C, (n_base, L)))
  for l in range(L):
    Md, Od = nj.pure(lambda d: fn(d, base, l))(agent.params, jnp.asarray(starts),
                                               seed=jax.random.PRNGKey(l))[1]
    M[l], O[l] = np.asarray(Md), np.asarray(Od)
  np.savez_compressed(out, M=M, O=O, task=str(config.task), run=os.path.basename(run_dir), ckpt=ckpt)
  gap = np.abs(np.arange(C)[:, None] - np.arange(C)[None])
  iu = np.triu_indices(C, 1)
  def rho(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]
  steps_ = np.array([[M[l, c, c + 1] for c in range(C - 1)] for l in range(L)])
  pos = np.concatenate([np.zeros((L, 1)), np.cumsum(steps_ / steps_.mean(), 1)], 1)
  ri = np.mean([rho(gap[iu], O[l][iu]) for l in range(L)])
  rp = np.mean([rho(np.abs(pos[l][:, None] - pos[l][None])[iu], O[l][iu]) for l in range(L)])
  # Across blocks: does a block's decoded strength predict its outcome strength?
  rb = rho(M.mean((1, 2)), O.mean((1, 2)))
  print(f'{os.path.basename(out)}: within-block rank corr with outcome: index {ri:.3f} '
        f'measured {rp:.3f}; across blocks (strength) {rb:.3f}', flush=True)


def fit(out_dir):
  """Grid over (metric, h_goal, h_state) on every collected checkpoint.

  metric 'index' places classes at 0..7; 'measured' at the cumulative decoded
  steps from that checkpoint's ``_metric.npz`` (mean step 1), as
  ``goal_ae.class_positions`` does online."""
  import json
  import jax
  import jax.numpy as jnp
  files = sorted(f for f in glob.glob(os.path.join(out_dir, '*.npz'))
                 if not f.endswith('_metric.npz'))
  rows = []

  @jax.jit
  def counts(x, buf, hs, hg):
    L = x.shape[1] // 2
    def chunk(acc, yb):
      def part(a, b, h):
        d2 = (jnp.square(a).sum(-1)[:, None] + jnp.square(b).sum(-1)[None]
              - 2 * jnp.matmul(a, b.T, precision=jax.lax.Precision.HIGHEST))
        return jnp.maximum(d2, 0) / (2 * h * h)
      e = part(x[:, :L], yb[:, :L], hs) + part(x[:, L:], yb[:, L:], hg)
      return acc + jnp.exp(-e).sum(-1), None
    acc, _ = jax.lax.scan(chunk, jnp.zeros(x.shape[0]), buf.reshape(-1, 4096, 2 * L))
    return acc * (MASS / buf.shape[0])

  for f in files:
    z = np.load(f)
    bufi = z['buf'].astype(np.int64)
    L = bufi.shape[1] // 2
    C = 8
    pg = z['probe_goal'].astype(np.int64)
    ps = np.repeat(z['probe_start'].astype(np.int64)[:, None], K, 1)
    xi = np.concatenate([ps, pg], -1).reshape(-1, 2 * L)
    logp = np.asarray(z['probe_logp'])
    variants = {'index': np.tile(np.arange(C, dtype=np.float32), (L, 1))}
    mf = f.replace('.npz', '_metric.npz')
    if os.path.exists(mf):
      M = np.load(mf)['M']
      st = np.array([[M[l, c, c + 1] for c in range(C - 1)] for l in range(L)])
      variants['measured'] = np.concatenate(
          [np.zeros((L, 1)), np.cumsum(st / st.mean(), 1)], 1).astype(np.float32)
    for metric, pos in variants.items():
      table = np.tile(pos, (2, 1))                                   # (2L, C)
      coord = lambda ids: table[np.arange(2 * L)[None], ids]
      x, buf = jnp.asarray(coord(xi)), jnp.asarray(coord(bufi))
      for hg in H_GOAL:
        for hs in H_STATE:
          n = np.asarray(counts(x, buf, hs, hg)).reshape(-1, K)
          b = 1 / np.sqrt(n + 1)
          r = dict(file=os.path.basename(f), task=str(z['task']), ckpt=str(z['ckpt']),
                   metric=metric, hg=hg, hs=hs, n_med=float(np.median(n)), b_mean=b.mean(),
                   b_spread=b.std(1).mean(), b_range=(b.max(1) - b.min(1)).mean(),
                   frac_novel=(n < 1).mean())
          for c in CS:
            w = np.exp(c * (b - b.max(1, keepdims=True)))
            w /= w.sum(1, keepdims=True)
            r[f'shift_c{c:g}'] = ((w * logp).sum(1) - logp.mean(1)).mean()
            r[f'gain_c{c:g}'] = ((w * b).sum(1) / b.mean(1)).mean()
          rows.append(r)
    print('done', os.path.basename(f), list(variants), flush=True)
  json.dump(rows, open(os.path.join(out_dir, 'fit.json'), 'w'), default=float)
  keys = ['n_med', 'b_mean', 'b_spread', 'b_range', 'frac_novel'] + \
         [f'{m}_c{c:g}' for c in CS for m in ('shift', 'gain')]
  for metric in ('index', 'measured'):
    print(f'== {metric}')
    print('hg   hs   ' + ' '.join(f'{k:>10s}' for k in keys) + '  worst b_spread')
    for hg in H_GOAL:
      for hs in H_STATE:
        sel = [r for r in rows if r['hg'] == hg and r['hs'] == hs and r['metric'] == metric]
        if sel:
          print(f'{hg:<4g} {hs:<4g} ' + ' '.join(f'{np.mean([r[k] for r in sel]):10.3g}' for k in keys)
                + f'  {min(r["b_spread"] for r in sel):.3f}')


if __name__ == '__main__':
  if sys.argv[1] == 'collect':
    collect(*sys.argv[2:5])
  elif sys.argv[1] == 'metric':
    metric(*sys.argv[2:5])
  else:
    fit(sys.argv[2])
