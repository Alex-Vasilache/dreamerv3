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


def fit(out_dir):
  import jax
  import jax.numpy as jnp
  files = sorted(glob.glob(os.path.join(out_dir, '*.npz')))
  rows = []
  for f in files:
    z = np.load(f)
    buf = jnp.asarray(z['buf'], jnp.float32)
    L = buf.shape[1] // 2
    pg = jnp.asarray(z['probe_goal'], jnp.float32)                      # (P, K, L)
    ps = jnp.asarray(z['probe_start'], jnp.float32)[:, None].repeat(K, 1)
    logp = np.asarray(z['probe_logp'])
    x = jnp.concatenate([ps, pg], -1).reshape(-1, 2 * L)

    @jax.jit
    def counts(x, hs, hg):
      def chunk(acc, yb):
        def part(a, b, h):
          d2 = (jnp.square(a).sum(-1)[:, None] + jnp.square(b).sum(-1)[None]
                - 2 * jnp.matmul(a, b.T, precision=jax.lax.Precision.HIGHEST))
          return jnp.maximum(d2, 0) / (2 * h * h)
        e = part(x[:, :L], yb[:, :L], hs) + part(x[:, L:], yb[:, L:], hg)
        return acc + jnp.exp(-e).sum(-1), None
      acc, _ = jax.lax.scan(chunk, jnp.zeros(x.shape[0]), buf.reshape(-1, 4096, 2 * L))
      return acc * (MASS / buf.shape[0])

    for hg in H_GOAL:
      for hs in H_STATE:
        n = np.asarray(counts(x, hs, hg)).reshape(-1, K)
        b = 1 / np.sqrt(n + 1)
        r = dict(file=os.path.basename(f), task=str(z['task']), ckpt=str(z['ckpt']),
                 hg=hg, hs=hs, n_med=np.median(n), b_mean=b.mean(),
                 b_spread=b.std(1).mean(), b_range=(b.max(1) - b.min(1)).mean(),
                 frac_novel=(n < 1).mean())
        for c in CS:
          w = np.exp(c * (b - b.max(1, keepdims=True)))
          w /= w.sum(1, keepdims=True)
          r[f'shift_c{c:g}'] = ((w * logp).sum(1) - logp.mean(1)).mean()
          r[f'gain_c{c:g}'] = ((w * b).sum(1) / b.mean(1)).mean()
        rows.append(r)
    print('done', os.path.basename(f), flush=True)
  import json
  json.dump(rows, open(os.path.join(out_dir, 'fit.json'), 'w'), default=float)
  # Summary: mean over files per width pair.
  keys = ['n_med', 'b_mean', 'b_spread', 'b_range', 'frac_novel'] + \
         [f'{m}_c{c:g}' for c in CS for m in ('shift', 'gain')]
  print('hg   hs   ' + ' '.join(f'{k:>10s}' for k in keys) + '  worst-file b_spread')
  for hg in H_GOAL:
    for hs in H_STATE:
      sel = [r for r in rows if r['hg'] == hg and r['hs'] == hs]
      print(f'{hg:<4g} {hs:<4g} ' + ' '.join(f'{np.mean([r[k] for r in sel]):10.3g}' for k in keys)
            + f'  {min(r["b_spread"] for r in sel):.3f}')


if __name__ == '__main__':
  if sys.argv[1] == 'collect':
    collect(*sys.argv[2:5])
  else:
    fit(sys.argv[2])
