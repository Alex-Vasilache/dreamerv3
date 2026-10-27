"""Fast offline fit of the table imagination UCB (arm B: start 2 x goal 4 bins).

For V+SOM+even checkpoints. Like ucb_kde_fit.py, but

  * held-out probes: states come from 16 parallel policy rollouts; the
    buffer of picks uses envs 0-12 and the probe states envs 13-15, so a probe
    never has its own neighbouring time steps in the buffer (ucb_kde_fit drew
    both from one pool and was optimistic);
  * the counts are the REAL ImagCounts table (hrl/explore.py), filled with the
    buffer picks and scaled to the steady-state pick mass.

Per smear width h: candidate counts and bonus spread among one state's K
candidates; per c: the tilt's log-prob shift and novelty gain.

  collect: python tools/collapse_study/ucb_table_fit.py collect <run_dir> <ckpt|final> <out.npz>
  fit:     python tools/collapse_study/ucb_table_fit.py fit <out_dir>
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K, PER_STATE, N_ENVS, HOLD = 32, 4, 16, 3
N_STATES = 32768
MASS = 9.2e7
HS = [0.5, 1.0, 1.5]
CS = [1.0, 2.0, 3.0, 5.0, 10.0]


def collect(run_dir, ckpt, out):
  sys.path.insert(0, os.path.join(REPO, 'experiments', 'goal_geometry'))
  from diag_goal_geometry import build, collect_states
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  import embodied.jax.nets as nets
  from dreamerv3.hrl.manager import mgr_as_dict
  ck = None if ckpt == 'final' else os.path.join(run_dir, 'logdir', 'ckpt_milestones', ckpt)
  config, agent = build(run_dir, ck)
  deter = collect_states(config, agent, N_ENVS, 4, N_STATES)
  per_env = len(deter) // N_ENVS                     # rows are env-major
  buf_states = deter[:(N_ENVS - HOLD) * per_env]
  probe_states = deter[(N_ENVS - HOLD) * per_env:N_ENVS * per_env]
  probe_states = probe_states[np.random.default_rng(0).permutation(len(probe_states))[:2048]]
  model = agent.model
  cdt = nets.COMPUTE_DTYPE

  def fn(d, nsamp):
    d = d.astype(cdt)
    logit = model.dyn._prior(d)
    stoch = jax.nn.one_hot(jnp.argmax(logit, -1), logit.shape[-1], dtype=cdt)
    dist = mgr_as_dict(model.manager_pol(model.feat2tensor(dict(deter=d, stoch=stoch)), 1))['skill']
    samples = [dist.sample(nj.seed()) for _ in range(nsamp)]
    goal = jnp.stack([jnp.argmax(s, -1) for s in samples], 1)
    logp = jnp.stack([dist.logp(s) for s in samples], 1)
    enc = model.goal_enc(d, 1)
    code = enc['skill'] if isinstance(enc, dict) else enc
    return goal, logp, jnp.argmax(code.pred(), -1)

  def run(states, k):
    outs = []
    for i in range(0, len(states), 1024):
      outs.append([np.asarray(x) for x in nj.pure(lambda d: fn(d, k))(
          agent.params, jnp.asarray(states[i:i + 1024]), seed=jax.random.PRNGKey(i))[1]])
    return [np.concatenate(z, 0) for z in zip(*outs)]

  bg, _, bs = run(buf_states, PER_STATE)
  pg, plogp, ps = run(probe_states, K)
  np.savez_compressed(out, buf_goal=bg.reshape(-1, bg.shape[-1]).astype(np.int8),
                      buf_start=np.repeat(bs, PER_STATE, 0).astype(np.int8),
                      probe_goal=pg.astype(np.int8), probe_logp=plogp.astype(np.float32),
                      probe_start=ps.astype(np.int8), task=str(config.task),
                      run=os.path.basename(run_dir), ckpt=ckpt)
  print('wrote', out, bg.shape, pg.shape, flush=True)


def fit(out_dir):
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  from dreamerv3.hrl.explore import ImagCounts
  rows = []
  for f in sorted(glob.glob(os.path.join(out_dir, '*.npz'))):
    z = np.load(f)
    bg, bs = jnp.asarray(z['buf_goal'], jnp.int32), jnp.asarray(z['buf_start'], jnp.int32)
    pg = jnp.asarray(z['probe_goal'], jnp.int32)                     # (P, K, L)
    ps = jnp.asarray(z['probe_start'], jnp.int32)[:, None].repeat(K, 1)
    logp = np.asarray(z['probe_logp'])
    for h in HS:
      ic = ImagCounts(decay=1.0, goal_bins=4, state_bins=2, h=h, name='ic')
      params = nj.init(lambda: ic.update(bg[:1], bs[:1], jnp.zeros(1)))({}, seed=0)
      params, _ = nj.pure(lambda: ic.update(bg, bs))(params, seed=0)
      _, n = nj.pure(lambda: ic.count(pg, ps))(params, seed=0)
      n = np.asarray(n) * (MASS / bg.shape[0])
      b = 1 / np.sqrt(n + 1)
      r = dict(task=str(z['task']), run=str(z['run'])[:5], ckpt=str(z['ckpt']), h=h,
               n_med=float(np.median(n)), b_mean=float(b.mean()),
               b_spread=float(b.std(1).mean()), frac_novel=float((n < 1).mean()))
      for c in CS:
        w = np.exp(c * (b - b.max(1, keepdims=True)))
        w /= w.sum(1, keepdims=True)
        r[f'shift_c{c:g}'] = float(((w * logp).sum(1) - logp.mean(1)).mean())
        r[f'gain_c{c:g}'] = float(((w * b).sum(1) / b.mean(1)).mean())
      rows.append(r)
    print('done', os.path.basename(f), flush=True)
  json.dump(rows, open(os.path.join(out_dir, 'fit.json'), 'w'))
  for h in HS:
    sel = [r for r in rows if r['h'] == h]
    print(f'\n== h {h}: n_med {np.mean([r["n_med"] for r in sel]):.3g}  b_mean '
          f'{np.mean([r["b_mean"] for r in sel]):.3f}  spread {np.mean([r["b_spread"] for r in sel]):.3f} '
          f'(min {min(r["b_spread"] for r in sel):.3f})  novel {np.mean([r["frac_novel"] for r in sel]):.2f}')
    for c in CS:
      print(f'   c {c:4g}: shift {np.mean([r[f"shift_c{c:g}"] for r in sel]):+.2f} '
            f'(worst {min(r[f"shift_c{c:g}"] for r in sel):+.2f})  gain '
            f'{np.mean([r[f"gain_c{c:g}"] for r in sel]):.2f}')
    for t in sorted({r['task'] for r in sel}):
      s = [r for r in sel if r['task'] == t]
      print(f'   {t[:16]:16s} spread {np.mean([r["b_spread"] for r in s]):.3f}  b_mean '
            f'{np.mean([r["b_mean"] for r in s]):.2f}  shift c3 {np.mean([r["shift_c3"] for r in s]):+.2f}')


if __name__ == '__main__':
  if sys.argv[1] == 'collect':
    collect(*sys.argv[2:5])
  else:
    fit(sys.argv[2])
