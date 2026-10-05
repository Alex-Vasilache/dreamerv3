"""Check the on-device numpy policy against the real JAX modules.

The numpy port in `dreamerv3/deploy/numpy_policy.py` is hand-written, so the
only thing standing between it and a robot that quietly acts on a subtly wrong
policy is this test. It drives the actual `rssm.Encoder`, `rssm.RSSM` and
`heads.MLPHead` objects -- the same classes the learner and the current actor
use -- with the exported weights, and compares every deterministic tensor:

  tokens   encoder output
  deter    the blocked GRU recurrence
  logit    the stochastic posterior
  mean     actor mean, after tanh
  stddev   actor stddev, after the bounded_normal squash

The stochastic sample is deliberately taken from the JAX side and injected into
the numpy side, so the actor comparison is apples to apples rather than a
distribution check that would pass with the wrong weights.

  .venv/bin/python tools/test_numpy_policy.py --weights /tmp/policy.npz
"""

import argparse
import pathlib
import sys

import numpy as np
import ruamel.yaml as yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import elements  # noqa: E402


def build_reference(config, obs_space, act_space):
  """The real modules, wrapped in a ninjax pure function."""
  import jax
  import ninjax as nj
  import dreamerv3.rssm as rssm
  import embodied.jax
  import embodied.jax.nets as nn

  exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
  enc_space = {k: v for k, v in obs_space.items() if k not in exclude}

  enc = rssm.Encoder(enc_space, **config.agent.enc.simple, name='enc')
  dyn = rssm.RSSM(act_space, **config.agent.dyn.rssm, name='dyn')
  pol_outs = {k: config.agent.policy_dist_cont for k in act_space}
  pol = embodied.jax.MLPHead(
      act_space, pol_outs, **config.agent.policy, name='pol')

  def probe(obs, prevact, carry):
    reset = obs['is_first']
    _, _, tokens = enc({}, obs, reset, training=False, single=True)
    dyn_carry, _, feat = dyn.observe(
        carry, tokens, prevact, reset, training=False, single=True)
    x = nn.cast(feat['deter'])
    s = nn.cast(feat['stoch'])
    x = jax.numpy.concatenate([x, s.reshape((*s.shape[:-2], -1))], -1)
    dist = pol(x, bdims=1)
    key = list(act_space)[0]
    # DictHead wraps each output in outs.Agg; the Normal underneath is what
    # carries the mean and stddev this test compares.
    out = dist[key]
    out = getattr(out, 'output', out)
    return dict(
        tokens=tokens, deter=feat['deter'], stoch=feat['stoch'],
        logit=feat['logit'], mean=out.mean, stddev=out.stddev,
        carry=dyn_carry)

  return nj.pure(probe), dyn


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--weights', required=True)
  ap.add_argument('--configs', nargs='+', default=['robot_daydreamer'])
  ap.add_argument('--steps', type=int, default=6)
  ap.add_argument('--tol', type=float, default=2e-2)
  args = ap.parse_args()

  import jax
  import dreamerv3.main as main_mod
  from dreamerv3.deploy.numpy_policy import NumpyPolicy

  configs = yaml.YAML(typ='safe').load(
      elements.Path(main_mod.folder / 'configs.yaml').read())
  config = elements.Config(configs['defaults'])
  for name in args.configs:
    config = config.update(configs[name])
  # Compare in float32. The actor really runs bfloat16, which is a separate
  # (and looser) question; a port that is wrong will be wrong in float32 too.
  config = config.update({'jax.compute_dtype': 'float32'})
  import jax.numpy as jnp
  import embodied.jax.nets as nn
  nn.COMPUTE_DTYPE = jnp.float32

  f32 = np.float32
  obs_space = {
      'orientation': elements.Space(f32, (2,)),
      'wheels': elements.Space(f32, (2,)),
      'reward': elements.Space(f32),
      'is_first': elements.Space(bool),
      'is_last': elements.Space(bool),
      'is_terminal': elements.Space(bool),
  }
  act_space = {'drive': elements.Space(f32, (2,), -1.0, 1.0)}

  probe, dyn = build_reference(config, obs_space, act_space)

  npol = NumpyPolicy(args.weights)
  params = {k: np.asarray(v) for k, v in np.load(args.weights).items()
            if k != '__meta__'}

  # ninjax wants the param names the modules will ask for, which are exactly
  # the keys the exporter kept.
  rng = np.random.default_rng(0)
  carry = jax.tree.map(lambda x: np.asarray(x), _initial_dyn(dyn))
  ncarry = npol.initial()
  worst = {}
  for step in range(args.steps):
    obs = {
        'orientation': rng.normal(size=(1, 2)).astype(f32),
        'wheels': rng.normal(size=(1, 2)).astype(f32),
        'reward': np.zeros((1,), f32),
        'is_first': np.array([step == 0]),
        'is_last': np.array([False]),
        'is_terminal': np.array([False]),
    }
    prevact = {'drive': ncarry['prevact'][None].astype(f32)}
    _, out = probe(params, obs, prevact, carry, seed=step)
    out = jax.tree.map(lambda x: np.asarray(x), out)
    carry = out['carry']

    # numpy side, with the JAX sample injected so the actor sees the same state
    nobs = {'orientation': obs['orientation'][0], 'wheels': obs['wheels'][0]}
    tokens = npol._encode(nobs)
    deter = npol._core(
        np.zeros_like(ncarry['deter']) if step == 0 else ncarry['deter'],
        np.zeros_like(ncarry['stoch']) if step == 0 else ncarry['stoch'],
        np.zeros_like(ncarry['prevact']) if step == 0 else ncarry['prevact'])
    stoch_np, logit = npol._obs_stoch(deter, tokens)
    jax_stoch = out['stoch'][0]
    x = np.concatenate([deter, jax_stoch.reshape(-1)], -1)
    mean, stddev = _numpy_actor(npol, x)

    for name, a, b in [
        ('tokens', out['tokens'][0], tokens),
        ('deter', out['deter'][0], deter),
        ('logit', out['logit'][0], logit),
        ('mean', out['mean'][0], mean),
        ('stddev', out['stddev'][0], stddev),
    ]:
      err = float(np.abs(np.asarray(a) - np.asarray(b)).max())
      rel = err / max(1e-6, float(np.abs(np.asarray(a)).max()))
      worst[name] = max(worst.get(name, 0.0), rel)

    ncarry = dict(deter=deter, stoch=jax_stoch, prevact=np.clip(
        np.asarray(out['mean'][0]), -1, 1).astype(f32))

  print(f'{"tensor":10s} {"max rel err":>12s}')
  bad = False
  for name, rel in worst.items():
    flag = '' if rel < args.tol else '   <-- FAIL'
    bad |= rel >= args.tol
    print(f'{name:10s} {rel:12.2e}{flag}')
  print()
  if bad:
    print('MISMATCH: the numpy port does not reproduce the JAX modules.')
    raise SystemExit(1)
  print(f'OK: numpy port matches the JAX modules within {args.tol:g} relative.')


def _numpy_actor(npol, x):
  from dreamerv3.deploy.numpy_policy import linear, rms_norm, silu, sigmoid
  w, m = npol.w, npol.meta
  for i in range(m['pol_layers']):
    x = linear(x, w[f'pol/mlp/linear{i}/kernel'], w[f'pol/mlp/linear{i}/bias'])
    x = silu(rms_norm(x, w[f'pol/mlp/norm{i}/scale']))
  k = m['act_key']
  mean = linear(x, w[f'pol/head/{k}/mean/kernel'], w[f'pol/head/{k}/mean/bias'])
  raw = linear(
      x, w[f'pol/head/{k}/stddev/kernel'], w[f'pol/head/{k}/stddev/bias'])
  lo, hi = m['minstd'], m['maxstd']
  return np.tanh(mean), (hi - lo) * sigmoid(raw + 2.0) + lo


def _initial_dyn(dyn):
  import numpy as np
  return dict(
      deter=np.zeros((1, dyn.deter), np.float32),
      stoch=np.zeros((1, dyn.stoch, dyn.classes), np.float32))


if __name__ == '__main__':
  main()
