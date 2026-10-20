"""Which goals does the manager propose, and how spread are they? Same probe for every arm.

Only the UCB-in-imagination arm has a count table, so arms are compared on a
quantity all of them have: the manager policy itself. For each milestone
checkpoint, a fixed set of probe states (random-walk episodes of the task,
identical for every run) is encoded with the run's own world model exactly as
the policy step does it (enc -> dyn.observe, single steps), and K goal codes are
sampled from the manager at every probe state. Untilted policy samples: what
the manager has LEARNED to propose, which is what any exploration tilt during
training is meant to change.

  python tools/collapse_study/goal_spread_probe.py <task> <out.npz> <run_dir> [...]

Per run and milestone it stores the sampled class indices (S states x K x L).
Run with dreamerv3_env (checkpoints are numpy-2 pickles).
"""
import glob
import os
import pickle
import sys

import numpy as np

os.environ.pop('DREAMERV3_CONV_IMPL', None)
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'osmesa')
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tools', 'goal_demo'))

import elements  # noqa: E402
import ninjax as nj  # noqa: E402

N_ENVS, T, BURN, EVERY, K = 16, 300, 50, 5, 16


def probe_episodes(config, seed=0):
  """Random-walk observations, (N_ENVS, T, ...) per key, plus the actions taken."""
  from dreamerv3 import main as mainmod
  rng = np.random.default_rng(seed)
  envs = [mainmod.make_env(config, i) for i in range(N_ENVS)]
  act_space = {k: v for k, v in envs[0].act_space.items() if k != 'reset'}
  obs_seq, act_seq = [], []
  obs = [e.step({**{k: np.zeros(v.shape, v.dtype) for k, v in act_space.items()},
                 'reset': True}) for e in envs]
  prev = {k: np.zeros((N_ENVS, *v.shape), v.dtype) for k, v in act_space.items()}
  for _ in range(T):
    obs_seq.append({k: np.stack([o[k] for o in obs]) for k in obs[0]
                    if not k.startswith('log/')})
    act_seq.append({k: v.copy() for k, v in prev.items()})
    acts = {k: (rng.integers(0, int(v.high), (N_ENVS, *v.shape)).astype(v.dtype)
                if v.discrete else rng.uniform(-1, 1, (N_ENVS, *v.shape)).astype(v.dtype))
            for k, v in act_space.items()}
    obs = [e.step({**{k: v[i] for k, v in acts.items()}, 'reset': False})
           for i, e in enumerate(envs)]
    prev = acts
  for e in envs:
    e.close()
  stack = lambda seq: {k: np.stack([s[k] for s in seq], 1) for k in seq[0]}
  return stack(obs_seq), stack(act_seq)


def build(run, ckpt):
  import jax
  import jax.numpy as jnp
  from embodied.jax import internal
  from dreamerv3.agent import Agent as AgentCls
  import render_core
  config = elements.Config.load(os.path.join(run, 'logdir', 'config.yaml'))
  setup = {k: v for k, v in config.jax.items() if k not in render_core._option_keys()}
  setup.update(platform='cpu', prealloc=False, transfer_guard=False, compute_dtype='float32')
  internal.setup(**setup)
  obs_space, act_space = render_core._env_spaces(config.task, config)
  agent_cfg = elements.Config(
      **config.agent, logdir=config.logdir, seed=config.seed, jax=config.jax,
      batch_size=config.batch_size, batch_length=config.batch_length,
      replay_context=config.replay_context, report_length=config.report_length,
      replica=config.replica, replicas=config.replicas,
      train_ratio=config.run.train_ratio, replay_size=config.replay.size)
  model = object.__new__(AgentCls)
  model.__init__(obs_space, act_space, agent_cfg)
  return config, model, obs_space


def run_probe(run, obs, act, out):
  import jax
  import jax.numpy as jnp
  import render_core
  ckpts = sorted(glob.glob(os.path.join(run, 'logdir', 'ckpt_milestones', '0*')))
  if not ckpts:
    return
  config, model, obs_space = build(run, ckpts[0])
  keys = [k for k in obs_space if k in obs]

  def step(carry, o, a):
    enc_c, dyn_c = carry
    reset = o['is_first']
    enc_c, _, tokens = model.enc(enc_c, o, reset, training=False, single=True)
    dyn_c, _, feat = model.dyn.observe(dyn_c, tokens, a, reset, training=False, single=True)
    return (enc_c, dyn_c), feat

  def goals(feat):
    if model.mgr_cond_goalcode:
      code = model._encode_goal_code(model.feat2deter(feat), 1)
      tensor = model._mgr_input(feat, {'goal_code': code})
    else:
      tensor = model.feat2tensor(feat)
    from dreamerv3.hrl.manager import mgr_as_dict
    dist = mgr_as_dict(model.manager_pol(tensor, 1))['skill']
    return jnp.stack([jnp.argmax(dist.sample(nj.seed()), -1) for _ in range(K)], 1)

  def init_carry():
    return (model.enc.initial(N_ENVS), model.dyn.initial(N_ENVS))

  o0 = {k: jnp.asarray(obs[k][:, 0]) for k in keys}
  a0 = {k: jnp.asarray(v[:, 0]) for k, v in act.items()}

  def full(o, a):
    carry, feat = step(init_carry(), o, a)
    return goals(feat)

  params = nj.init(full)({}, o0, a0, seed=0)
  pstep = jax.jit(lambda p, c, o, a, s: nj.pure(step)(p, c, o, a, seed=s)[1])
  pgoal = jax.jit(lambda p, f, s: nj.pure(goals)(p, f, seed=s)[1])
  pinit = nj.pure(init_carry)
  res = {}
  for ck in ckpts:
    loaded, _ = render_core._load_params(os.path.join(ck, 'agent.pkl'), params)
    p = jax.device_put(loaded)
    _, carry = pinit(p)
    ids = []
    for t in range(T):
      o = {k: jnp.asarray(obs[k][:, t]) for k in keys}
      a = {k: jnp.asarray(v[:, t]) for k, v in act.items()}
      carry, feat = pstep(p, carry, o, a, jax.random.PRNGKey(10_000 + t))
      if t >= BURN and (t - BURN) % EVERY == 0:
        ids.append(np.asarray(pgoal(p, feat, jax.random.PRNGKey(t))))
    res[os.path.basename(ck)] = np.stack(ids, 1).reshape(-1, K, ids[0].shape[-1]).astype(np.int8)
    print(os.path.basename(run)[:6], os.path.basename(ck), res[os.path.basename(ck)].shape, flush=True)
  out[os.path.basename(run)] = res


def main():
  task, out_path, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
  cache = out_path.replace('.npz', '_probe_obs.pkl')
  if os.path.exists(cache):
    obs, act = pickle.load(open(cache, 'rb'))
  else:
    config = elements.Config.load(os.path.join(runs[0], 'logdir', 'config.yaml'))
    obs, act = probe_episodes(config)
    pickle.dump((obs, act), open(cache, 'wb'))
  done = {}
  if os.path.exists(out_path):
    z = np.load(out_path, allow_pickle=True)
    done = {k: z[k].item() for k in z.files}
  for run in runs:
    name = os.path.basename(run)
    if name in done and len(done[name]) == len(glob.glob(os.path.join(run, 'logdir', 'ckpt_milestones', '0*'))):
      continue
    run_probe(run, obs, act, done)
    np.savez(out_path, **{k: np.array(v, dtype=object) for k, v in done.items()})
  # Plain arrays for the plotting env (numpy 1 cannot unpickle numpy-2 objects).
  np.savez_compressed(out_path.replace('.npz', '_flat.npz'),
                      **{f'{k}|{ck}': v for k, d in done.items() for ck, v in d.items()})


if __name__ == '__main__':
  main()
