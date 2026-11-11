"""Step 1 (handoff 11-11): calibrate 'goal reached' = cos_max(goal, deter) >= tau. Offline, CPU.
Logs per step: world-model deter, the decoded goal in force, and decode(encode(deter)) (the goal the
AE would express for the state we are in: the ceiling cos_max can reach when standing on the goal).
Phase A: normal play. Phase B (pin pad only): goal fixed by hand to each pad, 48 steps, manager frozen.
usage: reach_probe.py <run basename> <ckpt step> <out.npz> [episodes]"""
import sys, glob, re, numpy as np
R = '/apps/unit/DoyaU/vasilache/apps/code/dreamerv3'
sys.path.insert(0, R); sys.path.insert(0, R + '/dreamerv3'); sys.path.insert(0, '/work/DoyaU/vasilache/work/meetings/tce_investigation')
import jax, jax.numpy as jnp
import ruamel.yaml as yaml, elements
from dreamerv3 import main as M
from dreamerv3 import rssm as RS, agent as AG
from dreamerv3.hrl import manager as MG

_HOLD = {}
_o = RS.RSSM.observe
def _observe(self, *a, **k):
  c, e, f = _o(self, *a, **k); _HOLD['feat'] = f; return c, e, f
RS.RSSM.observe = _observe
mcls = [c for c in vars(MG).values() if isinstance(c, type) and '_manager_skill_step' in vars(c)][0]
_m = mcls._manager_skill_step
def _mstep(self, *a, **k):
  r = _m(self, *a, **k); _HOLD['goal'] = r[1]; return r
mcls._manager_skill_step = _mstep
_p = AG.Agent.policy
def _policy(self, *a, **k):
  c, act, out = _p(self, *a, **k)
  out = dict(out)
  deter = self.feat2deter(_HOLD['feat'])
  enc = self.goal_enc(deter, 1); dist = enc['skill'] if isinstance(enc, dict) else enc
  logits = AG._head_inner(dist).dist.logits
  code = jax.nn.one_hot(jnp.argmax(logits, -1), logits.shape[-1])
  out['log/rp_self'] = self.goal_dec(code, 1).pred()
  out['log/rp_code'] = jnp.argmax(logits, -1).astype(jnp.int32)
  out['log/rp_goal'] = _HOLD['goal']
  out['log/rp_deter'] = deter
  return c, act, out
AG.Agent.policy = _policy

run, step, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
eps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
d = (glob.glob('/bucket/DoyaU/vasilache/bucket/results/dreamerv3/' + run) + glob.glob('/work/DoyaU/vasilache/work/' + run))[0]
task = re.search(r'_((?:pinpad|dmc)_[a-z_]+?)_director', run).group(1)
cfgs = re.search(task + r'_(.*)_s\d+_j', run).group(1).split('-')
C = yaml.YAML(typ='safe').load(elements.Path(R + '/dreamerv3/configs.yaml').read())
config = elements.Config(C['defaults'])
for n in cfgs: config = config.update(C[n])
if not task.startswith('pinpad'):   # DMC runs carry task settings (image-only keys) that the name does not: use the saved config
  saved = yaml.YAML(typ='safe').load(elements.Path(d + '/logdir/config.yaml').read())
  def merge(base, new):
    return {k: (merge(v, new[k]) if isinstance(v, dict) and isinstance(new.get(k), dict) else new.get(k, v)) for k, v in base.items()}
  config = elements.Config(merge(dict(C['defaults']), saved))
config = elements.Flags(config).parse(['--task', task, '--logdir', outp + '_logdir', '--jax.platform', 'cpu',
  '--jax.prealloc', 'False', '--run.envs', '1', '--agent.policy_struct_diag', 'True'])
agent = M.make_agent(config)
cp = elements.Checkpoint(); cp.agent = agent; cp.load(f'{d}/logdir/ckpt_milestones/{step:012d}', keys=['agent'])
K = int(config.agent.manager_sample_freq) if hasattr(config.agent, 'manager_sample_freq') else 8
pin = task.startswith('pinpad')
f16 = lambda x: np.asarray(x).reshape(-1).astype(np.float16)

def policy(carry, o):
  ob = {k: np.asarray(o[k])[None] for k in agent.obs_space}
  carry, act, info = agent.policy(carry, ob, mode='train')
  return carry, act, info

rec = {k: [] for k in ['ep', 't', 'deter', 'goal', 'self', 'code', 'rew', 'pos', 'seq', 'goalimg', 'obsvec']}
if pin:
  from embodied.envs.pinpad import PinPad
  env = PinPad(task.split('_')[1], length=2000); env.random = np.random.RandomState(1)
else:
  env = M.make_env(config, 0)
for ep in range(eps):
  carry = agent.init_policy(1)
  if pin:
    o = env.step({'action': 0, 'reset': True})
  else:
    a0 = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items() if k != 'reset'}
    o = env.step({**a0, 'reset': True})
  score = 0.0
  for t in range(2000):
    carry, act, info = policy(carry, o)
    rec['ep'].append(ep); rec['t'].append(t)
    rec['deter'].append(f16(info['log/rp_deter'])); rec['goal'].append(f16(info['log/rp_goal']))
    rec['self'].append(f16(info['log/rp_self'])); rec['code'].append(np.asarray(info['log/rp_code']).reshape(-1))
    if pin:
      rec['pos'].append(env.player); rec['seq'].append(''.join(env.sequence))
      gi = np.asarray(info['log/image_with_goal'])[0]
      rec['goalimg'].append(gi[64:] if gi.shape[0] == 128 else gi[:, 64:])
      a = np.asarray(act['action']).reshape(-1); a = int(a[0]) if a.size == 1 else int(np.argmax(a))
      o = env.step({'action': a, 'reset': False})
    else:
      ph = env._dmenv.physics.data   # wrappers forward attribute access down to the DMC env
      rec['obsvec'].append(np.concatenate([np.asarray(ph.qpos).reshape(-1), np.asarray(ph.qvel).reshape(-1)]).astype(np.float32))
      o = env.step({**{k: np.asarray(v)[0] for k, v in act.items()}, 'reset': False})
    rec['rew'].append(float(o['reward'])); score += float(o['reward'])
    if o['is_last']: break
  print(f'episode {ep}: steps {t + 1} score {score:.0f}', flush=True)
out = {k: np.array(v) for k, v in rec.items() if len(v)}
out['K'] = K
np.savez_compressed(outp, **out)

if pin:
  from analyze import G, PADS
  def put(old, val):
    if isinstance(old, (list, tuple)): return type(old)(put(o, val) for o in old)
    v = np.broadcast_to(np.asarray(val), np.shape(old)) if np.ndim(val) == 0 else np.asarray(val).reshape(np.shape(old))
    v = np.ascontiguousarray(v).astype(np.asarray(old).dtype)
    return jax.device_put(v, old.sharding) if hasattr(old, 'sharding') else v
  def inject(carry, goal, code):
    c = list(carry); ms = dict(c[4])
    sk = ms['skill'][0] if isinstance(ms['skill'], (list, tuple)) else ms['skill']
    ms['goal_deter'] = put(ms['goal_deter'], goal)
    ms['skill'] = put(ms['skill'], np.eye(np.shape(sk)[-1], dtype=np.float32)[code])
    c[4] = ms; c[5] = put(c[5], 1)   # step 1 of K: the manager never re-decides
    return tuple(c)
  henv = PinPad('six', length=10 ** 9); rng = np.random.RandomState(0); henv.random = rng
  hand = {k: [] for k in ['pad', 'trial', 't', 'deter', 'goal', 'pos', 'onpad', 'paddist']}
  for p in PADS:
    goals = []
    for cell in G.padcells[p][::2]:
      carry = agent.init_policy(1); o = henv.step({'action': 0, 'reset': True}); henv.player = cell; o = henv._obs(0.0, is_first=True)
      for t in range(4):
        carry, _, info = policy(carry, o); o = henv.step({'action': 0, 'reset': False})
      goals.append((np.asarray(info['log/rp_self']).reshape(-1), np.asarray(info['log/rp_code']).reshape(-1)))
    for trial in range(24):
      g, code = goals[trial % len(goals)]
      carry = agent.init_policy(1); o = henv.step({'action': 0, 'reset': True})
      carry, act, info = policy(carry, o)
      for t in range(48):
        a = np.asarray(act['action']).reshape(-1); a = int(a[0]) if a.size == 1 else int(np.argmax(a))
        o = henv.step({'action': a, 'reset': False})
        carry = inject(carry, g, code)
        carry, act, info = policy(carry, o)
        assert np.allclose(np.asarray(info['log/rp_goal']).reshape(-1), g, atol=1e-3), 'goal not held'
        hand['pad'].append(p); hand['trial'].append(trial); hand['t'].append(t)
        hand['deter'].append(f16(info['log/rp_deter'])); hand['goal'].append(f16(g)); hand['pos'].append(henv.player)
        hand['onpad'].append(G.lay[henv.player] == p)
        hand['paddist'].append(min(G.D[G.idx[tuple(henv.player)], G.idx[c]] for c in G.padcells[p]))
    print(f'hand pad {p} done', flush=True)
  out.update({'hand_' + k: np.array(v) for k, v in hand.items()})
np.savez_compressed(outp, **out)
print('saved', outp)
