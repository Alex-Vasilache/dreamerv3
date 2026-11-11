"""Point 4 test, offline: can the manager express 'go to pad X' and can the worker do it?
(a) encode real states standing on each pad -> code -> decode: does the decoded goal land on that pad?
(b) 3000 uniform random codes -> decoded goal positions: how much of the goal space are pads?
(c) fix the goal to each pad's code (manager overridden) and roll out the worker from random spawns.
usage: handgoal.py <run> <step> <out.npz>"""
import sys, glob, re, numpy as np
R = '/apps/unit/DoyaU/vasilache/apps/code/dreamerv3'
sys.path.insert(0, R); sys.path.insert(0, R + '/dreamerv3'); sys.path.insert(0, '.')
import ruamel.yaml as yaml, elements
from dreamerv3 import main as M
from dreamerv3.hrl import manager as MG
from embodied.envs.pinpad import PinPad
from analyze import G, PADS, goal_pos, dist
# Manager override: re-emit whatever code is in the carry (we write it from outside).
_orig = MG.ManagerMixin._emit_manager if hasattr(MG, 'ManagerMixin') else None
cls = [c for c in vars(MG).values() if isinstance(c, type) and '_emit_manager' in vars(c)][0]
_orig = cls._emit_manager
def _forced(self, tensor, bdims, *a, prev_mgr_skill=None, **kw):
  out = _orig(self, tensor, bdims, *a, prev_mgr_skill=prev_mgr_skill, **kw)
  if prev_mgr_skill is not None and 'skill' in prev_mgr_skill and 'skill' in out:
    out = dict(out); out['skill'] = prev_mgr_skill['skill'].astype(out['skill'].dtype)
  return out
cls._emit_manager = _forced

run, step, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = (glob.glob('/bucket/DoyaU/vasilache/bucket/results/dreamerv3/' + run) + glob.glob('/work/DoyaU/vasilache/work/' + run))[0]
task = re.search(r'_(pinpad_[a-z]+)_', run).group(1); cfgs = re.search(task + r'_(.*)_s\d+_j', run).group(1).split('-')
C = yaml.YAML(typ='safe').load(elements.Path(R + '/dreamerv3/configs.yaml').read())
config = elements.Config(C['defaults'])
for n in cfgs: config = config.update(C[n])
config = elements.Flags(config).parse(['--task', task, '--logdir', outp + '_logdir', '--jax.platform', 'cpu',
  '--jax.prealloc', 'False', '--run.envs', '1', '--agent.policy_struct_diag', 'True'])
agent = M.make_agent(config)
cp = elements.Checkpoint(); cp.agent = agent; cp.load(f'{d}/logdir/ckpt_milestones/{step:012d}', keys=['agent'])
L, Cn = [int(x) for x in config.agent.skill_shape]
env = PinPad('six', length=10 ** 9); rng = np.random.RandomState(0); env.random = rng

def setcode(carry, code):
  import jax
  oh = np.eye(Cn, dtype=np.float32)[code]
  def conv(old, val):
    if isinstance(old, list):
      return [conv(o, val) for o in old]
    v = val if np.ndim(old) == val.ndim else val[None]
    v = v.astype(np.asarray(old).dtype)
    return jax.device_put(v, old.sharding) if hasattr(old, 'sharding') else v
  c = list(carry); ms = dict(c[4]); ms['skill'] = conv(ms['skill'], oh); c[4] = ms
  return type(carry)(c) if isinstance(carry, tuple) else c
def act(carry, o):
  ob = {k: np.asarray(o[k])[None] for k in agent.obs_space}
  carry, a, info = agent.policy(carry, ob, mode='eval')
  return carry, int(np.asarray(a['action']).reshape(-1)[0]), info
def reset_at(cell):
  o = env.step({'action': 0, 'reset': True})
  if cell is not None: env.player = cell; o = env._obs(0.0, is_first=True)
  return o

res = {}
# (a) pad states -> code -> decoded goal
enc_codes = {}; roundtrip = {}
for p in PADS:
  hits = []
  for cell in G.padcells[p][::2]:
    carry = agent.init_policy(1); o = reset_at(cell)
    for t in range(4):   # stand still a few steps so the posterior settles
      carry, _, info = act(carry, o); o = env.step({'action': 0, 'reset': False})
    probs = np.asarray(info['log/struct_diag_probs']).reshape(L, Cn); code = probs.argmax(-1)
    # decode: inject and read the fresh goal image at a reset step
    carry = setcode(agent.init_policy(1), code); carry, _, info = act(carry, reset_at(cell))
    gimg = np.asarray(info['log/image_with_goal'])[0]; gimg = gimg[64:] if gimg.shape[0] == 128 else gimg[:, 64:]
    gp, dk = goal_pos(gimg)
    hits.append((tuple(code), G.lay[gp] == p, dk))
  enc_codes[p] = [h[0] for h in hits]; roundtrip[p] = np.mean([h[1] for h in hits])
  print(f'(a) pad {p}: {len(hits)} states, encode->decode lands on the same pad {roundtrip[p]:.0%}, clarity {np.median([h[2] for h in hits]):.0f}', flush=True)
# (b) random codes
gps = []
for i in range(1500):
  code = rng.randint(Cn, size=L); carry = setcode(agent.init_policy(1), code)
  carry, _, info = act(carry, reset_at(None))
  gimg = np.asarray(info['log/image_with_goal'])[0]; gimg = gimg[64:] if gimg.shape[0] == 128 else gimg[:, 64:]
  gps.append(goal_pos(gimg))
onpad = [G.lay[g] for g, _ in gps]
print('(b) uniform random codes: on a pad ' + f'{np.mean([x in PADS for x in onpad]):.0%}; per pad ' +
      str({p: f'{np.mean([x == p for x in onpad]):.1%}' for p in PADS}) + f'; pad area {sum(len(v) for v in G.padcells.values()) / len(G.free):.0%} of cells', flush=True)
# (c) worker test with each pad's own code
for p in PADS:
  codes = [c for c, ok in zip(enc_codes[p], [True] * 99)]
  succ, steps_ = [], []
  for trial in range(24):
    code = np.array(codes[trial % len(codes)]); carry = setcode(agent.init_policy(1), code); o = reset_at(None)
    start = env.player; reached = None
    for t in range(48):
      carry, a, info = act(carry, o); o = env.step({'action': a, 'reset': False})
      if G.lay[env.player] == p: reached = t + 1; break
    succ.append(reached is not None); steps_.append(reached or 99)
    d0 = min(dist(start, c) for c in G.padcells[p])
  print(f'(c) pad {p}: worker reaches the pad within 8 steps {np.mean([s <= 8 for s in steps_]):.0%}, within 48 steps {np.mean(succ):.0%}; median steps {np.median(steps_):.0f}', flush=True)
