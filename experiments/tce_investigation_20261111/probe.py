"""Roll out a saved pin pad checkpoint (no training) and log per step: player cell, pad window,
reward, manager switch, goal code, decoded goal image, world-model deter.
usage: probe.py <run basename> <ckpt step> <out.npz> [episodes]"""
import sys, glob, re, numpy as np
R = '/apps/unit/DoyaU/vasilache/apps/code/dreamerv3'
sys.path.insert(0, R); sys.path.insert(0, R + '/dreamerv3')
import ruamel.yaml as yaml, elements
from dreamerv3 import main as M
from embodied.envs.pinpad import PinPad
run, step, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
eps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
d = (glob.glob('/bucket/DoyaU/vasilache/bucket/results/dreamerv3/' + run) + glob.glob('/work/DoyaU/vasilache/work/' + run))[0]
task = re.search(r'_(pinpad_[a-z]+)_', run).group(1)
cfgs = re.search(task + r'_(.*)_s\d+_j', run).group(1).split('-')
C = yaml.YAML(typ='safe').load(elements.Path(R + '/dreamerv3/configs.yaml').read())
config = elements.Config(C['defaults'])
for n in cfgs: config = config.update(C[n])
config = elements.Flags(config).parse(['--task', task, '--logdir', out + '_logdir', '--jax.platform', 'cpu',
  '--jax.prealloc', 'False', '--run.envs', '1', '--agent.policy_struct_diag', 'True'])
agent = M.make_agent(config)
cp = elements.Checkpoint(); cp.agent = agent; cp.load(f'{d}/logdir/ckpt_milestones/{step:012d}', keys=['agent'])
env = PinPad(task.split('_')[1], length=2000); env.random = np.random.RandomState(1)
rec = {k: [] for k in ['pos', 'seq', 'rew', 'switch', 'gid', 'deter', 'goalimg', 'act', 'countdown']}
for ep in range(eps):
  carry = agent.init_policy(1)
  o = env.step({'action': 0, 'reset': True})
  for t in range(2000):
    ob = {k: np.asarray(o[k])[None] for k in agent.obs_space}
    carry, act, info = agent.policy(carry, ob, mode='train')
    a = int(np.asarray(act['action']).reshape(-1)[0]) if np.asarray(act['action']).size == 1 else int(np.argmax(np.asarray(act['action']).reshape(-1)))
    rec['pos'].append(env.player); rec['seq'].append(''.join(env.sequence)); rec['countdown'].append(env.countdown)
    rec['switch'].append(float(np.asarray(info.get('log/fromto_switch', [t % 8 == 0])).ravel()[0]))
    rec['gid'].append(np.asarray(info.get('log/fromto_goal', [-1])).ravel().astype(int))
    rec['deter'].append(np.asarray(info['log/struct_diag_deter']).ravel().astype(np.float16))
    gi = np.asarray(info['log/image_with_goal'])[0]
    rec['goalimg'].append(gi[64:] if gi.shape[0] == 128 else gi[:, 64:])
    rec['act'].append(a)
    o = env.step({'action': a, 'reset': False})
    rec['rew'].append(float(o['reward']))
  print(f'episode {ep}: score {sum(rec["rew"][-2000:]):.0f}', flush=True)
np.savez_compressed(out, **{k: np.array(v) for k, v in rec.items()})
print('saved', out)
