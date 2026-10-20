"""Replay a trained policy in the simulated robot and record its body state.

Loads a run's config and final checkpoint, rebuilds the agent, and drives
`embodied.envs.robot_sim.RobotSim` (no real-time pacing) with the policy in
eval mode. Every step records the true body state (tilt, rate, wheel speeds,
distance, heading) plus the command, reward and action, for
`sim_video.py` to animate.

The scenario is fixed so two agents can be compared on the same episode:
the body starts on its back bumper, gets pushed at fixed times (balance), and
follows a fixed command schedule (command task; `schedule` below).

  python tools/robot/dev/sim_rollout.py RUN_DIR OUT.npz [--seconds 30] [--seed 0]
"""

import argparse
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# (start s, forward, turn): held until the next entry; ramped by the env.
SCHEDULE = [(0, 0.0, 0.0), (5, 0.6, 0.0), (11, 0.0, 0.0), (15, 0.0, 0.6),
            (20, -0.5, 0.0), (25, 0.4, 0.4), (30, 0.0, 0.0)]
# (time s, tilt-rate kick rad/s) for the balance task.
PUSHES = [(8.0, 1.5), (14.0, -1.5), (20.0, 2.0), (26.0, -2.0)]


def scheduled(t):
  cmd = (0.0, 0.0)
  for start, f, turn in SCHEDULE:
    if t >= start:
      cmd = (f, turn)
  return cmd


def unwrap(env):
  while hasattr(env, 'env'):
    env = env.env
  return env


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('run')
  parser.add_argument('out')
  parser.add_argument('--seconds', type=float, default=32.0)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--gpu', action='store_true')
  args = parser.parse_args()

  import elements
  import ruamel.yaml as yaml
  from dreamerv3 import main as dmain

  logdir = pathlib.Path(args.run) / 'logdir'
  raw = yaml.YAML(typ='safe').load((logdir / 'config.yaml').read_text())
  config = elements.Config(raw)
  import tempfile
  config = elements.Flags(config).parse([
      '--env.realtime_hz', '0', '--logdir', tempfile.mkdtemp(),
      '--jax.platform', 'cuda' if args.gpu else 'cpu', '--jax.prealloc', 'False',
      '--env.robot.length', '1000000', '--seed', str(args.seed)])
  agent = dmain.make_agent(config)
  cp = elements.Checkpoint()
  cp.agent = agent
  ckpt = logdir / 'ckpt' / (logdir / 'ckpt' / 'latest').read_text().strip()
  cp.load(str(ckpt), keys=['agent'])

  env = dmain.make_env(config, 0)
  sim = unwrap(env)
  hz = sim.hz
  task = sim.task
  # Same body for every agent: fixed balance point, start on the back bumper.
  sim.theta_eq = sim.theta_zero
  sim.theta = sim.theta_zero + sim.bumper_lo
  sim.omega = 0.0
  sim.sim_rng = np.random.RandomState(1234)
  if task == 'command':
    sim.command_sampler = dict(sim.command_sampler, hold_min=1e9, hold_max=1e9)
    sim.cmd_left = 1e9

  keys = [k for k in agent.obs_space]
  act_key = [k for k in agent.act_space][0]
  carry = agent.init_policy(1)
  zero = {act_key: np.zeros(agent.act_space[act_key].shape, np.float32)}
  obs = env.step({**zero, 'reset': True})
  rec = {k: [] for k in ('t', 'theta', 'omega', 'v_l', 'v_r', 'x', 'yaw',
                         'cmd_f', 'cmd_t', 'reward', 'act_l', 'act_r',
                         'pushed')}
  x = yaw = 0.0
  pushes = list(PUSHES) if task != 'command' else []
  steps = int(args.seconds * hz)
  for i in range(steps):
    t = i / hz
    if task == 'command':
      sim.cmd_target = scheduled(t)
    pushed = 0.0
    if pushes and t >= pushes[0][0]:
      sim.omega += pushes[0][1]
      pushed = pushes[0][1]
      pushes.pop(0)
    batch = {k: np.asarray(obs[k])[None] for k in keys}
    carry, acts, _ = agent.policy(carry, batch, mode='eval')
    act = {k: np.asarray(v[0]) for k, v in acts.items() if k == act_key}
    obs = env.step({**act, 'reset': False})
    # Forward is negative wheel speed on this robot; mm/s -> m.
    v = -0.5 * (sim.speed[0] + sim.speed[1]) * 1e-3
    x += v / hz
    yaw += -1e-3 * (sim.speed[0] - sim.speed[1]) / sim.track / hz
    rec['t'].append(t)
    rec['theta'].append(sim.theta - sim.theta_zero)
    rec['omega'].append(sim.omega)
    rec['v_l'].append(sim.speed[0])
    rec['v_r'].append(sim.speed[1])
    rec['x'].append(x)
    rec['yaw'].append(yaw)
    rec['cmd_f'].append(sim._cmd[0])
    rec['cmd_t'].append(sim._cmd[1])
    rec['reward'].append(float(obs['reward']))
    a = np.resize(act[act_key], 2)
    rec['act_l'].append(float(a[0]))
    rec['act_r'].append(float(a[1]))
    rec['pushed'].append(pushed)
  np.savez(args.out, hz=hz, task=task, run=pathlib.Path(args.run).name,
           bumper_lo=sim.bumper_lo, bumper_hi=sim.bumper_hi,
           command_speed=sim.command_speed, command_turn=sim.command_turn,
           **{k: np.array(v) for k, v in rec.items()})
  r = np.array(rec['reward'])
  print(f'{pathlib.Path(args.run).name}: {task}, {steps} steps, '
        f'reward/step {r.mean():.3f} (after 3 s: {r[int(3 * hz):].mean():.3f})')


if __name__ == '__main__':
  main()
