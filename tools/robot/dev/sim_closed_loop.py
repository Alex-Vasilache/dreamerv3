"""Run a trained policy in the MuJoCo sim, closed loop, as it runs on the robot.

For comparing a policy's behaviour on the robot (a recording made with
training paused) with the sim's, and for finding which change to the sim
reproduces the robot. The policy samples its actions ('train' mode, as the
phone's runner does) and nothing pushes it; each variant is a fresh episode
of --minutes from the back bumper, with perturbations the sim does not
otherwise have:

  eq=DEG          balance point, degrees from theta_zero (default: the run's
                  own draw, which is fixed for a whole training run)
  gain_l, gain_r  scale each wheel's output at the motor (phone order: _l is
                  the robot's right wheel)
  bias_l, bias_r  add to each wheel's output at the motor
  delay=N         N extra ticks between an action and the motor
  noise=X         sim_noise (sensor noise scale; 1 is the default)
  surface=NAME    floor

  python tools/robot/dev/sim_closed_loop.py RUN_DIR OUT.json [--ckpt STEP] \
      [--minutes 5] --variant base: --variant back:eq=-0.9 \
      --variant asym:gain_l=0.9,bias_r=0.03

Writes per-variant summary statistics (same as the s2r comparison) and, with
--traces, the per-step observations and actions to OUT.npz.
"""

import argparse
import collections
import json
import math
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def unwrap(env):
  while hasattr(env, 'env'):
    env = env.env
  return env


def parse_variant(spec):
  name, _, rest = spec.partition(':')
  opts = {}
  for part in [p for p in rest.split(',') if p]:
    k, _, v = part.partition('=')
    opts[k] = v if k == 'surface' else float(v)
  return name, opts


def summary(rec, r, hz, skip_s=30.0):
  th = np.degrees(np.array(rec['theta']))
  rate = np.degrees(np.array(rec['rate']))
  w = np.array(rec['wheels'])
  a = np.array(rec['act'])
  s = int(skip_s * hz)
  th, rate, w, a = th[s:], rate[s:], w[s:], a[s:]
  bump = ((th > math.degrees(r['bumper_hi']) - 1.5)
          | (th < math.degrees(r['bumper_lo']) + 1.5))
  up = ~bump
  fwd = 0.5 * (w[:, 0] + w[:, 1])
  turn = 0.5 * (w[:, 0] - w[:, 1])
  med = lambda x: float(np.median(np.abs(x))) if len(x) else float('nan')
  return dict(
      bumper_frac=float(bump.mean()),
      tilt_mean=float(th[up].mean()), tilt_std=float(th[up].std()),
      rate_std=float(rate[up].std()),
      fwd_abs_med=med(fwd[up]), fwd_mean=float(fwd[up].mean()),
      turn_abs_med=med(turn[up]), turn_mean=float(turn[up].mean()),
      act_mean=float(a[up].mean()), act_std=float(a[up].mean(1).std()),
      act_jump=float(np.abs(np.diff(a[up], axis=0)).mean()),
      act_lr=float(0.5 * (a[up, 0] - a[up, 1]).mean()),
      reward=float(np.mean(rec['reward'][s:])))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('run')
  ap.add_argument('out')
  ap.add_argument('--ckpt', default='latest')
  ap.add_argument('--minutes', type=float, default=5.0)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--gpu', action='store_true')
  ap.add_argument('--variant', action='append', default=None)
  ap.add_argument('--traces', action='store_true')
  args = ap.parse_args()
  variants = [parse_variant(v) for v in (args.variant or ['base:'])]

  import elements
  import ruamel.yaml as yaml
  from dreamerv3 import main as dmain

  logdir = pathlib.Path(args.run) / 'logdir'
  raw = yaml.YAML(typ='safe').load((logdir / 'config.yaml').read_text())
  r = raw['env']['robot']
  base = [
      '--env.realtime_hz', '0', '--logdir', tempfile.mkdtemp(),
      '--jax.platform', 'cuda' if args.gpu else 'cpu', '--jax.prealloc',
      'False', '--env.robot.length', '0', '--seed', str(args.seed)]
  if 'sim_schedule' in r:
    base += ['--env.robot.sim_schedule', '']
  config = elements.Flags(elements.Config(raw)).parse(base)
  agent = dmain.make_agent(config)
  cp = elements.Checkpoint()
  cp.agent = agent
  if args.ckpt == 'latest':
    ckpt = logdir / 'ckpt' / (logdir / 'ckpt' / 'latest').read_text().strip()
  else:
    ckpt = logdir / 'ckpt_milestones' / f'{int(args.ckpt):012d}'
  cp.load(str(ckpt), keys=['agent'])
  keys = list(agent.obs_space)
  act_key = list(agent.act_space)[0]

  results, traces = {}, {}
  for vi, (name, opts) in enumerate(variants):
    over = []
    if 'surface' in opts:
      over += ['--env.robot.sim_surface', opts['surface']]
    if 'noise' in opts:
      over += ['--env.robot.sim_noise', str(opts['noise'])]
    cfg = elements.Flags(config).parse(over) if over else config
    env = dmain.make_env(cfg, 0)
    sim = unwrap(env)
    hz = sim.hz
    if 'eq' in opts:
      sim.theta_eq = sim.theta_zero + math.radians(opts['eq'])
    sim.sim_rng = np.random.RandomState(1000 + vi + args.seed)
    gain = np.array([opts.get('gain_l', 1.0), opts.get('gain_r', 1.0)])
    bias = np.array([opts.get('bias_l', 0.0), opts.get('bias_r', 0.0)])
    queue = collections.deque([np.zeros(2)] * int(opts.get('delay', 0)))
    step = sim.physics.step

    def patched(sent, n, step=step, gain=gain, bias=bias, queue=queue):
      sent = np.clip(np.asarray(sent, float) * gain + bias, -1, 1)
      if queue:
        queue.append(sent)
        sent = queue.popleft()
      return step(sent, n)

    sim.physics.step = patched
    carry = agent.init_policy(1)
    zero = {act_key: np.zeros(agent.act_space[act_key].shape, np.float32)}
    obs = env.step({**zero, 'reset': True})
    rec = {k: [] for k in ('theta', 'rate', 'wheels', 'act', 'reward')}
    for _ in range(int(args.minutes * 60 * hz)):
      batch = {k: np.asarray(obs[k])[None] for k in keys}
      carry, acts, _ = agent.policy(carry, batch, mode='train')
      act = {act_key: np.asarray(acts[act_key][0])}
      obs = env.step({**act, 'reset': False})
      if obs['is_last']:
        obs = env.step({**zero, 'reset': True})
      rec['theta'].append(sim.theta - sim.theta_zero)
      rec['rate'].append(sim.omega)
      rec['wheels'].append(np.array(sim.speed))
      rec['act'].append(np.resize(act[act_key], 2).astype(float))
      rec['reward'].append(float(obs['reward']))
    results[name] = dict(opts=opts, theta_eq_deg=math.degrees(
        sim.theta_eq - sim.theta_zero), **summary(rec, r, hz))
    s = results[name]
    print(f'{name:14s} eq {s["theta_eq_deg"]:+.2f} | bumper {s["bumper_frac"]:.2f} '
          f'tilt {s["tilt_mean"]:+.2f}±{s["tilt_std"]:.2f} rate±{s["rate_std"]:5.1f} '
          f'|fwd| {s["fwd_abs_med"]:5.0f} (mean {s["fwd_mean"]:+5.0f}) '
          f'|turn| {s["turn_abs_med"]:5.0f} (mean {s["turn_mean"]:+5.0f}) '
          f'act lr {s["act_lr"]:+.3f} jump {s["act_jump"]:.3f}', flush=True)
    if args.traces:
      traces.update({f'{name}/{k}': np.array(v) for k, v in rec.items()})
  pathlib.Path(args.out).write_text(json.dumps(results, indent=1))
  if args.traces:
    np.savez(str(pathlib.Path(args.out).with_suffix('.npz')), hz=hz, **traces)


if __name__ == '__main__':
  main()
