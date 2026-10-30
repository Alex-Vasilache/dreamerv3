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
  stick=X, slide=X  the gearbox's dry friction, as fractions of stall torque
                  (fitted 0.385 / 0.098)
  rstatic=X       rolling stiction, as a multiple of the floor's rolling
                  resistance (1: none; fitted carpet 4.5, table 6)
  sim_<key>=X     any env.robot.sim_* key, e.g. sim_inertia=4
  fit=PATH        a sysid_fit.py result: its motor, body and floor (the
                  run's surface, or surface=) as env.robot.sim_* keys

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
    opts[k] = v if k in ('surface', 'fit', 'sim_rand') else float(v)
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
  extra = {}
  if rec.get('cmd'):
    c = np.array(rec['cmd'])[s:][up]
    f, t = r['forward_sign'] * fwd[up] / r['command_speed'], turn[up] / r['command_turn']
    extra = dict(corr_f=float(np.corrcoef(f, c[:, 0])[0, 1]),
                 corr_t=float(np.corrcoef(t, c[:, 1])[0, 1]))
  return dict(
      **extra,
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
  ap.add_argument('--mode', default='train',
                  help="'eval' plays the most likely action, as the phone's "
                       'recorder does for saved policies')
  args = ap.parse_args()
  variants = [parse_variant(v) for v in (args.variant or ['base:'])]

  import elements
  import ruamel.yaml as yaml
  from dreamerv3 import main as dmain

  logdir = pathlib.Path(args.run) / 'logdir'
  raw = yaml.YAML(typ='safe').load((logdir / 'config.yaml').read_text())
  r = raw['env']['robot']
  r.setdefault('sim_inertia', 1.0)   # runs from before the key
  for k, v in dict(sim_backlash=0.0, sim_gear_k=2.0, sim_friction=1.0, sim_rand='',
                   sim_rand_every=0.0, sim_tilt_lag=0.0).items():
    r.setdefault(k, v)
  base = [
      '--env.realtime_hz', '0', '--logdir', tempfile.mkdtemp(),
      '--jax.platform', 'cuda' if args.gpu else 'cpu', '--jax.prealloc',
      'False', '--env.robot.length', '0', '--seed', str(args.seed)]
  if 'sim_schedule' in r:
    base += ['--env.robot.sim_schedule', '']
  if r.get('sim_eq_every'):
    base += ['--env.robot.sim_eq_every', '0']   # each variant holds its own
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
    if 'fit' in opts:
      fit = json.loads(pathlib.Path(opts['fit']).read_text())
      m, fl = fit['motor'], fit['floors'][opts.get('surface', r['sim_surface'])]
      fitted = dict(sim_wheel_omega=m['wheel_omega'], sim_stall_torque=m['stall'],
                    sim_stick=m['stick'], sim_slide=m['slide'],
                    sim_armature=m['armature'], sim_inertia=m.get('inertia', 1.0),
                    sim_backlash=m.get('backlash', 0.0), sim_gear_k=m.get('gear_k', 2.0),
                    sim_tilt_lag=m.get('tilt_lag', 0.0))
      opts = {**fitted, **opts}
    for k, v in opts.items():
      if k.startswith('sim_'):
        over += [f'--env.robot.{k}', str(v)]
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
    ph = sim.physics
    if 'fit' in opts:   # the floor: not env keys in older runs' configs
      ph.set_floor(fl['rolling'], fl['contact_tc'], fl['bumper_friction'],
                   fl['drag'], rolling_static=fl['rolling'] * fl.get('stiction', 1.0),
                   noslip=20 if fl.get('stiction', 1.0) > 1.0 else 0)
    if 'stick' in opts:
      ph.stick = opts['stick'] * ph.stall
    if 'slide' in opts:
      ph.slide = opts['slide'] * ph.stall
    if 'rstatic' in opts:
      ph.rolling_static = opts['rstatic'] * ph.rolling
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
    rec = {k: [] for k in ('theta', 'rate', 'wheels', 'act', 'reward', 'cmd')}
    for _ in range(int(args.minutes * 60 * hz)):
      batch = {k: np.asarray(obs[k])[None] for k in keys}
      carry, acts, _ = agent.policy(carry, batch, mode=args.mode)
      act = {act_key: np.asarray(acts[act_key][0])}
      obs = env.step({**act, 'reset': False})
      if obs['is_last']:
        obs = env.step({**zero, 'reset': True})
      rec['theta'].append(sim.theta - sim.theta_zero)
      rec['rate'].append(sim.omega)
      rec['wheels'].append(np.array(sim.speed))
      rec['act'].append(np.resize(act[act_key], 2).astype(float))
      rec['reward'].append(float(obs['reward']))
      if 'command' in obs:
        rec['cmd'].append(np.array(obs['command'], float))
    results[name] = dict(opts=opts, theta_eq_deg=math.degrees(
        sim.theta_eq - sim.theta_zero), **summary(rec, r, hz))
    s = results[name]
    print(f'{name:14s} eq {s["theta_eq_deg"]:+.2f} | bumper {s["bumper_frac"]:.2f} '
          f'tilt {s["tilt_mean"]:+.2f}±{s["tilt_std"]:.2f} rate±{s["rate_std"]:5.1f} '
          f'|fwd| {s["fwd_abs_med"]:5.0f} (mean {s["fwd_mean"]:+5.0f}) '
          f'|turn| {s["turn_abs_med"]:5.0f} (mean {s["turn_mean"]:+5.0f}) '
          f'act lr {s["act_lr"]:+.3f} jump {s["act_jump"]:.3f}'
          + (f' | track fwd {s["corr_f"]:+.2f} turn {s["corr_t"]:+.2f}'
             if 'corr_f' in s else ''), flush=True)
    if args.traces:
      traces.update({f'{name}/{k}': np.array(v) for k, v in rec.items() if v})
  pathlib.Path(args.out).write_text(json.dumps(results, indent=1))
  if args.traces:
    np.savez(str(pathlib.Path(args.out).with_suffix('.npz')), hz=hz, **traces)


if __name__ == '__main__':
  main()
