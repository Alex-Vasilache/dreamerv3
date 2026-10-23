"""How well a command-following run follows its commands, over real time.

Reads each run's experience (logdir/online_shared/experience/*.npz: the
command the policy saw, the wheel speeds and tilt it produced) and, in
sliding windows of real time, measures:

  forward  correlation of commanded with measured forward speed
  turn     the same for the turn rate
  gain     least-squares slope of measured on commanded forward speed
           (1: drives as fast as asked; 0: ignores the command)
  bumper   fraction of steps within the bumper margin

Works on real-robot and simulated runs alike: measured speeds come back
from the observation through the run's own obs_wheel_scale, forward_sign,
command_speed and command_turn, so they are in command units.

  python tools/robot/dev/plot_cmd_tracking.py --runs DIR [--runs DIR2] \
      --series 'current:black:2070,2071' --series 'slow:blue:2072,2073' \
      --out docs/robot/figs/sim_cmd_r1 [--title ...]

Prints a table of the last-window values per run as well.
"""

import argparse
import glob
import pathlib
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from plot_sim_sweep import check_colours  # noqa: E402

COLOURS = dict(
    black='#000000', blue='#0072B2', orange='#E69F00', green='#009E73',
    sky='#56B4E9', vermilion='#D55E00', pink='#CC79A7')
WINDOW_MIN = 3.0   # minutes per window
GRID = 0.5         # minutes between windows
ROOTS = []


def find(exp):
  for root in ROOTS:
    hits = sorted(pathlib.Path(root).glob(f'e{exp}_*'))
    if hits:
      return hits[-1]
  raise FileNotFoundError(f'e{exp} not under {ROOTS}')


def load(run):
  import yaml
  logdir = pathlib.Path(run) / 'logdir'
  cfg = yaml.safe_load((logdir / 'config.yaml').read_text())
  r = cfg['env']['robot']
  files = sorted(glob.glob(str(logdir / 'online_shared/experience/*.npz')))
  keys = ('wheels', 'orientation', 'command', 'reward')
  data = {k: [] for k in keys}
  for f in files:
    try:
      chunk = np.load(f)
      parts = [chunk[k] for k in keys]
    except Exception:
      continue  # a chunk being written
    for k, p in zip(keys, parts):
      data[k].append(p)
  data = {k: np.concatenate(v) for k, v in data.items()}
  wheels = data['wheels'] / r['obs_wheel_scale']
  fwd = r['forward_sign'] * 0.5 * (wheels[:, 0] + wheels[:, 1]) / r['command_speed']
  turn = 0.5 * (wheels[:, 0] - wheels[:, 1]) / r['command_turn']
  theta = data['orientation'][:, 0] * r['obs_theta_scale']
  margin = r.get('bumper_margin', 0.0262)
  near = ((theta >= r.get('bumper_hi', 0.161) - margin)
          | (theta <= r.get('bumper_lo', -0.120) + margin))
  hz = r.get('hz', 50.0)
  return dict(cmd=data['command'], fwd=fwd, turn=turn, near=near,
              reward=data['reward'], hz=hz)


def _corr(a, b):
  if a.std() < 1e-3 or b.std() < 1e-6:
    return np.nan
  return float(np.corrcoef(a, b)[0, 1])


def windows(d):
  n = len(d['reward'])
  width = int(WINDOW_MIN * 60 * d['hz'])
  stride = int(GRID * 60 * d['hz'])
  out = dict(minute=[], forward=[], turn=[], gain=[], bumper=[], reward=[])
  for end in range(width, n + 1, stride):
    s = slice(end - width, end)
    cf, ct = d['cmd'][s, 0], d['cmd'][s, 1]
    out['minute'].append(end / d['hz'] / 60)
    out['forward'].append(_corr(cf, d['fwd'][s]))
    out['turn'].append(_corr(ct, d['turn'][s]))
    out['gain'].append(float(np.dot(cf, d['fwd'][s]) / max(np.dot(cf, cf), 1e-6)))
    out['bumper'].append(float(d['near'][s].mean()))
    out['reward'].append(float(d['reward'][s].mean()))
  return {k: np.array(v) for k, v in out.items()}


PANELS = [('forward', 'Forward speed: corr(command, measured)', (-0.3, 1)),
          ('turn', 'Turn rate: corr(command, measured)', (-0.3, 1)),
          ('gain', 'Forward gain (measured / commanded)', (-0.3, 1.2)),
          ('bumper', 'Time at a bumper', (0, 1))]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--runs', action='append', required=True)
  parser.add_argument('--series', action='append', required=True,
                      help='label:colour:exp,exp,...')
  parser.add_argument('--out', required=True)
  parser.add_argument('--title', default='')
  args = parser.parse_args()
  ROOTS.extend(args.runs)
  series = []
  for spec in args.series:
    label, colour, exps = spec.rsplit(':', 2)
    series.append((label, COLOURS[colour], [int(e) for e in exps.split(',')]))
  check_colours([c for _, c, _ in series], args.out)

  results = {}
  print(f'{"run":>6} {"min":>5} {"fwd":>6} {"turn":>6} {"gain":>6} '
        f'{"bumper":>6} {"reward":>7}')
  for _, _, exps in series:
    for e in exps:
      w = windows(load(find(e)))
      results[e] = w
      if len(w['minute']):
        late = slice(-4, None)  # last ~2 minutes of windows
        print(f'e{e:>5} {w["minute"][-1]:5.1f} '
              f'{np.nanmean(w["forward"][late]):+6.2f} '
              f'{np.nanmean(w["turn"][late]):+6.2f} '
              f'{np.nanmean(w["gain"][late]):+6.2f} '
              f'{np.nanmean(w["bumper"][late]):6.2f} '
              f'{np.nanmean(w["reward"][late]):+7.3f}')

  fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
  for ax, (key, title, ylim) in zip(axes.flat, PANELS):
    for label, colour, exps in series:
      runs = [results[e] for e in exps if len(results[e]['minute'])]
      if not runs:
        continue
      end = min(r['minute'][-1] for r in runs)
      grid = np.arange(WINDOW_MIN, end + 1e-9, GRID)
      curves = np.array([np.interp(grid, r['minute'], r[key]) for r in runs])
      with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean, std = np.nanmean(curves, 0), np.nanstd(curves, 0)
      ax.plot(grid, mean, color=colour, lw=2.4, label=f'{label} (n={len(runs)})')
      if len(runs) > 1:
        ax.fill_between(grid, mean - std, mean + std, color=colour, alpha=0.15)
    ax.set_title(title, fontsize=17)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=13)
    ax.grid(color='#d5d5d2', lw=0.7)
    ax.set_axisbelow(True)
    ax.axhline(0, color='#808080', lw=0.8)
  for ax in axes[:, 0]:
    ax.set_ylabel('')
  fig.text(0.5, 0.075, 'Minutes of real time', ha='center', fontsize=15)
  if args.title:
    fig.suptitle(args.title, fontsize=17)
  handles, labels = axes.flat[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center', ncol=min(4, len(labels)),
             frameon=False, fontsize=14)
  fig.tight_layout(rect=(0, 0.09, 1, 0.97 if args.title else 1))
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out.with_suffix('.png'), dpi=200, facecolor='white')
  fig.savefig(out.with_suffix('.pdf'), facecolor='white')
  print('wrote', out.with_suffix('.png'))


if __name__ == '__main__':
  main()
