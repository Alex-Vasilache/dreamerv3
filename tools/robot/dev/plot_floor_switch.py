"""Balance across a change of floor: reward, time at a bumper and wheel speed
against minutes of real time, from each run's experience.

The floor schedule is read from the run's env.robot.sim_schedule (the sim's
'carpet:12,table:25,carpet') and drawn as shaded bands; runs without one sit
on env.robot.sim_surface throughout. A summary table gives, per series and
per floor segment, the mean reward over the segment's last 3 minutes and the
minutes after the switch until the 1-minute reward first holds above
--level (mean +- std over seeds).

  python tools/robot/dev/plot_floor_switch.py --runs DIR \
      --series 'switch:black:2080,2081,2082' --series 'table only:blue:2083,2084' \
      --out docs/robot/figs/floor_switch_r1 [--title ...] [--level 0.6]
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
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from plot_sim_sweep import check_colours  # noqa: E402

COLOURS = dict(
    black='#000000', blue='#0072B2', orange='#E69F00', green='#009E73',
    sky='#56B4E9', vermilion='#D55E00', pink='#CC79A7')
FLOOR_SHADE = {'carpet': '#f1e6d6', 'table': '#ffffff'}
WINDOW_MIN = 1.0
GRID = 0.25
ROOTS = []


def find(exp):
  for root in ROOTS:
    hits = sorted(pathlib.Path(root).glob(f'e{exp}_*'))
    if hits:
      return hits[-1]
  raise FileNotFoundError(f'e{exp} not under {ROOTS}')


def schedule(r):
  sched = str(r.get('sim_schedule') or '')
  out, at = [], 0.0
  for part in [p for p in sched.split(',') if p.strip()]:
    name, _, minutes = part.strip().partition(':')
    out.append((at, name))
    at += float(minutes) if minutes else np.inf
  return out or [(0.0, r.get('sim_surface', 'carpet'))]


def load(run):
  logdir = pathlib.Path(run) / 'logdir'
  cfg = yaml.safe_load((logdir / 'config.yaml').read_text())
  r = cfg['env']['robot']
  keys = ('wheels', 'orientation', 'reward')
  parts = {k: [] for k in keys}
  for f in sorted(glob.glob(str(logdir / 'online_shared/experience/*.npz'))):
    try:
      z = np.load(f)
      got = [z[k] for k in keys]
    except Exception:
      continue
    for k, v in zip(keys, got):
      parts[k].append(v)
  d = {k: np.concatenate(v) for k, v in parts.items()}
  hz = r.get('hz', 65.0)
  theta = d['orientation'][:, 0] * r['obs_theta_scale']
  margin = r.get('bumper_margin', 0.0175)
  near = ((theta >= r.get('bumper_hi', 0.161) - margin)
          | (theta <= r.get('bumper_lo', -0.120) + margin))
  speed = np.abs(d['wheels'] / r['obs_wheel_scale']).mean(1)
  width, stride = int(WINDOW_MIN * 60 * hz), int(GRID * 60 * hz)
  out = dict(minute=[], reward=[], bumper=[], speed=[])
  for end in range(width, len(theta) + 1, stride):
    s = slice(end - width, end)
    out['minute'].append(end / hz / 60)
    out['reward'].append(d['reward'][s].mean())
    out['bumper'].append(near[s].mean())
    out['speed'].append(np.median(speed[s]))
  out = {k: np.array(v) for k, v in out.items()}
  out['schedule'] = schedule(r)
  return out


PANELS = [('reward', 'Reward per step (1-min mean)', (-0.3, 1.0)),
          ('bumper', 'Time at a bumper', (0, 1)),
          ('speed', 'Median |wheel speed| (phone mm/s)', (0, 4000))]


def recovery(run, level):
  """Per floor segment: minutes to hold `level` and the late mean reward."""
  rows = []
  sched = run['schedule'] + [(np.inf, None)]
  for (start, name), (end, _) in zip(sched[:-1], sched[1:]):
    end = min(end, run['minute'][-1])
    seg = (run['minute'] > start + WINDOW_MIN) & (run['minute'] <= end)
    if seg.sum() < 4:
      continue
    m, rew = run['minute'][seg], run['reward'][seg]
    above = rew >= level
    # First time from which the next 2 minutes all stay above.
    hold = np.nan
    k = int(2 / GRID)
    for i in range(len(above) - k + 1):
      if above[i:i + k].all():
        hold = m[i] - start
        break
    late = rew[m > end - 3].mean()
    rows.append((name, start, hold, late))
  return rows


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--runs', action='append', required=True)
  parser.add_argument('--series', action='append', required=True)
  parser.add_argument('--out', required=True)
  parser.add_argument('--title', default='')
  parser.add_argument('--level', type=float, default=0.6)
  args = parser.parse_args()
  ROOTS.extend(args.runs)
  series = []
  for spec in args.series:
    label, colour, exps = spec.rsplit(':', 2)
    series.append((label, COLOURS[colour], [int(e) for e in exps.split(',')]))
  check_colours([c for _, c, _ in series], args.out)
  data = {e: load(find(e)) for _, _, exps in series for e in exps}
  data = {e: d for e, d in data.items() if len(d['minute']) > 4}
  series = [(l, c, [e for e in exps if e in data]) for l, c, exps in series]
  series = [s for s in series if s[2]]

  print(f'{"series":24s} {"segment":16s} {"min to hold":>14s} '
        f'{"late reward":>14s}  n')
  for label, _, exps in series:
    table = {}
    for e in exps:
      for name, start, hold, late in recovery(data[e], args.level):
        table.setdefault((start, name), []).append((hold, late))
    for (start, name), vals in sorted(table.items()):
      h = np.array([v[0] for v in vals])
      lt = np.array([v[1] for v in vals])
      done = np.isfinite(h)
      hs = (f'{np.mean(h[done]):5.1f} ± {np.std(h[done]):3.1f}'
            if done.any() else '   never')
      if done.any() and not done.all():
        hs += f' ({done.sum()}/{len(h)})'
      print(f'{label:24s} {name + " @" + format(start, ".0f"):16s} '
            f'{hs:>14s} {np.mean(lt):+6.2f} ± {np.std(lt):4.2f}  {len(vals)}')

  fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
  sched = data[series[0][2][0]]['schedule']
  xmax = max(min(data[e]['minute'][-1] for e in exps) for _, _, exps in series)
  for ax, (key, title, ylim) in zip(axes, PANELS):
    for (start, name), nxt in zip(sched, sched[1:] + [(xmax, None)]):
      ax.axvspan(start, min(nxt[0], xmax), color=FLOOR_SHADE.get(name, '#eeeeee'),
                 zorder=0, lw=0)
      if key == 'reward' and start < xmax:
        ax.text(start + 0.3, ylim[1] - 0.08, name, fontsize=13, va='top',
                color='#606060')
    for label, colour, exps in series:
      runs = [data[e] for e in exps if len(data[e]['minute'])]
      end = min(r['minute'][-1] for r in runs)
      grid = np.arange(WINDOW_MIN, end + 1e-9, GRID)
      curves = np.array([np.interp(grid, r['minute'], r[key]) for r in runs])
      with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean, std = curves.mean(0), curves.std(0)
      ax.plot(grid, mean, color=colour, lw=2.4, label=f'{label} (n={len(runs)})')
      if len(runs) > 1:
        ax.fill_between(grid, mean - std, mean + std, color=colour, alpha=0.15)
    ax.set_title(title, fontsize=17)
    ax.set_ylim(*ylim)
    ax.set_xlim(0, xmax)
    ax.tick_params(labelsize=13)
    ax.grid(color='#d5d5d2', lw=0.7)
    ax.set_axisbelow(True)
  axes[-1].set_xlabel('Minutes of real time (shaded: carpet)', fontsize=15)
  if args.title:
    fig.suptitle(args.title, fontsize=17)
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center', ncol=min(3, len(labels)),
             frameon=False, fontsize=14)
  rows = (len(labels) + 2) // 3
  fig.tight_layout(rect=(0, 0.03 + 0.035 * rows, 1, 0.97 if args.title else 1))
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out.with_suffix('.png'), dpi=200, facecolor='white')
  fig.savefig(out.with_suffix('.pdf'), facecolor='white')
  print('wrote', out.with_suffix('.png'))


if __name__ == '__main__':
  main()
