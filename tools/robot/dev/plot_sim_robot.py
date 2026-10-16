"""Figures and table for the simulated-robot sweep (docs/SIM_ROBOT.md).

Each run is one simulated robot paced to real time, so the actor's step
counter is a clock: minutes = step / hz / 60. Episodes are 20 s at every
rate, so the score is shown per step (episode score / length): balance pays
at most 1 per step; command (bumper_track) pays tracking in [0, 1] and -2 per
step near a bumper.

  python tools/robot/dev/plot_sim_robot.py --runs DIR [--runs DIR2] table 1480-1495
  python tools/robot/dev/plot_sim_robot.py --runs DIR figure \
      --spec docs/robot/figs/sim_robot_final.json --out docs/robot/figs/sim_robot_final

--runs: directories holding the e<N>_simrobot_* run dirs, searched in order
(e.g. the live run directory, then the archive).

A figure spec is JSON: {"panels": [{"title": ..., "series": [[label, colour,
[exp, ...]], ...]}, ...]} with colours named black/blue/orange/green/sky/
vermilion/pink, two rows of panels.
"""

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

# Directories holding the e<N>_simrobot_* run dirs; set with --runs.
ROOTS = []
COLOURS = dict(
    black='#000000', blue='#0072B2', orange='#E69F00', green='#009E73',
    sky='#56B4E9', vermilion='#D55E00', pink='#CC79A7')
WINDOW = 5
GRID = 0.25  # minutes

sys.path.insert(0, str(pathlib.Path(__file__).parent))


def find(exp):
  for root in ROOTS:
    runs = sorted(root.glob(f'e{exp}_simrobot_*'))
    runs = [r for r in runs if (r / 'logdir/metrics.jsonl').exists()]
    if runs:
      return runs[-1]
  raise FileNotFoundError(f'no run e{exp}_simrobot_*')


_CACHE = {}


def load(exp):
  if exp in _CACHE:
    return _CACHE[exp]
  import ruamel.yaml as yaml
  run = find(exp)
  config = yaml.YAML(typ='safe').load((run / 'logdir/config.yaml').read_text())
  hz = float(config['env']['realtime_hz'])
  steps, scores, fps = [], [], []
  for line in (run / 'logdir/metrics.jsonl').read_text().splitlines():
    try:
      row = json.loads(line)
    except json.JSONDecodeError:
      continue
    if 'episode/score' in row:
      steps.append(row['step'])
      scores.append(row['episode/score'] / max(row['episode/length'], 1))
    if 'fps/train' in row:
      fps.append(row['fps/train'])
  minutes = np.array(steps) / hz / 60
  scores = np.array(scores)
  _CACHE[exp] = dict(run=run.name, hz=hz, minutes=minutes, scores=scores,
                     fps=np.median(fps) if fps else np.nan)
  return _CACHE[exp]


def trailing(exp):
  r = load(exp)
  if len(r['scores']) < WINDOW:
    return r['minutes'][:0], r['scores'][:0]
  trail = np.convolve(r['scores'], np.ones(WINDOW) / WINDOW, mode='valid')
  return r['minutes'][WINDOW - 1:], trail


def first_hold(exp, threshold):
  m, t = trailing(exp)
  hit = np.nonzero(t >= threshold)[0]
  return float(m[hit[0]]) if len(hit) else None


def late_mean(exp, start, end):
  r = load(exp)
  keep = (r['minutes'] >= start) & (r['minutes'] <= end)
  return float(r['scores'][keep].mean()) if keep.any() else np.nan


def parse_exps(args):
  exps = []
  for a in args:
    if '-' in a:
      lo, hi = map(int, a.split('-'))
      exps += list(range(lo, hi + 1))
    else:
      exps.append(int(a))
  return exps


def table(exps, thresholds, late):
  head = ['exp', 'run', 'min', 'eps'] + [f'>{t:g}' for t in thresholds] + [
      f'mean {late[0]:g}-{late[1]:g}', 'last5', 'trainfps']
  print('\t'.join(head))
  for e in exps:
    try:
      r = load(e)
    except FileNotFoundError as err:
      print(f'e{e}\t{err}')
      continue
    m, s = r['minutes'], r['scores']
    cells = [f'e{e}', r['run'].split('_2026')[0], f'{m[-1]:.1f}' if len(m) else '-',
             str(len(s))]
    for t in thresholds:
      h = first_hold(e, t)
      cells.append('-' if h is None else f'{h:.1f}')
    cells.append(f'{late_mean(e, *late):.3f}')
    cells.append(f'{s[-5:].mean():.3f}' if len(s) else '-')
    cells.append(f"{r['fps']:.0f}")
    print('\t'.join(cells))


def figure(spec, out):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from plot_sim_sweep import check_colours
  spec = json.loads(pathlib.Path(spec).read_text())
  panels = spec['panels']
  cols = (len(panels) + 1) // 2
  fig, axes = plt.subplots(2, cols, figsize=(5.2 * cols, 8.6), squeeze=False)
  handles = {}
  for ax, panel in zip(axes.flat, panels):
    ax.set_title(panel['title'], fontsize=17)
    ax.tick_params(labelsize=13)
    ax.grid(color='#d5d5d2', lw=0.7)
    ax.set_axisbelow(True)
    series = panel['series']
    check_colours([COLOURS[c] for _, c, _ in series], panel['title'])
    # Cut every curve where the first (main) series' shortest seed ends.
    xmax = panel.get('xmax') or min(trailing(e)[0][-1] for e in series[0][2])
    grid = np.arange(0, xmax + 1e-9, GRID)
    for label, colour, exps in series:
      curves = []
      for e in exps:
        m, t = trailing(e)
        c = np.interp(grid, m, t, left=np.nan, right=np.nan)
        curves.append(c)
      curves = np.array(curves)
      end = min(trailing(e)[0][-1] for e in exps)
      curves[:, grid > end] = np.nan
      with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean, std = np.nanmean(curves, 0), np.nanstd(curves, 0)
      name = f'{label} (n={len(exps)})'
      line, = ax.plot(grid, mean, color=COLOURS[colour], lw=2.4, label=name)
      if len(exps) > 1:
        ax.fill_between(grid, mean - std, mean + std, color=COLOURS[colour],
                        alpha=0.15, lw=0)
      handles.setdefault(label, line)
    ax.set_xlim(0, xmax)
    if 'ylim' in panel:
      ax.set_ylim(*panel['ylim'])
    if panel.get('legend', True):
      ax.legend(fontsize=11, frameon=False, loc='lower right')
  for ax in axes.flat[len(panels):]:
    ax.axis('off')
  for ax in axes[:, 0]:
    ax.set_ylabel('Reward per step\n(trailing 5 episodes)', fontsize=15)
  bottom = 0.05
  if spec.get('shared_legend'):
    ncol = min(len(handles), 3)
    rows = -(-len(handles) // ncol)
    fig.legend(handles.values(), handles.keys(), loc='lower center',
               ncol=ncol, fontsize=14, frameon=False)
    bottom = 0.06 + 0.04 * rows
  fig.tight_layout(rect=(0, bottom, 1, 1), h_pad=2)
  fig.text(0.5, bottom - 0.02, 'Real time (minutes)', ha='center', fontsize=15)
  out = pathlib.Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out.with_suffix('.png'), dpi=200, facecolor='white')
  fig.savefig(out.with_suffix('.pdf'), facecolor='white')
  print('wrote', out.with_suffix('.png'))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--runs', action='append', required=True,
                      help='directory of e<N>_simrobot_* run dirs (repeatable)')
  sub = parser.add_subparsers(dest='cmd', required=True)
  t = sub.add_parser('table')
  t.add_argument('exps', nargs='+')
  t.add_argument('--thresholds', type=float, nargs='+', default=[0.3, 0.5, 0.7])
  t.add_argument('--late', type=float, nargs=2, default=[40, 55])
  f = sub.add_parser('figure')
  f.add_argument('--spec', required=True)
  f.add_argument('--out', required=True)
  args = parser.parse_args()
  ROOTS[:] = [pathlib.Path(r) for r in args.runs]
  if args.cmd == 'table':
    table(parse_exps(args.exps), args.thresholds, args.late)
  else:
    figure(args.spec, args.out)


if __name__ == '__main__':
  main()
