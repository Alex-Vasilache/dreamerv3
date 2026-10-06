"""Score against real time for `sim_phone` runs.

Every env step is paced to 1 / env.realtime_hz seconds (wrappers.RealTime), so
the actor's step counter is a clock. It counts simulator frames (agent steps x
action repeat), so minutes = step / (hz * repeat) / 60. The headline
number is how many minutes of robot time it took to *hold* a score, i.e. the
first point where the trailing mean of `--window` episodes reaches each
threshold, plus the mean over the last `--window` episodes.

  python tools/sim_curve.py /work/DoyaU/vasilache/work/e1113_*
  python tools/sim_curve.py --plot out.png RUN [RUN ...]
"""

import argparse
import json
import pathlib

import numpy as np
import ruamel.yaml as yaml


def load(run):
  run = pathlib.Path(run)
  logdir = run / 'logdir' if (run / 'logdir').exists() else run
  config = yaml.YAML(typ='safe').load((logdir / 'config.yaml').read_text())
  hz = float(config['env']['realtime_hz'])
  # The logged step counts simulator frames: agent steps times action repeat.
  frames_hz = hz * (config['env']['dmc']['repeat']
                    if config['task'].startswith('dmc_') else 1)
  steps, scores, lags, train_fps = [], [], [], []
  for line in (logdir / 'metrics.jsonl').read_text().splitlines():
    try:
      row = json.loads(line)
    except json.JSONDecodeError:
      continue
    if 'episode/score' in row:
      steps.append(row['step'])
      scores.append(row['episode/score'])
    for key in ('epstats/log/rt_lag_ms/max', 'epstats/log/rt_lag_ms/avg'):
      if key in row:
        lags.append(row[key])
        break
    if 'fps/train' in row:
      train_fps.append(row['fps/train'])
  return dict(
      name=run.name, hz=hz, config=config,
      minutes=np.array(steps) / frames_hz / 60, scores=np.array(scores),
      lag=max(lags) if lags else float('nan'),
      train_fps=np.median(train_fps) if train_fps else float('nan'))


def first_hold(minutes, scores, threshold, window):
  if len(scores) < window:
    return None
  trail = np.convolve(scores, np.ones(window) / window, mode='valid')
  hit = np.nonzero(trail >= threshold)[0]
  return float(minutes[hit[0] + window - 1]) if len(hit) else None


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('runs', nargs='+')
  parser.add_argument('--window', type=int, default=5)
  parser.add_argument('--thresholds', type=float, nargs='+',
                      default=[500, 700, 800])
  parser.add_argument('--plot', default='')
  args = parser.parse_args()
  rows = []
  for run in args.runs:
    try:
      rows.append(load(run))
    except (FileNotFoundError, KeyError) as e:
      print(f'skip {run}: {e}')
  head = ['run', 'hz', 'eps', 'min'] + [
      f't>{int(t)}' for t in args.thresholds] + [
      f'last{args.window}', 'best', 'trainfps', 'maxlag']
  print('  '.join(f'{h:>9}' for h in head))
  for r in rows:
    m, s = r['minutes'], r['scores']
    cells = [r['name'][:40], f"{r['hz']:g}", str(len(s)),
             f'{m[-1]:.1f}' if len(m) else '-']
    for t in args.thresholds:
      hit = first_hold(m, s, t, args.window)
      cells.append('-' if hit is None else f'{hit:.1f}')
    cells.append(f'{s[-args.window:].mean():.0f}' if len(s) else '-')
    cells.append(f'{s.max():.0f}' if len(s) else '-')
    cells.append(f"{r['train_fps']:.0f}")
    cells.append(f"{r['lag']:.1f}")
    print(cells[0])
    print('  '.join(f'{c:>9}' for c in [''] + cells[1:]))
  if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in rows:
      s = r['scores']
      if len(s) >= args.window:
        trail = np.convolve(s, np.ones(args.window) / args.window, 'valid')
        ax.plot(r['minutes'][args.window - 1:], trail, label=r['name'][:32])
    ax.set_xlabel('minutes of real time (robot time)')
    ax.set_ylabel(f'episode score, trailing {args.window}')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120)
    print('wrote', args.plot)


if __name__ == '__main__':
  main()
