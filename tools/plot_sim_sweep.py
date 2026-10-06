"""Figures for the real-time cartpole sweep (docs/SIM_PHONE.md, e1113-e1196).

Reads the archived runs from the bucket and plots trailing-5 episode return
against minutes of real time, mean +-1 std over seeds:

  sim_sweep_hparams  one panel per question the sweep asked (cartpole swingup)
  sim_sweep_tasks    the recipe against the base on four DMC tasks
  sim_sweep_round14  delay, lr split, width, entropy, rate (e1215-e1222)

  python tools/plot_sim_sweep.py --out docs/figs
"""

import argparse
import json
import pathlib
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BUCKET = pathlib.Path('/bucket/DoyaU/vasilache/bucket/results/dreamerv3')

BLACK, BLUE, ORANGE, GREEN = '#000000', '#0072B2', '#E69F00', '#009E73'
SKY, VERMILION, PINK = '#56B4E9', '#D55E00', '#CC79A7'

WINDOW = 5
GRID = 0.25  # minutes


# --- colour check: deuteranopia, Machado et al. 2009 severity 1 -------------

MACHADO_DEUTAN = np.array([
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501, 0.047413],
    [-0.011820, 0.042940, 0.968881]])


def _linear(hex_):
  rgb = np.array([int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5)])
  return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _oklab(lin):
  lms = np.array([
      [0.4122214708, 0.5363325363, 0.0514459929],
      [0.2119034982, 0.6806995451, 0.1073969566],
      [0.0883024619, 0.2817188376, 0.6299787005]]) @ lin
  return np.array([
      [0.2104542553, 0.7936177850, -0.0040720468],
      [1.9779984951, -2.4285922050, 0.4505937099],
      [0.0259040371, 0.7827717662, -0.8086757660]]) @ np.cbrt(lms)


def cvd_distance(a, b):
  sim = lambda h: _oklab(np.clip(MACHADO_DEUTAN @ _linear(h), 0, 1))
  return 100 * np.linalg.norm(sim(a) - sim(b))


def check_colours(colours, where):
  for i, a in enumerate(colours):
    for b in colours[i + 1:]:
      d = cvd_distance(a, b)
      assert d >= 8, f'{where}: {a} vs {b} deuteranopia dE {d:.1f} < 8'
      if d < 15:
        print(f'  note {where}: {a} vs {b} deuteranopia dE {d:.1f} (< 15)')


# --- data -------------------------------------------------------------------

def load(exp):
  import ruamel.yaml as yaml  # noqa: only needed here
  # Experiment numbers have collided with other workflows (e1197-e1204), so
  # take the run that was paced to real time, not just the first match.
  for run in sorted(BUCKET.glob(f'e{exp}_*')):
    config = yaml.YAML(typ='safe').load(
        (run / 'logdir/config.yaml').read_text())
    if config['env'].get('realtime_hz', 0):
      break
  else:
    raise FileNotFoundError(f'no real-time run e{exp} in {BUCKET}')
  frames_hz = config['env']['realtime_hz'] * config['env']['dmc']['repeat']
  steps, scores = [], []
  for line in (run / 'logdir/metrics.jsonl').read_text().splitlines():
    try:
      row = json.loads(line)
    except json.JSONDecodeError:
      continue
    if 'episode/score' in row:
      steps.append(row['step'])
      scores.append(row['episode/score'])
  minutes = np.array(steps) / frames_hz / 60
  trail = np.convolve(scores, np.ones(WINDOW) / WINDOW, mode='valid')
  return minutes[WINDOW - 1:], trail


def mean_std(exps, xmax):
  grid = np.arange(0, xmax + 1e-9, GRID)
  curves = []
  for exp in exps:
    m, s = load(exp)
    curve = np.interp(grid, m, s, left=np.nan, right=np.nan)
    curves.append(curve)
  curves = np.array(curves)
  # End the mean where the shortest seed ends, so it never jumps when a seed
  # drops out of the average.
  curves[:, grid > end_minute(exps)] = np.nan
  # Grid points before a run's first trailing-5 value are NaN in every seed.
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    return grid, np.nanmean(curves, 0), np.nanstd(curves, 0), len(exps)


def end_minute(exps):
  return min(load(e)[0][-1] for e in exps)


# --- panels -----------------------------------------------------------------

def style_axes(ax, title):
  ax.set_title(title, fontsize=17)
  ax.tick_params(labelsize=13)
  ax.grid(color='#d5d5d2', lw=0.7)
  ax.set_axisbelow(True)
  for spine in ax.spines.values():
    spine.set_color('#404040')
  ax.set_ylim(0, 900)


def draw(ax, series, xmax=None, legend=True):
  check_colours([c for _, c, _ in series], ax.get_title())
  if xmax is None:
    # Cut every curve where the first (main comparison) series ends.
    xmax = min(40.0, end_minute(series[0][2]))
  for label, colour, exps in series:
    grid, mean, std, n = mean_std(exps, xmax)
    name = f'{label} (n={n})'
    ax.plot(grid, mean, color=colour, lw=2.4, label=name)
    if n > 1:
      ax.fill_between(grid, mean - std, mean + std, color=colour, alpha=0.15,
                      lw=0)
  ax.axhline(800, color='#8a8a85', lw=1.0, zorder=0)
  ax.set_xlim(0, xmax)
  if legend:
    ax.legend(fontsize=11, frameon=False, loc='lower right')


BASE50 = [1113, 1118, 1121, 1128]  # 50 Hz defaults; train_ratio 512 = 2048 (saturated)


def fig_hparams(out):
  panels = [
      ('Control rate (defaults)', [
          ('50 Hz', BLACK, BASE50),
          ('100 Hz', BLUE, [1117]),
          ('25 Hz', ORANGE, [1115]),
          ('12.5 Hz', GREEN, [1116])]),
      ('Learner saturation (50 Hz)', [
          ('train ratio 512-2048', BLACK, BASE50),
          ('train ratio 128', VERMILION, [1119])]),
      ('Return horizon (50 Hz)', [
          ('333 steps (6.7 s)', BLACK, BASE50),
          ('100 (2 s)', BLUE, [1126, 1129, 1130]),
          ('50 (1 s)', ORANGE, [1134])]),
      ('Batch (50 Hz, 2 s horizon)', [
          ('16x64', BLACK, [1126, 1129, 1130]),
          ('32x64', BLUE, [1133, 1137, 1138]),
          ('64x64', ORANGE, [1142, 1146, 1147])]),
      ('lr and width (12.5 Hz, 2 s)', [
          ('lr 1e-4, MLP 64', BLACK, [1145, 1151]),
          ('lr 1e-4, MLP 128', SKY, [1155, 1165]),
          ('lr 3e-4, MLP 64', ORANGE, [1156, 1158, 1159]),
          ('lr 3e-4, MLP 256', BLUE, [1162, 1167, 1168, 1177])]),
      ('Rate with lr 3e-4 + wide MLP', [
          ('12.5 Hz, MLP 256', BLUE, [1162, 1167, 1168, 1177]),
          ('10 Hz, MLP 256', BLACK, [1172, 1178]),
          ('25 Hz, MLP 256', ORANGE, [1170, 1193]),
          ('50 Hz, MLP 128', PINK, [1164])]),
      ('Weight push interval (recipe)', [
          ('30 s', BLACK, [1162]),
          ('5 s', BLUE, [1174]),
          ('10 s', ORANGE, [1175]),
          ('60 s', GREEN, [1176])]),
      ('Start vs. best', [
          ('sim_phone (50 Hz)', BLACK, BASE50),
          ('50 Hz, 2 s, 32x64', ORANGE, [1133, 1137, 1138]),
          ('sim_phone_fast (12.5 Hz)', BLUE, [1162, 1167, 1168, 1177])]),
  ]
  fig, axes = plt.subplots(2, 4, figsize=(20, 8.6))
  for ax, (title, series) in zip(axes.flat, panels):
    style_axes(ax, title)
    draw(ax, series)
  for ax in axes[:, 0]:
    ax.set_ylabel('Episode return\n(trailing 5)', fontsize=15)
  fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=2)
  fig.text(0.5, 0.015, 'Real time (minutes)', ha='center', fontsize=15)
  save(fig, out / 'sim_sweep_hparams')


RECIPE12 = [1162, 1167, 1168, 1177]  # sim_phone_fast, 12.5 Hz
RECIPE25 = [1170, 1193, 1220]        # same knobs at 25 Hz, horizon 50


def fig_round14(out):
  panels = [
      ('Action delay, 12.5 Hz', [
          ('no delay', BLACK, RECIPE12),
          ('1 step (80 ms)', VERMILION, [1215])]),
      ('Action delay, 25 Hz', [
          ('no delay', BLACK, RECIPE25),
          ('1 step (40 ms)', VERMILION, [1216])]),
      ('Which half gets lr 3e-4', [
          ('both', BLACK, RECIPE12),
          ('world model only', BLUE, [1218]),
          ('actor-critic only', ORANGE, [1219])]),
      ('MLP width (lr 3e-4, 12.5 Hz)', [
          ('256', BLACK, RECIPE12),
          ('128', BLUE, [1160, 1166]),
          ('512', ORANGE, [1217])]),
      ('Actor entropy (12.5 Hz)', [
          ('3e-4', BLACK, RECIPE12),
          ('1e-3', BLUE, [1221])]),
      ('Full recipe by control rate', [
          ('12.5 Hz', BLACK, RECIPE12),
          ('10 Hz', BLUE, [1172, 1178]),
          ('25 Hz', ORANGE, RECIPE25),
          ('50 Hz', GREEN, [1222])]),
  ]
  fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
  for ax, (title, series) in zip(axes.flat, panels):
    style_axes(ax, title)
    draw(ax, series, xmax=30.0)
  for ax in axes[:, 0]:
    ax.set_ylabel('Episode return\n(trailing 5)', fontsize=15)
  fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=2)
  fig.text(0.5, 0.015, 'Real time (minutes)', ha='center', fontsize=15)
  save(fig, out / 'sim_sweep_round14')


def fig_tasks(out):
  # (task, base 50 Hz, recipe 12.5 Hz, recipe 25 Hz); 12.5 Hz reacher fails.
  tasks = [
      ('cartpole swingup', BASE50, [1162, 1167, 1168, 1177], [1170, 1193]),
      ('cartpole balance', [1182], [1183], [1191]),
      ('pendulum swingup', [1180, 1194], [1181, 1195], [1190, 1196]),
      ('reacher easy', [1184], [1185], [1187, 1192]),
  ]
  labels = ('sim_phone: 50 Hz defaults',
            'recipe at 12.5 Hz (sim_phone_fast)',
            'recipe at 25 Hz')
  colours = (BLACK, BLUE, ORANGE)
  fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
  for ax, (task, *arms) in zip(axes.flat, tasks):
    style_axes(ax, task)
    ax.set_ylim(0, 1000)
    series = [(l, c, e) for l, c, e in zip(labels, colours, arms)]
    draw(ax, series, xmax=min(28.0, min(end_minute(e) for e in arms)),
         legend=False)
  for ax in axes[:, 0]:
    ax.set_ylabel('Episode return (trailing 5)', fontsize=15)
  handles = [plt.Line2D([], [], color=c, lw=2.4) for c in colours]
  fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
             fontsize=14)
  fig.tight_layout(rect=(0, 0.12, 1, 1), h_pad=2)
  fig.text(0.5, 0.095, 'Real time (minutes)', ha='center', fontsize=15)
  save(fig, out / 'sim_sweep_tasks')


def save(fig, stem):
  fig.savefig(f'{stem}.png', dpi=200, facecolor='white')
  fig.savefig(f'{stem}.pdf')
  print('wrote', f'{stem}.png', f'{stem}.pdf')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--out', default='docs/figs')
  args = parser.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  fig_hparams(out)
  fig_tasks(out)
  fig_round14(out)


if __name__ == '__main__':
  main()
