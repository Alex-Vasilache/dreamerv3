"""Extracts worker-success (``train/wkr_goal_rew``, the Director ``cosine_max``
goal-reward the worker actor-critic is trained on) across training from the
metrics.jsonl of pure-Director baseline runs.

Answers the paper's \\todo (`code/26_04_HRL-paper/sections/motivation.tex`,
Sec. "Worker success"): "Across training, worker success is relatively low."
No live rollout needed -- ``train/wkr_goal_rew`` (dreamerv3/hrl/losses.py:109,
``imag_loss_wkr``) is already logged at every training step in the archived
runs' metrics.jsonl.

Uses ../baselines_common.py to discover all 4 tasks x however many seeds
have finished training (see EXPERIMENTS.md e390-e409) -- currently 2 seeds,
growing to 5 as e398-e409 land, no code changes needed to pick them up.

Usage:
  python3 extract_worker_success.py     # reads bucket/work metrics.jsonl for
                                         # all discovered baselines, writes
                                         # results/*.json
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from baselines_common import TASKS, discover_baselines  # noqa: E402

SHORT_TASK = {
    'dmc_cartpole_swingup': 'cartpole',
    'dmc_hopper_hop': 'hopper',
    'dmc_acrobot_swingup': 'acrobot',
    'dmc_cheetah_run': 'cheetah',
}

KEY = 'train/wkr_goal_rew'


def load(run_dir):
  run_dir = pathlib.Path(run_dir)
  logdir = run_dir / 'logdir'
  logdir = logdir if logdir.exists() else run_dir
  path = logdir / 'metrics.jsonl'
  steps, vals = [], []
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      d = json.loads(line)
      if KEY in d:
        steps.append(d['step'])
        vals.append(d[KEY])
  order = sorted(range(len(steps)), key=lambda i: steps[i])
  steps = [steps[i] for i in order]
  vals = [vals[i] for i in order]
  return steps, vals


def frac_mean(steps, vals, lo, hi):
  smin, smax = steps[0], steps[-1]
  span = smax - smin
  sel = [v for s, v in zip(steps, vals) if smin + lo * span <= s <= smin + hi * span]
  return sum(sel) / len(sel), len(sel)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out_dir', default=str(HERE / 'results'))
  args = ap.parse_args()
  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  found = discover_baselines()
  summary = {}
  for task in TASKS:
    short = SHORT_TASK[task]
    seeds = found[task]
    if not seeds:
      print(f'WARNING: no complete baseline for {task} -- skipping')
      continue
    per_seed = []
    for seed, run_dir, tag in seeds:
      steps, vals = load(run_dir)
      overall_mean = sum(vals) / len(vals)
      early_mean, early_n = frac_mean(steps, vals, 0.0, 0.1)
      late_mean, late_n = frac_mean(steps, vals, 0.9, 1.0)
      result = {
          'task': task, 'seed': seed, 'tag': tag, 'run_dir': str(run_dir),
          'n_points': len(vals), 'step_min': steps[0], 'step_max': steps[-1],
          'overall_mean': overall_mean,
          'overall_min': min(vals), 'overall_max': max(vals),
          'early10pct_mean': early_mean, 'early10pct_n': early_n,
          'late10pct_mean': late_mean, 'late10pct_n': late_n,
          'final_value': vals[-1], 'final_step': steps[-1],
          'steps': steps, 'values': vals,
      }
      per_seed.append(result)
      out_name = f'{tag}_{short}_seed{seed}.json'
      with open(out_dir / out_name, 'w') as f:
        json.dump(result, f)
      print(f'{task:22s} seed{seed}  n={len(vals):4d}  overall={overall_mean:.3f}  '
            f'early10%={early_mean:.3f} (n={early_n})  late10%={late_mean:.3f} (n={late_n})  '
            f'final={vals[-1]:.3f} @ step {steps[-1]}')

    # Cross-seed summary stats (mean/std over each seed's own overall/late mean).
    overall_means = [r['overall_mean'] for r in per_seed]
    late_means = [r['late10pct_mean'] for r in per_seed]
    finals = [r['final_value'] for r in per_seed]
    import numpy as np
    summary[task] = {
        'short': short,
        'n_seeds': len(per_seed),
        'seeds': [r['seed'] for r in per_seed],
        'overall_mean_of_seeds': float(np.mean(overall_means)),
        'overall_std_of_seeds': float(np.std(overall_means)),
        'late10pct_mean_of_seeds': float(np.mean(late_means)),
        'late10pct_std_of_seeds': float(np.std(late_means)),
        'final_mean_of_seeds': float(np.mean(finals)),
        'final_std_of_seeds': float(np.std(finals)),
    }

  with open(out_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
  print('wrote', out_dir / 'summary.json')
  for task, s in summary.items():
    print(f'{task:22s} n_seeds={s["n_seeds"]}  '
          f'overall={s["overall_mean_of_seeds"]:.3f}+-{s["overall_std_of_seeds"]:.3f}  '
          f'late10%={s["late10pct_mean_of_seeds"]:.3f}+-{s["late10pct_std_of_seeds"]:.3f}  '
          f'final={s["final_mean_of_seeds"]:.3f}+-{s["final_std_of_seeds"]:.3f}')


if __name__ == '__main__':
  main()
