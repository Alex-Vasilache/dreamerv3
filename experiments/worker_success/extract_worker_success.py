"""Extracts worker-success (``train/wkr_goal_rew``, the Director ``cosine_max``
goal-reward the worker actor-critic is trained on) across training from the
metrics.jsonl of pure-Director baseline runs.

Answers the paper's open \\todo (`code/26_04_HRL-paper/sections/introduction.tex`,
Sec. "Efficiency: Worker success"): "Across training, worker success is
relatively low [Check this with actual numbers]." No live rollout needed --
``train/wkr_goal_rew`` (dreamerv3/hrl/losses.py:109, ``imag_loss_wkr``) is
already logged at every training step in the archived runs' metrics.jsonl.

Usage:
  python3 extract_worker_success.py     # reads bucket metrics.jsonl for the 3
                                         # baselines below, writes results/*.json
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
BUCKET = pathlib.Path('/bucket/DoyaU/vasilache/bucket/results/dreamerv3')

# Same three pure-Director baselines used by ../goal_code_struct_corr/ (fixed
# K=8, no masking/variable-goal-length, goal_struct_weight=0.0).
RUNS = {
    'e124_hopper': 'e124_dmc_hopper_hop_director_baseline_multiv100_20260706_094230_C3AT29',
    'e123_cheetah': 'e123_dmc_cheetah_run_director_baseline_20260706_155117_Vk7oa4',
    'e180_acrobot': 'e180_dmc_acrobot_swingup_director_baseline_j4660140',
}

KEY = 'train/wkr_goal_rew'


def load(run_dir):
  path = BUCKET / run_dir / 'logdir' / 'metrics.jsonl'
  steps, vals = [], []
  with open(path) as f:
    for line in f:
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

  summary = {}
  for tag, run_dir in RUNS.items():
    steps, vals = load(run_dir)
    overall_mean = sum(vals) / len(vals)
    overall_min, overall_max = min(vals), max(vals)
    early_mean, early_n = frac_mean(steps, vals, 0.0, 0.1)
    late_mean, late_n = frac_mean(steps, vals, 0.9, 1.0)
    final_val = vals[-1]
    final_step = steps[-1]
    result = {
        'run_dir': run_dir,
        'n_points': len(vals),
        'step_min': steps[0],
        'step_max': steps[-1],
        'overall_mean': overall_mean,
        'overall_min': overall_min,
        'overall_max': overall_max,
        'early10pct_mean': early_mean,
        'early10pct_n': early_n,
        'late10pct_mean': late_mean,
        'late10pct_n': late_n,
        'final_value': final_val,
        'final_step': final_step,
        'steps': steps,
        'values': vals,
    }
    summary[tag] = result
    with open(out_dir / f'{tag}.json', 'w') as f:
      json.dump(result, f)
    print(f'{tag:14s} n={len(vals):4d}  overall={overall_mean:.3f}  '
          f'early10%={early_mean:.3f} (n={early_n})  '
          f'late10%={late_mean:.3f} (n={late_n})  '
          f'final={final_val:.3f} @ step {final_step}')

  with open(out_dir / 'summary.json', 'w') as f:
    json.dump(
        {k: {kk: vv for kk, vv in v.items() if kk not in ('steps', 'values')}
         for k, v in summary.items()},
        f, indent=2)
  print('wrote', out_dir / 'summary.json')


if __name__ == '__main__':
  main()
