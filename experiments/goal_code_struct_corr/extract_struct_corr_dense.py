"""Extracts the fully-logged, in-training goal-code/goal-space correlation
trajectory from the one baseline seed that had ``goal_struct_diag: true`` set
for its *entire* training run (seed 4, e406-e409 -- see EXPERIMENTS.md).

Unlike the other seeds (0-3), which only have this correlation measured
after the fact (offline rollout via diag_goal_struct_corr.py, at just the
final checkpoint, or a handful of mid-training snapshots via
watch_and_diag_running.py), seed 4's runs computed
``train/goal/struct_corr_hard`` (and the soft/differentiable
``train/goal/struct_corr``) directly from the replay batch already sampled
for the goal-VAE loss, at *every* train step -- see
``Agent._goal_autoencoder_loss`` in ``dreamerv3/agent.py`` (``struct_diag``
gated on ``goal_struct_diag`` OR ``goal_struct_weight>0``/``goal_struct_adapt``;
seed 4 is a pure-Director baseline, so it's ``goal_struct_diag`` alone driving
this, with the struct loss itself never applied to training).

No cluster job needed -- reads metrics.jsonl directly (same pattern as
../worker_success/extract_worker_success.py).
"""
import json
import pathlib

WORK = pathlib.Path('/work/DoyaU/vasilache/work')
BUCKET = pathlib.Path('/bucket/DoyaU/vasilache/bucket/results/dreamerv3')
HERE = pathlib.Path(__file__).resolve().parent

SEED4_RUNS = {
    'dmc_cartpole_swingup': 'e406',
    'dmc_hopper_hop': 'e407',
    'dmc_acrobot_swingup': 'e408',
    'dmc_cheetah_run': 'e409',
}


def find_run_dir(tag):
  for base in (BUCKET, WORK):
    matches = sorted(base.glob(f'{tag}_*'))
    if matches:
      return matches[0]
  return None


def main():
  out_dir = HERE / 'results'
  out_dir.mkdir(exist_ok=True)
  for task, tag in SEED4_RUNS.items():
    run_dir = find_run_dir(tag)
    if run_dir is None:
      print(f'{task} ({tag}): run dir not found, skipping')
      continue
    logdir = run_dir / 'logdir'
    logdir = logdir if logdir.exists() else run_dir
    metrics = logdir / 'metrics.jsonl'
    steps, soft, hard = [], [], []
    with open(metrics) as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          d = json.loads(line)
        except json.JSONDecodeError:
          continue
        if 'step' not in d or 'train/goal/struct_corr_hard' not in d:
          continue
        steps.append(d['step'])
        hard.append(d['train/goal/struct_corr_hard'])
        soft.append(d.get('train/goal/struct_corr', float('nan')))
    out_path = out_dir / f'{tag}_{task.split("_")[1]}_seed4_dense.json'
    payload = {
        'task': task, 'tag': tag, 'seed': 4,
        'steps': steps, 'corr_soft': soft, 'corr_hard': hard,
    }
    out_path.write_text(json.dumps(payload))
    print(f'{task} ({tag}): n={len(steps)} steps, '
          f'first hard r={hard[0]:.3f} @ {steps[0]}, '
          f'last hard r={hard[-1]:.3f} @ {steps[-1]} -> wrote {out_path}')


if __name__ == '__main__':
  main()
