"""Shared discovery of the pure-Director baseline checkpoints (e390+ series,
see EXPERIMENTS.md) across all 4 tasks and however many seeds have actually
finished training so far.

Both ``goal_code_struct_corr/`` and ``worker_success/`` read the *same* set
of checkpoints from here, so a run is only ever added to one figure once its
training is far enough along, and both figures grow to more seeds together
as e398-e409 (seeds 2-4) finish -- no per-experiment hardcoded checkpoint
lists to keep in sync.

Naming convention (EXPERIMENTS.md e390-e409): experiment id =
390 + seed*4 + task_index, task_index over TASKS below (cartpole, hopper,
acrobot, cheetah). Seeds 0-1 (e390-e397) are complete; seeds 2-4
(e398-e409) are running/queued as of 2026-08-03 and will be picked up
automatically by ``discover_baselines`` once their metrics.jsonl reaches
the configured step budget.
"""
import json
import pathlib

WORK = pathlib.Path('/work/DoyaU/vasilache/work')
BUCKET = pathlib.Path('/bucket/DoyaU/vasilache/bucket/results/dreamerv3')

TASKS = ['dmc_cartpole_swingup', 'dmc_hopper_hop', 'dmc_acrobot_swingup', 'dmc_cheetah_run']
TASK_LABELS = {
    'dmc_cartpole_swingup': 'Cartpole Swingup',
    'dmc_hopper_hop': 'Hopper Hop',
    'dmc_acrobot_swingup': 'Acrobot Swingup',
    'dmc_cheetah_run': 'Cheetah Run',
}
# Colorblind-safe palette (Okabe-Ito), shared across both experiments' figures.
TASK_COLORS = {
    'dmc_cartpole_swingup': '#CC79A7',
    'dmc_hopper_hop': '#0072B2',
    'dmc_acrobot_swingup': '#009E73',
    'dmc_cheetah_run': '#D55E00',
}
MAX_SEED = 4  # seeds 0..4 -> 5 seeds total once e406-e409 finish


def exp_id(task, seed):
  return 390 + seed * 4 + TASKS.index(task)


def _find_run_dir(exp):
  # Prefer the bucket (archived, stable) copy if present, else the live /work
  # one (e390-e409 have not been archived yet as of 2026-08-03).
  for base in (BUCKET, WORK):
    matches = sorted(base.glob(f'e{exp}_*'))
    if matches:
      return matches[0]
  return None


def _logdir(run_dir):
  d = run_dir / 'logdir'
  return d if d.exists() else run_dir


def _last_step(metrics_path):
  last = None
  with open(metrics_path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        d = json.loads(line)
      except json.JSONDecodeError:
        continue
      if 'step' in d:
        last = d['step']
  return last


def _run_steps_target(cfg_path, default=4_000_000):
  if not cfg_path.exists():
    return default
  import ruamel.yaml as yaml
  try:
    saved = yaml.YAML(typ='safe').load(cfg_path.read_text())
    return int(saved.get('run', {}).get('steps', default))
  except Exception:
    return default


def is_complete(run_dir, min_frac=0.99):
  """(complete, last_step) -- complete iff metrics.jsonl has reached
  >=min_frac of the run's configured run.steps budget."""
  logdir = _logdir(run_dir)
  metrics = logdir / 'metrics.jsonl'
  if not metrics.exists():
    return False, None
  last_step = _last_step(metrics)
  if last_step is None:
    return False, None
  target = _run_steps_target(logdir / 'config.yaml')
  return last_step >= min_frac * target, last_step


def discover_baselines(tasks=TASKS, max_seed=MAX_SEED, require_complete=True):
  """{task: [(seed, run_dir_str, tag), ...]} sorted by seed, only for runs
  found on disk (and, by default, whose training has actually finished) --
  so still-running seeds (e398-e409 as of 2026-08-03) are silently skipped
  until they land, and rerunning the pipeline later picks them up with no
  code changes."""
  out = {t: [] for t in tasks}
  for t in tasks:
    for seed in range(max_seed + 1):
      exp = exp_id(t, seed)
      run_dir = _find_run_dir(exp)
      if run_dir is None:
        continue
      if require_complete:
        complete, _ = is_complete(run_dir)
        if not complete:
          continue
      out[t].append((seed, str(run_dir), f'e{exp}'))
  return out


if __name__ == '__main__':
  found = discover_baselines()
  total = sum(len(v) for v in found.values())
  print(f'{total} usable baseline checkpoints (>=99% trained):')
  for t, runs in found.items():
    print(f'  {t}: {[(s, tag) for s, _, tag in runs]}')
  print()
  print('all runs (including incomplete, for visibility):')
  for t, runs in discover_baselines(require_complete=False).items():
    for seed, run_dir, tag in runs:
      complete, last_step = is_complete(pathlib.Path(run_dir))
      status = 'DONE' if complete else f'running (step={last_step})'
      print(f'  {t} seed{seed} {tag}: {status}')
