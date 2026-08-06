"""Watches currently-training pure-Director baseline seeds (e398-e409 as of
2026-08-03, see ../baselines_common.py) and, the first time each crosses
1M/2M/3M env steps, snapshots its LIVE checkpoint and submits a
diag_goal_struct_corr sbatch job against that snapshot -- giving the
goal-code/goal-space correlation figure at multiple points in training for
runs that haven't finished yet, not just their final checkpoint.

Why a snapshot instead of pointing the diag script at ckpt/latest directly:
elements.Checkpoint (see ../../embodied dep) defaults to keep=1 -- every new
save deletes the previous one. By the time the diag SLURM job actually runs
(queued, then loads a ~1GB checkpoint), the live run may already have saved
past the milestone we wanted, silently measuring the wrong step. Copying the
checkpoint folder aside the moment we see the milestone crossed freezes it.

This script itself needs no GPU/dreamerv3_env -- it only reads metrics.jsonl,
copies checkpoint files, and calls `sbatch`. Run it on the login node:
  nohup python3 watch_and_diag_running.py > watch.log 2>&1 &
It exits on its own once every currently-training seed has reached (or
skipped past, e.g. if the run finishes early) all three milestones.

State (which milestones have already been captured, and their sbatch job
ids) is persisted to results/milestones_state.json, so re-running this
script after an interruption does not resubmit already-captured milestones.
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from baselines_common import TASKS, discover_baselines, is_complete  # noqa: E402

SHORT_TASK = {
    'dmc_cartpole_swingup': 'cartpole',
    'dmc_hopper_hop': 'hopper',
    'dmc_acrobot_swingup': 'acrobot',
    'dmc_cheetah_run': 'cheetah',
}
MILESTONES = [1_000_000, 2_000_000, 3_000_000]
STATE_PATH = HERE / 'results' / 'milestones_state.json'
# /work, NOT under the repo (/apps): each snapshot is a full ~1GB checkpoint
# copy, and /apps is a 50GB filesystem shared by the whole project -- it hit
# 100% full and crashed this watcher after ~20GB of snapshots accumulated
# there (2026-08-03). /work is 10TB scratch, the correct place for this.
SNAP_DIR = pathlib.Path('/work/DoyaU/vasilache/work/goal_struct_corr_snapshots')
SBATCH_SCRIPT = HERE / 'run_diag_struct_corr_snapshot.sbatch'


def logdir_of(run_dir):
  run_dir = pathlib.Path(run_dir)
  d = run_dir / 'logdir'
  return d if d.exists() else run_dir


def last_step(metrics_path):
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


def load_state():
  if STATE_PATH.exists():
    return json.loads(STATE_PATH.read_text())
  return {}


def save_state(state):
  STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
  STATE_PATH.write_text(json.dumps(state, indent=2))


def snapshot_ckpt(run_dir, dest):
  """Copies the run's CURRENT ckpt/latest folder to `dest`. Returns the dest
  path, or None if no checkpoint exists yet. Idempotent (skips if dest
  already exists, so a re-run after interruption doesn't redo the copy)."""
  ckptdir = logdir_of(run_dir) / 'ckpt'
  latest_file = ckptdir / 'latest'
  if not latest_file.exists():
    return None
  folder = latest_file.read_text().strip()
  src = ckptdir / folder
  if not src.exists():
    return None
  dest = pathlib.Path(dest)
  if dest.exists():
    return dest
  dest.parent.mkdir(parents=True, exist_ok=True)
  tmp = dest.with_name(dest.name + '.tmp')
  if tmp.exists():
    shutil.rmtree(tmp)
  shutil.copytree(src, tmp)
  tmp.rename(dest)
  return dest


def submit_diag(run_dir, ckpt_path, out_name, train_step):
  out_path = HERE / 'results' / f'{out_name}.npz'
  env = f'ALL,RUN_DIR={run_dir},CKPT_PATH={ckpt_path},OUT_PATH={out_path},TRAIN_STEP={train_step}'
  cmd = ['sbatch', '--parsable', f'--export={env}', str(SBATCH_SCRIPT)]
  # stdout=PIPE/universal_newlines instead of capture_output/text: this
  # script targets the login node's system python3 (3.6), which predates
  # both kwargs (added in 3.7).
  result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, check=True)
  return result.stdout.strip()


def find_targets():
  """(task, short, seed, run_dir, tag) for every baseline seed that has NOT
  yet finished training -- these are the ones worth watching for milestone
  crossings. Already-finished seeds belong to the regular (../submit_diag_
  goal_struct_corr.sh) pipeline instead."""
  out = []
  found = discover_baselines(require_complete=False)
  for task in TASKS:
    for seed, run_dir, tag in found[task]:
      complete, _ = is_complete(pathlib.Path(run_dir))
      if complete:
        continue
      out.append((task, SHORT_TASK[task], seed, run_dir, tag))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--poll_interval', type=float, default=120.0,
                   help='seconds between checks of each run''s metrics.jsonl')
  args = ap.parse_args()

  state = load_state()
  print(f'watching for milestones {MILESTONES} among currently-training baselines '
        f'(poll every {args.poll_interval:.0f}s)', flush=True)
  while True:
    targets = find_targets()
    if not targets:
      print('no currently-training baseline seeds found (all complete, or none launched yet)',
            flush=True)
      break
    pending = False
    for task, short, seed, run_dir, tag in targets:
      key = f'{tag}_{short}_seed{seed}'
      metrics = logdir_of(run_dir) / 'metrics.jsonl'
      if not metrics.exists():
        pending = True
        continue
      step = last_step(metrics)
      if step is None:
        pending = True
        continue
      for m in MILESTONES:
        mlabel = f'{m // 1_000_000}M'
        donekey = f'{key}_{mlabel}'
        if donekey in state:
          continue
        if step < m:
          pending = True
          continue
        out_name = f'{key}_step{mlabel}'
        print(f'{key}: crossed {mlabel} (step={step}) -- snapshotting checkpoint', flush=True)
        dest = SNAP_DIR / out_name
        snap = snapshot_ckpt(run_dir, dest)
        if snap is None:
          print(f'  WARNING: no checkpoint on disk yet for {key} -- will retry next poll',
                flush=True)
          pending = True
          continue
        jobid = submit_diag(run_dir, snap, out_name, step)
        state[donekey] = {'step': step, 'job': jobid, 'snapshot': str(snap)}
        save_state(state)
        print(f'  snapshotted -> {snap}, submitted diag job {jobid} -> results/{out_name}.npz',
              flush=True)
    if not pending:
      print('all currently-training baselines have reached every milestone -- done watching',
            flush=True)
      break
    time.sleep(args.poll_interval)


if __name__ == '__main__':
  main()
