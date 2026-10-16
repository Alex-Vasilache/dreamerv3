"""Health check for the goal-mask batch (e1600-e1659) and its smoke runs.

Prints one line per run dir matching a glob: env step, steps/hour since the
previous check, late score, the mask diagnostics, and red flags (NaN/inf in
metrics, a traceback in the SLURM .err, mask head stuck at chance, the manager
collapsing onto 1-group or full goals, worker goal reward dead).

  python tools/watch_pubmask.py                      # the batch
  python tools/watch_pubmask.py '/work/.../smoke_mask_cpu_*'
"""
import glob
import json
import math
import os
import re
import sys
import time

WORK = '/work/DoyaU/vasilache/work'
STATE = os.path.join(WORK, 'slurm_logs', 'watch_pubmask_state.json')
KEYS = {
    'acc': 'train/goal/mask_acc',
    'mloss': 'train/goal/mask_loss',
    'keep': 'train/goal/mask_keep_mgr',
    'full': 'train/goal/mask_full_mgr',
    'rec': 'train/goal/rec_mean',
    'wrew': 'train/wkr_goal_rew',
}


def tail_rows(path, n=40):
  """Last ``n`` parseable JSON rows of a (possibly large) jsonl file."""
  try:
    size = os.path.getsize(path)
  except OSError:
    return []
  with open(path, 'rb') as f:
    f.seek(max(0, size - 400_000))
    lines = f.read().decode('utf-8', 'replace').splitlines()[1:]
  rows = []
  for line in lines[-4 * n:]:
    try:
      rows.append(json.loads(line))
    except ValueError:
      pass
  return rows[-n:] if n else rows


def bad(v):
  return isinstance(v, float) and (math.isnan(v) or math.isinf(v))


def err_flag(run_dir):
  """Traceback/NaN text in the run's newest SLURM .err, if any."""
  env = os.path.join(run_dir, 'job.env')
  job = None
  if os.path.exists(env):
    for line in open(env):
      if line.startswith('JOB='):
        job = line.strip().split('=', 1)[1]
  if not job:
    return ''
  errs = glob.glob(os.path.join(WORK, 'slurm_logs', f'*-{job}.err'))
  if not errs:
    return ''
  with open(errs[0], 'rb') as f:
    f.seek(max(0, os.path.getsize(errs[0]) - 20_000))
    text = f.read().decode('utf-8', 'replace')
  for pat in ('Traceback', 'FloatingPointError', 'nan detected', 'Killed',
              'OUT_OF_MEMORY', 'RESOURCE_EXHAUSTED'):
    if pat in text:
      return pat
  return ''


def main():
  pattern = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, 'e16[0-5][0-9]_*')
  prev = {}
  if os.path.exists(STATE):
    try:
      prev = json.load(open(STATE))
    except ValueError:
      prev = {}
  now = time.time()
  state = {}
  for d in sorted(glob.glob(pattern)):
    name = os.path.basename(d)
    exp = name.split('_')[0]
    arm = 'som' if 'goal_som_line' in name else 'now'
    task = re.sub(r'^e\d+_|_director.*$', '', name)
    mrows = tail_rows(os.path.join(d, 'logdir', 'metrics.jsonl'))
    srows = tail_rows(os.path.join(d, 'logdir', 'scores.jsonl'), 50)
    step = max([r.get('step', 0) for r in mrows + srows] or [0])
    rate = ''
    if name in prev and now > prev[name]['t'] + 600:
      dh = (now - prev[name]['t']) / 3600
      rate = f'{(step - prev[name]["step"]) / dh / 1e3:.0f}k/h'
    state[name] = {'t': now, 'step': step} if name not in prev or rate else prev[name]
    vals = {}
    for k, key in KEYS.items():
      xs = [r[key] for r in mrows if key in r]
      vals[k] = xs[-1] if xs else None
    score = sum(r['episode/score'] for r in srows) / len(srows) if srows else float('nan')
    flags = []
    # train/ only: timer/*/min is NaN in every run until the timer has fired.
    if any(bad(v) for r in mrows for k, v in r.items() if k.startswith('train/')):
      flags.append('NaN/inf')
    e = err_flag(d)
    if e:
      flags.append(e)
    if vals['acc'] is not None and step > 300_000 and vals['acc'] < 0.75:
      flags.append('mask head near chance')
    if vals['keep'] is not None and step > 300_000:
      if vals['keep'] < 0.15:
        flags.append('manager keeps ~1 group')
      if vals['full'] is not None and vals['full'] > 0.98:
        flags.append('manager never masks')
    if vals['wrew'] is not None and step > 500_000 and vals['wrew'] < 0.05:
      flags.append('worker goal reward dead')
    fmt = lambda v: '   -' if v is None else f'{v:5.2f}'
    print(f'{exp} {arm:3s} {task:21s} {step / 1e6:5.2f}M {rate:>7s} '
          f'score {score:6.1f} | acc{fmt(vals["acc"])} mloss{fmt(vals["mloss"])} '
          f'keep{fmt(vals["keep"])} full{fmt(vals["full"])} '
          f'rec{fmt(vals["rec"])} wrew{fmt(vals["wrew"])}'
          + (f'  !! {", ".join(flags)}' if flags else ''))
  if pattern.endswith('e16[0-5][0-9]_*'):
    json.dump(state, open(STATE, 'w'))


if __name__ == '__main__':
  main()
