"""Will the benchmark finish by its deadline, and what to move if not.

The constraint is not total GPU count, it is that `size50m` is a 24 GB model
and a V100/P100 has 16 GB, so those arms can only run on A100. Everything else
can run anywhere. So the question is always the same one: does the A100-only
work fit in the A100 time left, and if not, can size6m work be moved off A100
to make room.

Wall time per RUN is roughly constant within a GPU class, because the 4M-step
arms run at replay ratio 64 and the 1.1M ones at 256 -- 3.6x the env steps at
4x the env-fps. That is why this counts runs, not steps.

  python3 sbatch/bench_forecast.py            # verdict
  python3 sbatch/bench_forecast.py --commands # + the scontrol lines to run

Prints REBALANCE only when the projection misses the deadline AND a 16 GB
partition has spare capacity; otherwise it says on track and does nothing.
"""
import collections
import datetime
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_coverage import (ARMS, ARRAY_PARAMS, BUCKET, PROJECT, SEEDS, TASKS,
                            WORK, env_of, last_step, params_rows, queued_cells)

DEADLINE = os.environ.get('BENCH_DEADLINE', '2026-09-28')
CAP = 8                      # GrpTRES gres/gpu per partition
OCC_LOG = os.path.join('/work/DoyaU/vasilache/work', 'bench_occupancy.tsv')
OCC_WINDOW_H = 24            # hours of history the effective capacity uses
# Days per run, measured 2026-09-09. Re-measure if the arms change shape.
RATE = {('gpu-a100', 'size6m'): 0.53, ('gpu-a100', 'size50m'): 0.85,
        ('gpu-v100', 'size6m'): 2.45, ('gpu-p100', 'size6m'): 2.65}


def big(arm):
    return 'size50m' in arm


def remaining_runs():
    """Runs not yet at target, split by whether they need an A100."""
    best = {}
    for root in (WORK, BUCKET):
        for d in sorted(glob.glob(os.path.join(root, 'e[0-9]*_j*'))):
            g = env_of(d)
            if g.get('WANDB_PROJECT', '').strip() != PROJECT:
                continue
            key = (g.get('CONFIG', '').strip(), g.get('TASK', '').strip(),
                   g.get('SEED', '').strip())
            s = last_step(d)
            if key not in best or s > best[key]:
                best[key] = s
    small = a100_only = 0
    for arm, target in ARMS:
        for task in TASKS:
            for seed in SEEDS:
                s = best.get((arm, task, seed), 0)
                if s >= 0.99 * target:
                    continue
                frac = 1.0 - min(1.0, s / target)   # part-done runs count part
                if big(arm):
                    a100_only += frac
                else:
                    small += frac
    return small, a100_only


def occupancy():
    try:
        out = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%P %T'],
            stdout=subprocess.PIPE, universal_newlines=True, timeout=30).stdout
    except Exception:
        return {}
    c = collections.Counter()
    for line in out.splitlines():
        f = line.split()
        if len(f) == 2 and f[1] == 'RUNNING':
            c[f[0]] += 1
    return c


def movable_a100_tasks():
    """Pending A100 array tasks whose arm is size6m -- the ones that could be
    moved to a 16 GB partition. Returns (arrayid, count, first_arm)."""
    try:
        ids = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', ''), '-h', '-t', 'PENDING',
             '-o', '%i %P'], stdout=subprocess.PIPE, universal_newlines=True,
            timeout=30).stdout.splitlines()
    except Exception:
        return []
    out = []
    for line in ids:
        f = line.split()
        if len(f) != 2 or f[1] != 'gpu-a100' or '_' not in f[0]:
            continue
        arr, task = f[0].split('_', 1)
        rows = params_rows(ARRAY_PARAMS.get(arr, ''))
        if not rows:
            continue
        n = 0
        if task.startswith('['):
            for part in task.strip('[]').split('%')[0].split(','):
                if '-' in part:
                    lo, hi = part.split('-')
                    n += int(hi) - int(lo) + 1
                elif part.isdigit():
                    n += 1
        elif task.isdigit():
            n = 1
        if n and not big(rows[0][0]):
            out.append((f[0], n, rows[0][0]))
    return out


def effective_cap(occ):
    """How many GPUs we ACTUALLY hold, not how many we are entitled to.

    The quota is 8 per partition, but the partition is shared: on 2026-09-11 all
    32 A100 GPUs were allocated and we held 3, with the rest going to other
    users. Projecting from the quota in that state reports ON TRACK while the
    work is in fact stretching out. Samples are appended on every run and the
    median of the last OCC_WINDOW_H hours is used, so a momentary dip between
    one job ending and the next starting does not swing the forecast.
    """
    now = datetime.datetime.now()
    try:
        with open(OCC_LOG, 'a') as f:
            f.write('%s\t%d\t%d\t%d\n' % (
                now.isoformat(timespec='seconds'), occ.get('gpu-a100', 0),
                occ.get('gpu-v100', 0), occ.get('gpu-p100', 0)))
    except Exception:
        pass
    hist = {'gpu-a100': [], 'gpu-v100': [], 'gpu-p100': []}
    cutoff = now - datetime.timedelta(hours=OCC_WINDOW_H)
    try:
        for line in open(OCC_LOG):
            f = line.rstrip('\n').split('\t')
            if len(f) != 4:
                continue
            try:
                when = datetime.datetime.fromisoformat(f[0])
            except ValueError:
                continue
            if when < cutoff:
                continue
            for k, v in zip(('gpu-a100', 'gpu-v100', 'gpu-p100'), f[1:]):
                hist[k].append(int(v))
    except Exception:
        pass
    out = {}
    for k, v in hist.items():
        if len(v) < 6:            # too little history to trust; assume the quota
            out[k] = CAP
        else:
            v = sorted(v)
            out[k] = max(1, v[len(v) // 2])
    return out


def main():
    small, a100_only = remaining_runs()
    left = (datetime.datetime.strptime(DEADLINE, '%Y-%m-%d')
            - datetime.datetime.now()).total_seconds() / 86400
    occ = occupancy()

    eff = effective_cap(occ)
    a100_n = eff['gpu-a100']
    vp_n = eff['gpu-v100'] + eff['gpu-p100']
    a100_days = a100_only * RATE[('gpu-a100', 'size50m')]
    vp_rate = (RATE[('gpu-v100', 'size6m')] + RATE[('gpu-p100', 'size6m')]) / 2
    # Split the size6m work so both pools land together.
    x = (vp_rate * small * a100_n - vp_n * a100_days) / (
        vp_n * RATE[('gpu-a100', 'size6m')] + vp_rate * a100_n)
    x = max(0.0, min(small, x))
    finish = (a100_days + RATE[('gpu-a100', 'size6m')] * x) / a100_n
    eta = datetime.datetime.now() + datetime.timedelta(days=finish)

    print('remaining      : %.0f size6m runs, %.0f size50m runs (A100-only)'
          % (small, a100_only))
    print('running now    : ' + ', '.join('%s=%d' % (p.replace('gpu-', ''), n)
                                          for p, n in sorted(occ.items())))
    print('held (24h med) : a100=%d v100=%d p100=%d of %d each -- the partition is '
          'shared, so this is what we actually get'
          % (eff['gpu-a100'], eff['gpu-v100'], eff['gpu-p100'], CAP))
    print('best split     : %.0f size6m on A100, %.0f on V100/P100' % (x, small - x))
    print('projected      : %.1f days -> %s' % (finish, eta.strftime('%Y-%m-%d')))
    print('deadline       : %s (%.1f days away)' % (DEADLINE, left))

    if finish <= left:
        print('\nON TRACK, %.1f days of slack. Nothing to do.' % (left - finish))
        return

    idle = {p: CAP - occ.get('gpu-' + p, 0) for p in ('v100', 'p100')}
    free = sum(v for v in idle.values() if v > 0)
    print('\nBEHIND by %.1f days.' % (finish - left))
    if free <= 0:
        print('V100/P100 are both at their cap, so there is nothing to move '
              'work onto. The only levers left are dropping seeds or arms -- '
              'ask before doing either.')
        return
    print('V100/P100 spare capacity: ' + ', '.join(
        '%s=%d' % (k, v) for k, v in idle.items() if v > 0))
    cand = movable_a100_tasks()
    if not cand:
        print('No pending size6m A100 tasks to move; the A100 queue is all '
              'size50m, which cannot run on a 16 GB card.')
        return
    print('\nREBALANCE -- move pending size6m tasks off A100:')
    target = 'gpu-v100' if idle.get('v100', 0) >= idle.get('p100', 0) else 'gpu-p100'
    for jid, n, arm in cand:
        print('  scontrol update jobid=%s Partition=%s   # %d task(s), %s'
              % (jid, target, n, arm))
    if target == 'gpu-p100':
        print('  (p100 also needs: scontrol update jobid=<id> '
              'ReqNodeList=saion-gpu[11-14])')
    print('\nMoving a PENDING task only changes where it starts; running jobs '
          'are untouched. Re-run this after the move to see the new date.')


if __name__ == '__main__':
    main()
