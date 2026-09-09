"""Completion of the whole benchmark matrix: 14 arms x 6 tasks x 3 seeds.

`bench_status.sh` reports the runs that exist on /work. That is the wrong
denominator once finished runs are archived away -- after the 2026-09-09 sweep
it read "0 of 36 at target" while 73 runs were in fact complete. This reports
against the DESIGN instead, counting a cell done wherever its data lives, and
naming the cells that are neither running nor queued so a gap cannot sit
unnoticed.

  python3 sbatch/bench_coverage.py [-v]
"""
import collections
import glob
import json
import os
import subprocess
import sys

PROJECT = 'dreamerv3-bench-2026-09'
WORK = '/work/DoyaU/vasilache/work'
BUCKET = '/bucket/DoyaU/vasilache/bucket/results/dreamerv3'
SB = os.path.dirname(os.path.abspath(__file__))

ARMS = [
    ('dreamerv3', 1.1e6), ('director', 1.1e6), ('som_lip', 1.1e6),
    ('director mgr_expl_perc01', 1.1e6), ('som_lip mgr_expl_perc01', 1.1e6),
    ('director mgr_expl_perc002', 1.1e6), ('som_lip mgr_expl_perc002', 1.1e6),
    ('dreamerv3 size50m', 1.1e6),
    ('director director_og', 4e6), ('director director_og size50m', 4e6),
    ('som_lip director_og', 4e6), ('som_lip director_og size50m', 4e6),
    ('director mgr_expl_perc002 size50m', 1.1e6),
    ('som_lip mgr_expl_perc002 size50m', 1.1e6),
]
TASKS = ['dmc_cartpole_swingup', 'dmc_cheetah_run', 'dmc_hopper_hop',
         'pinpad_four', 'pinpad_five', 'pinpad_six']
SEEDS = ['0', '1', '2']

# Which params file each live array indexes. The array's own PARAMS is in its
# environment, which scontrol does not print, so the mapping lives here.
ARRAY_PARAMS = {
    '4706736': 'bench_params_expl_a100.txt',
    '4707513': 'bench_params_expl002_a100.txt',
    '4707514': 'bench_params_expl002_v100.txt',
    '4707517': 'bench_params_expl002_p100.txt',
    '4709578': 'bench_params_v350m.txt',
    '4709678': 'bench_params_og_a100_v2.txt',
    '4709679': 'bench_params_og_v100_v2.txt',
    '4709680': 'bench_params_og_p100_v2.txt',
    '4709681': 'bench_params_og50m_v2.txt',
    '4710469': 'bench_params_fill4.txt',
    '4710487': 'bench_params_somog_v100.txt',
    '4710488': 'bench_params_somog_p100.txt',
    '4710489': 'bench_params_somog50m.txt',
    '4710490': 'bench_params_e002_50m_dir.txt',
    '4710491': 'bench_params_e002_50m_som.txt',
}


def env_of(d):
    try:
        return dict(l.split('=', 1) for l in
                    open(os.path.join(d, 'job.env')).read().splitlines()
                    if '=' in l)
    except Exception:
        return {}


def last_step(d):
    step = 0
    try:
        for line in open(os.path.join(d, 'logdir', 'metrics.jsonl')):
            line = line.strip()
            if line:
                try:
                    step = json.loads(line).get('step', step) or step
                except Exception:
                    pass
    except Exception:
        pass
    return step


def params_rows(fname):
    out = []
    try:
        for line in open(os.path.join(SB, fname)):
            p = line.split()
            if len(p) >= 4:
                cfg = p[0] + ((' ' + ' '.join(p[4:])) if len(p) > 4 else '')
                out.append((cfg, p[1], p[2]))
    except Exception:
        pass
    return out


def queued_cells():
    """Cells that have a pending or running array task behind them."""
    try:
        ids = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%i'],
            stdout=subprocess.PIPE, universal_newlines=True,
            timeout=30).stdout.split()
    except Exception:
        return set()
    idx = collections.defaultdict(set)
    for j in ids:
        if '_' not in j:
            continue
        arr, task = j.split('_', 1)
        if task.startswith('['):
            for part in task.strip('[]').split('%')[0].split(','):
                if '-' in part:
                    lo, hi = part.split('-')
                    idx[arr].update(range(int(lo), int(hi) + 1))
                elif part.isdigit():
                    idx[arr].add(int(part))
        elif task.isdigit():
            idx[arr].add(int(task))
    out = set()
    for arr, fname in ARRAY_PARAMS.items():
        rows = params_rows(fname)
        for i in idx.get(arr, ()):
            if i < len(rows):
                out.add(rows[i])
    return out


def live_jobs():
    try:
        return set(subprocess.run(
            ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%i'],
            stdout=subprocess.PIPE, universal_newlines=True,
            timeout=30).stdout.split())
    except Exception:
        return set()


def watchdog_jobs():
    out = {}
    try:
        for line in open(os.path.join(WORK, 'bench_watchdog_state.tsv')):
            f = line.rstrip('\n').split('\t')
            if len(f) >= 3 and f[0] != '__hwm__' and f[2] not in ('', '-'):
                out[f[0]] = f[2]
    except Exception:
        pass
    return out


def main():
    verbose = '-v' in sys.argv
    best, dirs = {}, {}
    for root in (WORK, BUCKET):
        for d in sorted(glob.glob(os.path.join(root, 'e[0-9]*_j*'))):
            g = env_of(d)
            if g.get('WANDB_PROJECT', '').strip() != PROJECT:
                continue
            key = (g.get('CONFIG', '').strip(), g.get('TASK', '').strip(),
                   g.get('SEED', '').strip())
            step = last_step(d)
            if key not in best or step > best[key]:
                best[key] = step
            dirs.setdefault(key, []).append((os.path.basename(d),
                                             g.get('JOB', '').strip()))
    queued = queued_cells()
    # A cell is also covered by a job that is not an array task: a watchdog
    # retry, or the run's own self-requeued job. Without this a resubmitted run
    # is reported as having nothing behind it, which is how a real gap gets
    # lost in noise.
    live = live_jobs()
    wd = watchdog_jobs()

    def has_live_job(key):
        for name, job in dirs.get(key, []):
            for j in (job, wd.get(name)):
                if not j:
                    continue
                if j in live or any(x.split('_')[0] == j for x in live):
                    return True
        return False

    done = part = pend = gap = 0
    gaps = []
    print('%-32s %5s %5s %5s %5s' % ('arm', 'done', 'part', 'queue', 'GAP'))
    for arm, target in ARMS:
        row = [0, 0, 0, 0]
        for task in TASKS:
            for seed in SEEDS:
                cell = (arm, task, seed)
                step = best.get(cell)
                if step is not None and step >= 0.99 * target:
                    row[0] += 1
                elif cell in queued or has_live_job(cell):
                    row[2] += 1
                elif step:
                    row[1] += 1          # started, stopped, nothing queued
                    gaps.append(('stalled %.2fM' % (step / 1e6), cell))
                else:
                    row[3] += 1
                    gaps.append(('never started', cell))
        done, part, pend, gap = (done + row[0], part + row[1],
                                 pend + row[2], gap + row[3])
        print('%-32s %5d %5d %5d %5d' % (arm, row[0], row[1], row[2], row[3]))
    total = len(ARMS) * len(TASKS) * len(SEEDS)
    print('%-32s %5d %5d %5d %5d   of %d' % ('TOTAL', done, part, pend, gap, total))
    print('\n%d of %d cells complete (%.0f%%)' % (done, total, 100.0 * done / total))
    if gaps:
        print('\ncells with nothing behind them:')
        for why, cell in sorted(gaps):
            print('  %-16s %s %s s%s' % (why, cell[0], cell[1], cell[2]))
    elif verbose:
        print('every incomplete cell has a running or queued job behind it.')


if __name__ == '__main__':
    main()
