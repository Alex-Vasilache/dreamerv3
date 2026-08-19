#!/usr/bin/env python3
"""Health check for the unimodal Poisson manager runs (e582-e586).

The e558-e573 watchdog already covers "the job died" and "the job stalled".
This covers the ways a Poisson manager can be *running fine and still be
worthless*, which is the failure class that cost us e478 (entropy loss silently
0 for a whole run) and e479 (entropy target below the reachable floor, so the
multiplier railed at 1e4x its healthy value).

Checks, per run:

  NAN       any non-finite metric (timer/* excluded -- `min` over a stage that
            never ran is +inf by design)
  ENT_DEAD  the manager entropy metric is missing entirely, i.e. losses.py is
            skipping the head because it has no minent/maxent
  ENT_PIN   normalized entropy pinned outside (0.2, 0.95): pinned high means the
            head cannot commit, pinned low means it collapsed to one class
  MULT_RAIL the mgr_actent multiplier sitting within 10x of its 1e2 cap, which
            means it is chasing an entropy it cannot reach
  FLAT      the manager never changes a block (implicit_sparsity_block ~ 0),
            i.e. the rate collapsed and the goal code is frozen

Usage:
  python3 tools/check_poisson_health.py                 # all e58[2-6] runs
  python3 tools/check_poisson_health.py --min-step 50000
"""
import argparse
import glob
import json
import math
import os
import time

WORK = os.environ.get('WORK_DIR', '/work/DoyaU/vasilache/work')
ACTENT_MAX = 1e2  # manager_actent_max


def read_rows(path, tail_bytes=4_000_000):
    rows = []
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            data = f.read().decode('utf8', 'ignore').splitlines()
    except FileNotFoundError:
        return rows
    # Seeking into the middle of the file almost certainly lands mid-line, so
    # drop the first fragment -- but only when we actually seeked.
    if size > tail_bytes and data:
        data = data[1:]
    for line in data:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if 'step' in r:
            rows.append(r)
    return rows


def series(rows, suffix):
    out = [(r['step'], r[k]) for r in rows for k in r
           if k.endswith(suffix) and isinstance(r[k], (int, float))]
    return sorted(out)


def check(run_dir, min_step, stale_minutes=60):
    name = os.path.basename(run_dir.rstrip('/'))
    rows = read_rows(os.path.join(run_dir, 'logdir', 'metrics.jsonl'))
    if not rows:
        # Not a failure *yet*. A fresh run spends the first several minutes in
        # XLA autotuning and prefill before the first metrics flush, so
        # treating this as a problem reports every launch as broken. It only
        # becomes a problem once it has had long enough.
        age_min = (time.time() - os.path.getmtime(run_dir)) / 60.0
        if age_min > stale_minutes:
            return name, 0, ['NODATA  no metrics after %.0f min -- not '
                             'training' % age_min], {}
        return name, 0, [], {'waiting_min': age_min}
    step = max(r['step'] for r in rows)
    problems, info = [], {'step': step}

    for r in rows[-40:]:
        for k, v in r.items():
            if k.startswith('timer/'):
                continue
            if isinstance(v, float) and not math.isfinite(v):
                problems.append('NAN     %s = %s at step %s' % (k, v, r['step']))
                break

    ent = series(rows, 'mgr_ent_norm_skill_mean')
    if not ent:
        problems.append('ENT_DEAD  no mgr_ent_norm_skill_mean -- the entropy '
                        'regularizer is skipping the manager head (e478)')
    else:
        info['ent'] = ent[-1][1]
        if step >= min_step:
            recent = [v for _, v in ent[-20:]]
            avg = sum(recent) / len(recent)
            if avg > 0.95:
                problems.append('ENT_PIN   normalized entropy pinned high '
                                '(%.3f) -- head cannot commit to a class' % avg)
            elif avg < 0.20:
                problems.append('ENT_PIN   normalized entropy pinned low '
                                '(%.3f) -- collapsed onto one class' % avg)

    mult = series(rows, 'mgr_actent_skill_scale_mean')
    if mult:
        info['mult'] = mult[-1][1]
        if step >= min_step and mult[-1][1] > ACTENT_MAX / 10:
            problems.append('MULT_RAIL entropy multiplier %.4g near the %.0f '
                            'cap -- target unreachable (e479)' % (mult[-1][1], ACTENT_MAX))

    spars = series(rows, 'goal/implicit_sparsity_block')
    if spars:
        info['blk_change'] = spars[-1][1]
        if step >= min_step and all(v < 1e-3 for _, v in spars[-20:]):
            problems.append('FLAT      manager never changes a block -- the '
                            'goal code is frozen')

    sc = series(rows, 'episode/score')
    if sc:
        info['score'] = sum(v for _, v in sc[-15:]) / len(sc[-15:])
    return name, step, problems, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--glob', default='e58[2-6]_*_som_line_poisson_*')
    p.add_argument('--stale-minutes', type=float, default=60.0,
                   help='a started run with no metrics after this long is a failure')
    p.add_argument('--min-step', type=int, default=50_000,
                   help='below this, only hard failures (NaN) are reported')
    a = p.parse_args()

    dirs = sorted(glob.glob(os.path.join(WORK, a.glob)))
    if not dirs:
        print('no run dirs matching %s' % a.glob)
        return 0
    bad = 0
    for d in dirs:
        name, step, problems, info = check(d, a.min_step, a.stale_minutes)
        tag = 'FAIL' if problems else 'ok  '
        extra = '  '.join('%s=%.4g' % (k, v) for k, v in sorted(info.items())
                          if k != 'step')
        print('%s %-52s step=%-9s %s' % (tag, name, step, extra))
        for msg in problems:
            print('       %s' % msg)
        bad += bool(problems)
    print('\n%d/%d runs healthy' % (len(dirs) - bad, len(dirs)))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
