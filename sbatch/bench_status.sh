#!/bin/bash -l
# Status of the 2026-09-04 benchmark matrix: queue state, per-run progress,
# and anything that looks wrong (no metrics, stalled step, non-zero exit).
#
#   bash sbatch/bench_status.sh          # summary
#   bash sbatch/bench_status.sh -v       # one line per run
set -uo pipefail
WD=/work/DoyaU/vasilache/work
VERBOSE="${1:-}"

echo "=== queue ==="
squeue -u "$(whoami)" -h -o "%T" | sort | uniq -c | awk '{printf "  %-10s %s\n", $2, $1}'
for p in a100 v100 p100; do
  r=$(squeue -u "$(whoami)" -h -n "bench_$p" -t RUNNING | wc -l)
  q=$(squeue -u "$(whoami)" -h -n "bench_$p" -t PENDING | wc -l)
  printf "  bench_%-5s running=%-3s pending=%s\n" "$p" "$r" "$q"
done

echo
echo "=== runs ==="
python3 - "$WD" "$VERBOSE" <<'PYEOF'
import json, os, sys, glob, time, subprocess
wd, verbose = sys.argv[1], sys.argv[2] == '-v'
# Runs deliberately abandoned. Without this they are reported as stalled in
# every snapshot, which over an unattended stretch buries real problems.
try:
    dropped = {l.strip() for l in open(os.path.join(wd, 'bench_droplist.txt')) if l.strip()}
except Exception:
    dropped = set()
PROJECT = 'dreamerv3-bench-2026-09'


def target_steps(d):
    """What this run was configured to reach.

    The size6m arms stop at 1.1M, the director_og and size50m arms at 4M, so a
    single hardcoded number would report the long arms as permanently stalled.
    0.99 because the driver checks between chunks and stops slightly short.
    """
    try:
        run = False
        for line in open(os.path.join(d, 'logdir', 'config.yaml')):
            if line.startswith('run:'):
                run = True
            elif run and line[:1].isalpha():
                break
            elif run and line.startswith('  steps:'):
                return float(line.split(':')[1]) * 0.99
    except Exception:
        pass
    return 1_090_000.0


def watchdog_jobs():
    """job the watchdog last submitted, per run. `job.env` is only rewritten
    when the replacement actually STARTS, so between a resubmit and its start
    the run's own job.env still names the dead job -- which is exactly the
    window a queued retry sits in while it waits for a node."""
    out = {}
    try:
        for line in open(os.path.join(wd, 'bench_watchdog_state.tsv')):
            f = line.rstrip('\n').split('\t')
            if len(f) >= 3 and f[0] != '__hwm__' and f[2] not in ('', '-'):
                out[f[0]] = f[2]
    except Exception:
        pass
    return out


WD_JOBS = watchdog_jobs()


def queued(d):
    """Is a job that owns this run dir still in the queue (running OR pending)?"""
    jobs = []
    try:
        with open(os.path.join(d, 'job.env')) as f:
            jobs.append(next(l.split('=', 1)[1].strip()
                             for l in f if l.startswith('JOB=')))
    except Exception:
        pass
    j = WD_JOBS.get(os.path.basename(d))
    if j:
        jobs.append(j)
    if not jobs:
        return False
    return any(_in_queue(job) for job in jobs)


def _in_queue(job):
    try:
        # 3.6-compatible: this runs under the login node's python3, which has
        # neither `capture_output` nor `text`. Getting that wrong is silent
        # here -- the except below turns it into "not queued" and the false
        # alarm stays.
        out = subprocess.run(
            ['squeue', '-j', job, '-h', '-o', '%i'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=30)
        return bool(out.stdout.strip())
    except Exception:
        return False  # squeue unavailable: report rather than hide


rows, bad = [], []
# Ours are the dirs whose job.env names the benchmark's wandb project; a job-id
# prefix glob silently excluded every array submitted after the first day.
for d in sorted(glob.glob(os.path.join(wd, 'e[0-9]*_j*'))):
    name = os.path.basename(d)
    try:
        if 'WANDB_PROJECT=%s' % PROJECT not in open(os.path.join(d, 'job.env')).read():
            continue
    except Exception:
        continue
    tgt = target_steps(d)
    mpath = os.path.join(d, 'logdir', 'metrics.jsonl')
    spath = os.path.join(d, 'logdir', 'scores.jsonl')
    step, score, age = None, None, None
    if os.path.exists(mpath):
        age = time.time() - os.path.getmtime(mpath)
        # Scan the file rather than seeking to a fixed tail: a metrics row here
        # carries 237 keys, so an 8 KB window can miss a complete line entirely
        # and silently report step=None. That made a 36% run look like 3.6%.
        try:
            with open(mpath) as f:
                for line in f:
                    if line.strip():
                        try:
                            step = json.loads(line).get('step', step)
                        except Exception:
                            pass
        except Exception:
            pass
    if os.path.exists(spath):
        try:
            sc = [json.loads(l) for l in open(spath) if l.strip()]
            if sc:
                tail = [r['episode/score'] for r in sc[-15:] if 'episode/score' in r]
                score = sum(tail) / len(tail) if tail else None
        except Exception:
            pass
    rows.append((name, step, score, age, tgt))
    # A run at target is DONE, not stalled -- and it stops a few thousand steps
    # short of run.steps because the driver checks between chunks, so compare
    # against what runs actually reach.
    done = step is not None and step >= tgt
    if done or name in dropped:
        continue
    if age is None or age <= 3600:
        continue
    # A quiet metrics file is not a stall if the run's job is still queued: a
    # job that hit its 48h walltime self-requeues and then sits PENDING for
    # hours waiting for a GPU. e788 was reported stalled for 269 minutes that
    # way while doing exactly the right thing. The watchdog already checks
    # this; the status script has to agree with it or every snapshot carries a
    # false alarm that buries the real ones.
    if queued(d):
        continue
    if step is None:
        bad.append(f'{name}: metrics file untouched for {age/60:.0f} min, no step')
    else:
        bad.append(f'{name}: no metric written for {age/60:.0f} min (step {step})')

done_ = [r for r in rows if r[1] and r[1] >= r[4]]
print(f'  {len(rows)} run dirs, {len(done_)} at their target step count')
if rows:
    tot = sum(r[1] for r in rows if r[1])
    want = sum(r[4] for r in rows)
    print(f'  aggregate progress: {tot/1e6:.2f}M of {want/1e6:.1f}M env steps '
          f'({100*tot/want:.1f}%)')
if verbose:
    for name, step, score, age, tgt in rows:
        s = f'{step:>9,}' if step else '        -'
        sc = f'{score:7.1f}' if score is not None else '      -'
        a = f'{age/60:5.1f}m' if age else '    -'
        print(f'  {name[:58]:58s} step={s}/{tgt/1e6:.1f}M last15={sc} idle={a}')
print()
if bad:
    print('  !! attention:')
    for b in bad:
        print(f'     {b}')
else:
    print('  no stalled runs')
PYEOF

echo
echo "=== non-zero exits (live run dirs only) ==="
# Filter to run dirs that still exist: a cancelled array leaves its old logs
# behind and they would otherwise be reported forever.
found=0
for f in "$WD"/slurm_logs/bench_*.out; do
  [ -f "$f" ] || continue
  rc=$(grep -aoE '^\[done\] rc=[0-9]+' "$f" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
  [ -n "$rc" ] && [ "$rc" != 0 ] || continue
  d=$(grep -aoE 'run_dir=\S+' "$f" 2>/dev/null | tail -1 | cut -d= -f2)
  [ -n "$d" ] && [ -d "$d" ] || continue
  # Only report a failure that is still the last word on this run. Two ways it
  # stops being that: the run has since moved to a different job, or a
  # replacement is already queued for it. Either way the incident is resolved
  # and reporting it forever turns it into a permanent alarm.
  logjob=$(basename "$f" .out); logjob=${logjob##*-}
  envjob=$(awk -F= '$1=="JOB"{print $2}' "$d/job.env" 2>/dev/null)
  [ -n "$envjob" ] && [ "${logjob%%_*}" != "${envjob%%_*}" ] && continue
  wdjob=$(awk -F'\t' -v k="$(basename "$d")" '$1==k{print $3}' \
    "$WD/bench_watchdog_state.tsv" 2>/dev/null)
  if [ -n "$wdjob" ] && [ "$wdjob" != - ] && \
     squeue -j "$wdjob" -h -o "%i" 2>/dev/null | grep -q .; then
    continue
  fi
  echo "  rc=$rc $(basename "$d")"
  found=1
done
[ "$found" -eq 0 ] && echo "  none"
