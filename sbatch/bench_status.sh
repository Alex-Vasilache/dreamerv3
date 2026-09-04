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
import json, os, sys, glob, time
wd, verbose = sys.argv[1], sys.argv[2] == '-v'
rows, bad = [], []
for d in sorted(glob.glob(os.path.join(wd, 'e7*_j4706*'))):
    name = os.path.basename(d)
    mpath = os.path.join(d, 'logdir', 'metrics.jsonl')
    spath = os.path.join(d, 'logdir', 'scores.jsonl')
    step, score, age = None, None, None
    if os.path.exists(mpath):
        age = time.time() - os.path.getmtime(mpath)
        try:
            with open(mpath, 'rb') as f:
                f.seek(max(0, f.seek(0, 2) - 8192))
                last = [l for l in f.read().decode(errors='ignore').split('\n') if l.strip()]
            step = json.loads(last[-1]).get('step')
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
    rows.append((name, step, score, age))
    if step is None and age is not None and age > 3600:
        bad.append(f'{name}: metrics file untouched for {age/60:.0f} min, no step')
    elif age is not None and age > 3600:
        bad.append(f'{name}: no metric written for {age/60:.0f} min (step {step})')

done_ = [r for r in rows if r[1] and r[1] >= 1_090_000]
print(f'  {len(rows)} run dirs, {len(done_)} at/over 1.09M steps')
if rows:
    withstep = [r for r in rows if r[1]]
    if withstep:
        tot = sum(r[1] for r in withstep)
        print(f'  aggregate progress: {tot/1e6:.2f}M of {len(rows)*1.1:.1f}M env steps '
              f'({100*tot/(len(rows)*1.1e6):.1f}%)')
if verbose:
    for name, step, score, age in rows:
        s = f'{step:>9,}' if step else '        -'
        sc = f'{score:7.1f}' if score is not None else '      -'
        a = f'{age/60:5.1f}m' if age else '    -'
        print(f'  {name[:58]:58s} step={s} last15={sc} idle={a}')
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
  echo "  rc=$rc $(basename "$d")"
  found=1
done
[ "$found" -eq 0 ] && echo "  none"
