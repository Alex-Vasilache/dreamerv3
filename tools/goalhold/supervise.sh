#!/bin/bash
# Status of the goal-hold array (h0374-h0403). Read-only.
A=${1:-4812978}
date
echo "== queue"; squeue -u $(whoami) -h -o "%i %T %M %R" | grep "^${A}_" | awk '{print $2}' | sort | uniq -c
echo "== sacct"; sacct -j $A -X -n -o State | sort | uniq -c
echo "== runs"
python3 - "$A" <<'P'
import glob, json, os, sys, math, time
A = sys.argv[1]
for d in sorted(glob.glob(f'/work/DoyaU/vasilache/work/h0[34]*_j{A}_*')):
  name = os.path.basename(d); tag = name.split('_')[0]; task = name.split('_director')[0][6:]
  mp = d + '/logdir/metrics.jsonl'
  if not os.path.exists(mp): print(tag, task, 'no metrics yet'); continue
  last = {}; bad = 0; rows = 0
  with open(mp) as f:
    for l in f:
      try: r = json.loads(l)
      except Exception: continue
      rows += 1; last.update(r)
      bad += any(isinstance(v, float) and not math.isfinite(v) and not k.startswith("timer/") for k, v in r.items())
  sc = []
  sp = d + '/logdir/scores.jsonl'
  if os.path.exists(sp):
    for l in open(sp):
      try: r = json.loads(l); sc.append((r['step'], r.get('episode/score', 0)))
      except Exception: pass
  g = lambda k: next((v for kk, v in last.items() if kk.endswith(k)), float('nan'))
  recent = [s for t, s in sc[-20:]]
  age = (time.time() - os.path.getmtime(mp)) / 60
  cap = '8 ' if 'cap8' in name else '16' if 'cap16' in name else '32'; seed = name.split('_s')[-1].split('_')[0]
  print(f"{tag} cap{cap} s{seed} {task:22s} step {last.get('step', 0)/1e6:5.2f}M fps {g('fps/policy'):5.0f} | hold imag {g('goal/hold_len_imag'):5.1f} repl {g('goal/hold_len_repl'):5.1f} reach {g('goal/hold_reach_frac'):4.2f} len1 {g('goal/hold_len1_frac'):4.2f} ceil {g('goal/reach_ceil'):4.2f} | wkr_rew {g('wkr_goal_rew'):5.2f} | score last20 {sum(recent)/max(len(recent),1):6.1f} | nonfinite(non-timer) {bad} | log age {age:4.0f}m")
P
echo "== errors"; grep -l "Traceback\|CUDA_ERROR\|out of memory\|Killed" /work/DoyaU/vasilache/work/slurm_logs/goalhold-*.err /work/DoyaU/vasilache/work/slurm_logs/goalhold-*.out 2>/dev/null | head
ls /work/DoyaU/vasilache/work/slurm_logs/ | grep -c "goalhold-"
