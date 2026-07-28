"""Compare the e372-e377 pre/post-refactor A/B arms.

Both arms run the same recipe at the same seed on different repo trees, so the
question is only "are these the same algorithm". Step-aligned scalar comparison
is meaningless here (the train loop is not run-to-run reproducible -- see
EXPERIMENTS.md 2026-07-28), so this compares the two things that ARE stable:

  * the score curve summaries the experiment log uses --
    ``best-15``    = highest mean over any 15 consecutive logged episode scores
    ``trailing-50``= mean of the last 50 logged episode scores
  * the mechanism metrics the refactor actually touches, averaged over the last
    third of the run (block-pooled var-K credit + implicit sparsity).

Usage:
  python tools/compare_refactor_ab.py [--root /work/DoyaU/vasilache/work]
"""
import argparse
import glob
import json
import os

PAIRS = [
    ('cartpole_swingup', 'e372', 'e373'),
    ('cheetah_run', 'e374', 'e375'),
    ('hopper_hop', 'e376', 'e377'),
]

# The quantities computed by the code paths this refactor restructured.
MECH_KEYS = [
    'train/goal/mgr_duration_mean',
    'train/mgr_switch_rate',
    'train/goal/implicit_sparsity_block',
    'train/goal/soft_reuse_overlap_mean',
    'train/loss/mgr_policy',
    'train/loss/wkr_policy',
]


def find(root, exp):
  hits = sorted(glob.glob(os.path.join(root, f'{exp}_*_refab_*', 'logdir', 'metrics.jsonl')))
  return hits[0] if hits else None


def load(path):
  scores, mech, last_step = [], {}, 0
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        rec = json.loads(line)
      except json.JSONDecodeError:
        continue  # a partially-flushed final line while the run is live
      last_step = max(last_step, rec.get('step', 0) or 0)
      if 'episode/score' in rec:
        scores.append(rec['episode/score'])
      for k in MECH_KEYS:
        if k in rec:
          mech.setdefault(k, []).append(rec[k])
  return scores, mech, last_step


def best_n(scores, n=15):
  if len(scores) < n:
    return float('nan')
  return max(sum(scores[i:i + n]) / n for i in range(len(scores) - n + 1))


def trailing(scores, n=50):
  if not scores:
    return float('nan')
  tail = scores[-n:]
  return sum(tail) / len(tail)


def mean(xs):
  return sum(xs) / len(xs) if xs else float('nan')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--root', default='/work/DoyaU/vasilache/work')
  args = ap.parse_args()

  for task, pre_exp, post_exp in PAIRS:
    pre_p, post_p = find(args.root, pre_exp), find(args.root, post_exp)
    print(f'\n=== {task}   {pre_exp}(pre) vs {post_exp}(post)')
    if not pre_p or not post_p:
      print(f'    missing metrics (pre={bool(pre_p)}, post={bool(post_p)})')
      continue
    ps, pm, pstep = load(pre_p)
    qs, qm, qstep = load(post_p)
    pct = lambda s: 100.0 * s / 4e6
    print(f'    progress   pre {pstep:>9,} ({pct(pstep):5.1f}%)   '
          f'post {qstep:>9,} ({pct(qstep):5.1f}%)   episodes {len(ps)}/{len(qs)}')
    for label, fn in (('best-15', best_n), ('trailing-50', trailing)):
      a, b = fn(ps), fn(qs)
      gap = abs(a - b) / (abs(b) + 1e-9) if b == b and a == a else float('nan')
      print(f'    {label:<12} pre {a:10.2f}   post {b:10.2f}   rel {gap:7.3f}')
    print('    -- mechanism metrics (last third) --')
    for k in MECH_KEYS:
      va, vb = pm.get(k, []), qm.get(k, [])
      if not va or not vb:
        continue
      a = mean(va[len(va) * 2 // 3:])
      b = mean(vb[len(vb) * 2 // 3:])
      rel = abs(a - b) / (abs(b) + 1e-12)
      flag = '  <-- CHECK' if rel > 0.05 else ''
      print(f'      {k:38s} pre {a:11.5g}  post {b:11.5g}  rel {rel:.2e}{flag}')

  print('\nReminder: both arms are the SAME recipe; the only variable is which '
        'tree ran it. Score gaps within seed noise are expected; a systematic '
        'gap, or a mechanism metric off by >5%, is the failure signal.')


if __name__ == '__main__':
  main()
