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

# task -> (pre arm, post arm, same-code replicate of the pre arm or None).
# The replicate is the calibration baseline: it differs from ``pre`` by nothing
# but run-to-run nondeterminism, so a post-vs-pre gap is only evidence of a
# behavior change if it materially EXCEEDS the pre-vs-replicate gap.
PAIRS = [
    ('cartpole_swingup', 'e372', 'e373', 'e378'),
    ('cheetah_run', 'e374', 'e375', 'e379'),
    ('hopper_hop', 'e376', 'e377', None),
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
  """(scores, mech, last_step) with every sample tagged by its env step.

  Values are kept as ``(step, value)`` so arms can be compared over a COMMON
  step window. Arms launched minutes apart sit at different progress, and a
  plain 'last third of each file' average would then compare different amounts
  of training -- which is a confound, not a measurement.
  """
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
      step = rec.get('step', 0) or 0
      last_step = max(last_step, step)
      if 'episode/score' in rec:
        scores.append((step, rec['episode/score']))
      for k in MECH_KEYS:
        if k in rec:
          mech.setdefault(k, []).append((step, rec[k]))
  return scores, mech, last_step


def window(pairs, lo, hi):
  """Values whose step falls in [lo, hi]."""
  return [v for s, v in pairs if lo <= s <= hi]


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


def verdict(gap, self_gap):
  """Is ``gap`` (post vs pre) large relative to same-code run-to-run noise?"""
  if gap != gap:
    return '-'
  if self_gap != self_gap:
    return 'no control yet'
  if gap <= max(self_gap * 1.5, self_gap + 0.02):
    return 'within noise'
  if gap <= max(self_gap * 3.0, self_gap + 0.10):
    return 'inconclusive'
  return 'EXCEEDS NOISE'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--root', default='/work/DoyaU/vasilache/work')
  args = ap.parse_args()

  for task, pre_exp, post_exp, ctl_exp in PAIRS:
    pre_p, post_p = find(args.root, pre_exp), find(args.root, post_exp)
    ctl_p = find(args.root, ctl_exp) if ctl_exp else None
    print(f'\n=== {task}   {pre_exp}(pre) vs {post_exp}(post)'
          + (f'   [control {ctl_exp}: same code as pre]' if ctl_p else
             '   [no same-code control]'))
    if not pre_p or not post_p:
      print(f'    missing metrics (pre={bool(pre_p)}, post={bool(post_p)})')
      continue
    ps, pm, pstep = load(pre_p)
    qs, qm, qstep = load(post_p)
    cs, cm, cstep = load(ctl_p) if ctl_p else ([], {}, 0)

    # Compare only over the step range every arm has actually reached, and only
    # its second half -- early training is dominated by transients.
    common = min([pstep, qstep] + ([cstep] if ctl_p else []))
    lo, hi = common // 2, common
    pct = lambda s: 100.0 * s / 4e6
    line = (f'    progress   pre {pstep:>9,} ({pct(pstep):5.1f}%)   '
            f'post {qstep:>9,} ({pct(qstep):5.1f}%)')
    if ctl_p:
      line += f'   ctl {cstep:>9,} ({pct(cstep):5.1f}%)'
    print(line)
    print(f'    comparing over the COMMON window {lo:,}-{hi:,} steps')
    rel = lambda a, b: (abs(a - b) / (abs(b) + 1e-9)
                        if a == a and b == b else float('nan'))
    print(f'    {"":<12} {"pre":>10} {"post":>10} {"gap":>8} | '
          f'{"ctl":>10} {"self-gap":>9}   verdict')
    for label, fn in (('best-15', best_n), ('trailing-50', trailing)):
      a = fn(window(ps, lo, hi))
      b = fn(window(qs, lo, hi))
      c = fn(window(cs, lo, hi)) if ctl_p else float('nan')
      g, sg_ = rel(a, b), rel(a, c)
      print(f'    {label:<12} {a:10.2f} {b:10.2f} {g:8.3f} | '
            f'{c:10.2f} {sg_:9.3f}   {verdict(g, sg_)}')
    print('    -- mechanism metrics (common window) --')
    for k in MECH_KEYS:
      va = window(pm.get(k, []), lo, hi)
      vb = window(qm.get(k, []), lo, hi)
      if not va or not vb:
        continue
      vc = window(cm.get(k, []), lo, hi) if ctl_p else []
      a, b = mean(va), mean(vb)
      c = mean(vc) if vc else float('nan')
      g, sg_ = rel(a, b), rel(a, c)
      print(f'      {k:36s} {a:11.5g} {b:11.5g} {g:8.3f} | '
            f'{c:11.5g} {sg_:9.3f}   {verdict(g, sg_)}')

  print('\nBoth arms are the SAME recipe; the only variable is which tree ran '
        'it.\n"gap" = post vs pre. "self-gap" = a SAME-CODE replicate vs pre, '
        'i.e. pure\nrun-to-run nondeterminism. A gap only indicates a behavior '
        'change if it\nclearly exceeds the self-gap -- raw gap size alone means '
        'nothing here.')


if __name__ == '__main__':
  main()
