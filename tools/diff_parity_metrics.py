"""Compare two parity-smoke runs' metrics.jsonl step-for-step.

The 2026-07-28 modularization is a pure restructuring of the retained code
paths, so every leg of ``sbatch/run_parity_smoke.sbatch`` must reproduce its
pre-refactor numbers exactly (same seed, single-threaded CPU XLA). This walks
the per-leg ``metrics.jsonl`` files and reports the largest disagreement on any
shared scalar key.

Usage:
  python tools/diff_parity_metrics.py <pre_dir> <post_dir> [--rtol 1e-6]
"""
import argparse
import json
import math
import pathlib
import sys


def load(path):
  """{step: {key: value}} for the scalar columns of one metrics.jsonl."""
  rows = {}
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      step = rec.get('step')
      if step is None:
        continue
      rows[step] = {
          k: v for k, v in rec.items()
          if isinstance(v, (int, float)) and not isinstance(v, bool)}
  return rows


def close(a, b, rtol, atol):
  if a == b:
    return True
  if math.isnan(a) and math.isnan(b):
    return True
  return abs(a - b) <= atol + rtol * abs(b)


def compare_leg(pre_path, post_path, rtol, atol):
  pre, post = load(pre_path), load(post_path)
  steps = sorted(set(pre) & set(post))
  if not steps:
    return None, f'no overlapping steps ({len(pre)} pre rows, {len(post)} post rows)'
  worst = (0.0, None, None)
  n_keys = 0
  mismatches = []
  for step in steps:
    shared = set(pre[step]) & set(post[step])
    n_keys = max(n_keys, len(shared))
    for key in shared:
      a, b = pre[step][key], post[step][key]
      if close(a, b, rtol, atol):
        continue
      rel = abs(a - b) / (abs(b) + 1e-12)
      mismatches.append((rel, step, key, a, b))
      if rel > worst[0]:
        worst = (rel, key, step)
  only_pre = set().union(*[set(pre[s]) for s in steps]) - set().union(
      *[set(post[s]) for s in steps])
  only_post = set().union(*[set(post[s]) for s in steps]) - set().union(
      *[set(pre[s]) for s in steps])
  return dict(
      steps=len(steps), keys=n_keys, mismatches=sorted(mismatches, reverse=True),
      worst=worst, only_pre=sorted(only_pre), only_post=sorted(only_post)), None


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('pre')
  ap.add_argument('post')
  ap.add_argument('--rtol', type=float, default=1e-6)
  ap.add_argument('--atol', type=float, default=1e-8)
  ap.add_argument('--show', type=int, default=5)
  args = ap.parse_args()

  pre_root, post_root = pathlib.Path(args.pre), pathlib.Path(args.post)
  legs = sorted(p.name for p in pre_root.iterdir() if p.is_dir())
  if not legs:
    sys.exit(f'no leg directories under {pre_root}')

  failed = False
  for leg in legs:
    pre_f = pre_root / leg / 'metrics.jsonl'
    post_f = post_root / leg / 'metrics.jsonl'
    if not pre_f.exists() or not post_f.exists():
      print(f'{leg:>16}: MISSING (pre={pre_f.exists()}, post={post_f.exists()})')
      failed = True
      continue
    res, err = compare_leg(pre_f, post_f, args.rtol, args.atol)
    if err:
      print(f'{leg:>16}: {err}')
      failed = True
      continue
    tag = 'IDENTICAL' if not res['mismatches'] else 'DIFFERS'
    print(f'{leg:>16}: {tag}  ({res["steps"]} steps x ~{res["keys"]} keys)')
    if res['only_pre']:
      print(f'{"":>18}keys only in pre : {res["only_pre"]}')
    if res['only_post']:
      print(f'{"":>18}keys only in post: {res["only_post"]}')
    for rel, step, key, a, b in res['mismatches'][:args.show]:
      print(f'{"":>18}step {step} {key}: pre={a!r} post={b!r} (rel {rel:.3e})')
    if res['mismatches']:
      failed = True

  print('\nRESULT:', 'MISMATCH' if failed else 'ALL LEGS IDENTICAL')
  sys.exit(1 if failed else 0)


if __name__ == '__main__':
  main()
