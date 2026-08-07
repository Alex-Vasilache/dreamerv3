#!/usr/bin/env python3
"""Measure how far a goal-AE codebook is ordered along its topology.

Reports rho, the mean distance between entries adjacent in index divided by the
mean distance between arbitrary distinct entries, per block and pooled. rho is a
ratio of distances, so it is invariant to the codebook's scale and says nothing
about whether the entries are used -- read it together with the perplexity
logged during training.

Reference values for C evenly spaced points, computed here rather than hardcoded:
a ring's wrap keeps its farthest pairs closer than a line's ends, so a ring
compresses the top of the distance scale (0.533 at C=8, against a line's 0.333).
Both are references, not bounds: the entries live in R^d and either arrangement
can fold.

Reads the codebook straight out of agent.pkl, so it needs no GPU and does not
touch the running job.

  experiments/goal_code_struct_corr/diag_codebook_ratio.py e481 --topology line
"""

import argparse
import glob
import itertools
import pickle
import sys

import numpy as np


def ideal_ratio(classes, topology):
  """rho for `classes` points spaced evenly on the given topology."""
  if topology == 'ring':
    ang = 2 * np.pi * np.arange(classes) / classes
    pts = np.stack([np.cos(ang), np.sin(ang)], -1)
    adj = [(k, (k + 1) % classes) for k in range(classes)]
  else:
    pts = np.arange(classes, dtype=np.float64)[:, None]
    adj = [(k, k + 1) for k in range(classes - 1)]
  d = lambda i, j: float(np.linalg.norm(pts[i] - pts[j]))
  near = np.mean([d(i, j) for i, j in adj])
  allp = np.mean([d(i, j) for i, j in itertools.combinations(range(classes), 2)])
  return near / allp


def rho(table, topology):
  """table: (L, C, D). Returns (pooled, per_block)."""
  blocks, classes, _ = table.shape
  per = []
  for l in range(blocks):
    e = table[l]
    if topology == 'ring':
      adj = [(k, (k + 1) % classes) for k in range(classes)]
    else:
      adj = [(k, k + 1) for k in range(classes - 1)]
    near = np.mean([np.linalg.norm(e[i] - e[j]) for i, j in adj])
    allp = np.mean([np.linalg.norm(e[i] - e[j])
                    for i, j in itertools.combinations(range(classes), 2)])
    per.append(float(near / allp) if allp > 0 else float('nan'))
  return float(np.mean(per)), per


def find_table(params):
  """The (L, C, D) codebook, located by shape rather than by name."""
  # The optimizer state carries same-shaped copies of every parameter (momentum,
  # second moment), so matching on 'codebook' alone returns three arrays.
  hits = {k: np.asarray(v) for k, v in params.items()
          if 'codebook' in k.lower() and np.asarray(v).ndim == 3
          and '/opt/' not in k}
  if not hits:
    hits = {k: np.asarray(v) for k, v in params.items()
            if np.asarray(v).ndim == 3 and np.asarray(v).shape[0] == 8
            and np.asarray(v).shape[1] == 8}
  if len(hits) != 1:
    raise SystemExit(f'expected exactly one codebook, found {sorted(hits)}')
  return next(iter(hits.items()))


def load_params(run_glob):
  hits = sorted(glob.glob(
      f'/work/DoyaU/vasilache/work/{run_glob}_*/logdir/ckpt/*/agent.pkl'))
  hits = [h for h in hits if 'latest' not in h]
  if not hits:
    raise SystemExit(f'no checkpoint for {run_glob}')
  path = hits[-1]
  with open(path, 'rb') as f:
    blob = pickle.load(f)
  # embodied stores {group: {name: array}}; flatten one level if needed.
  flat = {}
  def walk(d, prefix=''):
    for k, v in d.items():
      if isinstance(v, dict):
        walk(v, f'{prefix}{k}/')
      else:
        flat[f'{prefix}{k}'] = v
  walk(blob if isinstance(blob, dict) else {'root': blob})
  return path, flat


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('runs', nargs='+')
  ap.add_argument('--topology', default='ring', choices=['ring', 'line'])
  args = ap.parse_args()
  for run in args.runs:
    try:
      path, params = load_params(run)
      name, table = find_table(params)
    except SystemExit as e:
      print(f'{run}: {e}')
      continue
    pooled, per = rho(table, args.topology)
    L, C, D = table.shape
    ref = ideal_ratio(C, args.topology)
    print(f'{run}  ({args.topology}, {L}x{C}x{D}, {name})')
    print(f'   rho = {pooled:.3f}   ideal {args.topology} = {ref:.3f}   '
          f'unordered = 1.000')
    print(f'   per block: ' + ' '.join(f'{v:.2f}' for v in per))
    print(f'   ckpt: {path.split("/ckpt/")[-1].split("/")[0]}')


if __name__ == '__main__':
  main()
