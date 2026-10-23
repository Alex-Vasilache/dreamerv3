#!/usr/bin/env python3
"""Does a block's class index behave the way the UCB count kernel assumes?

The imagination UCB table (hrl/explore.py ImagCounts) bins each block's class
index into contiguous bins (8, 4 or 2 per block) and smears every count onto
neighbouring bins of the SAME block with a Gaussian in index distance. That is
only right if, for every block,

  * moving a block's class by one changes the decoded goal a little, and by
    more classes changes it more (locality / order),
  * one class step means about the same change everywhere (even spacing, so
    one kernel width fits all positions and all blocks),
  * the classes merged into one bin decode to similar goals (bin coherence).

This measures the decoded goal directly. For N uniform base codes, every block
l and every class c, it decodes the base code with block l set to c, and
stores the mean goal dissimilarity 1 - cosine_max between classes c and c' of
the same block over the N bases:

  M[l, c, c']   (L x C x C)

Everything the figure needs is derived from M.

  python -u ucb_kernel_geometry.py --run_dir <run> --out <file stem>
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from code_sim_to_state_sim import onehot_of
from diag_goal_geometry import build, make_fns


def cosmax_pairs(g):
  """g: (N, C, D) -> (C, C) mean over N of 1 - cosine_max between classes."""
  g = g.astype(np.float64)
  n = np.linalg.norm(g, axis=-1)                                   # (N, C)
  dot = np.einsum('ncd,ned->nce', g, g)
  m = np.maximum(n[:, :, None], n[:, None, :]) + 1e-12
  return (1 - dot / (m * m)).mean(0)


def run(run_dir, out, n_base=256, seed=0):
  config, agent = build(run_dir)
  _, _, decode = make_fns(config, agent)
  L, C = [int(x) for x in config.agent.skill_shape]
  rng = np.random.default_rng(seed)
  base = rng.integers(0, C, (n_base, L))
  M = np.zeros((L, C, C))
  for l in range(L):
    ids = np.repeat(base[:, None], C, 1)                           # (N, C, L)
    ids[:, :, l] = np.arange(C)[None]
    g = decode(onehot_of(ids.reshape(-1, L), C)).reshape(n_base, C, -1)
    M[l] = cosmax_pairs(g)
  name = pathlib.Path(str(run_dir).rstrip('/')).name
  pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(out + '.npz', M=M, task=str(config.task), run=name)
  adj = np.array([M[:, c, c + 1] for c in range(C - 1)])
  print(f'{name[:6]} {config.task}: one-step {adj.mean():.4f}  max {M.max():.4f}')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run_dir', required=True)
  p.add_argument('--out', required=True)
  p.add_argument('--n_base', type=int, default=256)
  a = p.parse_args()
  run(a.run_dir, a.out, a.n_base)


if __name__ == '__main__':
  main()
