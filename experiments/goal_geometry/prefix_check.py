#!/usr/bin/env python3
"""Does the goal code of a pin pad state tell different pad-sequence prefixes apart?

The from-to novelty (dreamerv3/hrl/fromto.py) keys counts by the start state's
goal code. On pin pad the task is a sequence, so the start code must separate
"at pad C after A, B" from "at pad C after D, E", or from-to cannot tell which
next step is new from here. Pin pad draws the recent pad history in the
image's bottom band, so the prefix is read off the observation.

Rolls out the run's own policy, records (prefix of the last ``k`` pads, goal
code), and reports, at full class resolution and at the start binning:
  purity   share of states whose code's most common prefix is their own prefix
  split    of prefix pairs that differ, how often their states share a code

  python -u prefix_check.py --run_dir <pinpad run> --bins 2 --k 2
"""
import argparse
import collections
import pathlib
import sys
from functools import partial as bind

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from diag_goal_geometry import build

COLORS = {(255, 0, 0): '1', (0, 255, 0): '2', (0, 0, 255): '3', (255, 255, 0): '4',
          (255, 0, 255): '5', (0, 255, 255): '6', (128, 0, 128): '7', (0, 128, 128): '8'}


def prefix(image, k):
  """Pad history from the bottom band: grid cell (2i+1, 14) -> image pixels."""
  seq = []
  for i in range(8):
    px = tuple(int(v) for v in image[14 * 4 + 1, (2 * i + 1) * 4 + 1])
    if px in COLORS:
      seq.append(COLORS[px])
  return tuple(seq[-k:]) if k else tuple(seq)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run_dir', required=True)
  p.add_argument('--steps', type=int, default=40000)
  p.add_argument('--k', type=int, default=2)
  p.add_argument('--bins', type=int, nargs='+', default=[8, 4, 2])
  a = p.parse_args()
  import embodied
  import dreamerv3.main as m
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  config, agent = build(a.run_dir)
  rows = []
  driver = embodied.Driver([bind(m.make_env, config, i) for i in range(8)], parallel=True)
  driver.reset(agent.init_policy)
  driver.on_step(lambda tran, w: 'log/struct_diag_deter' in tran and rows.append(
      (prefix(tran['image'], a.k), np.asarray(tran['log/struct_diag_deter']).copy())))
  driver(agent.policy, steps=a.steps)
  driver.close()
  model = agent.model
  deter = np.stack([r[1] for r in rows])
  pre = [r[0] for r in rows]

  def enc(d):
    out = model.goal_enc(d, 1)
    out = out['skill'] if isinstance(out, dict) else out
    return jnp.argmax(out.pred(), -1)
  codes = np.concatenate([np.asarray(nj.pure(enc)(agent.params, jnp.asarray(deter[i:i + 4096]))[1])
                          for i in range(0, len(deter), 4096)])
  C = 8
  print(f'{len(rows)} states, {len(set(pre))} distinct last-{a.k} prefixes')
  for bins in a.bins:
    key = [tuple(np.minimum(c * bins // C, bins - 1)) for c in codes]
    by = collections.defaultdict(collections.Counter)
    for kk, pp in zip(key, pre):
      by[kk][pp] += 1
    purity = sum(c.most_common(1)[0][1] for c in by.values()) / len(rows)
    # Of state pairs with different prefixes, share with the same code.
    pc = collections.Counter(zip(key, pre))
    kc = collections.Counter(key)
    same_code = sum(n * n for n in kc.values())
    same_both = sum(n * n for n in pc.values())
    prc = collections.Counter(pre)
    same_pre = sum(n * n for n in prc.values())
    tot = len(rows) ** 2
    diff_pre = tot - same_pre
    split = (same_code - same_both) / max(diff_pre, 1)
    print(f'  {bins} bins/block: {len(kc)} codes used, purity {purity:.3f}, '
          f'different-prefix pairs sharing a code {split:.4f} (random baseline {same_code / tot:.4f})')


if __name__ == '__main__':
  main()
