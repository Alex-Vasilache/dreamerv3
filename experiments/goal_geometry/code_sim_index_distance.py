#!/usr/bin/env python3
"""Goal-state similarity against TOTAL index distance, the code's 57 bins.

The block-count measurement treats every changed row alike. On a SOM the class
index is ordered, so moving a row one class is a smaller edit than moving it
seven, and the distance that respects that is

    D = sum_l |c'_l - c_l|,     0 <= D <= L * (C - 1) = 56

L=8 rows each contributing at most C-1=7. D=0 is identical codes; D=56 is every
row different by the full 7.

Pairs are sampled JOINTLY, not by perturbing a base code. Perturbing cannot
reach the top of the axis: a row sitting at class c can only move
max(c, C-1-c), so a uniformly drawn base code has a total capacity of 44 on
average and only a code with every row at 0 or 7 -- probability (2/8)^8, about
1.5e-5 -- can reach 56 at all. Sampling pairs directly gives every bin the same
number of pairs:

  * split D into per-row distances d_l by drawing D of the L*(C-1) unit slots
    without replacement, which enforces sum d_l = D and d_l <= C-1 at once;
  * per row pick the pair (c, c+d_l) with c uniform over the 8-d_l starts that
    admit it.

Reports the mean and the full spread per bin, because at fixed code distance
the goal similarity varies by more than the mean moves across the whole axis.

  python -u code_sim_index_distance.py --run_dir /work/.../e534_... --out out/e534
"""
import argparse
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent /
                      'goal_code_struct_corr'))

import numpy as np

from code_sim_to_state_sim import cosine_max, onehot_of
from diag_goal_geometry import build, make_fns


def sample_pairs(n, D, blocks, classes, rng):
    """n code pairs at total index distance exactly D."""
    span = classes - 1
    # per-row distances: draw D of the blocks*span unit slots without
    # replacement, so the counts sum to D and none exceeds span
    slots = np.argsort(rng.random((n, blocks * span)), axis=1)[:, :D]
    d = np.zeros((n, blocks), np.int64)
    if D:
        row = slots // span
        np.add.at(d, (np.arange(n)[:, None], row), 1)
    # per row: c uniform over the starts that admit distance d, partner c+d
    c = (rng.random((n, blocks)) * (classes - d)).astype(np.int64)
    return c, c + d


def run(run_dir, out_path, n_pairs=100000, chunk=32768, seed=0):
    config, agent = build(run_dir)
    impl, encode, decode = make_fns(config, agent)
    blocks, classes = [int(x) for x in config.agent.skill_shape]
    dmax = blocks * (classes - 1)
    name = pathlib.Path(str(run_dir).rstrip('/')).name
    print(f'=== {name} | impl={impl} L={blocks} C={classes} | '
          f'task={config.task} | D = 0..{dmax}, {n_pairs} pairs per bin')

    rng = np.random.default_rng(seed)
    mean = np.zeros(dmax + 1)
    sd = np.zeros(dmax + 1)
    quant = np.zeros((dmax + 1, 5))
    mean[0], quant[0] = 1.0, 1.0
    for D in range(1, dmax + 1):
        vals = np.empty(n_pairs)
        for lo in range(0, n_pairs, chunk):
            hi = min(lo + chunk, n_pairs)
            a, b = sample_pairs(hi - lo, D, blocks, classes, rng)
            assert (np.abs(b - a).sum(1) == D).all()
            assert a.min() >= 0 and b.max() < classes
            vals[lo:hi] = cosine_max(decode(onehot_of(a, classes)),
                                     decode(onehot_of(b, classes)))
        mean[D], sd[D] = vals.mean(), vals.std()
        quant[D] = np.percentile(vals, [5, 25, 50, 75, 95])
    print('  D -> goal-state cosine_max (every 7th bin)')
    print('   ' + '  '.join(f'{D}:{mean[D]:.4f}' for D in range(0, dmax + 1, 7)))
    print('  spread (std) at those bins')
    print('   ' + '  '.join(f'{D}:{sd[D]:.4f}' for D in range(0, dmax + 1, 7)))
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + '.npz', curve=mean, pair_sd=sd,
                        pair_q=quant, d=np.arange(dmax + 1),
                        task=str(config.task), run=name, n_pairs=n_pairs)
    print('wrote', str(out) + '.npz')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n_pairs', type=int, default=100000)
    p.add_argument('--chunk', type=int, default=32768)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    run(a.run_dir, a.out, a.n_pairs, a.chunk, a.seed)


if __name__ == '__main__':
    main()
