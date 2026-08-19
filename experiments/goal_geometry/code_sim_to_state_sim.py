#!/usr/bin/env python3
"""Code similarity -> goal-state similarity, on the code's own 9 bins.

The measurement the paragraph describes: sample ``n_codes`` hard codes, pair
each with a partner at every one of the L+1 = 9 similarities the code admits
(0 blocks different through all 8 different), decode both, and record the
``cosine_max`` similarity of the generated goal states -- the quantity the
low-level policy is actually rewarded on.

Sampling (``--sampling``):

  uniform   (default) ``n_codes`` codes drawn uniformly over the C^L grid --
            every block an independent uniform class. This measures the
            decoder's geometry over the whole code space the manager can
            address, and needs no environment at all.
  encoded   codes obtained by encoding real world-model states from a live
            rollout, i.e. the codes the manager actually emits on-policy. A
            different question: the geometry of the region the encoder uses.

Either way partners are *constructed* at each exact Hamming distance rather
than drawn as random pairs -- random pairs of uniform codes land almost
entirely in the 7- and 8-block bins and would leave the near end of the axis
empty. One partner per base code per bin, so every bin gets exactly
``n_codes`` pairs.

m=0 is a self-pair, so cosine_max is exactly 1 by construction and serves as the
sanity check on the whole path.

  python -u code_sim_to_state_sim.py --run_dir /work/.../e502_... --out out/e502
"""
import argparse
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent /
                      'goal_code_struct_corr'))

import numpy as np

from diag_goal_geometry import build, make_fns, collect_states


def cosine_max(x, y):
    """cosine_max(x, y) = (x/m).(y/m), m = max(|x|, |y|) -- rowwise."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    nx = np.linalg.norm(x, axis=-1)
    ny = np.linalg.norm(y, axis=-1)
    m = np.maximum(nx, ny) + 1e-12
    return (x * y).sum(-1) / (m * m)


def onehot_of(ids, classes):
    return np.eye(classes, dtype=np.float32)[ids]


def run(run_dir, out_path, n_codes=1024, n_envs=4, stride=8, seed=0,
        sampling='uniform'):
    config, agent = build(run_dir)
    impl, encode, decode = make_fns(config, agent)
    blocks, classes = [int(x) for x in config.agent.skill_shape]
    name = pathlib.Path(str(run_dir).rstrip('/')).name
    print(f'=== {name} | impl={impl} L={blocks} C={classes} | '
          f'task={config.task} | sampling={sampling}')

    rng = np.random.default_rng(seed)
    if sampling == 'uniform':
        # Uniform over the C^L grid: each block an independent uniform class.
        # No environment needed -- this only touches the decoder.
        ref_ids = rng.integers(0, classes, size=(n_codes, blocks))
        onehot = onehot_of(ref_ids, classes)
    else:
        deters = collect_states(config, agent, n_envs, stride, n_codes)
        _, _, _, onehot, _ = encode(deters[:n_codes])
        onehot = np.asarray(onehot)
        ref_ids = onehot.argmax(-1)
    ref_goal = decode(onehot)
    print(f'{len(onehot)} hard codes, goal dim {ref_goal.shape[-1]}')

    M = len(onehot)
    sim = np.full((M, blocks + 1), np.nan)
    sim[:, 0] = 1.0                     # self-pair, exactly 1 by construction
    for m in range(1, blocks + 1):
        # exactly one partner per base code, at Hamming distance exactly m
        mod = onehot.copy()
        for i in range(M):
            for b in rng.choice(blocks, size=m, replace=False):
                c = int(rng.integers(classes - 1))
                c += int(c >= ref_ids[i, b])   # uniform over the OTHER classes
                mod[i, b, :] = 0.0
                mod[i, b, c] = 1.0
        assert (mod.argmax(-1) != ref_ids).sum(-1).min() == m
        assert (mod.argmax(-1) != ref_ids).sum(-1).max() == m
        sim[:, m] = cosine_max(decode(mod), ref_goal)

    curve = sim.mean(0)
    print('  blocks different -> goal-state cosine_max')
    print('   ' + '  '.join(f'{m}:{v:.4f}' for m, v in enumerate(curve)))
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + '.npz', sim=sim, curve=curve,
                        blocks=np.arange(blocks + 1), task=str(config.task),
                        run=name, n_codes=M, sampling=sampling)
    print('wrote', str(out) + '.npz')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n_codes', type=int, default=1024)
    p.add_argument('--sampling', choices=['uniform', 'encoded'],
                   default='uniform')
    p.add_argument('--n_envs', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    run(a.run_dir, a.out, a.n_codes, a.n_envs, seed=a.seed,
        sampling=a.sampling)


if __name__ == '__main__':
    main()
