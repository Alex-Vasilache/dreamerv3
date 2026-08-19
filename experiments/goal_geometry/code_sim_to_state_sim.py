#!/usr/bin/env python3
"""Code similarity -> goal-state similarity, on the code's own 9 bins.

The measurement the paragraph describes: sample ``n_codes`` hard codes, pair
each with a partner at every one of the L+1 = 9 similarities the code admits
(0 blocks different through all 8 different), decode both, and record the
``cosine_max`` similarity of the generated goal states -- the quantity the
low-level policy is actually rewarded on.

Codes are sampled by encoding real world-model states from a live rollout, so
they are codes the manager would actually emit, not uniform draws over the
8^8 grid. Partners are then *constructed* at each exact Hamming distance,
because uniform pairs land almost entirely in the 7- and 8-block bins and would
leave the near end of the axis empty.

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


def run(run_dir, out_path, n_codes=1024, n_draws=8, n_envs=4, stride=8, seed=0):
    config, agent = build(run_dir)
    impl, encode, decode = make_fns(config, agent)
    blocks, classes = [int(x) for x in config.agent.skill_shape]
    name = pathlib.Path(str(run_dir).rstrip('/')).name
    print(f'=== {name} | impl={impl} L={blocks} C={classes} | task={config.task}')

    deters = collect_states(config, agent, n_envs, stride, n_codes)
    deters = deters[:n_codes]
    _, _, _, onehot, _ = encode(deters)
    onehot = np.asarray(onehot)
    ref_ids = onehot.argmax(-1)
    ref_goal = decode(onehot)
    print(f'{len(onehot)} hard codes, goal dim {ref_goal.shape[-1]}')

    rng = np.random.default_rng(seed)
    M = len(onehot)
    sim = np.full((M, blocks + 1), np.nan)
    sim[:, 0] = 1.0                     # identical codes, by construction
    for m in range(1, blocks + 1):
        mod = np.repeat(onehot, n_draws, axis=0)
        ids_rep = np.repeat(ref_ids, n_draws, axis=0)
        for i in range(len(mod)):
            for b in rng.choice(blocks, size=m, replace=False):
                c = int(rng.integers(classes - 1))
                c += int(c >= ids_rep[i, b])   # uniform over the other classes
                mod[i, b, :] = 0.0
                mod[i, b, c] = 1.0
        assert (mod.argmax(-1) != ids_rep).sum(-1).min() == m
        s = cosine_max(decode(mod), np.repeat(ref_goal, n_draws, axis=0))
        sim[:, m] = s.reshape(M, n_draws).mean(-1)

    curve = sim.mean(0)
    print('  blocks different -> goal-state cosine_max')
    print('   ' + '  '.join(f'{m}:{v:.4f}' for m, v in enumerate(curve)))
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + '.npz', sim=sim, curve=curve,
                        blocks=np.arange(blocks + 1), task=str(config.task),
                        run=name, n_codes=M, n_draws=n_draws)
    print('wrote', str(out) + '.npz')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n_codes', type=int, default=1024)
    p.add_argument('--n_draws', type=int, default=8)
    p.add_argument('--n_envs', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    run(a.run_dir, a.out, a.n_codes, a.n_draws, a.n_envs, seed=a.seed)


if __name__ == '__main__':
    main()
